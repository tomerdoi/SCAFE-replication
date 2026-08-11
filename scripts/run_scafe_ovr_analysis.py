#!/usr/bin/env python3
"""Run vulnerability-specific one-vs-rest SCAFE heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.baseline_comparison import (
    CANONICAL_VULNERABILITY_TYPES,
    DatasetSpec,
    add_canonical_vulnerability_types,
    load_scored_frame,
    run_leave_one_dataset_out_ovr,
    source_feature_columns,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"
)
DATASET_SPECS = {
    "DAppSCAN": DatasetSpec("DAppSCAN", group_by_project=True),
    "DIVE": DatasetSpec("DIVE"),
    "Empirical-Analysis": DatasetSpec("Empirical-Analysis"),
    "FORGE": DatasetSpec("FORGE", accepted_labels=("unknown", "vulnerable")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(CANONICAL_VULNERABILITY_TYPES),
        default=[t for t in CANONICAL_VULNERABILITY_TYPES if t != "other"],
    )
    parser.add_argument("--models", nargs="+", default=["rf", "et", "hgb", "xgb"])
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Keep the historical fixed 0.5 threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    specs = [DATASET_SPECS[name] for name in args.datasets]
    features = source_feature_columns(frame)
    metrics, predictions = run_leave_one_dataset_out_ovr(
        frame,
        specs,
        features,
        vulnerability_types=args.types,
        model_names=args.models,
        calibrate_threshold=not args.no_calibration,
    )
    normalized = add_canonical_vulnerability_types(frame)
    taxonomy_counts = {}
    for vulnerability_type in args.types:
        taxonomy_counts[vulnerability_type] = int(
            normalized["taxonomy_types"]
            .fillna("")
            .map(lambda value: vulnerability_type in str(value).split(";"))
            .sum()
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "scafe_ovr_leave_one_dataset_out.csv"
    predictions_path = args.output_dir / "scafe_ovr_leave_one_dataset_out_predictions.csv"
    json_path = args.output_dir / "scafe_ovr_leave_one_dataset_out.json"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "types": args.types,
                "models": args.models,
                "feature_count": len(features),
                "taxonomy_counts": taxonomy_counts,
                "protocol": {
                    "source_only": True,
                    "held_out_dataset": True,
                    "one_vs_rest": True,
                    "exclude_cross_dataset_source_duplicates": True,
                    "threshold": (
                        "fixed_0.5" if args.no_calibration else "train_only_f1"
                    ),
                    "test_threshold_tuning": False,
                },
                "metrics": metrics,
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
