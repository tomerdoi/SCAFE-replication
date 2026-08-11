#!/usr/bin/env python3
"""Run leave-one-dataset-out SCAFE evaluation and error analysis."""

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
    DatasetSpec,
    load_scored_frame,
    run_leave_one_dataset_out,
    source_feature_columns,
    vulnerability_error_analysis,
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
    metrics, predictions = run_leave_one_dataset_out(
        frame,
        specs,
        features,
        model_names=args.models,
        calibrate_threshold=not args.no_calibration,
    )
    errors = vulnerability_error_analysis(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "scafe_leave_one_dataset_out.csv"
    errors_path = args.output_dir / "scafe_leave_one_dataset_out_errors.csv"
    predictions_path = args.output_dir / "scafe_leave_one_dataset_out_predictions.csv"
    json_path = args.output_dir / "scafe_leave_one_dataset_out.json"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    errors.to_csv(errors_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "models": args.models,
                "feature_count": len(features),
                "protocol": {
                    "source_only": True,
                    "held_out_dataset": True,
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
    print(f"Wrote {errors_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
