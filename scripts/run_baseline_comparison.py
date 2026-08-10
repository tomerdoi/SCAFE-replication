#!/usr/bin/env python3
"""Run the SCAFE-only external baseline comparison.

Example:
    python3 scripts/run_baseline_comparison.py

The script consumes the already-scored unified SCAFE CSV.  It never invokes
any external vulnerability detector or model API.
"""

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
    run_dataset_cv,
    run_transfer,
    source_feature_columns,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"

DATASETS = {
    "DAppSCAN": DatasetSpec("DAppSCAN", group_by_project=True),
    "DIVE": DatasetSpec("DIVE"),
    "Empirical-Analysis": DatasetSpec("Empirical-Analysis"),
    "FORGE": DatasetSpec("FORGE", accepted_labels=("unknown", "vulnerable")),
    "LISA-Bench": DatasetSpec("LISA-Bench", accepted_labels=("vulnerable",)),
    "AccessControlVulnerabilities": DatasetSpec(
        "AccessControlVulnerabilities", accepted_labels=("vulnerable",)
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["rf", "et", "hgb", "xgb"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "DAppSCAN",
            "DIVE",
            "Empirical-Analysis",
            "FORGE",
            "LISA-Bench",
            "AccessControlVulnerabilities",
        ],
        choices=sorted(DATASETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    features = source_feature_columns(frame)
    specs = [DATASETS[name] for name in args.datasets]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in specs:
        rows.extend(
            run_dataset_cv(
                frame,
                spec,
                features,
                model_names=args.models,
                n_splits=args.folds,
            )
        )

    source = DATASETS["DAppSCAN"]
    if source.name in args.datasets:
        for target in specs:
            if target.name == source.name:
                continue
            rows.extend(
                run_transfer(
                    frame,
                    source,
                    target,
                    features,
                    model_names=args.models,
                )
            )

    result = pd.DataFrame(rows)
    csv_path = args.output_dir / "scafe_only_metrics.csv"
    json_path = args.output_dir / "scafe_only_metrics.json"
    result.to_csv(csv_path, index=False)
    metadata = {
        "input": str(args.input),
        "feature_count": len(features),
        "features": features,
        "models": args.models,
        "folds": args.folds,
        "protocol": {
            "source_only": True,
            "grouped_cv": True,
            "transfer_train": "DAppSCAN",
            "decision_threshold": 0.5,
            "test_threshold_tuning": False,
            "third_party_tools_run": False,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(metadata, indent=2, allow_nan=True))
    print(f"Wrote {csv_path} ({len(result)} rows)")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
