#!/usr/bin/env python3
"""Evaluate SCAFE structural features with source-only semantic context."""

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
    run_leave_one_dataset_out_ovr,
    source_feature_columns,
)
from src.ml.semantic_features import add_semantic_features_to_frame

INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
OUTPUT = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"
SPECS = [
    DatasetSpec("DAppSCAN", group_by_project=True),
    DatasetSpec("DIVE"),
    DatasetSpec("Empirical-Analysis"),
    DatasetSpec("FORGE", accepted_labels=("unknown", "vulnerable")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--models", nargs="+", default=["rf", "et", "hgb", "xgb"])
    parser.add_argument("--ovr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = add_semantic_features_to_frame(
        load_scored_frame(str(args.input))
    )
    features = source_feature_columns(frame)
    if args.ovr:
        metrics, predictions = run_leave_one_dataset_out_ovr(
            frame,
            SPECS,
            features,
            model_names=args.models,
            calibrate_threshold=True,
        )
        stem = "scafe_semantic_ovr_leave_one_dataset_out"
    else:
        metrics, predictions = run_leave_one_dataset_out(
            frame,
            SPECS,
            features,
            model_names=args.models,
            calibrate_threshold=True,
        )
        stem = "scafe_semantic_leave_one_dataset_out"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / f"{stem}.csv", index=False)
    predictions.to_csv(
        args.output_dir / f"{stem}_predictions.csv", index=False
    )
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "datasets": [spec.name for spec in SPECS],
                "models": args.models,
                "feature_count": len(features),
                "semantic_features": [
                    feature for feature in features if feature.startswith("semantic_")
                ],
                "protocol": {
                    "source_only": True,
                    "held_out_dataset": True,
                    "exclude_cross_dataset_source_duplicates": True,
                    "identifier_semantics": True,
                    "threshold": "train_only_f1",
                    "test_threshold_tuning": False,
                    "third_party_tools_run": False,
                },
                "metrics": metrics,
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Wrote {args.output_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
