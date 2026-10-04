#!/usr/bin/env python3
"""Run repeated, duplicate-safe SCAFE LODO with confidence intervals."""

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
    run_repeated_leave_one_dataset_out,
    source_feature_columns,
    summarize_repeated_lodo,
)
from src.ml.semantic_features import add_semantic_features_to_frame

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"
)
DATASET_SPECS = [
    DatasetSpec("DAppSCAN", group_by_project=True),
    DatasetSpec("DIVE"),
    DatasetSpec("Empirical-Analysis"),
    DatasetSpec("FORGE", accepted_labels=("unknown", "vulnerable")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--models", nargs="+", default=["rf", "et", "hgb", "xgb"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DAppSCAN", "DIVE", "Empirical-Analysis"],
    )
    parser.add_argument(
        "--train-threshold-sensitivity",
        action="store_true",
        help="Optimize an F1 threshold on a training-only split (secondary sensitivity only).",
    )
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Add the 87 source-only identifier/context features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    if args.include_semantic:
        frame = add_semantic_features_to_frame(frame)
    features = source_feature_columns(frame)
    specs = [spec for spec in DATASET_SPECS if spec.name in args.datasets]
    if len(specs) < 2:
        raise ValueError("At least two datasets are required for LODO")
    metrics, predictions = run_repeated_leave_one_dataset_out(
        frame,
        specs,
        features,
        seeds=args.seeds,
        model_names=args.models,
        calibrate_threshold=args.train_threshold_sensitivity,
    )
    metrics_frame = pd.DataFrame(metrics)
    summary = summarize_repeated_lodo(metrics_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "scafe_repeated_lodo.csv"
    summary_path = args.output_dir / "scafe_repeated_lodo_summary.csv"
    predictions_path = args.output_dir / "scafe_repeated_lodo_predictions.csv"
    json_path = args.output_dir / "scafe_repeated_lodo.json"
    metrics_frame.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "datasets": [spec.name for spec in specs],
                "models": args.models,
                "seeds": args.seeds,
                "feature_count": len(features),
                "feature_names": features,
                "canonical_counts": {
                    str(dataset): int(group["n_samples"].iloc[0])
                    for dataset, group in metrics_frame.groupby("dataset", sort=False)
                },
                "protocol": {
                    "source_only": True,
                    "identifier_context_extension": args.include_semantic,
                    "canonicalize_within_dataset": True,
                    "exclude_within_dataset_label_conflicts": True,
                    "held_out_dataset": True,
                    "exclude_cross_dataset_source_duplicates": True,
                    "exclude_cross_training_label_conflicts": True,
                    "collapse_cross_training_same_label_duplicates": True,
                    "threshold": (
                        "train_only_f1"
                        if args.train_threshold_sensitivity
                        else "fixed_0.5"
                    ),
                    "test_threshold_tuning": False,
                    "primary_model": "rf_selected_by_dappscan_project_group_cv",
                    "development_dataset": "DAppSCAN",
                    "external_targets": ["DIVE", "Empirical-Analysis"],
                    "uncertainty": "seed_mean_and_standard_deviation",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
