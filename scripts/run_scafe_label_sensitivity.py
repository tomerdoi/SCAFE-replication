#!/usr/bin/env python3
"""Measure label-policy and duplicate-policy sensitivity under LODO."""

from __future__ import annotations

import argparse
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
)

INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
OUTPUT = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--models", nargs="+", default=["rf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    features = source_feature_columns(frame)
    common = [
        DatasetSpec("DAppSCAN", group_by_project=True),
        DatasetSpec("DIVE"),
        DatasetSpec("Empirical-Analysis"),
    ]
    scenarios = {
        "forge_vulnerable_vs_unknown": DatasetSpec(
            "FORGE", accepted_labels=("unknown", "vulnerable")
        ),
        "forge_vulnerable_vs_unknown_suspicious": DatasetSpec(
            "FORGE", accepted_labels=("suspicious", "unknown", "vulnerable")
        ),
    }
    rows: list[dict] = []
    for label_policy, forge_spec in scenarios.items():
        specs = [*common, forge_spec]
        for duplicate_policy in (True, False):
            metrics, _ = run_leave_one_dataset_out(
                frame,
                specs,
                features,
                model_names=args.models,
                calibrate_threshold=not args.no_calibration,
                exclude_duplicate_sources=duplicate_policy,
                random_state=args.seed,
            )
            for result in metrics:
                result["label_policy"] = label_policy
                result["exclude_duplicate_sources"] = duplicate_policy
                rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "scafe_label_sensitivity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
