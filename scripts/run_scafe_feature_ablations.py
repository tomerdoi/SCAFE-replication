#!/usr/bin/env python3
"""Run source-feature-group ablations under duplicate-safe LODO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.experiment_manifest import feature_group_columns
from src.ml.baseline_comparison import (
    DatasetSpec,
    load_scored_frame,
    run_leave_one_dataset_out,
)

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


def _variants(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    all_features = sorted(feature for values in groups.values() for feature in values)
    component = groups.get("structural_components", [])
    graph = [
        feature
        for group, values in groups.items()
        if group in {"structural", "structural_components"}
        for feature in values
        if feature.startswith(("ast_", "cg_"))
    ]
    return {
        "structural_all": all_features,
        "structural_without_components": [
            feature for feature in all_features if feature not in component
        ],
        "structural_components_only": component,
        "ast_call_graph_only": sorted(graph),
        "structural_without_ast_call_graph": [
            feature for feature in all_features if feature not in graph
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["rf", "hgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    groups = feature_group_columns(frame)
    variants = _variants(groups)
    all_metrics: list[dict] = []
    for variant, features in variants.items():
        if not features:
            continue
        metrics, _ = run_leave_one_dataset_out(
            frame,
            DATASET_SPECS,
            features,
            model_names=args.models,
            calibrate_threshold=not args.no_calibration,
            random_state=args.seed,
        )
        for row in metrics:
            row["feature_variant"] = variant
            row["feature_count"] = len(features)
        all_metrics.extend(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scafe_feature_ablations.csv"
    json_path = args.output_dir / "scafe_feature_ablations.json"
    pd.DataFrame(all_metrics).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "variants": {
                    name: {
                        "feature_count": len(features),
                        "features": features,
                    }
                    for name, features in variants.items()
                },
                "models": args.models,
                "seed": args.seed,
                "protocol": {
                    "source_only": True,
                    "held_out_dataset": True,
                    "exclude_cross_dataset_source_duplicates": True,
                    "threshold": (
                        "fixed_0.5" if args.no_calibration else "train_only_f1"
                    ),
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
