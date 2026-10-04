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

from src.ml.baseline_comparison import (
    DatasetSpec,
    load_scored_frame,
    run_leave_one_dataset_out,
    source_feature_columns,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"
)
DATASET_SPECS = [
    DatasetSpec("DAppSCAN"),
    DatasetSpec("DIVE"),
    DatasetSpec("Empirical-Analysis"),
]


def _variants(all_features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "relational": [],
        "complexity_distribution": [],
        "lexical_statistics": [],
        "security_patterns": [],
        "syntax_organization": [],
    }
    complexity_tokens = (
        "cyclomatic", "halstead", "_loc", "loc_", "nesting", "concentration",
        "gini", "skew", "iqr", "_p75", "_p90", "params", "fanout",
        "inheritance",
    )
    lexical_tokens = ("token", "bigram", "comment", "duplication", "duplicated")
    security_tokens = (
        "dangerous", "safety", "risky", "unprotected", "guard", "owner",
        "delegatecall", "external_call", "unchecked", "reentrancy", "oracle",
        "proxy", "interaction_surface", "accesscontrol", "safe_", "timelock",
        "selfdestruct", "tx_origin",
    )
    for feature in all_features:
        if feature.startswith(("ast_", "cg_", "df_")):
            group = "relational"
        elif any(token in feature for token in complexity_tokens):
            group = "complexity_distribution"
        elif any(token in feature for token in lexical_tokens):
            group = "lexical_statistics"
        elif (
            feature.startswith(("has_", "uses_", "vt_", "sp_"))
            or any(token in feature for token in security_tokens)
        ):
            group = "security_patterns"
        else:
            group = "syntax_organization"
        groups[group].append(feature)

    variants = {
        f"raw_structural_{len(all_features)}": all_features,
        "ast_call_dataflow_only": sorted(groups["relational"]),
        "without_ast_call_dataflow": [
            feature for feature in all_features if feature not in groups["relational"]
        ],
    }
    for group, features in groups.items():
        if group == "relational":
            continue
        variants[f"without_{group}"] = [
            feature for feature in all_features if feature not in features
        ]
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["rf"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    variants = _variants(source_feature_columns(frame))
    all_metrics: list[dict] = []
    for variant, features in variants.items():
        if not features:
            continue
        metrics, _ = run_leave_one_dataset_out(
            frame,
            DATASET_SPECS,
            features,
            model_names=args.models,
            calibrate_threshold=False,
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
                    "exclude_cross_training_label_conflicts": True,
                    "collapse_cross_training_same_label_duplicates": True,
                    "threshold": "fixed_0.5",
                    "model_selection": "fixed_rf_selected_on_dappscan_development",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
