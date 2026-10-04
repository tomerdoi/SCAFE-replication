#!/usr/bin/env python3
"""Validate publication-critical properties of generated SCAFE artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"


def _load(name: str) -> object:
    path = OUTPUT / name
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")
    return json.loads(path.read_text())


def main() -> None:
    primary = OUTPUT / "source_only_raw_canonical"
    protocol_record = json.loads(
        (primary / "scafe_repeated_lodo.json").read_text()
    )
    protocol = protocol_record["protocol"]
    assert protocol_record["feature_count"] == 244
    assert protocol["canonicalize_within_dataset"] is True
    assert protocol["exclude_within_dataset_label_conflicts"] is True
    assert protocol["held_out_dataset"] is True
    assert protocol["exclude_cross_dataset_source_duplicates"] is True
    assert protocol["exclude_cross_training_label_conflicts"] is True
    assert protocol["collapse_cross_training_same_label_duplicates"] is True
    assert protocol["test_threshold_tuning"] is False
    assert protocol["primary_model"] == "rf_selected_by_dappscan_project_group_cv"
    assert protocol["external_targets"] == ["DIVE", "Empirical-Analysis"]

    metrics = pd.read_csv(primary / "scafe_repeated_lodo.csv")
    assert not metrics.empty
    assert (metrics["n_duplicate_train_excluded"] >= 0).all()
    assert metrics["dataset"].nunique() == 3
    assert metrics["model"].nunique() == 4
    assert metrics["random_state"].nunique() == 5
    summary = pd.read_csv(primary / "scafe_repeated_lodo_summary.csv")
    assert (summary["n_seeds"] == 5).all()

    ablations = pd.read_csv(
        OUTPUT / "source_only_raw_canonical_ablations/scafe_feature_ablations.csv"
    )
    assert set(ablations["feature_variant"]) == {
        "raw_structural_244",
        "ast_call_dataflow_only",
        "without_ast_call_dataflow",
        "without_complexity_distribution",
        "without_lexical_statistics",
        "without_security_patterns",
        "without_syntax_organization",
    }

    print(
        "Integrity checks passed: "
        f"{len(metrics)} transfer rows, "
        f"{protocol_record['feature_count']} raw source features, "
        f"{metrics['random_state'].nunique()} repeated seeds, "
        f"{len(protocol['external_targets'])} external targets."
    )


if __name__ == "__main__":
    main()
