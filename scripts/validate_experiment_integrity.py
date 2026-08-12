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
    manifest = _load("scafe_experiment_manifest.json")
    feature_manifest = manifest["feature_manifest"]
    assert feature_manifest["evaluated_feature_count"] == 257
    assert not any(
        item["name"] in feature_manifest["source_only_policy"]["excluded_columns"]
        for item in feature_manifest["evaluated_features"]
    )

    lodo = _load("scafe_leave_one_dataset_out.json")
    protocol = lodo["protocol"]
    assert protocol["held_out_dataset"] is True
    assert protocol["exclude_cross_dataset_source_duplicates"] is True
    assert protocol["test_threshold_tuning"] is False

    metrics = pd.read_csv(OUTPUT / "scafe_leave_one_dataset_out.csv")
    assert not metrics.empty
    assert (metrics["n_duplicate_train_excluded"] >= 0).all()
    assert metrics["dataset"].nunique() == 4
    assert metrics["model"].nunique() >= 2

    audit = pd.read_csv(OUTPUT / "dataset_duplicate_audit.csv")
    conflicts = audit[audit["cross_dataset"] & audit["label_conflict"]]
    assert len(conflicts) == 263

    repeated = pd.read_csv(OUTPUT / "scafe_repeated_lodo_five_seed.csv")
    assert repeated["random_state"].nunique() == 5
    assert repeated["dataset"].nunique() == 4
    summary = pd.read_csv(OUTPUT / "scafe_repeated_lodo_five_seed_summary.csv")
    assert (summary["n_seeds"] == 5).all()

    ablations = pd.read_csv(OUTPUT / "scafe_feature_ablations.csv")
    assert {"structural_all", "structural_components_only"}.issubset(
        set(ablations["feature_variant"])
    )

    gate = _load("gnn_gate_decision.json")
    assert gate["status"] == "deferred"
    assert gate["decision"] == "do_not_add_to_primary_paper"

    strict_dir = OUTPUT / "strict_three_dataset_repeated"
    strict_metrics = pd.read_csv(strict_dir / "scafe_repeated_lodo.csv")
    strict_summary = pd.read_csv(strict_dir / "scafe_repeated_lodo_summary.csv")
    assert strict_metrics["random_state"].nunique() == 5
    assert strict_metrics["dataset"].nunique() == 3
    assert (strict_summary["n_seeds"] == 5).all()

    print(
        "Integrity checks passed: "
        f"{len(metrics)} LODO rows, "
        f"{len(conflicts)} cross-dataset conflict groups, "
        f"{feature_manifest['evaluated_feature_count']} source features, "
        f"{repeated['random_state'].nunique()} repeated seeds, "
        f"{strict_metrics['dataset'].nunique()} strict primary datasets."
    )


if __name__ == "__main__":
    main()
