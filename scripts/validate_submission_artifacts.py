#!/usr/bin/env python3
"""Validate submission-critical manuscript claims against sanitized artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SANITIZED = REPO / "data" / "derived" / "sanitized_v1"
RESULTS = SANITIZED / "ml_results"
PAPER = REPO.parent / "sc_risk_score_overleaf"
TEX = (PAPER / "sc_vulnerability_detection.tex").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> bool:
    return math.isclose(actual, expected, abs_tol=tolerance)


def main() -> None:
    manifest = json.loads((SANITIZED / "manifest.json").read_text())
    primary_dir = RESULTS / "primary_lodo"
    primary_protocol = json.loads(
        (primary_dir / "scafe_repeated_lodo.json").read_text()
    )
    primary = pd.read_csv(primary_dir / "scafe_repeated_lodo_summary.csv")
    metrics = pd.read_csv(primary_dir / "scafe_repeated_lodo.csv")
    bootstrap = pd.read_csv(primary_dir / "primary_record_bootstrap.csv")
    development = pd.read_csv(
        RESULTS / "primary_development/scafe_within_corpus_summary.csv"
    )
    nuisance = pd.read_csv(RESULTS / "nuisance_baseline/summary.csv")
    ablations = pd.read_csv(
        RESULTS / "feature_ablations/scafe_feature_ablations.csv"
    )
    lexical = pd.read_csv(
        RESULTS / "lexical_extension/scafe_repeated_lodo_summary.csv"
    )
    ensemble_lock = json.loads(
        (RESULTS / "sampled_ensemble/ensemble_lock.json").read_text()
    )
    sampled_ensemble = pd.read_csv(
        RESULTS / "sampled_ensemble/ensemble_metrics.csv"
    )
    shap_metadata = json.loads(
        (SANITIZED / "shap_results/primary/metadata.json").read_text()
    )

    require(
        primary_protocol["input_sha256"] == manifest["scored_csv_sha256"],
        "primary LODO input does not match the sanitized table",
    )
    require(primary_protocol["feature_count"] == 244, "primary feature count")
    require(
        primary_protocol["protocol"]["primary_model"]
        == "et_selected_by_sanitized_dappscan_project_group_cv",
        "primary head metadata",
    )
    require(metrics["model"].unique().tolist() == ["et"], "primary model rows")
    require(metrics["random_state"].nunique() == 5, "primary seed coverage")
    require(len(metrics) == 15, "primary metric row count")

    selected = development.sort_values(
        ["roc_auc_mean", "pr_auc_mean"],
        ascending=False,
    ).iloc[0]
    require(selected["model"] == "et", "sanitized development winner")
    require(close(selected["roc_auc_mean"], 0.8646764595), "development AUC")

    indexed = primary.set_index("dataset")
    require(close(indexed.loc["DIVE", "roc_auc_mean"], 0.5230929174), "DIVE AUC")
    require(
        close(
            indexed.loc["Empirical-Analysis", "roc_auc_mean"],
            0.5816122948,
        ),
        "Empirical AUC",
    )
    require((primary["n_seeds"] == 5).all(), "primary summary seed coverage")

    boot = bootstrap.set_index("dataset")
    require(close(boot.loc["DIVE", "roc_auc_bootstrap95_low"], 0.5106841660),
            "DIVE bootstrap low")
    require(close(boot.loc["DIVE", "roc_auc_bootstrap95_high"], 0.5348643490),
            "DIVE bootstrap high")
    require(
        close(
            boot.loc["Empirical-Analysis", "roc_auc_bootstrap95_low"],
            0.5569649118,
        ),
        "Empirical bootstrap low",
    )

    nuisance_index = nuisance.set_index("dataset")
    require(close(nuisance_index.loc["DIVE", "roc_auc_mean"], 0.5420272893),
            "DIVE nuisance AUC")
    require(
        close(
            nuisance_index.loc["Empirical-Analysis", "roc_auc_mean"],
            0.4746763027,
        ),
        "Empirical nuisance AUC",
    )
    require(set(ablations["model"]) == {"et"}, "ablation head")
    require(ablations["feature_variant"].nunique() == 7, "ablation variants")
    require(set(lexical["model"]) == {"et"}, "lexical-extension head")
    require(ensemble_lock["scafe_model"] == "et", "sampled ensemble head")
    require(
        set(sampled_ensemble["dataset"]) == {"DIVE", "Empirical-Analysis"},
        "sampled ensemble targets",
    )
    require(shap_metadata["model"] == "et", "SHAP head")
    require(shap_metadata["feature_count"] == 244, "SHAP feature count")

    for phrase in (
        "0.523 on DIVE",
        "0.582 on Empirical-Analysis",
        "size/formatting",
        "Extra Trees",
        "conditional iid-record",
    ):
        require(phrase in TEX, f"manuscript missing current claim: {phrase}")
    for stale in (
        "0.594 on DIVE",
        "0.591 on Empirical-Analysis",
        "fixed Random Forest head",
    ):
        require(stale not in TEX, f"stale manuscript claim remains: {stale}")
    for figure in (
        "fig_feature_ablation.pdf",
        "fig_sampled_model_comparison.png",
        "fig_shap_cases.pdf",
    ):
        require((PAPER / "figures" / figure).exists(), f"missing figure: {figure}")

    print(
        "PASS: sanitized lineage, primary metrics, controls, sampled artifacts, "
        "SHAP cases, and manuscript claims are consistent"
    )


if __name__ == "__main__":
    main()
