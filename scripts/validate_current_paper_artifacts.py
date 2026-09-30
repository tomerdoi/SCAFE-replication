#!/usr/bin/env python3
"""Validate current manuscript claims against current publication artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.baseline_comparison import source_feature_columns


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / "sc_risk_score_overleaf"
TEX = (PAPER / "sc_vulnerability_detection.tex").read_text(encoding="utf-8")
ERRORS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


corrected = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "source_only_raw_canonical/scafe_repeated_lodo.json").read_text()
)
fixed_summary = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo_summary.csv"
)
predictions = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo_predictions.csv"
)
scored = pd.read_csv(REPO / "data/derived/unified_contract_eval_scored.csv", low_memory=False)
source_columns = source_feature_columns(scored)
shap_metadata = json.loads(
    (REPO / "data/derived/shap_results/fold_consistent_primary/metadata.json").read_text()
)
parser_audit = json.loads(
    (REPO / "data/derived/parser_compatibility_audit/"
     "parser_compatibility_summary.json").read_text()
)

require(corrected["feature_count"] == 244, "corrected transfer feature count is not 244")
require(corrected["seeds"] == [42, 43, 44, 45, 46], "corrected LODO seed list is incomplete")
require(corrected["protocol"]["threshold"] == "fixed_0.5", "corrected LODO is not fixed-threshold")
require(corrected["protocol"]["development_dataset"] == "DAppSCAN", "development dataset is not recorded")
require(corrected["protocol"]["external_targets"] == ["DIVE", "Empirical-Analysis"],
        "external targets are not recorded")
require(len(fixed_summary) == 12, "corrected fixed summary does not contain 12 model/dataset rows")
require(not {"has_verified_source", "is_solc_pre_08"}.intersection(source_columns),
        "metadata feature entered source feature columns")
require(len(source_columns) == 244, "source feature selector does not return 244 raw features")
require(not any(column.startswith("comp_") for column in source_columns),
        "label-informed component entered source feature columns")
require(not {"has_payload", "is_proxy"}.intersection(source_columns),
        "provenance metadata entered source feature columns")
require("244 evaluated raw features" in TEX, "manuscript does not state 244 evaluated raw features")
require("0.594 on DIVE" in TEX, "manuscript lacks corrected DIVE result")
require("0.589 on Empirical-Analysis" in TEX, "manuscript lacks corrected Empirical result")
counts = (
    predictions[predictions["model"] == "rf"]
    .groupby(["dataset", "random_state"]).size()
    .groupby("dataset").first().to_dict()
)
require(counts == {"DAppSCAN": 14791, "DIVE": 21939, "Empirical-Analysis": 2172},
        f"canonical prediction counts differ: {counts}")
require(shap_metadata["model"] == "rf", "SHAP cases do not explain the fixed RF head")
require(shap_metadata["feature_count"] == 244, "SHAP feature count is not 244")
rates = {row["dataset"]: row["parse_success_rate"] for row in parser_audit["summary"]}
require(rates == {"DAppSCAN": 0.954, "DIVE": 0.652, "Empirical-Analysis": 0.926},
        f"parser audit rates differ: {rates}")
require((PAPER / "sc_vulnerability_detection.pdf").exists(), "compiled manuscript PDF is missing")
for figure in (
    "fig_model_comparison.pdf", "fig_confusion_matrices.pdf",
    "fig_feature_ablation.pdf", "fig_shap_cases.pdf"
):
    require((PAPER / "figures" / figure).exists(), f"corrected figure missing: {figure}")

if ERRORS:
    for error in ERRORS:
        print(f"FAIL: {error}")
    raise SystemExit(1)

print("PASS: current manuscript and corrected SCAFE artifacts are consistent")
