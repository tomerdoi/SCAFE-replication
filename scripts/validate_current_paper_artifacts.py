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


manifest = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "scafe_experiment_manifest_source_only.json").read_text()
)
corrected = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "source_only_metadata_excluded/scafe_repeated_lodo.json").read_text()
)
fixed_summary = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_metadata_excluded/scafe_repeated_lodo_summary.csv"
)
calibrated_summary = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_metadata_excluded_calibrated/scafe_repeated_lodo_summary.csv"
)
scored = pd.read_csv(REPO / "data/derived/unified_contract_eval_scored.csv", nrows=1)
source_columns = source_feature_columns(scored)

require(manifest["feature_manifest"]["evaluated_feature_count"] == 255, "manifest feature count is not 255")
require(corrected["feature_count"] == 255, "corrected LODO feature count is not 255")
require(corrected["seeds"] == [42, 43, 44, 45, 46], "corrected LODO seed list is incomplete")
require(corrected["protocol"]["threshold"] == "fixed_0.5", "corrected LODO is not fixed-threshold")
require(len(fixed_summary) == 12, "corrected fixed summary does not contain 12 model/dataset rows")
require(len(calibrated_summary) == 12, "corrected calibrated summary does not contain 12 model/dataset rows")
require(not {"has_verified_source", "is_solc_pre_08"}.intersection(source_columns),
        "metadata feature entered source feature columns")
require("255 evaluated features" in TEX, "manuscript does not state 255 evaluated features")
require("Structural core & 246" in TEX, "feature inventory does not state 246 structural core features")
require("Structural core & 248" not in TEX, "stale 248 structural-core count remains")
require("best five-seed means of 0.825" in TEX, "manuscript lacks corrected DAppSCAN result")
require("0.627 on DIVE" in TEX, "manuscript lacks corrected DIVE result")
require("0.572 on Empirical-Analysis" in TEX, "manuscript lacks corrected Empirical-Analysis result")
require((PAPER / "sc_vulnerability_detection.pdf").exists(), "compiled manuscript PDF is missing")
for figure in ("fig_model_comparison.pdf", "fig_confusion_matrices.pdf", "fig_shap_cases.pdf"):
    require((PAPER / "figures" / figure).exists(), f"corrected figure missing: {figure}")
require(
    (REPO / "data/derived/shap_results/fold_consistent_primary/metadata.json").exists(),
    "fold-consistent SHAP metadata is missing",
)

if ERRORS:
    for error in ERRORS:
        print(f"FAIL: {error}")
    raise SystemExit(1)

print("PASS: current manuscript and corrected SCAFE artifacts are consistent")
