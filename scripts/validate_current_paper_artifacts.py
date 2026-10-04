#!/usr/bin/env python3
"""Validate current manuscript claims against current publication artifacts."""

from __future__ import annotations

import json
import math
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
fixed_metrics = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo.csv"
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
parser_strata = json.loads(
    (REPO / "data/derived/parser_compatibility_audit/"
     "parser_stratified_performance.json").read_text()
)
feature_audit = json.loads(
    (REPO / "data/derived/feature_representation_audit/summary.json").read_text()
)
manifest = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "scafe_experiment_manifest.json").read_text()
)
within_summary = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_within_corpus/scafe_within_corpus_summary.csv"
)
within_metadata = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "source_only_within_corpus/scafe_within_corpus_metadata.json").read_text()
)
lexical_protocol = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "source_only_raw_plus_lexical_canonical/scafe_repeated_lodo.json").read_text()
)
lexical_summary = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_plus_lexical_canonical/scafe_repeated_lodo_summary.csv"
)
ablation_protocol = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "source_only_raw_canonical_ablations/scafe_feature_ablations.json").read_text()
)
ablation_metrics = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical_ablations/scafe_feature_ablations.csv"
)
label_sensitivity = pd.read_csv(
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical_label_sensitivity/scafe_label_sensitivity.csv"
)
exact_benchmarks = json.loads(
    (REPO / "data/derived/ml_results/baseline_comparison/"
     "exact_paper_benchmark_metrics.json").read_text()
)
slither_summary_path = REPO / "data/derived/slither_reliability/summary.json"
slither_summary_text = slither_summary_path.read_text()
slither_summary = json.loads(slither_summary_text)
slither_constructs = pd.read_csv(
    REPO / "data/derived/slither_reliability/by_source_construct.csv"
)
slither_correlations = pd.read_csv(
    REPO / "data/derived/slither_correlation/correlation_results.csv"
)

require(corrected["feature_count"] == 244, "corrected transfer feature count is not 244")
require(corrected["seeds"] == [42, 43, 44, 45, 46], "corrected LODO seed list is incomplete")
require(corrected["protocol"]["threshold"] == "fixed_0.5", "corrected LODO is not fixed-threshold")
require(corrected["protocol"]["development_dataset"] == "DAppSCAN", "development dataset is not recorded")
require(corrected["protocol"]["external_targets"] == ["DIVE", "Empirical-Analysis"],
        "external targets are not recorded")
require(corrected["protocol"]["exclude_cross_training_label_conflicts"] is True,
        "cross-training label conflicts are not excluded")
require(corrected["protocol"]["collapse_cross_training_same_label_duplicates"] is True,
        "cross-training same-label duplicates are not collapsed")
require(len(fixed_summary) == 12, "corrected fixed summary does not contain 12 model/dataset rows")
require({
    "n_cross_training_duplicates_collapsed",
    "n_cross_training_conflict_rows_excluded",
}.issubset(fixed_metrics.columns), "cross-training canonicalization counts are absent")
primary_rf = fixed_summary[fixed_summary["model"] == "rf"].set_index("dataset")
require(math.isclose(primary_rf.loc["DIVE", "roc_auc_mean"], 0.5935472750),
        "DIVE primary RF result changed")
require(math.isclose(
    primary_rf.loc["Empirical-Analysis", "roc_auc_mean"], 0.5909068430
), "Empirical primary RF result changed")
require(len(source_columns) == 244, "source feature selector does not return 244 raw features")
require(not any(column.startswith("comp_") for column in source_columns),
        "label-informed component entered source feature columns")
require(
    not {"has_payload", "has_verified_source", "is_proxy", "is_solc_pre_08"}
    .intersection(source_columns),
    "one of four provenance fields entered the source feature matrix",
)
require("244 evaluated raw features" in TEX, "manuscript does not state 244 evaluated raw features")
require("0.594 on DIVE" in TEX, "manuscript lacks corrected DIVE result")
require("0.591 on Empirical-Analysis" in TEX, "manuscript lacks corrected Empirical result")
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
require(parser_audit["dataset_seeds"] == {
    "DAppSCAN": 42, "DIVE": 43, "Empirical-Analysis": 44
}, "parser audit does not record per-dataset sampling seeds")
require(parser_strata["matched_records"] == 1442,
        "parser-stratified prediction audit does not contain 1,442 records")
require(feature_audit["evaluated_feature_count"] == 244,
        "feature-representation audit is not based on 244 features")
require(feature_audit["registry_rows"] == 56960,
        "feature-representation audit registry count changed")
require(feature_audit["source_available_rows"] == 56943,
        "feature-representation audit does not exclude 17 source-missing rows")
require(feature_audit["high_correlation_pair_count"] == 80,
        "feature-representation audit does not contain 80 high-correlation pairs")
require(math.isclose(feature_audit["pairwise_absolute_spearman"]["median"], 0.2604083345),
        "feature-correlation median changed")
require(math.isclose(feature_audit["pairwise_absolute_spearman"]["p95"], 0.7256539189),
        "feature-correlation p95 changed")
require(manifest["feature_manifest"]["evaluated_feature_count"] == 244,
        "experiment manifest is stale")
require(manifest["feature_manifest"]["excluded_numeric_count"] == 57,
        "experiment manifest numeric-exclusion count is not 57")
require(manifest["protocol"]["threshold"] == "fixed_0.5",
        "experiment manifest does not record the fixed cutoff")
rf_development = within_summary[
    (within_summary["dataset"] == "DAppSCAN") & (within_summary["model"] == "rf")
].iloc[0]
require(within_metadata["feature_count"] == 244,
        "DAppSCAN model-selection metadata is not based on 244 features")
require(math.isclose(rf_development["roc_auc_mean"], 0.8662907429),
        "DAppSCAN model-selection result changed")
require("0.866 ROC-AUC" in TEX, "manuscript lacks current model-selection result")
require(lexical_protocol["feature_count"] == 331,
        "lexical/context comparison does not use 331 features")
require(lexical_protocol["protocol"]["identifier_context_extension"] is True,
        "lexical/context protocol flag is absent")
require(lexical_protocol["protocol"]["exclude_cross_training_label_conflicts"] is True,
        "lexical/context run retains cross-training label conflicts")
lexical_roc = lexical_summary.set_index("dataset")["roc_auc_mean"].to_dict()
require(math.isclose(lexical_roc["DIVE"], 0.5861150791),
        "DIVE lexical/context result changed")
require(math.isclose(lexical_roc["Empirical-Analysis"], 0.5939294487),
        "Empirical lexical/context result changed")
ablation_counts = {
    variant: details["feature_count"]
    for variant, details in ablation_protocol["variants"].items()
}
require(ablation_counts == {
    "raw_structural_244": 244,
    "ast_call_dataflow_only": 17,
    "without_ast_call_dataflow": 227,
    "without_complexity_distribution": 210,
    "without_lexical_statistics": 231,
    "without_security_patterns": 181,
    "without_syntax_organization": 127,
}, f"feature-ablation variants changed: {ablation_counts}")
full_ablation_set = set(ablation_protocol["variants"]["raw_structural_244"]["features"])
removed_groups = [
    full_ablation_set - set(details["features"])
    for name, details in ablation_protocol["variants"].items()
    if name.startswith("without_")
]
require(
    set().union(*removed_groups) == full_ablation_set
    and sum(map(len, removed_groups)) == len(full_ablation_set),
    "major ablation groups do not form a disjoint partition of the 244 features",
)
ablation_roc = ablation_metrics.set_index(["dataset", "feature_variant"])["roc_auc"]
require(math.isclose(ablation_roc["DIVE", "without_ast_call_dataflow"], 0.5449430628),
        "DIVE relational ablation changed")
require(math.isclose(ablation_roc["DIVE", "without_syntax_organization"], 0.6164457939),
        "DIVE syntax/organization ablation changed")
require(math.isclose(
    ablation_roc["Empirical-Analysis", "without_lexical_statistics"], 0.5874279140
), "Empirical lexical-statistics ablation changed")
require("syntax/organization (117)" in TEX,
        "manuscript lacks the current major-group ablation")
reported_label_sensitivity = (
    label_sensitivity[label_sensitivity["exclude_duplicate_sources"]]
    .set_index(["dataset", "label_policy"])["f1"]
)
require(math.isclose(
    reported_label_sensitivity["DIVE", "forge_vulnerable_vs_unknown"],
    0.6602280968,
), "DIVE weak-label sensitivity changed")
require(math.isclose(
    reported_label_sensitivity[
        "Empirical-Analysis", "forge_vulnerable_vs_unknown_suspicious"
    ],
    0.7580383241,
), "Empirical weak-label sensitivity changed")
require("56{,}943" in TEX, "manuscript does not distinguish source-bearing rows")
require("All numeric exclusions" in TEX,
        "manuscript does not account for all 57 excluded numeric columns")
require(exact_benchmarks["feature_count"] == 244,
        "contextual benchmark slices do not use 244 features")
require(exact_benchmarks["protocol"]["model"] == "rf_selected_by_dappscan_project_group_cv",
        "contextual benchmark slices do not use the fixed RF head")
exact_recall = {
    row["dataset"]: row["recall"]
    for row in exact_benchmarks["metrics"]
    if row["dataset"] in {"TRACE-CVE", "PropertyGPT-CVE", "LLM-SmartAudit-CVE"}
}
require(exact_recall == {
    "TRACE-CVE": 8 / 15,
    "PropertyGPT-CVE": 2 / 13,
    "LLM-SmartAudit-CVE": 2 / 13,
}, f"contextual benchmark recall changed: {exact_recall}")
require(
    (slither_summary["input_rows"], slither_summary["successes"], slither_summary["failures"])
    == (6712, 6341, 371),
    "compiler-assisted reliability counts changed",
)
require("/Users/" not in slither_summary_text,
        "compiler-assisted reliability summary leaks an absolute user path")
construct_totals = {
    column: (
        int(slither_constructs.loc[slither_constructs[column], "n"].sum()),
        int(slither_constructs.loc[slither_constructs[column], "n_success"].sum()),
    )
    for column in ("interfaces", "inline_assembly", "delegatecall", "proxy_indicators")
}
require(construct_totals == {
    "interfaces": (3701, 3371),
    "inline_assembly": (735, 689),
    "delegatecall": (168, 150),
    "proxy_indicators": (530, 499),
}, f"compiler-assisted construct strata changed: {construct_totals}")
correlations = slither_correlations.set_index("scafe_feature")["spearman_r"].to_dict()
require(math.isclose(correlations["n_functions"], 0.9315),
        "Slither function-count agreement changed")
require(math.isclose(correlations["halstead_volume"], 0.356),
        "Slither Halstead-volume agreement changed")
require("0.533 on TRACE-CVE" in TEX, "manuscript lacks current TRACE slice result")
require("0.154 on PropertyGPT-CVE" in TEX,
        "manuscript lacks current PropertyGPT slice result")
require("anonymous.4open.science" not in TEX,
        "manuscript claims an inaccessible anonymous artifact URL")
require((PAPER / "sc_vulnerability_detection.pdf").exists(), "compiled manuscript PDF is missing")
for figure in (
    "fig_model_comparison.pdf", "fig_confusion_matrices.pdf",
    "fig_feature_ablation.pdf", "fig_shap_cases.pdf"
):
    require((PAPER / "figures" / figure).exists(), f"corrected figure missing: {figure}")
for obsolete_table in (
    "tab_contextual_baselines.tex",
    "tab_fair_dl_baselines_seed42.tex",
    "tab_threshold_sensitivity.tex",
):
    require(not (PAPER / "figures" / obsolete_table).exists(),
            f"obsolete table remains in submission tree: {obsolete_table}")

if ERRORS:
    for error in ERRORS:
        print(f"FAIL: {error}")
    raise SystemExit(1)

print("PASS: current manuscript and corrected SCAFE artifacts are consistent")
