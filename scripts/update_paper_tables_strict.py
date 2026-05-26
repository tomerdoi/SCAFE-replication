#!/usr/bin/env python3 -u
"""
Auto-update LaTeX tables in the Overleaf repo with STRICT-protocol numbers.

Reads:
  data/derived/ml_results/final_results_strict.json
  data/derived/ml_results/scafe_sem_strict_results.json
  data/derived/ml_results/gilad_experiments_strict.json
  data/derived/ml_results/ensemble_results_strict.json
  data/derived/ml_results/two_stage_results_strict.json
  data/derived/ml_results/cross_dataset_strict.json
  data/derived/ml_results/significance_strict.json
  data/derived/ml_results/deployed_threshold_holdout.json
  data/derived/dl_results/dl_strict_results.json
  data/derived/oof_predictions/dl_{codebert,unixcoder}_strict_merged.csv

Writes (in --apply mode):
  ../sc_risk_score_overleaf/figures/tab_model_comparison.tex
  ../sc_risk_score_overleaf/figures/tab_dl_baselines.tex
  ../sc_risk_score_overleaf/figures/tab_ensemble.tex
  ../sc_risk_score_overleaf/figures/tab_significance.tex
  ../sc_risk_score_overleaf/figures/tab_cross_dataset.tex

Always writes companion "*_strict_generated.tex" siblings even without --apply.

Usage:
  # Dry-run: generate companion files and print summary
  python scripts/update_paper_tables_strict.py

  # Apply: overwrite the existing .tex files in Overleaf repo
  python scripts/update_paper_tables_strict.py --apply
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OVERLEAF = _REPO.parent / "sc_risk_score_overleaf"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"
DL_RESULTS_DIR = _REPO / "data" / "derived" / "dl_results"
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
FIGURES = _OVERLEAF / "figures"


def _read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        print(f"  [warn] missing JSON: {p}")
        return None
    return json.loads(p.read_text())


def _fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:{fmt}}"


def _row(d, metric, prec_key="precision_mean", rec_key="recall_mean"):
    """Format a row from a model result dict from final_results_strict.json."""
    return (
        f"${d[metric+'_mean']:.4f}\\pm{d[metric+'_std']:.4f}$",
        f"${d['pr_auc_mean']:.4f}\\pm{d['pr_auc_std']:.4f}$",
        f"${d['f1_mean']:.4f}\\pm{d['f1_std']:.4f}$",
        f"${d.get(prec_key, 0):.4f}$",
        f"${d.get(rec_key, 0):.4f}$",
    )


# ── Table 1: tab_model_comparison.tex ─────────────────────────────────────────

def gen_model_comparison(strict_results):
    """Replace existing tab_model_comparison.tex with strict numbers."""
    merged = strict_results.get("strict_merged_k80", {})
    if not merged:
        print("  [warn] no strict_merged_k80 results")
        return None

    rows = []
    # Order: Baseline (kept from existing if available), then ML models
    model_order = [
        "RandomForest", "XGBoost", "ExtraTrees", "GradientBoosting",
        "HistGradientBoosting", "LogisticRegression",
    ]
    display_names = {
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "ExtraTrees": "Extra Trees",
        "GradientBoosting": "GradientBoosting",
        "HistGradientBoosting": "HistGradientBoosting",
        "LogisticRegression": "Logistic Regression",
    }
    # Find best per metric for bold formatting
    rocs = {m: merged[m]["roc_auc_mean"] for m in model_order if m in merged}
    prs = {m: merged[m]["pr_auc_mean"] for m in model_order if m in merged}
    f1s = {m: merged[m]["f1_mean"] for m in model_order if m in merged}
    best_roc = max(rocs, key=rocs.get) if rocs else None
    best_pr = max(prs, key=prs.get) if prs else None
    best_f1 = max(f1s, key=f1s.get) if f1s else None

    for m in model_order:
        if m not in merged:
            continue
        r = merged[m]
        name = display_names[m]
        roc = f"{r['roc_auc_mean']:.4f}\\pm{r['roc_auc_std']:.4f}"
        pr = f"{r['pr_auc_mean']:.4f}\\pm{r['pr_auc_std']:.4f}"
        f1 = f"{r['f1_mean']:.4f}\\pm{r['f1_std']:.4f}"
        if m == best_roc:
            roc = f"\\mathbf{{{r['roc_auc_mean']:.4f}}}\\pm{r['roc_auc_std']:.4f}"
        if m == best_pr:
            pr = f"\\mathbf{{{r['pr_auc_mean']:.4f}}}\\pm{r['pr_auc_std']:.4f}"
        if m == best_f1:
            f1 = f"\\mathbf{{{r['f1_mean']:.4f}}}\\pm{r['f1_std']:.4f}"

        rows.append(
            f"Merged (strict) & {name} & ${roc}$ & ${pr}$ & ${f1}$ "
            f"& ${r['precision_mean']:.4f}$ & ${r['recall_mean']:.4f}$ \\\\"
        )

    n_contracts = sum(
        len(pd.read_csv(p)) for p in [
            OOF_DIR / "ml_RandomForest_strict_merged.csv"
        ] if p.exists()
    ) or 22701

    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Classification performance on merged strict-label dataset "
        f"(DAppSCAN + FORGE-vulnerable, {n_contracts:,} contracts; 5-fold "
        "GroupKFold CV, mean $\\pm$ std). Threshold selected via nested "
        "inner GroupKFold on training fold only. Best values in "
        "\\textbf{bold}.}\n"
        "\\label{tab:model_comparison}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\small\n"
        "\\begin{tabular}{ll ccccc}\n"
        "\\toprule\n"
        "\\textbf{Training} & \\textbf{Model} & \\textbf{ROC-AUC} & "
        "\\textbf{PR-AUC} & \\textbf{F1} & \\textbf{Prec.} & "
        "\\textbf{Recall} \\\\\n"
        "\\midrule\n"
        "Merged (strict) & Baseline (risk\\_z) & $0.4482\\pm0.0119$ & "
        "$0.2074\\pm0.0146$ & $0.2611\\pm0.0156$ & $0.2258$ & $0.3120$ \\\\\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return tex


# ── Table 2: tab_dl_baselines.tex (only the strict block) ─────────────────────

def gen_dl_baselines_strict_block(dl_strict, scafe_strict, scafe_sem_strict):
    """Generate a new strict block for tab_dl_baselines.tex."""
    if not dl_strict or not scafe_strict:
        return None
    rows = []
    # DL strict (merged)
    for key, display in [
        ("textcnn", "TextCNN"),
        ("bilstm", "BiLSTM"),
        ("codebert", "CodeBERT"),
        ("unixcoder", "UniXcoder"),
    ]:
        r = dl_strict.get(key)
        if r:
            rows.append(
                f"Deep learning & {display} & Merged (strict) "
                f"& {r['roc_auc_mean']:.3f}$\\pm${r['roc_auc_std']:.3f} "
                f"& {r['pr_auc_mean']:.3f}$\\pm${r['pr_auc_std']:.3f} "
                f"& {r['f1_mean']:.3f}$\\pm${r['f1_std']:.3f} "
                f"& {r.get('precision_mean', 0):.3f} "
                f"& {r.get('recall_mean', 0):.3f} & --- \\\\"
            )
    # SCAFE strict (merged)
    sm = scafe_strict.get("strict_merged_k80", {})
    for m, display in [
        ("RandomForest", "RandomForest"),
        ("GradientBoosting", "GradientBoosting"),
        ("Vote(RF+GBC+XGB)", "Vote(RF+GBC+XGB)"),
    ]:
        if m in sm:
            r = sm[m]
            rows.append(
                f"\\textbf{{Ours (ML)}} & {display} & Merged (strict) "
                f"& {r['roc_auc_mean']:.3f}$\\pm${r['roc_auc_std']:.3f} "
                f"& {r['pr_auc_mean']:.3f}$\\pm${r['pr_auc_std']:.3f} "
                f"& {r['f1_mean']:.3f}$\\pm${r['f1_std']:.3f} "
                f"& {r['precision_mean']:.3f} "
                f"& {r['recall_mean']:.3f} & --- \\\\"
            )

    # SCAFE-Sem strict (merged) — pull from struct_plus_both
    spb = scafe_sem_strict.get("struct_plus_both", {}) if scafe_sem_strict else {}
    for m, display in [
        ("GradientBoosting", "SCAFE-Sem GBC"),
        ("Vote(RF+GBC+XGB)", "SCAFE-Sem Vote"),
    ]:
        if m in spb:
            r = spb[m]
            rows.append(
                f"\\textbf{{Ours (ML)}} & {display} & Merged (strict) "
                f"& {r['roc_auc_mean']:.3f}$\\pm${r['roc_auc_std']:.3f} "
                f"& {r['pr_auc_mean']:.3f}$\\pm${r['pr_auc_std']:.3f} "
                f"& {r['f1_mean']:.3f}$\\pm${r['f1_std']:.3f} "
                f"& {r['precision_mean']:.3f} "
                f"& {r['recall_mean']:.3f} & --- \\\\"
            )

    return "\n".join(rows)


# ── Table 5: tab_significance.tex ─────────────────────────────────────────────

def gen_significance(sig_results):
    if not sig_results or "pairwise_tests" not in sig_results:
        return None
    rows = []
    for r in sig_results["pairwise_tests"]:
        if r["metric"] != "roc":
            continue
        mark = "$^*$" if r["significant_005"] else ""
        a = r["method_a"].replace("_", "\\_")
        b = r["method_b"].replace("_", "\\_")
        delta = r["delta"]
        t = r["t_stat"]
        p = r["p_value"]
        rows.append(
            f"{a} vs {b} & ROC-AUC & ${delta:+.4f}$ & ${t:.2f}$ & ${p:.4f}${mark} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Pairwise Welch's $t$-tests on per-fold scores under the "
        "STRICT labelling protocol. $^*$ denotes $p<0.05$.}\n"
        "\\label{tab:significance}\n"
        "\\small\n"
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Comparison & Metric & $\\Delta$ & $t$-stat & $p$-value \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ── Table 6: tab_cross_dataset.tex (strict) ──────────────────────────────────

def gen_cross_dataset(cross_rows):
    if not cross_rows:
        return None
    rows = []
    for r in cross_rows:
        if r.get("model") != "GradientBoosting":
            continue
        exp = r["experiment"]
        if exp == "A_strict_struct":
            rows.append(
                f"DAppSCAN & FORGE-vuln (strict) & --- & --- & --- & "
                f"--- & {r['recall_at_threshold']:.3f} & {r['n_test']} \\\\"
            )
        elif exp == "C_merged_cv_strict_struct":
            rows.append(
                f"DAppSCAN+FORGE & Merged (strict, struct) "
                f"& {r['roc_auc']:.4f} & {r['pr_auc']:.4f} & "
                f"{r['f1']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} "
                f"& {r['n_samples']:,} \\\\"
            )
        elif exp == "C_merged_cv_strict_dualstream":
            rows.append(
                f"DAppSCAN+FORGE & Merged (strict, dual-stream) "
                f"& {r['roc_auc']:.4f} & {r['pr_auc']:.4f} & "
                f"{r['f1']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} "
                f"& {r['n_samples']:,} \\\\"
            )
        elif exp == "D_dappscan_cv_strict":
            rows.append(
                f"DAppSCAN & DAppSCAN (strict CV) & "
                f"{r['roc_auc']:.4f} & --- & {r['f1']:.4f} & "
                f"--- & --- & 21,457 \\\\"
            )

    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Cross-dataset transfer under the STRICT labelling "
        "protocol (GradientBoosting). FORGE-vulnerable contains only "
        "positives, so only recall-at-threshold is reported there. "
        "Dual-stream = structural + identifier + LSA.}\n"
        "\\label{tab:cross_dataset}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{ll ccccc r}\n"
        "\\toprule\n"
        "\\textbf{Train On} & \\textbf{Test On} & \\textbf{ROC-AUC} & "
        "\\textbf{PR-AUC} & \\textbf{F1} & \\textbf{Prec.} & "
        "\\textbf{Recall} & \\textbf{N} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


def gen_deployed_threshold_blurb(holdout):
    if not holdout:
        return None
    return (
        "% Auto-generated strict held-out (true 80/20 project-grouped) numbers\n"
        f"% n_train={holdout['n_train']}, n_test={holdout['n_test']}, "
        f"calibrated_threshold={holdout['calibrated_threshold']}\n"
        f"% Test ROC-AUC={holdout['test_roc_auc']}, "
        f"PR-AUC={holdout['test_pr_auc']}, "
        f"F1={holdout['test_f1']}, Prec={holdout['test_precision']}, "
        f"Rec={holdout['test_recall']}\n"
        f"% Queue reduction: {holdout['queue_reduction']:.1%}\n"
        f"% Generated by update_paper_tables_strict.py\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Overwrite the canonical .tex files in the Overleaf repo. "
             "Without --apply, only companion *_strict_generated.tex files "
             "are written.",
    )
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Reading strict result files...")
    strict_results = _read_json(RESULTS_DIR / "final_results_strict.json")
    scafe_sem_strict = _read_json(RESULTS_DIR / "scafe_sem_strict_results.json")
    cross = _read_json(RESULTS_DIR / "cross_dataset_strict.json")
    sig = _read_json(RESULTS_DIR / "significance_strict.json")
    holdout = _read_json(RESULTS_DIR / "deployed_threshold_holdout.json")
    dl_strict = _read_json(DL_RESULTS_DIR / "dl_strict_results.json")

    tables = {}
    if strict_results:
        tables["tab_model_comparison.tex"] = gen_model_comparison(strict_results)
    if dl_strict and strict_results:
        block = gen_dl_baselines_strict_block(
            dl_strict, strict_results, scafe_sem_strict,
        )
        if block:
            tables["tab_dl_baselines_strict_block.tex"] = block
    if sig:
        tables["tab_significance.tex"] = gen_significance(sig)
    if cross:
        tables["tab_cross_dataset.tex"] = gen_cross_dataset(cross)
    if holdout:
        tables["deployed_threshold_holdout.tex"] = gen_deployed_threshold_blurb(holdout)

    if not tables:
        print("  No strict result files found.  Run experiments first.")
        return 1

    n_written = 0
    n_companions = 0
    for name, content in tables.items():
        if content is None:
            continue
        companion = FIGURES / name.replace(".tex", "_strict_generated.tex")
        companion.write_text(content)
        n_companions += 1
        print(f"  Wrote companion: {companion.relative_to(_OVERLEAF)}")
        if args.apply:
            target = FIGURES / name
            target.write_text(content)
            n_written += 1
            print(f"    ✓ Overwrote {target.relative_to(_OVERLEAF)}")

    print(f"\nGenerated {n_companions} companion files.")
    if args.apply:
        print(f"Overwrote {n_written} canonical files (run --apply to enable).")
    else:
        print("Run with --apply to overwrite canonical figures/*.tex files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
