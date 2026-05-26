#!/usr/bin/env python3 -u
"""
Pairwise significance tests on STRICT-label OOF predictions.

For every (model_a, model_b) pair, recomputes per-fold ROC/PR/F1 from the
saved OOF CSVs, then runs Welch's t-test on the 5 fold-level scores.
This is stronger than running the test on summary statistics because we
use the actual fold scores.

Output:
  data/derived/ml_results/significance_strict.json
  prints a LaTeX-ready table snippet
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

# All OOF files we want to compare under STRICT protocol.
# Each entry: (display_name, filename_or_pattern).
# Files that don't exist are silently skipped.
STRICT_OOFS = [
    # SCAFE structural-only (Vote ensemble)
    ("SCAFE Vote",       "ml_Vote_RF_GBC_XGB_strict_merged.csv"),
    ("SCAFE RF",         "ml_RandomForest_strict_merged.csv"),
    ("SCAFE GBC",        "ml_GradientBoosting_strict_merged.csv"),
    ("SCAFE XGB",        "ml_XGBoost_strict_merged.csv"),
    # SCAFE-Sem (dual-stream)
    ("SCAFE-Sem Vote",   "sem_Vote_RF_GBC_XGB_strict_struct_plus_both.csv"),
    ("SCAFE-Sem GBC",    "sem_GradientBoosting_strict_struct_plus_both.csv"),
    # DL baselines
    ("CodeBERT",         "dl_codebert_strict_merged.csv"),
    ("UniXcoder",        "dl_unixcoder_strict_merged.csv"),
    ("TextCNN",          "dl_textcnn_strict_merged.csv"),
    ("BiLSTM",           "dl_bilstm_strict_merged.csv"),
]


def _optimal_f1(y_true, y_prob):
    best, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 199):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, t
    return best_t, best


def per_fold_metrics(df):
    """Compute per-fold ROC/PR/F1 from an OOF DataFrame."""
    if "fold" not in df.columns:
        return None
    rocs, prs, f1s = [], [], []
    for fid in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fid]
        if sub["y_true"].nunique() < 2:
            continue
        rocs.append(roc_auc_score(sub["y_true"], sub["y_prob"]))
        prs.append(average_precision_score(sub["y_true"], sub["y_prob"]))
        _, f1 = _optimal_f1(sub["y_true"].values, sub["y_prob"].values)
        f1s.append(f1)
    return {"roc": np.array(rocs), "pr": np.array(prs), "f1": np.array(f1s)}


def load_strict_oofs(strict_oofs=None):
    strict_oofs = strict_oofs or STRICT_OOFS
    methods = {}
    for name, fname in strict_oofs:
        p = OOF_DIR / fname
        if not p.exists():
            print(f"  [skip] {name}: {fname} not found")
            continue
        df = pd.read_csv(p)
        m = per_fold_metrics(df)
        if m is None or len(m["roc"]) < 2:
            print(f"  [skip] {name}: not enough folds")
            continue
        methods[name] = m
        print(
            f"  {name:20s}  "
            f"ROC={m['roc'].mean():.4f}±{m['roc'].std():.4f}  "
            f"PR={m['pr'].mean():.4f}  "
            f"F1={m['f1'].mean():.4f}  "
            f"(n_folds={len(m['roc'])})"
        )
    return methods


def welch_from_folds(a, b):
    """Welch's two-sample t-test on per-fold scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), 0.0
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p), float(len(a) + len(b) - 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key-pairs-only", action="store_true",
        help="Only run the focal pairwise comparisons used in the paper",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("STRICT-LABEL SIGNIFICANCE TESTS (per-fold scores, Welch's t-test)")
    print("=" * 90)

    methods = load_strict_oofs()
    print()

    if args.key_pairs_only:
        focal = [
            ("SCAFE Vote",     "SCAFE-Sem Vote"),
            ("SCAFE Vote",     "CodeBERT"),
            ("SCAFE Vote",     "UniXcoder"),
            ("SCAFE-Sem Vote", "CodeBERT"),
            ("SCAFE-Sem Vote", "UniXcoder"),
            ("SCAFE-Sem Vote", "TextCNN"),
            ("SCAFE-Sem Vote", "BiLSTM"),
            ("CodeBERT",       "UniXcoder"),
        ]
        pairs = [(a, b) for a, b in focal if a in methods and b in methods]
    else:
        pairs = list(combinations(sorted(methods.keys()), 2))

    rows = []
    print(f"{'A':<22} {'B':<22} {'Metric':<7} {'Δmean':>8} "
          f"{'t-stat':>8} {'p-value':>10} {'Sig*':>6}")
    print("-" * 90)
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        ma, mb = methods[a], methods[b]
        for metric in ("roc", "pr", "f1"):
            sa = ma[metric]
            sb = mb[metric]
            # Use only the folds both sides have
            n = min(len(sa), len(sb))
            sa, sb = sa[:n], sb[:n]
            t, p, df_ = welch_from_folds(sa, sb)
            delta = sa.mean() - sb.mean()
            sig = "YES" if (not np.isnan(p) and p < 0.05) else "no"
            rows.append({
                "method_a": a, "method_b": b, "metric": metric,
                "mean_a": round(float(sa.mean()), 4),
                "mean_b": round(float(sb.mean()), 4),
                "delta": round(float(delta), 4),
                "t_stat": (round(float(t), 4) if not np.isnan(t) else None),
                "p_value": (round(float(p), 4) if not np.isnan(p) else None),
                "df": df_,
                "significant_005": (
                    bool(p < 0.05) if not np.isnan(p) else False
                ),
                "n_folds": int(n),
            })
            print(f"{a:<22} {b:<22} {metric:<7} {delta:>+8.4f} "
                  f"{t:>8.3f} {p:>10.4f} {sig:>6}")

    out = RESULTS_DIR / "significance_strict.json"
    out.write_text(json.dumps({
        "protocol": "strict",
        "test": "Welch's t-test on per-fold scores",
        "n_methods": len(methods),
        "method_means": {
            k: {
                "roc_mean": float(v["roc"].mean()),
                "roc_std": float(v["roc"].std()),
                "pr_mean": float(v["pr"].mean()),
                "f1_mean": float(v["f1"].mean()),
                "n_folds": int(len(v["roc"])),
            }
            for k, v in methods.items()
        },
        "pairwise_tests": rows,
    }, indent=2, default=str))
    print(f"\nSaved → {out}")

    # LaTeX snippet
    print("\nLaTeX table snippet:")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"A vs B & Metric & $\Delta$ & $t$ & $p$ \\")
    print(r"\midrule")
    for r in rows:
        if r["metric"] != "roc":
            continue
        mark = r"$^*$" if r["significant_005"] else ""
        print(
            f"{r['method_a']} vs {r['method_b']} & ROC-AUC & "
            f"{r['delta']:+.4f} & {r['t_stat']:.2f} & {r['p_value']:.4f}{mark} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
