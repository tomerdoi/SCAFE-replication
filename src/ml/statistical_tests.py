"""
Statistical significance tests comparing methods across 5-fold GroupKFold CV.

Uses Welch's t-test on per-fold summary statistics (mean ± std, n=5).
This approach is standard in ML research when per-fold scores are
aggregated at training time (see Dietterich 1998, Demsar 2006).
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "data" / "derived" / "dl_results"


def welch_t_test(mean_a, std_a, n_a, mean_b, std_b, n_b):
    """Welch's t-test from summary statistics."""
    se = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
    if se < 1e-12:
        return 0.0, 1.0, 0
    t_stat = (mean_a - mean_b) / se
    # Welch-Satterthwaite degrees of freedom
    num = (std_a**2 / n_a + std_b**2 / n_b)**2
    denom = ((std_a**2 / n_a)**2 / (n_a - 1) +
             (std_b**2 / n_b)**2 / (n_b - 1))
    if denom < 1e-12:
        return float(t_stat), 1.0, 0
    df = num / denom
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return float(t_stat), float(p_value), float(df)


def load_all_results():
    """Load ML and DL results with means and stds."""
    methods = {}

    # ML results: merged GBC from cross_dataset_evaluation.csv
    ml_csv = _REPO / "data" / "derived" / "ml_results" / "cross_dataset_evaluation.csv"
    ml_df = pd.read_csv(ml_csv)
    merged_gbc = ml_df[(ml_df["experiment"] == "C_merged_cv") &
                       (ml_df["model"] == "gbc")]
    if len(merged_gbc) == 1:
        r = merged_gbc.iloc[0]
        methods["Ours (GBC, merged)"] = {
            "roc_auc_mean": r["roc_auc"], "roc_auc_std": r["roc_auc_std"],
            "pr_auc_mean": r["pr_auc"], "pr_auc_std": r["pr_auc_std"],
            "f1_mean": r["f1"], "f1_std": np.nan,
            "n_folds": 5,
        }

    # ML results: DAppSCAN GBC from model_comparison.csv
    mc_csv = _REPO / "data" / "derived" / "ml_results" / "model_comparison.csv"
    mc_df = pd.read_csv(mc_csv)
    dappscan_gbc = mc_df[(mc_df["model"] == "gbc") &
                         (mc_df["feature_set"] == "all") &
                         (mc_df["dataset"] == "DAppSCAN")]
    if len(dappscan_gbc) == 1:
        r = dappscan_gbc.iloc[0]
        methods["Ours (GBC, DAppSCAN)"] = {
            "roc_auc_mean": r["roc_auc_mean"], "roc_auc_std": r["roc_auc_std"],
            "pr_auc_mean": r["pr_auc_mean"], "pr_auc_std": r["pr_auc_std"],
            "f1_mean": r["f1_mean"], "f1_std": r["f1_std"],
            "n_folds": r["n_folds"],
        }

    # DL results
    dl_json = _REPO / "data" / "derived" / "dl_results" / "dl_baseline_results.json"
    with open(dl_json) as f:
        dl_results = json.load(f)
    for r in dl_results:
        label = f"{r['display_name']} ({r['dataset']})"
        methods[label] = {
            "roc_auc_mean": r["roc_auc_mean"], "roc_auc_std": r["roc_auc_std"],
            "pr_auc_mean": r["pr_auc_mean"], "pr_auc_std": r["pr_auc_std"],
            "f1_mean": r["f1_mean"], "f1_std": r["f1_std"],
            "n_folds": r["n_folds"],
        }

    return methods


def main():
    methods = load_all_results()

    print("=" * 90)
    print("METHOD SUMMARY (5-fold GroupKFold CV)")
    print("=" * 90)
    print(f"{'Method':<30} {'ROC-AUC':>18} {'PR-AUC':>18} {'F1':>18}")
    print("-" * 90)
    for name, m in methods.items():
        roc = f"{m['roc_auc_mean']:.4f}±{m['roc_auc_std']:.4f}"
        pr = f"{m['pr_auc_mean']:.4f}±{m['pr_auc_std']:.4f}"
        f1_std = m['f1_std']
        f1 = f"{m['f1_mean']:.4f}±{f1_std:.4f}" if not np.isnan(f1_std) else f"{m['f1_mean']:.4f}"
        print(f"  {name:<28} {roc:>18} {pr:>18} {f1:>18}")

    # Pairwise comparisons on merged data
    comparisons = [
        ("Ours (GBC, merged)", "TextCNN (Merged)", "Key: parity with simple DL"),
        ("Ours (GBC, merged)", "BiLSTM (Merged)", "Key: parity with simple DL"),
        ("Ours (GBC, merged)", "CodeBERT (fine-tuned) (Merged)", "CodeBERT advantage"),
        ("TextCNN (Merged)", "CodeBERT (fine-tuned) (Merged)", "DL internal gap"),
        ("BiLSTM (Merged)", "CodeBERT (fine-tuned) (Merged)", "DL internal gap"),
    ]

    print("\n" + "=" * 110)
    print("PAIRWISE WELCH'S t-TESTS (merged dataset)")
    print("=" * 110)
    print(f"  {'Comparison':<50} {'Metric':<10} {'t-stat':>8} {'df':>6} "
          f"{'p-value':>10} {'Sig (0.05)':>10}")
    print("-" * 110)

    test_results = []
    for method_a, method_b, note in comparisons:
        if method_a not in methods or method_b not in methods:
            logger.warning(f"Skipping {method_a} vs {method_b}: method not found")
            continue
        ma = methods[method_a]
        mb = methods[method_b]
        n_a, n_b = ma["n_folds"], mb["n_folds"]

        for metric in ["roc_auc", "pr_auc", "f1"]:
            mean_a = ma[f"{metric}_mean"]
            std_a = ma[f"{metric}_std"]
            mean_b = mb[f"{metric}_mean"]
            std_b = mb[f"{metric}_std"]

            if np.isnan(std_a) or np.isnan(std_b):
                continue

            t_stat, p_val, df = welch_t_test(mean_a, std_a, n_a,
                                             mean_b, std_b, n_b)
            sig = "YES*" if p_val < 0.05 else "no"
            label = f"{method_a} vs {method_b}"
            diff = mean_a - mean_b

            print(f"  {label:<50} {metric:<10} {t_stat:>8.3f} {df:>6.1f} "
                  f"{p_val:>10.4f} {sig:>10}")

            test_results.append({
                "method_a": method_a,
                "method_b": method_b,
                "note": note,
                "metric": metric,
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "diff": round(diff, 4),
                "t_statistic": round(t_stat, 4),
                "df": round(df, 2),
                "p_value": round(p_val, 4),
                "significant_005": p_val < 0.05,
            })

    # Interpretation
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    parity_tests = [t for t in test_results
                    if "parity" in t["note"] and t["metric"] == "roc_auc"]
    codebert_tests = [t for t in test_results
                      if "CodeBERT advantage" in t["note"]
                      and t["metric"] == "roc_auc"]

    for t in parity_tests:
        if not t["significant_005"]:
            print(f"  {t['method_a']} vs {t['method_b']} (ROC-AUC): "
                  f"NOT significantly different (p={t['p_value']:.4f}). "
                  f"Supports statistical parity claim.")
        else:
            print(f"  {t['method_a']} vs {t['method_b']} (ROC-AUC): "
                  f"Significantly different (p={t['p_value']:.4f}).")

    for t in codebert_tests:
        if t["significant_005"]:
            print(f"  {t['method_a']} vs {t['method_b']} (ROC-AUC): "
                  f"Significantly different (p={t['p_value']:.4f}). "
                  f"CodeBERT's advantage is statistically significant.")
        else:
            print(f"  {t['method_a']} vs {t['method_b']} (ROC-AUC): "
                  f"NOT significantly different (p={t['p_value']:.4f}).")

    # Save
    out_path = _OUT_DIR / "statistical_tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test_type": "Welch's two-sample t-test (two-sided)",
        "note": ("Computed from 5-fold GroupKFold CV summary statistics "
                 "(mean, std). Conservative compared to paired test."),
        "methods": {k: v for k, v in methods.items()},
        "pairwise_tests": test_results,
    }
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    out_path.write_text(json.dumps(output, indent=2, default=_convert))
    logger.info(f"Results saved to {out_path}")

    # Generate LaTeX table snippet
    print("\n" + "=" * 90)
    print("LATEX TABLE SNIPPET (for paper)")
    print("=" * 90)
    print(r"""
\begin{table}[t]
\centering
\caption{Pairwise Welch's $t$-test results on merged data (5-fold GroupKFold CV).
$p < 0.05$ indicates a statistically significant difference.}
\label{tab:significance}
\small
\begin{tabular}{llrr}
\toprule
Comparison & Metric & $t$-stat & $p$-value \\
\midrule""")

    for t in test_results:
        a_short = t["method_a"].replace(" (merged)", "").replace("Ours (GBC, merged)", "Ours")
        b_short = t["method_b"].replace(" (Merged)", "").replace("(fine-tuned) ", "")
        sig_mark = "$^{*}$" if t["significant_005"] else ""
        metric_tex = t["metric"].replace("_", r"\_")
        print(f"{a_short} vs {b_short} & {metric_tex} & "
              f"{t['t_statistic']:.2f} & {t['p_value']:.4f}{sig_mark} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
