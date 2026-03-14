#!/usr/bin/env python3
"""
Regenerate ALL publication figures from authoritative final_results.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_FINAL = _REPO / "data" / "derived" / "ml_results" / "final_results.json"
_DL = _REPO / "data" / "derived" / "dl_results" / "dl_baseline_results.json"
_CROSS_CSV = _REPO / "data" / "derived" / "ml_results" / "cross_dataset_evaluation.csv"
_SLITHER = _REPO / "data" / "derived" / "slither_results" / "slither_summary.json"
_OLD_COMP = _REPO / "data" / "derived" / "ml_results" / "model_comparison.csv"
_OUT = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

with open(_FINAL) as f:
    FINAL = json.load(f)
with open(_DL) as f:
    DL_LIST = json.load(f)

DL = {}
for d in DL_LIST:
    DL[(d["model"], d["dataset"])] = d

MODEL_DISPLAY = {
    "RandomForest": "Random Forest",
    "GradientBoosting": "GradientBoosting",
    "XGBoost": "XGBoost",
    "HistGradientBoosting": "HistGradientBoosting",
    "ExtraTrees": "Extra Trees",
    "LogisticRegression": "Logistic Reg.",
    "Vote(RF+GBC+XGB)": "Vote Ensemble",
}

COLORS_MODEL = {
    "RandomForest": "#2196F3",
    "XGBoost": "#FF9800",
    "GradientBoosting": "#4CAF50",
    "HistGradientBoosting": "#9C27B0",
    "ExtraTrees": "#F44336",
    "LogisticRegression": "#607D8B",
    "Vote(RF+GBC+XGB)": "#009688",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(_OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Model comparison — DAppSCAN vs Merged
# ─────────────────────────────────────────────────────────────────────────
def fig_model_comparison():
    print("Generating: fig_model_comparison")
    models_order = ["RandomForest", "GradientBoosting", "XGBoost", "HistGradientBoosting"]
    metrics = [("roc_auc_mean", "roc_auc_std", "ROC-AUC"),
               ("pr_auc_mean", "pr_auc_std", "PR-AUC"),
               ("f1_mean", "f1_std", "F1")]
    ylims = [(0.82, 0.96), (0.25, 0.58), (0.34, 0.68)]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))

    for ax, (met, std, label), (ylo, yhi) in zip(axes, metrics, ylims):
        x = np.arange(len(models_order))
        width = 0.35

        vals_d = [FINAL["dappscan_k80"][m][met] for m in models_order]
        errs_d = [FINAL["dappscan_k80"][m][std] for m in models_order]
        vals_m = [FINAL["merged_k80"][m][met] for m in models_order]
        errs_m = [FINAL["merged_k80"][m][std] for m in models_order]

        ax.bar(x - width/2, vals_d, width, yerr=errs_d, capsize=3,
               label="DAppSCAN only", color="#2196F3", alpha=0.85,
               edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})
        ax.bar(x + width/2, vals_m, width, yerr=errs_m, capsize=3,
               label="Merged (DAppSCAN+FORGE)", color="#FF9800", alpha=0.85,
               edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})

        for i, (dv, mv, de, me) in enumerate(zip(vals_d, vals_m, errs_d, errs_m)):
            ax.text(i - width/2, dv + de + 0.008, f"{dv:.3f}",
                    ha="center", fontsize=6.5, color="#1565C0")
            ax.text(i + width/2, mv + me + 0.008, f"{mv:.3f}",
                    ha="center", fontsize=6.5, color="#E65100")

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in models_order],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
              framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model Performance: DAppSCAN Only vs. Merged Training", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig_model_comparison")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Feature importance (top 20) — from tab_feature_importance.tex
# ─────────────────────────────────────────────────────────────────────────
def fig_feature_importance():
    print("Generating: fig_feature_importance")
    features = [
        ("token_diversity", 0.0402),
        ("comp_complexity", 0.0380),
        ("trigram_diversity", 0.0355),
        ("halstead_volume", 0.0312),
        ("trigram_entropy", 0.0298),
        ("cg_leaf_ratio", 0.0271),
        ("halstead_effort", 0.0254),
        ("n_total_tokens", 0.0243),
        ("bigram_diversity", 0.0229),
        ("n_unique_tokens", 0.0218),
        ("bigram_entropy", 0.0207),
        ("halstead_difficulty", 0.0195),
        ("loc", 0.0183),
        ("n_written_state_vars", 0.0174),
        ("token_entropy", 0.0168),
        ("n_risky_bigrams", 0.0156),
        ("halstead_volume_per_loc", 0.0144),
        ("avg_loc_per_function", 0.0138),
        ("n_require_assert", 0.0131),
        ("sum_cyclomatic", 0.0125),
    ]

    names = [f[0] for f in features][::-1]
    vals = [f[1] for f in features][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = [plt.cm.viridis(0.3 + 0.6 * i / len(names)) for i in range(len(names))]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    labels = [n.replace("_", " ").title() for n in names]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Importance (RF, 2-stage selection)")
    ax.set_title("Top 20 Features by Importance (306 $\\rightarrow$ 80)")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    max_val = max(vals)
    ax.set_xlim(0, max_val * 1.25)

    for i, (val, name) in enumerate(zip(vals, names)):
        ax.text(val + max_val * 0.015, i, f"{val:.4f}",
                va="center", fontsize=7, color="#333")

    fig.tight_layout()
    _save(fig, "fig_feature_importance")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Confusion matrices — from old cross_dataset_evaluation.csv
# ─────────────────────────────────────────────────────────────────────────
def fig_confusion_matrices():
    print("Generating: fig_confusion_matrices")
    if not _CROSS_CSV.exists():
        print("  SKIPPED (no cross_dataset_evaluation.csv)")
        return

    cross = pd.read_csv(_CROSS_CSV)

    d_cv = cross[(cross["experiment"] == "D_per_dataset_cv") &
                 (cross["model"] == "gbc") &
                 (cross["dataset"] == "DAppSCAN")].iloc[0]
    m_gbc = cross[(cross["experiment"] == "C_merged_cv") &
                  (cross["model"] == "gbc")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, data, title in [
        (axes[0], d_cv, "DAppSCAN Only (GBC)"),
        (axes[1], m_gbc, "Merged DAppSCAN+FORGE (GBC)"),
    ]:
        cm = np.array([[int(data["TN"]), int(data["FP"])],
                       [int(data["FN"]), int(data["TP"])]])
        cm_pct = cm / cm.sum() * 100

        cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#2196F3"])
        ax.imshow(cm_pct, cmap=cmap, aspect="auto", vmin=0, vmax=100)

        for i in range(2):
            for j in range(2):
                txt = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nClean", "Predicted\nVulnerable"])
        ax.set_yticklabels(["Actual\nClean", "Actual\nVulnerable"])
        ax.set_title(title, fontsize=11, pad=10)

        n = int(data["n_samples"])
        pos = int(data["n_pos"])
        ax.text(0.5, -0.22, f"N={n:,}  |  Pos={pos:,} ({pos/n*100:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=8, color="#666")

    fig.suptitle("Confusion Matrices — GradientBoosting", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_confusion_matrices")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Cross-dataset transfer heatmap
# ─────────────────────────────────────────────────────────────────────────
def fig_cross_dataset_heatmap():
    print("Generating: fig_cross_dataset_heatmap")
    if not _CROSS_CSV.exists():
        print("  SKIPPED (no cross_dataset_evaluation.csv)")
        return

    cross = pd.read_csv(_CROSS_CSV)

    labels_train = ["DAppSCAN", "FORGE", "DAppSCAN\n+FORGE"]
    labels_test = ["DAppSCAN", "FORGE", "Merged (CV)"]

    matrix_roc = np.full((3, 3), np.nan)
    matrix_f1 = np.full((3, 3), np.nan)

    # DAppSCAN→DAppSCAN: use authoritative data
    matrix_roc[0, 0] = FINAL["dappscan_k80"]["GradientBoosting"]["roc_auc_mean"]
    matrix_f1[0, 0] = FINAL["dappscan_k80"]["GradientBoosting"]["f1_mean"]

    # Merged CV: use authoritative data
    matrix_roc[2, 2] = FINAL["merged_k80"]["GradientBoosting"]["roc_auc_mean"]
    matrix_f1[2, 2] = FINAL["merged_k80"]["GradientBoosting"]["f1_mean"]

    scenarios = [
        ("A_cross_predict", "vuln_vs_unknown", 0, 1),
        ("B_reverse", None, 1, 0),
        ("D_per_dataset_cv", "FORGE", 1, 1),
    ]
    for exp, ds_filter, ti, tj in scenarios:
        if ds_filter:
            row = cross[(cross["experiment"] == exp) & (cross["model"] == "gbc") &
                        (cross["dataset"].str.contains(ds_filter))].iloc[0]
        else:
            row = cross[(cross["experiment"] == exp) & (cross["model"] == "gbc")].iloc[0]
        matrix_roc[ti, tj] = row["roc_auc"]
        matrix_f1[ti, tj] = row["f1"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, matrix, title, vmin in [(axes[0], matrix_roc, "ROC-AUC", 0.4),
                                     (axes[1], matrix_f1, "F1 Score", 0.1)]:
        cmap = LinearSegmentedColormap.from_list("rg", ["#FFCDD2", "#FFFFFF", "#C8E6C9", "#4CAF50"])
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=1.0)

        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.85 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#999")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels_test, fontsize=9)
        ax.set_yticklabels(labels_train, fontsize=9)
        ax.set_xlabel("Test On", fontsize=10)
        ax.set_ylabel("Train On", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-Dataset Transfer Performance (GBC)", fontsize=12, y=1.04)
    fig.tight_layout()
    _save(fig, "fig_cross_dataset_heatmap")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5: Feature set ablation
# ─────────────────────────────────────────────────────────────────────────
def fig_feature_ablation():
    print("Generating: fig_feature_ablation")
    if not _OLD_COMP.exists():
        print("  SKIPPED (no model_comparison.csv)")
        return

    comp = pd.read_csv(_OLD_COMP)

    fs_order = ["all", "static_only", "components_only"]
    fs_labels = ["All Features\n(341)", "Static Only\n(306)", "Components Only\n(9)"]
    models = ["rf", "gbc", "xgb", "hgb"]
    model_names = {"rf": "Random Forest", "gbc": "GradientBoosting",
                   "xgb": "XGBoost", "hgb": "HistGradientBoosting"}
    model_colors = {"rf": "#2196F3", "gbc": "#4CAF50", "xgb": "#FF9800", "hgb": "#9C27B0"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [("roc_auc_mean", "ROC-AUC"), ("pr_auc_mean", "PR-AUC"), ("f1_mean", "F1")]
    ylims = [(0.78, 0.96), (0.14, 0.42), (0.22, 0.46)]

    for ax, (metric, label), (ylo, yhi) in zip(axes, metrics, ylims):
        x = np.arange(len(fs_order))
        width = 0.18

        for i, model in enumerate(models):
            vals = []
            for fs in fs_order:
                row = comp[(comp["model"] == model) & (comp["feature_set"] == fs)]
                vals.append(row.iloc[0][metric] if len(row) else 0)
            ax.bar(x + i * width - 1.5 * width, vals, width,
                   label=model_names[model], color=model_colors[model], alpha=0.85,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(fs_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
              framealpha=0.9, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Feature Set Ablation Study (DAppSCAN)", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig_feature_ablation")


# ─────────────────────────────────────────────────────────────────────────
# Figure 6: DL baselines vs ML comparison (merged data)
# ─────────────────────────────────────────────────────────────────────────
def fig_dl_comparison():
    print("Generating: fig_dl_comparison")

    with open(_SLITHER) as f:
        slither = json.load(f)
    sm = slither["metrics_threshold_medium"]

    rows = []
    rows.append({"name": "Slither\n(static)", "roc_auc": np.nan, "pr_auc": np.nan,
                 "f1": sm["f1"], "category": "tool", "std_roc": 0, "std_pr": 0, "std_f1": 0})

    rows.append({"name": "Baseline\n(risk score)", "roc_auc": 0.815, "pr_auc": 0.168,
                 "f1": 0.255, "category": "baseline", "std_roc": 0.015, "std_pr": 0.008, "std_f1": 0.012})

    for model_key, display in [("textcnn", "TextCNN"), ("bilstm", "BiLSTM"), ("codebert", "CodeBERT")]:
        d = DL[(model_key, "Merged")]
        rows.append({"name": f"{display}\n(merged)", "roc_auc": d["roc_auc_mean"],
                     "pr_auc": d["pr_auc_mean"], "f1": d["f1_mean"],
                     "category": "dl", "std_roc": d["roc_auc_std"],
                     "std_pr": d["pr_auc_std"], "std_f1": d["f1_std"]})

    v = FINAL["merged_k80"]["Vote(RF+GBC+XGB)"]
    rows.append({"name": "Vote Ens.\n(ours)", "roc_auc": v["roc_auc_mean"],
                 "pr_auc": v["pr_auc_mean"], "f1": v["f1_mean"],
                 "category": "ours", "std_roc": v["roc_auc_std"],
                 "std_pr": v["pr_auc_std"], "std_f1": v["f1_std"]})

    g = FINAL["merged_k80"]["GradientBoosting"]
    rows.append({"name": "GBC\n(ours)", "roc_auc": g["roc_auc_mean"],
                 "pr_auc": g["pr_auc_mean"], "f1": g["f1_mean"],
                 "category": "ours", "std_roc": g["roc_auc_std"],
                 "std_pr": g["pr_auc_std"], "std_f1": g["f1_std"]})

    df = pd.DataFrame(rows)
    cat_colors = {"tool": "#9E9E9E", "baseline": "#BDBDBD", "dl": "#7E57C2", "ours": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics_info = [("roc_auc", "std_roc", "ROC-AUC"),
                    ("pr_auc", "std_pr", "PR-AUC"),
                    ("f1", "std_f1", "F1")]

    for ax, (metric, std_col, label) in zip(axes, metrics_info):
        vals = df[metric].values
        errs = df[std_col].values
        colors = [cat_colors[c] for c in df["category"]]
        names = df["name"].values

        x = np.arange(len(df))
        for i in x:
            if not np.isnan(vals[i]):
                ax.bar(i, vals[i], color=colors[i], alpha=0.85, edgecolor="white",
                       linewidth=0.5, width=0.7)
                if errs[i] > 0:
                    ax.errorbar(i, vals[i], yerr=errs[i], fmt="none", color="black",
                               capsize=3, linewidth=0.8)
                ax.text(i, vals[i] + max(errs[i], 0) + 0.015, f"{vals[i]:.3f}",
                        ha="center", fontsize=7, fontweight="bold")
            else:
                ax.text(i, 0.05, "N/A", ha="center", fontsize=8, color="#999")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5, rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    legend_elements = [
        Patch(facecolor=cat_colors["tool"], label="Static Analysis Tool"),
        Patch(facecolor=cat_colors["baseline"], label="Rule-based Baseline"),
        Patch(facecolor=cat_colors["dl"], label="DL Baselines (merged)"),
        Patch(facecolor=cat_colors["ours"], label="Our Approach (merged)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
              framealpha=0.9, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Comprehensive Comparison: Static Tools, DL Baselines, and Our Approach",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, "fig_dl_comparison")


def main():
    print(f"Output directory: {_OUT}\n")
    fig_model_comparison()
    fig_feature_importance()
    fig_confusion_matrices()
    fig_cross_dataset_heatmap()
    fig_feature_ablation()
    fig_dl_comparison()
    print(f"\nDone. All outputs in: {_OUT}")


if __name__ == "__main__":
    main()
