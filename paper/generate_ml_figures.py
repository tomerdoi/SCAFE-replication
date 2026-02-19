"""
Generate publication-ready figures and LaTeX tables for the ML evaluation.

Output: paper/figures/
  - fig_roc_curves.pdf          ROC curves (DAppSCAN-only vs merged)
  - fig_pr_curves.pdf           PR curves (DAppSCAN-only vs merged)
  - fig_feature_importance.pdf  Top-20 feature importances (horizontal bar)
  - fig_confusion_matrices.pdf  Confusion matrices (best model, DAppSCAN & merged)
  - fig_model_comparison.pdf    Grouped bar chart comparing models across experiments
  - fig_cross_dataset_heatmap.pdf  Cross-dataset transfer heatmap
  - tab_model_comparison.tex    LaTeX table: main model comparison
  - tab_cross_dataset.tex       LaTeX table: cross-dataset results
  - tab_dataset_overview.tex    LaTeX table: dataset overview
  - tab_feature_importance.tex  LaTeX table: top-20 features

Usage:
    /path/to/venv/bin/python paper/generate_ml_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_ML_DIR = _REPO / "data" / "derived" / "ml_results"
_OUT = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

# Publication style
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

COLORS = {
    "rf": "#2196F3",
    "xgb": "#FF9800",
    "gbc": "#4CAF50",
    "hgb": "#9C27B0",
    "et": "#F44336",
    "logreg": "#607D8B",
    "logreg_l1": "#795548",
    "baseline_risk_z": "#9E9E9E",
}
MODEL_NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "gbc": "GradientBoosting",
    "hgb": "HistGradientBoosting",
    "et": "Extra Trees",
    "logreg": "Logistic Reg.",
    "logreg_l1": "Logistic Reg. (L1)",
    "baseline_risk_z": "Baseline (risk score)",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(_OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Model comparison — grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_comparison():
    print("Generating: fig_model_comparison")
    comp = pd.read_csv(_ML_DIR / "model_comparison.csv")
    comp_all = comp[comp["feature_set"] == "all"].copy()

    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")
    merged = cross[cross["experiment"] == "C_merged_cv"].copy()

    models_order = ["rf", "xgb", "gbc", "hgb", "et", "logreg", "baseline_risk_z"]
    metrics = ["roc_auc_mean", "pr_auc_mean", "f1_mean"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(models_order))
        width = 0.35

        # DAppSCAN-only values
        vals_d = []
        errs_d = []
        for m in models_order:
            row = comp_all[comp_all["model"] == m]
            if len(row):
                vals_d.append(row.iloc[0][metric])
                std_col = metric.replace("_mean", "_std")
                errs_d.append(row.iloc[0].get(std_col, 0))
            else:
                vals_d.append(0)
                errs_d.append(0)

        # Merged values (only for models in merged)
        vals_m = []
        errs_m = []
        merged_metric_map = {"roc_auc_mean": "roc_auc", "pr_auc_mean": "pr_auc", "f1_mean": "f1"}
        merged_std_map = {"roc_auc_mean": "roc_auc_std", "pr_auc_mean": "pr_auc_std", "f1_mean": None}
        for m in models_order:
            row = merged[merged["model"] == m]
            if len(row):
                vals_m.append(row.iloc[0][merged_metric_map[metric]])
                std_col = merged_std_map[metric]
                errs_m.append(row.iloc[0].get(std_col, 0) if std_col else 0)
            else:
                vals_m.append(0)
                errs_m.append(0)

        bars1 = ax.bar(x - width/2, vals_d, width, label="DAppSCAN only",
                       color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width/2, vals_m, width, label="Merged (DAppSCAN+FORGE)",
                       color="#FF9800", alpha=0.8, edgecolor="white", linewidth=0.5)

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in models_order],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Model Performance: DAppSCAN Only vs. Merged Training", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_model_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Feature importance (top 20)
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance():
    print("Generating: fig_feature_importance")
    sel = pd.read_csv(_ML_DIR / "enhanced_selected_features.csv")
    top = sel.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))[::-1]
    bars = ax.barh(range(len(top)), top["importance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    labels = top["feature"].str.replace("_", " ").str.title()
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Importance (RF, 2-stage selection)")
    ax.set_title("Top 20 Features by Importance")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    for i, (val, name) in enumerate(zip(top["importance"], top["feature"])):
        ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=7, color="#333")

    fig.tight_layout()
    _save(fig, "fig_feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
def fig_confusion_matrices():
    print("Generating: fig_confusion_matrices")
    comp = pd.read_csv(_ML_DIR / "model_comparison.csv")
    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")

    # DAppSCAN-only best (GBC)
    d_gbc = comp[(comp["model"] == "gbc") & (comp["feature_set"] == "all")].iloc[0]
    # Get TP/FN/FP/TN from cross_dataset D_per_dataset_cv
    d_cv = cross[(cross["experiment"] == "D_per_dataset_cv") & (cross["model"] == "gbc") & (cross["dataset"] == "DAppSCAN")].iloc[0]

    # Merged best (GBC)
    m_gbc = cross[(cross["experiment"] == "C_merged_cv") & (cross["model"] == "gbc")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, data, title in [
        (axes[0], d_cv, "DAppSCAN Only (GBC)"),
        (axes[1], m_gbc, "Merged DAppSCAN+FORGE (GBC)"),
    ]:
        cm = np.array([[int(data["TN"]), int(data["FP"])],
                        [int(data["FN"]), int(data["TP"])]])
        cm_pct = cm / cm.sum() * 100

        cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#2196F3"])
        im = ax.imshow(cm_pct, cmap=cmap, aspect="auto", vmin=0, vmax=100)

        for i in range(2):
            for j in range(2):
                txt = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nClean", "Predicted\nVulnerable"])
        ax.set_yticklabels(["Actual\nClean", "Actual\nVulnerable"])
        ax.set_title(title, fontsize=11, pad=10)

        n = int(data["n_samples"])
        pos = int(data["n_pos"])
        ax.text(0.5, -0.22, f"N={n:,}  |  Pos={pos:,} ({pos/n*100:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=8, color="#666")

    fig.suptitle("Confusion Matrices — Best Model (GradientBoosting)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_confusion_matrices")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Cross-dataset transfer heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_cross_dataset_heatmap():
    print("Generating: fig_cross_dataset_heatmap")
    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")

    # Build a matrix: rows = train_on, cols = test_on, value = best ROC-AUC (using GBC)
    scenarios = [
        ("DAppSCAN", "DAppSCAN", "D_per_dataset_cv", "gbc"),
        ("DAppSCAN", "FORGE", "A_cross_predict", "gbc"),
        ("FORGE", "FORGE", "D_per_dataset_cv", "gbc"),
        ("FORGE", "DAppSCAN", "B_reverse", "gbc"),
        ("DAppSCAN+FORGE", "Merged (CV)", "C_merged_cv", "gbc"),
    ]

    labels_train = ["DAppSCAN", "FORGE", "DAppSCAN\n+FORGE"]
    labels_test = ["DAppSCAN", "FORGE", "Merged (CV)"]

    matrix_roc = np.full((3, 3), np.nan)
    matrix_f1 = np.full((3, 3), np.nan)

    train_map = {"DAppSCAN": 0, "FORGE": 1, "DAppSCAN+FORGE": 2}
    test_map = {"DAppSCAN": 0, "FORGE": 1, "Merged (CV)": 2}

    for train_on, test_on, exp, model in scenarios:
        if exp == "A_cross_predict":
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model) &
                        (cross["dataset"].str.contains("vuln_vs_unknown"))].iloc[0]
        elif exp == "D_per_dataset_cv":
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model) &
                        (cross["dataset"] == test_on.replace("\n", ""))].iloc[0]
        else:
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model)].iloc[0]

        ti, tj = train_map[train_on], test_map[test_on]
        matrix_roc[ti, tj] = row["roc_auc"]
        matrix_f1[ti, tj] = row["f1"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, matrix, title, vmin in [(axes[0], matrix_roc, "ROC-AUC", 0.4), (axes[1], matrix_f1, "F1 Score", 0.1)]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Metric improvement — DAppSCAN vs Merged
# ─────────────────────────────────────────────────────────────────────────────
def fig_improvement_radar():
    print("Generating: fig_metric_improvement")
    comp = pd.read_csv(_ML_DIR / "model_comparison.csv")
    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")

    # GBC: DAppSCAN-only vs merged
    d = comp[(comp["model"] == "gbc") & (comp["feature_set"] == "all")].iloc[0]
    m = cross[(cross["experiment"] == "C_merged_cv") & (cross["model"] == "gbc")].iloc[0]

    metrics = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]
    d_vals = [d["roc_auc_mean"], d["pr_auc_mean"], d["f1_mean"], d["precision_mean"], d["recall_mean"]]
    m_vals = [m["roc_auc"], m["pr_auc"], m["f1"], m["precision"], m["recall"]]
    improvements = [(mv - dv) / dv * 100 for dv, mv in zip(d_vals, m_vals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: side-by-side bars
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w/2, d_vals, w, label="DAppSCAN only", color="#2196F3", alpha=0.85, edgecolor="white")
    ax1.bar(x + w/2, m_vals, w, label="Merged (DAppSCAN+FORGE)", color="#FF9800", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.set_title("GBC: Absolute Performance")
    ax1.grid(axis="y", alpha=0.3)

    for i, (dv, mv) in enumerate(zip(d_vals, m_vals)):
        ax1.text(i - w/2, dv + 0.02, f"{dv:.3f}", ha="center", fontsize=7, color="#1565C0")
        ax1.text(i + w/2, mv + 0.02, f"{mv:.3f}", ha="center", fontsize=7, color="#E65100")

    # Right: improvement %
    colors = ["#4CAF50" if imp > 0 else "#F44336" for imp in improvements]
    ax2.bar(metrics, improvements, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Relative Improvement (%)")
    ax2.set_title("Improvement from Merging Datasets")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    for i, (imp, m_name) in enumerate(zip(improvements, metrics)):
        ax2.text(i, imp + 1, f"+{imp:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#2E7D32")

    fig.tight_layout()
    _save(fig, "fig_metric_improvement")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Feature set ablation
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_ablation():
    print("Generating: fig_feature_ablation")
    comp = pd.read_csv(_ML_DIR / "model_comparison.csv")

    # Best model per feature set
    fs_order = ["all", "static_only", "components_only"]
    fs_labels = ["All Features\n(292)", "Static Only\n(257)", "Components Only\n(9)"]

    models = ["rf", "xgb", "gbc", "hgb"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [("roc_auc_mean", "ROC-AUC"), ("pr_auc_mean", "PR-AUC"), ("f1_mean", "F1")]

    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(fs_order))
        width = 0.18

        for i, model in enumerate(models):
            vals = []
            for fs in fs_order:
                row = comp[(comp["model"] == model) & (comp["feature_set"] == fs)]
                vals.append(row.iloc[0][metric] if len(row) else 0)
            ax.bar(x + i * width - 1.5 * width, vals, width,
                   label=MODEL_NAMES.get(model, model), color=COLORS[model], alpha=0.85,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(fs_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Feature Set Ablation Study (DAppSCAN)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_feature_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Tables
# ─────────────────────────────────────────────────────────────────────────────
def tab_dataset_overview():
    print("Generating: tab_dataset_overview.tex")
    tex = r"""\begin{table}[t]
\centering
\caption{Overview of evaluation datasets.}
\label{tab:datasets}
\begin{tabular}{lrrrrr}
\hline
\textbf{Dataset} & \textbf{Contracts} & \textbf{Clean} & \textbf{Vulnerable} & \textbf{Other} & \textbf{Pos.\ Rate} \\
\hline
DAppSCAN & 21{,}457 & 20{,}509 & 948 & 0 & 4.4\% \\
FORGE & 4{,}872 & 0 & 1{,}244 & 3{,}628$^\dagger$ & 52.6\%$^\ddagger$ \\
AccessControlVuln & 94 & 0 & 94 & 0 & 100\% \\
\hline
\textbf{Total} & \textbf{26{,}423} & 20{,}509 & 2{,}286 & 3{,}628 & --- \\
\hline
\end{tabular}
\\[4pt]
{\footnotesize $^\dagger$2{,}505 suspicious + 1{,}123 unknown. $^\ddagger$Vulnerable vs.\ unknown only.}
\end{table}
"""
    (_OUT / "tab_dataset_overview.tex").write_text(tex)
    print("  Saved tab_dataset_overview.tex")


def tab_model_comparison():
    print("Generating: tab_model_comparison.tex")
    comp = pd.read_csv(_ML_DIR / "model_comparison.csv")
    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")

    comp_all = comp[comp["feature_set"] == "all"].sort_values("roc_auc_mean", ascending=False)
    merged = cross[cross["experiment"] == "C_merged_cv"].sort_values("roc_auc", ascending=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Classification performance on DAppSCAN (single-dataset) and merged (DAppSCAN + FORGE) training. Best values in \textbf{bold}.}",
        r"\label{tab:model_comparison}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll ccccc}",
        r"\hline",
        r"\textbf{Training} & \textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} \\",
        r"\hline",
    ]

    # DAppSCAN-only results
    best_d = {"roc": comp_all["roc_auc_mean"].max(), "pr": comp_all["pr_auc_mean"].max(),
              "f1": comp_all["f1_mean"].max(), "prec": comp_all["precision_mean"].max(),
              "rec": comp_all["recall_mean"].max()}

    for _, row in comp_all.iterrows():
        if row["model"] == "logreg_l1":
            continue
        name = MODEL_NAMES.get(row["model"], row["model"])
        roc = f"{row['roc_auc_mean']:.3f}$\\pm${row['roc_auc_std']:.3f}"
        pr = f"{row['pr_auc_mean']:.3f}"
        f1 = f"{row['f1_mean']:.3f}"
        prec = f"{row['precision_mean']:.3f}"
        rec = f"{row['recall_mean']:.3f}"
        lines.append(f"DAppSCAN & {name} & {roc} & {pr} & {f1} & {prec} & {rec} \\\\")

    lines.append(r"\hline")

    # Merged results
    for _, row in merged.iterrows():
        name = MODEL_NAMES.get(row["model"], row["model"])
        std_str = f"$\\pm${row['roc_auc_std']:.3f}" if pd.notna(row.get("roc_auc_std")) else ""
        roc = f"{row['roc_auc']:.3f}{std_str}"
        pr = f"{row['pr_auc']:.3f}"
        f1 = f"{row['f1']:.3f}"
        prec = f"{row['precision']:.3f}"
        rec = f"{row['recall']:.3f}"
        lines.append(f"Merged & {name} & {roc} & {pr} & {f1} & {prec} & {rec} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (_OUT / "tab_model_comparison.tex").write_text("\n".join(lines))
    print("  Saved tab_model_comparison.tex")


def tab_cross_dataset():
    print("Generating: tab_cross_dataset.tex")
    cross = pd.read_csv(_ML_DIR / "cross_dataset_evaluation.csv")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset transfer results (GradientBoosting). Models trained on one dataset and evaluated on another.}",
        r"\label{tab:cross_dataset}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll ccccc r}",
        r"\hline",
        r"\textbf{Train On} & \textbf{Test On} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{N} \\",
        r"\hline",
    ]

    scenarios = [
        ("D_per_dataset_cv", "gbc", "DAppSCAN", "DAppSCAN", "DAppSCAN"),
        ("A_cross_predict", "gbc", "DAppSCAN", "FORGE (vuln\\_vs\\_unknown)", "FORGE (vuln_vs_unknown)"),
        ("A_cross_predict", "gbc", "DAppSCAN", "AccessControlVuln", "AccessControlVuln"),
        ("B_reverse", "gbc", "FORGE", "DAppSCAN", "DAppSCAN"),
        ("D_per_dataset_cv", "gbc", "FORGE", "FORGE", "FORGE"),
        ("C_merged_cv", "gbc", "DAppSCAN+FORGE", "Merged (CV)", "merged"),
    ]

    for exp, model, train_label, test_label, ds_filter in scenarios:
        if exp == "A_cross_predict" and "vuln_vs_unknown" in ds_filter:
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model) &
                        (cross["dataset"].str.contains("vuln_vs_unknown"))]
        elif exp == "D_per_dataset_cv":
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model) &
                        (cross["dataset"] == ds_filter)]
        else:
            row = cross[(cross["experiment"] == exp) & (cross["model"] == model)]

        if len(row) == 0:
            continue
        r = row.iloc[0]

        roc = f"{r['roc_auc']:.3f}" if pd.notna(r["roc_auc"]) else "---"
        pr = f"{r['pr_auc']:.3f}" if pd.notna(r["pr_auc"]) else "---"
        f1 = f"{r['f1']:.3f}" if pd.notna(r["f1"]) else "---"
        prec = f"{r['precision']:.3f}" if pd.notna(r["precision"]) else "---"
        rec = f"{r['recall']:.3f}"
        n = f"{int(r['n_samples']):,}"

        lines.append(f"{train_label} & {test_label} & {roc} & {pr} & {f1} & {prec} & {rec} & {n} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (_OUT / "tab_cross_dataset.tex").write_text("\n".join(lines))
    print("  Saved tab_cross_dataset.tex")


def tab_feature_importance():
    print("Generating: tab_feature_importance.tex")
    sel = pd.read_csv(_ML_DIR / "enhanced_selected_features.csv")
    top = sel.head(20)

    feat_descriptions = {
        "comp_complexity": "Composite complexity score",
        "n_risky_bigrams": "Risky code bigram count",
        "n_written_state_vars": "State variables written",
        "halstead_effort": "Halstead effort metric",
        "halstead_difficulty": "Halstead difficulty metric",
        "n_assignments": "Assignment statement count",
        "n_arithmetic_ops": "Arithmetic operation count",
        "loc": "Lines of code",
        "std_cyclomatic": "Cyclomatic complexity (std)",
        "bigram_entropy": "Code bigram entropy",
        "risky_bigram_density": "Risky bigram density",
        "sv_fanout_entropy": "State-variable fanout entropy",
        "fn_loc_gini": "Function size Gini coefficient",
        "cyclomatic_variance_ratio": "Cyclomatic variance ratio",
        "arithmetic_density": "Arithmetic density",
        "fn_loc_cv": "Function size coeff. of variation",
        "n_state_variables": "State variable count",
        "avg_cyclomatic": "Avg. cyclomatic complexity",
        "std_fn_arith": "Arithmetic per-function (std)",
        "token_entropy": "Token entropy",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Top 20 features by importance (two-stage MI + RF selection, 292$\rightarrow$80).}",
        r"\label{tab:features}",
        r"\begin{tabular}{rlll}",
        r"\hline",
        r"\textbf{Rank} & \textbf{Feature} & \textbf{Description} & \textbf{Importance} \\",
        r"\hline",
    ]

    for _, row in top.iterrows():
        rank = int(row["rank"])
        feat = row["feature"].replace("_", r"\_")
        desc = feat_descriptions.get(row["feature"], "")
        imp = f"{row['importance']:.4f}"
        lines.append(f"{rank} & \\texttt{{{feat}}} & {desc} & {imp} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (_OUT / "tab_feature_importance.tex").write_text("\n".join(lines))
    print("  Saved tab_feature_importance.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Output directory: {_OUT}\n")

    # Figures
    fig_model_comparison()
    fig_feature_importance()
    fig_confusion_matrices()
    fig_cross_dataset_heatmap()
    fig_improvement_radar()
    fig_feature_ablation()

    # Tables
    tab_dataset_overview()
    tab_model_comparison()
    tab_cross_dataset()
    tab_feature_importance()

    print(f"\nDone. All outputs in: {_OUT}")


if __name__ == "__main__":
    main()
