"""Regenerate primary SCAFE figures from the corrected source-only artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / "sc_risk_score_overleaf" / "figures"
SUMMARY = (
    REPO
    / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo_summary.csv"
)
PREDICTIONS = (
    REPO
    / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo_predictions.csv"
)
ABLATIONS = (
    REPO
    / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical_ablations/scafe_feature_ablations.csv"
)
LABEL_SENSITIVITY = (
    REPO
    / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical_label_sensitivity/scafe_label_sensitivity.csv"
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    png = OUT / f"{name}.png"
    if png.exists():
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def model_comparison() -> None:
    summary = pd.read_csv(SUMMARY)
    summary = summary[summary["model"] == "rf"].set_index("dataset")
    datasets = ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    metrics = [("roc_auc_mean", "ROC-AUC"), ("pr_auc_mean", "PR-AUC"), ("f1_mean", "F1")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(datasets))
    width = 0.24
    for i, (metric, label) in enumerate(metrics):
        rows = summary.loc[datasets]
        ax.bar(
            x + (i - 1) * width,
            rows[metric],
            width,
            yerr=rows[metric.replace("_mean", "_std")],
            capsize=2,
            label=label,
        )
    ax.set_xticks(x, ["DAppSCAN\n(development)", "DIVE", "Empirical-\nAnalysis"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (mean ± seed SD)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "fig_model_comparison")


def confusion_matrices() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    best = {"DAppSCAN": "rf", "DIVE": "rf", "Empirical-Analysis": "rf"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, (dataset, model) in zip(axes, best.items()):
        frame = predictions[(predictions.dataset == dataset) & (predictions.model == model)]
        y = frame.y_true.astype(int)
        pred = frame.y_pred.astype(int)
        cm = pd.crosstab(y, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        pct = cm / max(cm.sum(), 1) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                        color="white" if pct[i, j] > 50 else "black", fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{dataset}\n{model}")
    fig.suptitle("Fixed-Random-Forest transfer confusion matrices", y=1.04)
    fig.tight_layout()
    save(fig, "fig_confusion_matrices")


def heatmap() -> None:
    summary = pd.read_csv(SUMMARY)
    rows = (
        summary[summary["model"] == "rf"]
        .set_index("dataset")
        .loc[["DAppSCAN", "DIVE", "Empirical-Analysis"]]
    )
    values = rows[["roc_auc_mean", "pr_auc_mean", "f1_mean"]].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(3), ["ROC-AUC", "PR-AUC", "F1"])
    ax.set_yticks(range(3), ["DAppSCAN (dev.)", "DIVE", "Empirical-Analysis"])
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    save(fig, "fig_cross_dataset_heatmap")


def feature_ablation() -> None:
    frame = pd.read_csv(ABLATIONS)
    frame = frame[frame["dataset"].isin(["DAppSCAN", "DIVE", "Empirical-Analysis"])]
    variants = ["raw_structural_244", "without_ast_call_dataflow", "ast_call_dataflow_only"]
    labels = ["All 244", "Without relational", "Relational only"]
    datasets = ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    matrix = (
        frame.pivot(index="dataset", columns="feature_variant", values="roc_auc")
        .reindex(index=datasets, columns=variants)
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    image = ax.imshow(matrix, cmap="magma", vmin=0.45, vmax=0.85, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(3), labels, rotation=18, ha="right")
    ax.set_yticks(range(3), ["DAppSCAN (dev.)", "DIVE", "Empirical-Analysis"])
    ax.set_title("Fixed-head feature-group ROC-AUC")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    save(fig, "fig_feature_ablation")


def feature_inventory() -> None:
    labels = ["Raw structural\n(primary)", "Composite scores\n(excluded)", "Identifier/context\n(secondary)"]
    values = [244, 11, 87]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(labels, values, color=["#1565c0", "#b71c1c", "#6a1b9a"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Feature count")
    ax.set_ylim(0, 280)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_feature_importance")


def label_sensitivity() -> None:
    frame = pd.read_csv(LABEL_SENSITIVITY)
    frame = frame[
        frame["exclude_duplicate_sources"]
        & frame["dataset"].isin(["DIVE", "Empirical-Analysis", "FORGE"])
    ].copy()
    labels = {
        "forge_vulnerable_vs_unknown": "Unknown as provisional negative",
        "forge_vulnerable_vs_unknown_suspicious": "Unknown+suspicious as negative",
    }
    frame["label_policy"] = frame["label_policy"].map(labels)
    pivot = frame.pivot(index="dataset", columns="label_policy", values="f1").reindex(
        ["DIVE", "Empirical-Analysis", "FORGE"]
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    pivot.plot.bar(ax=ax, color=["#5e35b1", "#00897b"], width=0.75)
    ax.set_ylabel("F1 at fixed 0.5 cutoff")
    ax.set_xlabel("")
    ax.set_xticklabels(["DIVE", "Empirical-Analysis", "FORGE"], rotation=0)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_ensemble_comparison")


if __name__ == "__main__":
    model_comparison()
    confusion_matrices()
    heatmap()
    feature_ablation()
    feature_inventory()
    label_sensitivity()
