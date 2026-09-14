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
    "source_only_metadata_excluded/scafe_repeated_lodo_summary.csv"
)
PREDICTIONS = (
    REPO
    / "data/derived/ml_results/baseline_comparison/"
    "source_only_metadata_excluded/scafe_repeated_lodo_predictions.csv"
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def model_comparison() -> None:
    summary = pd.read_csv(SUMMARY)
    models = ["rf", "et", "hgb", "xgb"]
    labels = {"rf": "Random Forest", "et": "Extra Trees", "hgb": "HistGradientBoosting", "xgb": "XGBoost"}
    datasets = ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    metrics = [("roc_auc_mean", "ROC-AUC"), ("pr_auc_mean", "PR-AUC"), ("f1_mean", "F1")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    x = np.arange(len(datasets))
    width = 0.18
    for ax, (metric, title) in zip(axes, metrics):
        for i, model in enumerate(models):
            rows = summary[(summary.dataset.isin(datasets)) & (summary.model == model)].set_index("dataset").loc[datasets]
            ax.bar(x + (i - 1.5) * width, rows[metric], width, label=labels[model])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["DAppSCAN", "DIVE", "Empirical-\nAnalysis"], rotation=0)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_model_comparison")


def confusion_matrices() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    best = {"DAppSCAN": "rf", "DIVE": "hgb", "Empirical-Analysis": "hgb"}
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
        ax.set_xticks([0, 1], ["Clean", "Vulnerable"])
        ax.set_yticks([0, 1], ["Clean", "Vulnerable"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{dataset}\n{model}")
    fig.suptitle("Corrected source-only LODO confusion matrices", y=1.04)
    fig.tight_layout()
    save(fig, "fig_confusion_matrices")


if __name__ == "__main__":
    model_comparison()
    confusion_matrices()
