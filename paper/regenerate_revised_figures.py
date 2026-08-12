#!/usr/bin/env python3
"""Generate manuscript figures from the revised, leakage-safe artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"
OUT = Path("/Users/tomer.doitshman/PycharmProjects/sc_risk_score_overleaf/figures")

plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 180})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def best_roc(summary: pd.DataFrame) -> pd.DataFrame:
    column = "roc_auc_mean" if "roc_auc_mean" in summary else "roc_auc"
    return summary.loc[summary.groupby("dataset")[column].idxmax()].copy()


def lodo_results() -> pd.DataFrame:
    path = RESULTS / "strict_three_dataset_repeated" / "scafe_repeated_lodo_summary.csv"
    return pd.read_csv(path)


def plot_model_comparison() -> None:
    df = best_roc(lodo_results())
    order = ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    df["dataset"] = pd.Categorical(df["dataset"], order, ordered=True)
    df = df.sort_values("dataset")
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (metric, label, color) in enumerate(
        [
            ("roc_auc_mean", "ROC-AUC", "#1565c0"),
            ("pr_auc_mean", "PR-AUC", "#2e7d32"),
            ("f1_mean", "F1", "#ef6c00"),
        ]
    ):
        ax.bar(
            x + (i - 1) * width,
            df[metric],
            width,
            yerr=df[metric.replace("_mean", "_std")],
            capsize=2,
            label=label,
            color=color,
        )
    ax.set_xticks(x, df["dataset"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (mean ± seed SD)")
    ax.set_title("Strict repeated LODO performance")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_model_comparison.pdf")


def plot_cross_dataset_heatmap() -> None:
    df = best_roc(lodo_results()).set_index("dataset").loc[
        ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    ]
    values = df[["roc_auc_mean", "pr_auc_mean", "f1_mean"]].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(3), ["ROC-AUC", "PR-AUC", "F1"])
    ax.set_yticks(range(len(df)), df.index)
    ax.set_title("Held-out-dataset results")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    save(fig, "fig_cross_dataset_heatmap.pdf")


def plot_confusion_matrices() -> None:
    path = RESULTS / "strict_three_dataset_lodo" / "scafe_leave_one_dataset_out.csv"
    df = pd.read_csv(path)
    rows = best_roc(df)
    order = ["DAppSCAN", "DIVE", "Empirical-Analysis"]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    for ax, dataset in zip(axes, order):
        row = rows[rows["dataset"] == dataset].iloc[0]
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Clean", "Vulnerable"])
        ax.set_yticks([0, 1], ["Clean", "Vulnerable"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(dataset)
    fig.suptitle("Representative strict LODO confusion matrices")
    fig.tight_layout()
    save(fig, "fig_confusion_matrices.pdf")


def plot_feature_ablation() -> None:
    df = pd.read_csv(RESULTS / "scafe_feature_ablations.csv")
    df = df[df["dataset"].isin(["DAppSCAN", "DIVE", "Empirical-Analysis", "FORGE"])]
    df = df.groupby(["dataset", "feature_variant"], as_index=False)[
        ["roc_auc", "pr_auc", "f1"]
    ].max()
    variants = ["structural_all", "structural_without_components", "structural_components_only", "ast_call_graph_only"]
    labels = ["All 257", "No components", "9 components", "14 AST/call"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc", "f1"], ["ROC-AUC", "PR-AUC", "F1"]):
        matrix = (
            df.pivot(index="dataset", columns="feature_variant", values=metric)
            .reindex(index=["DAppSCAN", "DIVE", "Empirical-Analysis", "FORGE"], columns=variants)
        )
        image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(4), ["DAppSCAN", "DIVE", "Empirical", "FORGE"], fontsize=7)
    fig.suptitle("Current feature-group ablation under duplicate-safe LODO")
    fig.tight_layout()
    save(fig, "fig_feature_ablation.pdf")


def plot_label_sensitivity() -> None:
    df = pd.read_csv(RESULTS / "scafe_label_sensitivity.csv")
    df = df[df["model"] == "rf"]
    pivot = df.groupby(["dataset", "label_policy"], as_index=False)["f1"].mean().pivot(
        index="dataset", columns="label_policy", values="f1"
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    pivot.plot.bar(ax=ax, color=["#5e35b1", "#00897b"], width=0.75)
    ax.set_ylabel("F1 at training-only threshold")
    ax.set_xlabel("")
    ax.set_title("Label-policy sensitivity (current audit)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="", frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ensemble_comparison.pdf")


def plot_feature_inventory() -> None:
    counts = {"Structural core": 248, "Components": 9, "Identifiers": 73, "Context": 14}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(counts.keys(), counts.values(), color=["#1565c0", "#ef6c00", "#6a1b9a", "#00897b"])
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Number of features")
    ax.set_title("Current source-only feature inventory")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_feature_importance.pdf")


def plot_semantic_diagnostic() -> None:
    structural = pd.read_csv(
        RESULTS / "strict_three_dataset_lodo" / "scafe_leave_one_dataset_out.csv"
    )
    semantic = pd.read_csv(RESULTS / "semantic_revised" / "scafe_semantic_leave_one_dataset_out.csv")
    structural = structural.groupby("dataset", as_index=False)["roc_auc"].max()
    semantic = semantic.groupby("dataset", as_index=False)["roc_auc"].max()
    datasets = ["DAppSCAN", "DIVE", "Empirical-Analysis", "FORGE"]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for offset, frame, label, color in [
        (-0.18, structural, "Structural LODO", "#1565c0"),
        (0.18, semantic, "Semantic/context diagnostic", "#6a1b9a"),
    ]:
        values = frame.set_index("dataset").reindex(datasets)["roc_auc"]
        ax.bar(x + offset, values, width=0.36, label=label, color=color)
    ax.axhline(0.5, color="#777", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(x, datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best ROC-AUC")
    ax.set_title("Structural versus semantic/context diagnostic")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_dl_comparison.pdf")


if __name__ == "__main__":
    plot_model_comparison()
    plot_cross_dataset_heatmap()
    plot_confusion_matrices()
    plot_feature_ablation()
    plot_label_sensitivity()
    plot_feature_inventory()
    plot_semantic_diagnostic()
