#!/usr/bin/env python3
"""Generate manuscript tables and figures for sampled model comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WITHIN = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1"
    / "model_comparison/model_metrics.csv"
)
DEFAULT_LODO = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_ensemble_v2"
    / "ensemble_metrics.csv"
)
DEFAULT_LODO_DL = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1"
    / "dl/metrics.csv"
)
DEFAULT_LODO_SCAFE = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1"
    / "scafe/scafe_repeated_lodo_summary.csv"
)
DEFAULT_OUTPUT = REPO / "data/derived/sanitized_v1/paper_artifacts"
BASE_MODELS = ("scafe", "textcnn", "bilstm", "codebert", "unixcoder")
DISPLAY = {
    "scafe": "SCAFE",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
    "codebert": "CodeBERT",
    "unixcoder": "UniXcoder",
    "soft_vote": "Soft vote",
    "stacking": "Stacking",
    "confidence_cascade": "Confidence cascade",
}


def _value(frame: pd.DataFrame, dataset: str, method: str, metric: str) -> float:
    row = frame[frame["dataset"].eq(dataset) & frame["method"].eq(method)]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {dataset}/{method}, found {len(row)}")
    return float(row.iloc[0][metric])


def model_table(within: pd.DataFrame, lodo: pd.DataFrame) -> str:
    within = within.rename(columns={"model": "method"})
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled sampled comparison using complete-source chunk aggregation. Within-corpus values pool the five project-group-disjoint DAppSCAN out-of-fold predictions (1,500 records). LODO values use 500 records per corpus; DAppSCAN is development evidence, while DIVE and Empirical-Analysis are external targets. All rows use seed 42.}",
        r"\label{tab:sampled_models}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{DAppSCAN within corpus} & \multicolumn{2}{c}{DIVE LODO} & \multicolumn{2}{c}{Empirical LODO} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & ROC-AUC & PR-AUC & ROC-AUC & PR-AUC & ROC-AUC & PR-AUC \\",
        r"\midrule",
    ]
    for model in BASE_MODELS:
        lines.append(
            f"{DISPLAY[model]} & "
            f"{_value(within, 'DAppSCAN', model, 'roc_auc'):.3f} & "
            f"{_value(within, 'DAppSCAN', model, 'pr_auc'):.3f} & "
            f"{_value(lodo, 'DIVE', model, 'roc_auc'):.3f} & "
            f"{_value(lodo, 'DIVE', model, 'pr_auc'):.3f} & "
            f"{_value(lodo, 'Empirical-Analysis', model, 'roc_auc'):.3f} & "
            f"{_value(lodo, 'Empirical-Analysis', model, 'pr_auc'):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def ensemble_table(lodo: pd.DataFrame) -> str:
    methods = (
        "scafe", "textcnn", "soft_vote", "stacking", "confidence_cascade"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen post-hoc ensemble diagnostic on sampled external targets. TextCNN, the cascade gate, and the stacker are selected or fitted using DAppSCAN within-corpus out-of-fold development predictions only. No DIVE or Empirical-Analysis labels are used for selection.}",
        r"\label{tab:sampled_ensembles}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{DIVE} & \multicolumn{3}{c}{Empirical-Analysis} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & ROC-AUC & PR-AUC & F1 & ROC-AUC & PR-AUC & F1 \\",
        r"\midrule",
    ]
    for method in methods:
        lines.append(
            f"{DISPLAY[method]} & "
            f"{_value(lodo, 'DIVE', method, 'roc_auc'):.3f} & "
            f"{_value(lodo, 'DIVE', method, 'pr_auc'):.3f} & "
            f"{_value(lodo, 'DIVE', method, 'f1'):.3f} & "
            f"{_value(lodo, 'Empirical-Analysis', method, 'roc_auc'):.3f} & "
            f"{_value(lodo, 'Empirical-Analysis', method, 'pr_auc'):.3f} & "
            f"{_value(lodo, 'Empirical-Analysis', method, 'f1'):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def comparison_figure(within: pd.DataFrame, lodo: pd.DataFrame, output: Path) -> None:
    within = within.rename(columns={"model": "method"})
    colors = ["#2f6690", "#3a7d44", "#d17b0f", "#7a5195", "#c44536"]
    labels = [DISPLAY[model] for model in BASE_MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    within_values = [
        _value(within, "DAppSCAN", model, "roc_auc") for model in BASE_MODELS
    ]
    axes[0].bar(np.arange(len(BASE_MODELS)), within_values, color=colors)
    axes[0].set_title("(a) DAppSCAN within-corpus OOF")
    axes[0].set_xticks(np.arange(len(BASE_MODELS)), labels, rotation=25, ha="right")
    axes[0].set_ylim(0.35, 0.9)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)

    x = np.arange(3)
    width = 0.16
    for index, (model, color) in enumerate(zip(BASE_MODELS, colors, strict=True)):
        values = [
            _value(lodo, dataset, model, "roc_auc")
            for dataset in ("DAppSCAN", "DIVE", "Empirical-Analysis")
        ]
        axes[1].bar(x + (index - 2) * width, values, width, label=DISPLAY[model],
                    color=color)
    axes[1].set_title("(b) Sampled leave-one-dataset-out")
    axes[1].set_xticks(x, ("DAppSCAN\n(dev.)", "DIVE", "Empirical"))
    axes[1].set_ylim(0.35, 0.75)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "fig_sampled_model_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        output / "fig_sampled_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--within-metrics", type=Path, default=DEFAULT_WITHIN)
    parser.add_argument("--lodo-metrics", type=Path, default=DEFAULT_LODO)
    parser.add_argument("--lodo-dl-metrics", type=Path, default=DEFAULT_LODO_DL)
    parser.add_argument("--lodo-scafe-metrics", type=Path, default=DEFAULT_LODO_SCAFE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    within = pd.read_csv(args.within_metrics)
    lodo = pd.read_csv(args.lodo_metrics)
    dl_development = pd.read_csv(args.lodo_dl_metrics)
    dl_development = dl_development[dl_development["dataset"].eq("DAppSCAN")].rename(
        columns={"model": "method"}
    )
    scafe_development = pd.read_csv(args.lodo_scafe_metrics)
    scafe_development = scafe_development[
        scafe_development["dataset"].eq("DAppSCAN")
    ].rename(
        columns={
            "model": "method",
            "roc_auc_mean": "roc_auc",
            "pr_auc_mean": "pr_auc",
            "f1_mean": "f1",
            "precision_mean": "precision",
            "recall_mean": "recall",
        }
    )
    scafe_development["method"] = "scafe"
    lodo_models = pd.concat(
        [lodo, dl_development, scafe_development], ignore_index=True, sort=False
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "tab_sampled_model_comparison.tex").write_text(
        model_table(within, lodo_models), encoding="utf-8"
    )
    (output / "tab_sampled_ensembles.tex").write_text(
        ensemble_table(lodo), encoding="utf-8"
    )
    comparison_figure(within, lodo_models, output)
    print(f"Wrote sampled manuscript artifacts to {output}")


if __name__ == "__main__":
    main()
