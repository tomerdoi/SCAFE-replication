#!/usr/bin/env python3
"""Generate the manuscript feature-ablation figure from frozen metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = (
    "raw_structural_244",
    "ast_call_dataflow_only",
    "without_ast_call_dataflow",
    "without_complexity_distribution",
    "without_lexical_statistics",
    "without_security_patterns",
    "without_syntax_organization",
)
LABELS = {
    "raw_structural_244": "All 244",
    "ast_call_dataflow_only": "Relational only",
    "without_ast_call_dataflow": "Without relational",
    "without_complexity_distribution": "Without complexity",
    "without_lexical_statistics": "Without lexical",
    "without_security_patterns": "Without security patterns",
    "without_syntax_organization": "Without syntax/organization",
}
DATASETS = ("DIVE", "Empirical-Analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[
        frame["dataset"].isin(DATASETS)
        & frame["feature_variant"].isin(ORDER)
    ]
    if len(frame) != len(DATASETS) * len(ORDER):
        raise ValueError("Expected one ablation row per target and variant")

    x = np.arange(len(ORDER))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    for offset, dataset in zip((-width / 2, width / 2), DATASETS, strict=True):
        values = [
            float(
                frame[
                    frame["dataset"].eq(dataset)
                    & frame["feature_variant"].eq(variant)
                ]["roc_auc"].iloc[0]
            )
            for variant in ORDER
        ]
        axis.bar(x + offset, values, width, label=dataset)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0.45, 0.61)
    axis.set_xticks(x, [LABELS[item] for item in ORDER], rotation=25, ha="right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
