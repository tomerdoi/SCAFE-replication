#!/usr/bin/env python3
"""Generate fixed-budget audit-queue metrics from canonical transfer predictions.

This diagnostic does not tune a decision threshold.  It measures how much the
fixed Random Forest SCAFE head enriches top-k review queues relative
to each held-out dataset's vulnerable-label prevalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / "sc_risk_score_overleaf"
PREDICTIONS = (
    REPO / "data/derived/ml_results/baseline_comparison/"
    "source_only_raw_canonical/scafe_repeated_lodo_predictions.csv"
)
OUTPUT = REPO / "data/derived/ml_results/baseline_comparison/source_only_raw_canonical"
PAPER_TABLE = PAPER / "figures/tab_operational_triage.tex"
FIXED_HEAD = {
    "DAppSCAN": "rf",
    "DIVE": "rf",
    "Empirical-Analysis": "rf",
}
BUDGETS = (0.01, 0.05, 0.10)


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    rows: list[dict[str, object]] = []
    for dataset, model in FIXED_HEAD.items():
        selected = predictions[
            (predictions["dataset"] == dataset) & (predictions["model"] == model)
        ]
        for seed, fold in selected.groupby("random_state", sort=True):
            fold = fold.sort_values("y_prob", ascending=False, kind="mergesort")
            n = len(fold)
            positives = int(fold["y_true"].sum())
            prevalence = positives / max(n, 1)
            for budget in BUDGETS:
                k = max(1, round(n * budget))
                reviewed = fold.iloc[:k]
                precision = float(reviewed["y_true"].mean())
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": int(seed),
                        "budget_fraction": budget,
                        "reviewed_contracts": k,
                        "positive_prevalence": prevalence,
                        "precision_at_budget": precision,
                        "recall_at_budget": float(reviewed["y_true"].sum() / max(positives, 1)),
                        "precision_lift": precision / prevalence if prevalence else 0.0,
                    }
                )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUTPUT / "scafe_operational_triage.csv", index=False)
    summary = (
        detailed.groupby(["dataset", "model", "budget_fraction"], as_index=False)
        .agg(
            reviewed_contracts=("reviewed_contracts", "mean"),
            positive_prevalence=("positive_prevalence", "mean"),
            precision_at_budget=("precision_at_budget", "mean"),
            recall_at_budget=("recall_at_budget", "mean"),
            precision_lift=("precision_lift", "mean"),
        )
    )
    summary.to_csv(OUTPUT / "scafe_operational_triage_summary.csv", index=False)
    payload = {
        "protocol": "fixed top-k retrieval; no threshold tuning",
        "heads": FIXED_HEAD,
        "budgets": list(BUDGETS),
        "rows": summary.to_dict(orient="records"),
    }
    (OUTPUT / "scafe_operational_triage.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    display = {
        "DAppSCAN": "DAppSCAN (development)",
        "DIVE": "DIVE",
        "Empirical-Analysis": "Empirical-Analysis",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fixed-budget audit-queue retrieval for the fixed Random Forest head "
        r"(five-seed means). A budget is the fraction reviewed after score ranking; "
        r"precision lift is precision divided by positive-label prevalence. "
        r"DAppSCAN is a development diagnostic.}",
        r"\label{tab:operational_triage}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Review budget} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{Precision lift} \\",
        r"\midrule",
    ]
    for dataset in FIXED_HEAD:
        for _, row in summary[summary["dataset"] == dataset].iterrows():
            lines.append(
                f"{display[dataset]} & {row['budget_fraction']:.0%}".replace("%", r"\%")
                + f" & "
                f"{row['precision_at_budget']:.3f} & {row['recall_at_budget']:.3f} & "
                f"${row['precision_lift']:.1f}\\times$ \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}", ""])
    PAPER_TABLE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
