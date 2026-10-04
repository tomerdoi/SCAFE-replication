#!/usr/bin/env python3
"""Report fixed-head performance by lightweight-parser execution outcome."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "data/derived/parser_compatibility_audit/parser_compatibility_sample.csv"
PREDICTIONS = (
    REPO
    / "data/derived/ml_results/baseline_comparison/source_only_raw_canonical/"
    "scafe_repeated_lodo_predictions.csv"
)
OUTPUT = (
    REPO
    / "data/derived/parser_compatibility_audit/"
    "parser_stratified_performance.json"
)


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[
        (predictions["model"] == "rf") & (predictions["random_state"] == 42)
    ]
    keys = ["source_dataset", "contract_id"]
    sample_counts = sample.groupby(keys).size()
    prediction_counts = (
        predictions.groupby(["dataset", "contract_id"]).size()
        .rename_axis(index=keys)
    )
    unique_keys = sample_counts[
        sample_counts.eq(1) & prediction_counts.reindex(sample_counts.index).eq(1)
    ].index
    sample = sample.set_index(keys).loc[unique_keys].reset_index()
    predictions = (
        predictions.rename(columns={"dataset": "source_dataset"})
        .set_index(keys)
        .loc[unique_keys]
        .reset_index()
    )
    merged = sample.merge(
        predictions[["source_dataset", "contract_id", "y_true", "y_prob"]],
        on=["source_dataset", "contract_id"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, parse_ok), group in merged.groupby(
        ["source_dataset", "ast_parse_ok"], sort=True
    ):
        y_true = group["y_true"].astype(int)
        row: dict[str, object] = {
            "dataset": dataset,
            "parse_success": bool(parse_ok),
            "n": int(len(group)),
            "positives": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "mean_score": float(group["y_prob"].mean()),
            "roc_auc": None,
            "pr_auc": None,
        }
        if y_true.nunique() == 2:
            row["roc_auc"] = float(roc_auc_score(y_true, group["y_prob"]))
            row["pr_auc"] = float(average_precision_score(y_true, group["y_prob"]))
        rows.append(row)
    payload = {
        "protocol": "seed-42 fixed-RF held-out predictions joined to the prespecified parser sample",
        "matched_records": int(len(merged)),
        "rows": rows,
        "limitations": (
            "Post-hoc stratification on a 500-record sample per corpus; small failure "
            "strata and label imbalance preclude causal interpretation."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
