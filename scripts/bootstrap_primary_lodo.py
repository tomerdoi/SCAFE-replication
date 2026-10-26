#!/usr/bin/env python3
"""Bootstrap primary LODO metrics after averaging repeated-seed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, low_memory=False)
    required = {"dataset", "contract_id", "y_true", "y_prob", "random_state"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction columns missing: {missing}")

    averaged = (
        predictions.groupby(["dataset", "contract_id"], as_index=False)
        .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for dataset, group in averaged.groupby("dataset", sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["y_prob"].to_numpy(dtype=float)
        if np.unique(y_true).size != 2:
            continue
        roc_samples = np.empty(args.repetitions, dtype=float)
        pr_samples = np.empty(args.repetitions, dtype=float)
        for repetition in range(args.repetitions):
            while True:
                indices = rng.integers(0, len(group), size=len(group))
                sampled_y = y_true[indices]
                if np.unique(sampled_y).size == 2:
                    break
            sampled_probability = y_prob[indices]
            roc_samples[repetition] = roc_auc_score(
                sampled_y,
                sampled_probability,
            )
            pr_samples[repetition] = average_precision_score(
                sampled_y,
                sampled_probability,
            )
        rows.append(
            {
                "dataset": dataset,
                "n_records": int(len(group)),
                "positive_rate": float(y_true.mean()),
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "roc_auc_bootstrap95_low": float(
                    np.percentile(roc_samples, 2.5)
                ),
                "roc_auc_bootstrap95_high": float(
                    np.percentile(roc_samples, 97.5)
                ),
                "pr_auc": float(average_precision_score(y_true, y_prob)),
                "pr_auc_bootstrap95_low": float(
                    np.percentile(pr_samples, 2.5)
                ),
                "pr_auc_bootstrap95_high": float(
                    np.percentile(pr_samples, 97.5)
                ),
            }
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions.resolve()),
                "seed_aggregation": "mean probability per record",
                "resampling_unit": "record",
                "repetitions": args.repetitions,
                "seed": args.seed,
                "limitations": (
                    "Conditional iid-record intervals; do not capture project, "
                    "corpus, annotation, model-selection, or temporal uncertainty."
                ),
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
