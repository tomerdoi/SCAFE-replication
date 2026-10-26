#!/usr/bin/env python3
"""Create aligned within-corpus model metrics with project-bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_sampled_prediction_ensembles import METRICS, metric_values  # noqa: E402


DEFAULT_SCAFE = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1"
    / "scafe/scafe_within_corpus_predictions.csv"
)
DEFAULT_DL = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1"
    / "dl/dl_within_corpus_oof_predictions.csv"
)
DEFAULT_OUTPUT = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1"
    / "model_comparison"
)
MODELS = ("scafe", "textcnn", "bilstm", "codebert", "unixcoder")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align(
    scafe: pd.DataFrame,
    neural: pd.DataFrame,
    *,
    scafe_model: str = "rf",
) -> pd.DataFrame:
    scafe = scafe[
        scafe["dataset"].eq("DAppSCAN")
        & scafe["model"].eq(scafe_model)
        & scafe["random_state"].eq(42)
    ].copy()
    if len(scafe) != 1_500 or scafe["contract_id"].duplicated().any():
        raise ValueError("Expected 1,500 unique sampled SCAFE OOF predictions")
    aligned = scafe[
        ["contract_id", "source_identity_hash", "y_true", "y_prob"]
    ].rename(columns={"y_prob": "scafe_prob"})
    groups = None
    for model in MODELS[1:]:
        part = neural[
            neural["dataset"].eq("DAppSCAN")
            & neural["model"].eq(model)
            & neural["random_state"].eq(42)
        ].copy()
        if len(part) != 1_500 or part["contract_id"].duplicated().any():
            raise ValueError(f"Expected 1,500 unique {model} OOF predictions")
        if groups is None:
            groups = part[["contract_id", "group"]].copy()
        aligned = aligned.merge(
            part[["contract_id", "source_identity_hash", "y_true", "y_prob"]]
            .rename(columns={
                "source_identity_hash": f"{model}_source_identity_hash",
                "y_true": f"{model}_y_true",
                "y_prob": f"{model}_prob",
            }),
            on="contract_id",
            how="inner",
            validate="one_to_one",
        )
        if not np.array_equal(aligned["y_true"], aligned.pop(f"{model}_y_true")):
            raise ValueError(f"Label mismatch for {model}")
        if not np.array_equal(
            aligned["source_identity_hash"],
            aligned.pop(f"{model}_source_identity_hash"),
        ):
            raise ValueError(f"Source-identity mismatch for {model}")
    if groups is None:
        raise ValueError("No neural predictions")
    aligned = aligned.merge(groups, on="contract_id", validate="one_to_one")
    if len(aligned) != 1_500 or aligned["group"].nunique() < 2:
        raise ValueError("Incomplete OOF alignment or project groups")
    return aligned.sort_values("contract_id").reset_index(drop=True)


def summarize(
    aligned: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = aligned["y_true"].to_numpy(dtype=int)
    probabilities = {
        model: aligned[f"{model}_prob"].to_numpy(dtype=float)
        for model in MODELS
    }
    metric_rows = [
        {
            "dataset": "DAppSCAN",
            "model": model,
            "n_contracts": len(aligned),
            "n_projects": int(aligned["group"].nunique()),
            "positive_prevalence": float(y_true.mean()),
            **metric_values(y_true, probability),
        }
        for model, probability in probabilities.items()
    ]

    group_to_indices = {
        group: indices.to_numpy(dtype=int)
        for group, indices in aligned.groupby("group").groups.items()
    }
    groups = np.asarray(list(group_to_indices), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = {model: {metric: [] for metric in METRICS} for model in MODELS}
    deltas = {
        model: {metric: [] for metric in METRICS}
        for model in MODELS if model != "scafe"
    }
    for _ in range(repetitions):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        sampled = {
            model: metric_values(sampled_y, probability[indices])
            for model, probability in probabilities.items()
        }
        for model, values in sampled.items():
            for metric in METRICS:
                estimates[model][metric].append(values[metric])
                if model != "scafe":
                    deltas[model][metric].append(
                        values[metric] - sampled["scafe"][metric]
                    )
    interval_rows = []
    for model, probability in probabilities.items():
        point = metric_values(y_true, probability)
        scafe_point = metric_values(y_true, probabilities["scafe"])
        for metric in METRICS:
            values = np.asarray(estimates[model][metric])
            row = {
                "dataset": "DAppSCAN",
                "model": model,
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_repetitions": int(len(values)),
            }
            if model != "scafe":
                delta_values = np.asarray(deltas[model][metric])
                row.update({
                    "delta_vs_scafe": float(point[metric] - scafe_point[metric]),
                    "delta_ci_lower": float(np.quantile(delta_values, 0.025)),
                    "delta_ci_upper": float(np.quantile(delta_values, 0.975)),
                })
            interval_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scafe-predictions", type=Path, default=DEFAULT_SCAFE)
    parser.add_argument("--dl-predictions", type=Path, default=DEFAULT_DL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    aligned = align(
        pd.read_csv(args.scafe_predictions, low_memory=False),
        pd.read_csv(args.dl_predictions, low_memory=False),
    )
    metrics, intervals = summarize(
        aligned,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output / "aligned_oof_predictions.csv", index=False)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    intervals.to_csv(output / "project_bootstrap_intervals.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps({
            "version": 1,
            "scope": "sampled DAppSCAN within-corpus model comparison",
            "publication_scope": "sampled estimate; not a full-corpus result",
            "outer_protocol": "five-fold project-group-disjoint OOF predictions",
            "models": list(MODELS),
            "threshold": 0.5,
            "bootstrap": {
                "unit": "DAppSCAN project group",
                "paired_against_scafe": True,
                "repetitions": int(args.bootstrap_repetitions),
                "seed": int(args.seed),
            },
            "inputs": {
                "scafe_predictions": str(args.scafe_predictions.resolve()),
                "scafe_sha256": _sha256(args.scafe_predictions.resolve()),
                "dl_predictions": str(args.dl_predictions.resolve()),
                "dl_sha256": _sha256(args.dl_predictions.resolve()),
            },
            "aligned_contracts": int(len(aligned)),
            "project_groups": int(aligned["group"].nunique()),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
