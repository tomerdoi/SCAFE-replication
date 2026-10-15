#!/usr/bin/env python3
"""Run repeated, source-safe within-corpus SCAFE validation.

This complements LODO: it estimates same-corpus generalization only. It never
mixes datasets and never uses held-out fold labels to choose features, models,
or thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ml.baseline_comparison import (  # noqa: E402
    _fit,
    load_scored_frame,
    source_feature_columns,
    source_identity_hash,
)


DEFAULT_INPUT = REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO / "data/derived/sanitized_v1/ml_results/source_only_within_corpus"
)
DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")
MODELS = ("rf", "et", "hgb", "xgb")
SEEDS = (42, 43, 44, 45, 46)


def _canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one consistently labelled source per dataset before folding."""
    result = frame.copy()
    result["_source_identity"] = result["local_source_path"].fillna("").astype(str).map(
        source_identity_hash
    )
    result["_source_identity"] = result["_source_identity"].where(
        result["_source_identity"].ne(""),
        "contract:" + result["contract_id"].astype(str),
    )
    key = ["source_dataset", "_source_identity"]
    conflicts = (
        result.groupby(key)["label_status"].nunique().loc[lambda count: count > 1].index
    )
    conflict_rows = 0
    if len(conflicts):
        indexed = result.set_index(key)
        conflict_rows = int(indexed.index.isin(conflicts).sum())
        result = indexed.loc[~indexed.index.isin(conflicts)].reset_index()
    before = len(result)
    result = result.drop_duplicates(key, keep="first")
    audit = {
        "conflicting_label_rows_excluded": conflict_rows,
        "same_dataset_duplicate_rows_excluded": int(before - len(result)),
    }
    return result.reset_index(drop=True), audit


def _groups(frame: pd.DataFrame, dataset: str) -> np.ndarray:
    """Use project groups where reliable; otherwise source identities."""
    source = frame["_source_identity"].astype(str)
    if dataset == "DAppSCAN" and "source_subdataset" in frame:
        project = frame["source_subdataset"].fillna("").astype(str)
        # Canonicalization above removes exact source duplicates first. Project
        # grouping then prevents closely related DAppSCAN contracts crossing folds.
        key = project.where(project.ne(""), source)
    else:
        key = source
    codes, _ = pd.factorize(key, sort=True)
    return codes


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "coverage": float(prediction.mean()),
    }


def _top_budget_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    rows: dict[str, float] = {}
    prevalence = float(y.mean())
    ordered = np.argsort(-probability, kind="stable")
    for budget in (0.01, 0.05, 0.10):
        count = max(1, round(len(y) * budget))
        reviewed = y[ordered[:count]]
        precision = float(reviewed.mean())
        rows[f"precision_at_{int(budget * 100)}pct"] = precision
        rows[f"recall_at_{int(budget * 100)}pct"] = float(reviewed.sum() / max(y.sum(), 1))
        rows[f"lift_at_{int(budget * 100)}pct"] = precision / prevalence if prevalence else 0.0
    return rows


def run(
    frame: pd.DataFrame, datasets: Iterable[str], models: Iterable[str], seeds: Iterable[int], folds: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame, duplicate_audit = _canonicalize(frame)
    features = source_feature_columns(frame)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    source_group_counts: dict[str, int] = {}
    for dataset in datasets:
        selected = frame[
            (frame["source_dataset"] == dataset)
            & frame["label_status"].isin(("clean", "vulnerable"))
        ].copy()
        selected["y"] = (selected["label_status"] == "vulnerable").astype(int)
        groups = _groups(selected, dataset)
        source_group_counts[dataset] = int(len(np.unique(groups)))
        n_splits = min(folds, len(np.unique(groups)))
        if n_splits < 2 or selected["y"].nunique() != 2:
            continue
        for model in models:
            for seed in seeds:
                probabilities = np.zeros(len(selected), dtype=float)
                for train_index, test_index in GroupKFold(n_splits=n_splits).split(
                    selected, selected["y"], groups
                ):
                    fitted = _fit(
                        selected.iloc[train_index], features, model, random_state=int(seed)
                    )
                    probabilities[test_index] = fitted.predict_probability(selected.iloc[test_index])
                y = selected["y"].to_numpy(dtype=int)
                result: dict[str, object] = {
                    "dataset": dataset,
                    "model": model,
                    "seed": int(seed),
                    "n_samples": int(len(selected)),
                    "n_pos": int(y.sum()),
                    "n_neg": int((1 - y).sum()),
                    "prevalence": float(y.mean()),
                    "folds": n_splits,
                    **_metrics(y, probabilities),
                    **_top_budget_metrics(y, probabilities),
                }
                metric_rows.append(result)
                for position, row in selected.reset_index(drop=True).iterrows():
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "random_state": int(seed),
                            "contract_id": row["contract_id"],
                            "source_identity_hash": row["_source_identity"],
                            "y_true": int(y[position]),
                            "y_prob": float(probabilities[position]),
                            "y_pred": int(probabilities[position] >= 0.5),
                        }
                    )
    metadata: dict[str, object] = {
        "protocol": {
            "scope": "within-corpus validation; not cross-dataset generalization",
            "source_only": True,
            "source_canonicalization_before_split": True,
            "project_groups_for_dappscan": True,
            "source_identity_groups_other_datasets": True,
            "threshold": "fixed_0.5",
            "test_threshold_tuning": False,
            "fixed_review_budgets": [0.01, 0.05, 0.10],
        },
        "feature_count": len(features),
        "seeds": list(seeds),
        "duplicate_audit": duplicate_audit,
        "group_counts": source_group_counts,
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    metrics, predictions, metadata = run(
        load_scored_frame(str(args.input)), args.datasets, args.models, args.seeds, args.folds
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "scafe_within_corpus_per_seed.csv", index=False)
    predictions.to_csv(args.output / "scafe_within_corpus_predictions.csv", index=False)
    means = metrics.groupby(["dataset", "model"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics.columns if metric in {
            "roc_auc", "pr_auc", "f1", "precision", "recall", "coverage",
            "precision_at_1pct", "recall_at_1pct", "lift_at_1pct",
            "precision_at_5pct", "recall_at_5pct", "lift_at_5pct",
            "precision_at_10pct", "recall_at_10pct", "lift_at_10pct",
        }},
        **{f"{metric}_std": (metric, "std") for metric in ("roc_auc", "pr_auc", "f1")},
    )
    means.to_csv(args.output / "scafe_within_corpus_summary.csv", index=False)
    metadata["rows"] = len(metrics)
    (args.output / "scafe_within_corpus_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
