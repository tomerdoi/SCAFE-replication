#!/usr/bin/env python3
"""Run strict full-source DL baselines under within-corpus grouped CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ml.within_corpus_dl_eval import DEFAULT_MODELS, run_within_corpus

DEFAULT_INPUT = REPO / "data/derived/unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = REPO / "data/derived/ml_results/baseline_comparison/source_only_within_corpus"
DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")
SEEDS = (42, 43, 44, 45, 46)


def _merge_prior(path: Path, current: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Retain completed models from prior sequential invocations."""
    if not path.is_file():
        return current
    prior = pd.read_csv(path, low_memory=False)
    if prior.empty or current.empty:
        return pd.concat([prior, current], ignore_index=True)
    current_keys = current[keys].drop_duplicates()
    retained = prior.merge(current_keys, on=keys, how="left", indicator=True)
    retained = retained.loc[retained["_merge"] == "left_only", prior.columns]
    return pd.concat([retained, current], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be strictly between 0 and 1")
    frame = pd.read_csv(args.input, low_memory=False)
    predictions, fold_metrics, metadata = run_within_corpus(
        frame, args.datasets, args.models, args.seeds, folds=args.folds,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        checkpoint_dir=args.output / "dl_within_corpus_checkpoints", resume=args.resume,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "dl_within_corpus_oof_predictions.csv"
    metrics_path = args.output / "dl_within_corpus_per_fold.csv"
    predictions = _merge_prior(
        predictions_path,
        predictions,
        ["dataset", "model", "random_state", "fold", "contract_id"],
    )
    fold_metrics = _merge_prior(
        metrics_path,
        fold_metrics,
        ["dataset", "model", "random_state", "fold"],
    )
    predictions.to_csv(predictions_path, index=False)
    fold_metrics.to_csv(metrics_path, index=False)
    seed_metrics = (fold_metrics.groupby(["dataset", "model", "random_state"], as_index=False)
                    .mean(numeric_only=True))
    seed_metrics.to_csv(args.output / "dl_within_corpus_per_seed.csv", index=False)
    summary = (seed_metrics.groupby(["dataset", "model"], as_index=False)
               .agg(["mean", "std"]).reset_index())
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column for column in summary.columns
    ]
    summary.to_csv(args.output / "dl_within_corpus_five_seed_summary.csv", index=False)
    metadata["output_rows"] = {"predictions": len(predictions), "fold_metrics": len(fold_metrics)}
    (args.output / "dl_within_corpus_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
