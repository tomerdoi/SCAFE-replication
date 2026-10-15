#!/usr/bin/env python3
"""Freeze a deterministic, audited sample for resumable model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ml.within_corpus_dl_eval import canonicalize, groups_for  # noqa: E402
from src.source_sanitization import SOURCE_SANITIZATION_VERSION  # noqa: E402


DEFAULT_INPUT = (
    REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
)
DEFAULT_OUTPUT = REPO / "data/derived/sanitized_v1/sampled_campaign_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proportional_allocation(counts: pd.Series, sample_size: int) -> dict[int, int]:
    """Allocate an exact stratified sample by largest fractional remainder."""

    integer_counts = {int(label): int(count) for label, count in counts.items()}
    total = sum(integer_counts.values())
    if not 0 < sample_size <= total:
        raise ValueError(f"sample_size must be in 1..{total}")
    quotas = {
        label: sample_size * count / total
        for label, count in integer_counts.items()
    }
    allocation = {
        label: min(integer_counts[label], int(np.floor(quota)))
        for label, quota in quotas.items()
    }
    remaining = sample_size - sum(allocation.values())
    order = sorted(
        quotas,
        key=lambda label: (
            quotas[label] - allocation[label],
            integer_counts[label],
            -label,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for label in order:
            if allocation[label] < integer_counts[label]:
                allocation[label] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Unable to complete proportional allocation")
    return allocation


def build_sample(
    canonical: pd.DataFrame,
    *,
    dataset: str,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, int]]:
    population = canonical[canonical["source_dataset"].eq(dataset)].copy()
    if population.empty:
        raise ValueError(f"No canonical records found for {dataset!r}")
    if population["y"].nunique() != 2:
        raise ValueError(f"{dataset!r} is not a binary-labelled population")
    allocation = proportional_allocation(population["y"].value_counts(), sample_size)
    parts = [
        group.sample(n=allocation[int(label)], random_state=seed + int(label))
        for label, group in population.groupby("y", sort=True)
    ]
    sample = pd.concat(parts).sort_index().reset_index(drop=True)
    if len(sample) != sample_size or sample["_source_identity"].duplicated().any():
        raise AssertionError("Sample size or source-identity uniqueness invariant failed")
    return sample, allocation


def fold_audit(
    sample: pd.DataFrame,
    *,
    dataset: str,
    folds: int,
    min_minority_per_test_fold: int,
) -> list[dict[str, int]]:
    groups = groups_for(sample, dataset)
    unique_groups = len(np.unique(groups))
    if folds < 2 or folds > unique_groups:
        raise ValueError(f"folds must be in 2..{unique_groups}")
    rows: list[dict[str, int]] = []
    for fold, (train, test) in enumerate(
        GroupKFold(folds).split(sample, sample["y"], groups)
    ):
        train_y = sample.iloc[train]["y"].to_numpy(dtype=int)
        test_y = sample.iloc[test]["y"].to_numpy(dtype=int)
        test_counts = np.bincount(test_y, minlength=2)
        if int(test_counts.min()) < min_minority_per_test_fold:
            raise ValueError(
                f"Fold {fold} has only {int(test_counts.min())} minority-label "
                f"test records; required {min_minority_per_test_fold}"
            )
        rows.append({
            "fold": fold,
            "train_records": int(len(train)),
            "train_positive": int(train_y.sum()),
            "train_negative": int((1 - train_y).sum()),
            "test_records": int(len(test)),
            "test_positive": int(test_y.sum()),
            "test_negative": int((1 - test_y).sum()),
            "test_groups": int(len(np.unique(groups[test]))),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="DAppSCAN")
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-minority-per-test-fold", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    frame = pd.read_csv(source_path, low_memory=False)
    canonical, duplicate_audit = canonicalize(frame)
    sample, allocation = build_sample(
        canonical,
        dataset=args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    folds = fold_audit(
        sample,
        dataset=args.dataset,
        folds=args.folds,
        min_minority_per_test_fold=args.min_minority_per_test_fold,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "scored_sample.csv"
    sample.to_csv(sample_path, index=False)
    population = canonical[canonical["source_dataset"].eq(args.dataset)]
    manifest = {
        "version": 1,
        "purpose": "sampled_within_corpus_model_comparison",
        "publication_scope": "sampled estimate; not a full-corpus result",
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "source_scored_csv": str(source_path),
        "source_scored_csv_sha256": _sha256(source_path),
        "sample_csv": str(sample_path),
        "sample_csv_sha256": _sha256(sample_path),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "sample_size": int(len(sample)),
        "population_size": int(len(population)),
        "population_labels": {
            str(int(label)): int(count)
            for label, count in population["y"].value_counts().sort_index().items()
        },
        "sample_labels": {
            str(label): count for label, count in sorted(allocation.items())
        },
        "sample_fraction": float(len(sample) / len(population)),
        "sampling": "without replacement; proportional binary-label strata",
        "grouping": (
            "DAppSCAN project groups" if args.dataset == "DAppSCAN"
            else "normalized source identities"
        ),
        "folds": int(args.folds),
        "min_minority_per_test_fold": int(args.min_minority_per_test_fold),
        "fold_audit": folds,
        "duplicate_audit": duplicate_audit,
        "selected_records": [
            {
                "contract_id": str(contract_id),
                "source_identity_hash": str(identity),
                "label": int(label),
            }
            for contract_id, identity, label in zip(
                sample["contract_id"],
                sample["_source_identity"],
                sample["y"],
                strict=True,
            )
        ],
    }
    manifest_path = output / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_csv": str(sample_path),
        "manifest": str(manifest_path),
        "sample_size": len(sample),
        "sample_labels": manifest["sample_labels"],
        "fold_audit": folds,
    }, indent=2))


if __name__ == "__main__":
    main()
