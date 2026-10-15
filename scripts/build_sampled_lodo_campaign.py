#!/usr/bin/env python3
"""Freeze an audited, deterministic multicorpus sample for LODO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_sampled_experiment_campaign import build_sample  # noqa: E402
from src.ml.within_corpus_dl_eval import canonicalize  # noqa: E402
from src.source_sanitization import SOURCE_SANITIZATION_VERSION  # noqa: E402


DEFAULT_INPUT = (
    REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
)
DEFAULT_OUTPUT = REPO / "data/derived/sanitized_v1/sampled_lodo_campaign_v1"
DEFAULT_DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_multicorpus_sample(
    canonical: pd.DataFrame,
    *,
    datasets: list[str],
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[int, int]]]:
    """Sample each binary corpus independently at its natural prevalence."""

    if len(set(datasets)) < 2:
        raise ValueError("LODO sampling requires at least two distinct datasets")
    parts: list[pd.DataFrame] = []
    allocations: dict[str, dict[int, int]] = {}
    for dataset_index, dataset in enumerate(datasets):
        part, allocation = build_sample(
            canonical,
            dataset=dataset,
            sample_size=sample_size,
            seed=seed + dataset_index * 1_000,
        )
        parts.append(part)
        allocations[dataset] = allocation
    sample = pd.concat(parts, ignore_index=True)
    if sample.groupby("source_dataset").size().to_dict() != {
        dataset: sample_size for dataset in datasets
    }:
        raise AssertionError("Per-corpus sample-size invariant failed")
    return sample, allocations


def cross_corpus_overlap_audit(sample: pd.DataFrame) -> dict[str, object]:
    """Describe source identities shared by sampled corpora."""

    memberships = (
        sample.groupby("_source_identity")["source_dataset"]
        .agg(lambda values: sorted(set(map(str, values))))
    )
    shared = memberships[memberships.map(len) > 1]
    pair_counts: dict[str, int] = {}
    for datasets in shared:
        for left_index, left in enumerate(datasets):
            for right in datasets[left_index + 1:]:
                key = f"{left} <-> {right}"
                pair_counts[key] = pair_counts.get(key, 0) + 1
    return {
        "shared_source_identities": int(len(shared)),
        "pair_counts": dict(sorted(pair_counts.items())),
        "policy": "retained in frozen input; evaluators remove target overlap from training",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 2:
        raise ValueError("sample-size must be at least 2")
    source_path = args.input.resolve()
    frame = pd.read_csv(source_path, low_memory=False)
    canonical, duplicate_audit = canonicalize(frame)
    sample, allocations = build_multicorpus_sample(
        canonical,
        datasets=list(args.datasets),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "scored_sample.csv"
    sample.to_csv(sample_path, index=False)
    manifest = {
        "version": 1,
        "purpose": "sampled_cross_corpus_model_comparison",
        "publication_scope": "sampled LODO estimate; not a full-corpus result",
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "source_scored_csv": str(source_path),
        "source_scored_csv_sha256": _sha256(source_path),
        "sample_csv": str(sample_path),
        "sample_csv_sha256": _sha256(sample_path),
        "datasets": list(args.datasets),
        "seed": int(args.seed),
        "per_dataset_sample_size": int(args.sample_size),
        "total_sample_size": int(len(sample)),
        "sampling": "without replacement; independent proportional binary-label strata per corpus",
        "population": {},
        "sample": {},
        "cross_corpus_overlap_audit": cross_corpus_overlap_audit(sample),
        "duplicate_audit": duplicate_audit,
        "selected_records": [],
    }
    for dataset in args.datasets:
        population = canonical[canonical["source_dataset"].eq(dataset)]
        selected = sample[sample["source_dataset"].eq(dataset)]
        manifest["population"][dataset] = {
            "records": int(len(population)),
            "labels": {
                str(int(label)): int(count)
                for label, count in population["y"].value_counts().sort_index().items()
            },
        }
        manifest["sample"][dataset] = {
            "records": int(len(selected)),
            "labels": {
                str(label): int(count)
                for label, count in sorted(allocations[dataset].items())
            },
            "fraction": float(len(selected) / len(population)),
        }
    manifest["selected_records"] = [
        {
            "dataset": str(dataset),
            "contract_id": str(contract_id),
            "source_identity_hash": str(identity),
            "label": int(label),
        }
        for dataset, contract_id, identity, label in zip(
            sample["source_dataset"],
            sample["contract_id"],
            sample["_source_identity"],
            sample["y"],
            strict=True,
        )
    ]
    manifest_path = output / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_csv": str(sample_path),
        "manifest": str(manifest_path),
        "total_sample_size": len(sample),
        "sample": manifest["sample"],
        "cross_corpus_overlap_audit": manifest["cross_corpus_overlap_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
