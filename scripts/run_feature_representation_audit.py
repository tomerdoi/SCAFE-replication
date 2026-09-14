#!/usr/bin/env python3
"""Audit feature redundancy and source-extraction cost.

This is a read-only audit of the frozen source-only representation.  It does
not change the production extractor or the primary evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.experiment_manifest import build_feature_manifest, feature_group_columns
from src.ml.features import load_scored_csv
from src.ml.semantic_features import compute_context_features, compute_identifier_features
from src.risk.contract_risk import ContractRiskUtil


DEFAULT_INPUT = REPO / "data/derived/unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = REPO / "data/derived/feature_representation_audit"


def _read_source(path_value: object) -> tuple[str, int]:
    path = Path(str(path_value or ""))
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), 1
    if path.is_dir():
        files = sorted(path.rglob("*.sol"))
        return (
            "\n\n".join(
                item.read_text(encoding="utf-8", errors="replace") for item in files
            ),
            len(files),
        )
    return "", 0


def _correlation_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = build_feature_manifest(frame)
    columns = [item["name"] for item in manifest["evaluated_features"]]
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr = numeric.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = (
        upper.stack()
        .rename("absolute_spearman")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
        .sort_values("absolute_spearman", ascending=False)
    )
    pairs["feature_group_a"] = pairs["feature_a"].map(
        {name: group for group, names in feature_group_columns(frame).items() for name in names}
    )
    pairs["feature_group_b"] = pairs["feature_b"].map(
        {name: group for group, names in feature_group_columns(frame).items() for name in names}
    )
    group_rows = []
    groups = feature_group_columns(frame)
    for group, names in groups.items():
        values = corr.loc[names, names].where(
            np.triu(np.ones((len(names), len(names)), dtype=bool), k=1)
        ).stack()
        group_rows.append(
            {
                "feature_group": group,
                "feature_count": len(names),
                "pair_count": int(len(values)),
                "median_absolute_spearman": float(values.median()) if len(values) else 0.0,
                "p95_absolute_spearman": float(values.quantile(0.95)) if len(values) else 0.0,
                "pairs_abs_spearman_ge_0_95": int((values >= 0.95).sum()) if len(values) else 0,
            }
        )
    return pairs, pd.DataFrame(group_rows)


def _timing_audit(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    rows = frame[frame["local_source_path"].fillna("").astype(str).ne("")]
    rows = rows.drop_duplicates("local_source_path")
    datasets = sorted(rows["source_dataset"].dropna().astype(str).unique())
    per_dataset = max(1, sample_size // max(len(datasets), 1))
    rows = (
        rows.assign(_dataset=rows["source_dataset"].fillna("").astype(str))
        .groupby("_dataset", sort=True, group_keys=False)
        .head(per_dataset)
        .head(sample_size)
    )
    extractor = ContractRiskUtil()
    timings = []
    for index, row in rows.iterrows():
        source, n_files = _read_source(row["local_source_path"])
        if not source:
            continue
        start = time.perf_counter()
        extractor.extract_features_from_source_text(
            str(row.get("contract_id", index)), source, n_files=n_files
        )
        structural_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compute_identifier_features(source)
        identifier_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compute_context_features(source)
        context_seconds = time.perf_counter() - start
        timings.append(
            {
                "source_dataset": str(row.get("source_dataset", "")),
                "source_path": str(row["local_source_path"]),
                "source_bytes": len(source.encode("utf-8")),
                "structural_seconds": structural_seconds,
                "semantic_identifier_seconds": identifier_seconds,
                "semantic_context_seconds": context_seconds,
            }
        )
    return pd.DataFrame(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    args = parser.parse_args()

    frame = load_scored_csv(args.input)
    pairs, group_summary = _correlation_audit(frame)
    timings = _timing_audit(frame, max(args.sample_size, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    pairs[pairs["absolute_spearman"] >= args.correlation_threshold].to_csv(
        args.output / "high_correlation_pairs.csv", index=False
    )
    group_summary.to_csv(args.output / "correlation_by_feature_group.csv", index=False)
    timings.to_csv(args.output / "extraction_timings.csv", index=False)
    summary = {
        "input_rows": int(len(frame)),
        "evaluated_feature_count": int(len(build_feature_manifest(frame)["evaluated_features"])),
        "timed_unique_sources": int(len(timings)),
        "correlation_method": "absolute Spearman",
        "high_correlation_threshold": args.correlation_threshold,
        "high_correlation_pair_count": int(
            (pairs["absolute_spearman"] >= args.correlation_threshold).sum()
        ),
        "timing_summary_seconds": {
            column: {
                "median": float(timings[column].median()) if len(timings) else 0.0,
                "p95": float(timings[column].quantile(0.95)) if len(timings) else 0.0,
                "total": float(timings[column].sum()) if len(timings) else 0.0,
            }
            for column in (
                "structural_seconds",
                "semantic_identifier_seconds",
                "semantic_context_seconds",
            )
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
