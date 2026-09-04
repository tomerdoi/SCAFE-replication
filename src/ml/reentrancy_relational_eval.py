"""Evaluation helpers for the isolated relational reentrancy PoC."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.ml.baseline_comparison import (
    DatasetSpec,
    add_canonical_vulnerability_types,
    run_leave_one_dataset_out_ovr,
    source_identity_hash,
    source_feature_columns,
)
from src.ml.reentrancy_relational_features import (
    RELATIONAL_FEATURE_COLUMNS,
    RELATIONAL_METADATA_COLUMNS,
    extract_relational_features_for_path,
)


POC_VERSION = 2
VARIANT_STRUCTURAL = "structural_only"
VARIANT_RELATIONAL = "relational_only"
VARIANT_COMBINED = "structural_plus_relational"


def feature_variants(
    structural_columns: Sequence[str],
) -> Mapping[str, list[str]]:
    """Return controlled feature sets for the PoC ablation."""

    structural = list(dict.fromkeys(structural_columns))
    relational = list(RELATIONAL_FEATURE_COLUMNS)
    return {
        VARIANT_STRUCTURAL: structural,
        VARIANT_RELATIONAL: relational,
        VARIANT_COMBINED: structural + [
            column for column in relational if column not in structural
        ],
    }


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    Path(temporary_name).unlink(missing_ok=True)
    try:
        frame.to_csv(temporary_name, index=False)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _cache_row(
    frame: pd.DataFrame, row_number: int, features: Mapping[str, float]
) -> dict[str, object]:
    row = frame.iloc[row_number]
    result: dict[str, object] = {
        "poc_version": POC_VERSION,
        "row_number": int(row_number),
    }
    for column in ("contract_id", "source_dataset", "local_source_path"):
        if column in frame:
            result[column] = str(row.get(column, "") or "")
    result.update(
        {
            column: float(features.get(column, 0.0))
            for column in RELATIONAL_FEATURE_COLUMNS + RELATIONAL_METADATA_COLUMNS
        }
    )
    return result


def _load_partial_rows(
    path: Path, frame: pd.DataFrame
) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    if not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    row_number = int(row["row_number"])
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                if row.get("poc_version") != POC_VERSION:
                    continue
                if row_number < 0 or row_number >= len(frame):
                    continue
                expected = frame.iloc[row_number]
                identity_matches = all(
                    str(row.get(column, "") or "")
                    == str(expected.get(column, "") or "")
                    for column in (
                        "contract_id",
                        "source_dataset",
                        "local_source_path",
                    )
                    if column in frame
                )
                if identity_matches and all(
                    column in row
                    for column in (
                        RELATIONAL_FEATURE_COLUMNS + RELATIONAL_METADATA_COLUMNS
                    )
                ):
                    rows[row_number] = row
    except OSError:
        return {}
    return rows


def build_relational_feature_cache(
    frame: pd.DataFrame,
    *,
    partial_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Extract PoC features with append-only, row-level resume support."""

    completed = (
        _load_partial_rows(partial_path, frame)
        if partial_path is not None
        else {}
    )
    handle = None
    try:
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            handle = partial_path.open("a", encoding="utf-8")
        for row_number in range(len(frame)):
            if row_number not in completed:
                path_value = str(
                    frame.iloc[row_number].get("local_source_path", "") or ""
                )
                row = _cache_row(
                    frame,
                    row_number,
                    extract_relational_features_for_path(path_value),
                )
                completed[row_number] = row
                if handle is not None:
                    handle.write(
                        json.dumps(row, separators=(",", ":")) + "\n"
                    )
                    handle.flush()
            if progress_callback is not None and (
                row_number == 0
                or (row_number + 1) % 250 == 0
                or row_number + 1 == len(frame)
            ):
                progress_callback(row_number + 1, len(frame))
    finally:
        if handle is not None:
            handle.close()
    return pd.DataFrame(
        [completed[row_number] for row_number in range(len(frame))]
    )


def attach_cached_relational_features(
    frame: pd.DataFrame,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a cache only when its row identity exactly matches the input."""

    if len(frame) != len(cache):
        raise ValueError("Relational feature cache row count does not match input")
    if "poc_version" not in cache or not (
        cache["poc_version"].astype(int) == POC_VERSION
    ).all():
        raise ValueError("Relational feature cache version is incompatible")
    if "row_number" not in cache or cache["row_number"].astype(int).tolist() != list(
        range(len(frame))
    ):
        raise ValueError("Relational feature cache row ordering is incompatible")
    for column in ("contract_id", "source_dataset", "local_source_path"):
        if column not in frame:
            continue
        if column not in cache:
            raise ValueError(f"Relational feature cache lacks identity column {column}")
        expected = frame[column].fillna("").astype(str).tolist()
        actual = cache[column].fillna("").astype(str).tolist()
        if expected != actual:
            raise ValueError(
                f"Relational feature cache does not match input column {column}"
            )

    result = frame.copy()
    for column in RELATIONAL_FEATURE_COLUMNS + RELATIONAL_METADATA_COLUMNS:
        if column not in cache:
            raise ValueError(f"Relational feature cache lacks feature {column}")
        result[column] = pd.to_numeric(
            cache[column], errors="coerce"
        ).fillna(0.0).to_numpy()
    return result


def load_or_build_relational_features(
    frame: pd.DataFrame,
    cache_path: Path | None,
    *,
    rebuild: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load an exact matching cache or build and persist a fresh one."""

    if cache_path is not None and cache_path.is_file() and not rebuild:
        cache = pd.read_csv(cache_path, low_memory=False)
        return attach_cached_relational_features(frame, cache), True

    partial_path = (
        cache_path.with_suffix(cache_path.suffix + ".partial.jsonl")
        if cache_path is not None
        else None
    )
    if rebuild and partial_path is not None:
        partial_path.unlink(missing_ok=True)
    cache = build_relational_feature_cache(
        frame,
        partial_path=partial_path,
        progress_callback=progress_callback,
    )
    if cache_path is not None:
        _atomic_to_csv(cache, cache_path)
    if partial_path is not None:
        partial_path.unlink(missing_ok=True)
    return attach_cached_relational_features(frame, cache), False


def canonicalize_reentrancy_source_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate exact sources before OVR splits and reject label conflicts."""

    normalized = add_canonical_vulnerability_types(frame)
    normalized = normalized.copy()
    normalized["_reentrancy_label"] = normalized["taxonomy_types"].map(
        lambda value: int("reentrancy" in str(value).split(";"))
    )
    hashes = normalized.get(
        "local_source_path", pd.Series("", index=normalized.index)
    ).fillna("").astype(str).map(source_identity_hash)
    fallback = pd.Series(
        [
            f"row:{index}:{contract_id}"
            for index, contract_id in enumerate(
                normalized["contract_id"].fillna("").astype(str)
            )
        ],
        index=normalized.index,
    )
    normalized["_poc_source_identity"] = hashes.where(hashes.ne(""), fallback)
    key = ["source_dataset", "_poc_source_identity"]
    label_counts = normalized.groupby(key, sort=False)[
        "_reentrancy_label"
    ].nunique()
    conflicting = label_counts[label_counts > 1].index
    conflict_mask = pd.MultiIndex.from_frame(normalized[key]).isin(conflicting)
    without_conflicts = normalized.loc[~conflict_mask].copy()
    deduplicated = without_conflicts.drop_duplicates(key, keep="first").copy()
    audit = {
        "input_rows": int(len(frame)),
        "conflicting_rows_excluded": int(conflict_mask.sum()),
        "duplicate_rows_excluded": int(
            len(without_conflicts) - len(deduplicated)
        ),
        "output_rows": int(len(deduplicated)),
    }
    return (
        deduplicated.drop(
            columns=["_reentrancy_label", "_poc_source_identity"]
        ).reset_index(drop=True),
        audit,
    )


def run_reentrancy_relational_poc(
    frame: pd.DataFrame,
    specs: Sequence[DatasetSpec],
    *,
    model_names: Iterable[str] = ("rf", "hgb"),
    seeds: Sequence[int] = (42,),
    calibrate_threshold: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare structural, relational, and combined duplicate-safe OVR LODO."""

    frame, _ = canonicalize_reentrancy_source_rows(frame)
    structural_columns = [
        column
        for column in source_feature_columns(frame)
        if column not in RELATIONAL_FEATURE_COLUMNS
        and column not in RELATIONAL_METADATA_COLUMNS
    ]
    variants = feature_variants(structural_columns)
    metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for seed in seeds:
        for variant, columns in variants.items():
            metrics, predictions = run_leave_one_dataset_out_ovr(
                frame,
                specs,
                columns,
                vulnerability_types=("reentrancy",),
                model_names=model_names,
                calibrate_threshold=calibrate_threshold,
                exclude_duplicate_sources=True,
                random_state=int(seed),
            )
            metrics_frame = pd.DataFrame(metrics)
            metrics_frame["feature_variant"] = variant
            metrics_frame["feature_count"] = len(columns)
            metrics_parts.append(metrics_frame)

            if not predictions.empty:
                predictions = predictions.copy()
                predictions["feature_variant"] = variant
                predictions["random_state"] = int(seed)
                prediction_parts.append(predictions)

    return (
        pd.concat(metrics_parts, ignore_index=True)
        if metrics_parts
        else pd.DataFrame(),
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame(),
    )


def summarize_poc_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed variation and deltas against structural-only SCAFE."""

    if metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "coverage",
    ]
    grouped = (
        metrics.groupby(
            ["dataset", "model", "feature_variant", "feature_count"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in grouped.columns.to_flat_index()
    ]
    keys = ["dataset", "model"]
    baseline = grouped[
        grouped["feature_variant"] == VARIANT_STRUCTURAL
    ].set_index(keys)
    for metric in ("roc_auc", "pr_auc", "f1"):
        baseline_values = baseline[f"{metric}_mean"]
        grouped[f"{metric}_delta_vs_structural"] = [
            float(row[f"{metric}_mean"] - baseline_values.loc[(row["dataset"], row["model"])])
            for _, row in grouped.iterrows()
        ]
    return grouped


def write_poc_metadata(
    path: Path,
    *,
    datasets: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    cache_reused: bool,
    calibrate_threshold: bool,
) -> None:
    """Write an auditable protocol record for the isolated PoC."""

    payload = {
        "poc_version": POC_VERSION,
        "purpose": "relational reentrancy feature feasibility test",
        "datasets": list(datasets),
        "models": list(models),
        "seeds": [int(seed) for seed in seeds],
        "variants": [
            VARIANT_STRUCTURAL,
            VARIANT_RELATIONAL,
            VARIANT_COMBINED,
        ],
        "relational_features": list(RELATIONAL_FEATURE_COLUMNS),
        "metadata_features_excluded_from_models": list(
            RELATIONAL_METADATA_COLUMNS
        ),
        "protocol": {
            "one_vs_rest": True,
            "held_out_dataset": True,
            "exclude_cross_dataset_source_duplicates": True,
            "threshold": (
                "train_only_f1" if calibrate_threshold else "fixed_0.5"
            ),
            "test_threshold_tuning": False,
            "cache_reused": bool(cache_reused),
            "production_scafe_modules_modified": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
