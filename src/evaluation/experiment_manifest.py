"""Build authoritative dataset and feature manifests for publication."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.baseline_comparison import source_feature_columns
from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.semantic_features import compute_context_features, compute_identifier_features


DATASET_PROVENANCE = {
    "DAppSCAN": "audit-derived clean/vulnerable labels",
    "DIVE": "released multi-label source corpus",
    "Empirical-Analysis": "replication corpus with normalized annotations",
    "FORGE": "released vulnerable/unknown/suspicious labels",
    "LISA-Bench": "released audit findings; vulnerable-only",
    "AccessControlVulnerabilities": "released access-control findings; vulnerable-only",
}


def _count(frame: pd.DataFrame, mask: pd.Series) -> int:
    return int(mask.sum())


def build_dataset_manifest(
    frame: pd.DataFrame,
    *,
    datasets: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Summarize rows, labels, source availability, and provenance."""

    selected = list(datasets) if datasets is not None else sorted(
        frame["source_dataset"].dropna().astype(str).unique()
    )
    records: list[dict[str, Any]] = []
    for dataset in selected:
        group = frame[frame["source_dataset"].astype(str) == dataset]
        labels = group["label_status"].fillna("").astype(str)
        paths = group.get(
            "local_source_path",
            pd.Series("", index=group.index),
        ).fillna("").astype(str)
        source_available = paths.map(lambda value: bool(value) and Path(value).exists())
        records.append(
            {
                "dataset": dataset,
                "provenance": DATASET_PROVENANCE.get(dataset, "not classified"),
                "rows": int(len(group)),
                "clean": _count(group, labels == "clean"),
                "vulnerable": _count(group, labels == "vulnerable"),
                "unknown": _count(group, labels == "unknown"),
                "suspicious": _count(group, labels == "suspicious"),
                "source_available_rows": int(source_available.sum()),
                "source_missing_rows": int((~source_available).sum()),
                "unique_contract_ids": int(group["contract_id"].nunique()),
            }
        )
    return records


def _feature_group(name: str) -> str:
    if name.startswith("semantic_"):
        if any(token in name for token in ("risk", "safety", "defi", "name_")):
            return "semantic_identifier"
        return "semantic_context"
    if name in _ONCHAIN_FEATURES:
        return "onchain_or_bytecode"
    if name.startswith("comp_"):
        return "structural_components"
    return "structural"


def build_feature_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe evaluated source features and excluded numeric columns."""

    evaluated = source_feature_columns(frame)
    records = [
        {
            "name": name,
            "group": _feature_group(name),
            "source_only": name not in _ONCHAIN_FEATURES,
        }
        for name in evaluated
    ]
    excluded_numeric = [
        {
            "name": name,
            "reason": (
                "onchain_or_bytecode"
                if name in _ONCHAIN_FEATURES
                else "identifier_label_path_or_risk_metadata"
            ),
        }
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name])
        and name not in evaluated
    ]
    identifier_names = sorted(compute_identifier_features("").keys())
    context_names = sorted(compute_context_features("").keys())
    semantic_catalog = [
        {"name": f"semantic_{name}", "group": "semantic_identifier"}
        for name in identifier_names
    ] + [
        {"name": f"semantic_{name}", "group": "semantic_context"}
        for name in context_names
    ]
    return {
        "evaluated_feature_count": len(records),
        "evaluated_features": records,
        "excluded_numeric_count": len(excluded_numeric),
        "excluded_numeric_features": excluded_numeric,
        "semantic_catalog_count": len(semantic_catalog),
        "semantic_catalog": semantic_catalog,
        "source_only_policy": {
            "excluded_onchain_and_bytecode": True,
            "excluded_columns": sorted(_EXCLUDE | _ONCHAIN_FEATURES),
            "third_party_detectors": False,
            "hosted_apis": False,
        },
    }


def feature_group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return evaluated numeric features partitioned into named groups."""

    groups: dict[str, list[str]] = {}
    for name in source_feature_columns(frame):
        groups.setdefault(_feature_group(name), []).append(name)
    return {group: sorted(names) for group, names in sorted(groups.items())}
