"""Normalize the Empirical Analysis Solidity replication corpus.

The raw GitHub checkout is intentionally kept immutable under ``data/raw``.
This module creates a separate, provenance-preserving manifest and content
addressed source files under ``data/derived/empirical_analysis``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.paths import DATA_DIR, DERIVED_DIR


RAW_DIR = (
    DATA_DIR
    / "raw"
    / "Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts"
)
OUTPUT_DIR = DERIVED_DIR / "empirical_analysis"
SOURCE_DIR = OUTPUT_DIR / "sources"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def normalize_source(source: Any) -> str:
    """Normalize source text only for hashing and storage."""
    if source is None or (isinstance(source, float) and pd.isna(source)):
        return ""
    text = str(source).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n" if text.strip() else ""


def source_hash(source: str) -> str:
    """Return a stable SHA-256 identifier for normalized Solidity."""
    return hashlib.sha256(normalize_source(source).encode("utf-8")).hexdigest()


def parse_annotation_value(value: Any) -> list[dict[str, Any]]:
    """Parse the repository's JSON/Python-literal vulnerability annotations."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def parse_tag_types(tag: Any) -> list[str]:
    """Preserve the source repository's vulnerability tag strings."""
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return []
    text = str(tag).strip()
    if not text or text.lower().rstrip(";") == "no":
        return []
    return sorted(set(
        part.strip() for part in re.split(r";+", text) if part.strip()
    ))


def _record(
    *,
    source: str,
    provenance: str,
    contract_id: str,
    label: str,
    vulnerability_types: Iterable[str],
    line_annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    text = normalize_source(source)
    if not text:
        return None
    types = sorted(set(str(item) for item in vulnerability_types if str(item)))
    return {
        "source_hash": source_hash(text),
        "contract_id": str(contract_id or ""),
        "provenance": provenance,
        "source_label": label,
        "label_status": "vulnerable" if types else "clean",
        "vulnerability_types": types,
        "line_annotations": line_annotations or [],
        "source": text,
    }


def load_records(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load source-bearing CSVs from the raw checkout."""
    records: list[dict[str, Any]] = []
    sample_path = raw_dir / "csvs" / "sample_of_interest_with_code.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, low_memory=False)
        for row in sample.to_dict("records"):
            record = _record(
                source=row.get("contract_code"),
                provenance="sample_of_interest_with_code",
                contract_id=row.get("contract"),
                label=row.get("label"),
                vulnerability_types=parse_tag_types(row.get("tag")),
            )
            if record:
                records.append(record)

    curated_path = raw_dir / "gpt_analysis" / "sb_curated" / "smartbugs_curated_revised.csv"
    if curated_path.exists():
        curated = pd.read_csv(curated_path, low_memory=False)
        for row in curated.to_dict("records"):
            annotations = parse_annotation_value(row.get("vulnerabilities"))
            types = [
                item.get("category", "")
                for item in annotations if isinstance(item, dict)
            ]
            record = _record(
                source=row.get("source_code"),
                provenance="smartbugs_curated_revised",
                contract_id=row.get("name"),
                label="smartbugs_curated",
                vulnerability_types=types,
                line_annotations=annotations,
            )
            if record:
                records.append(record)
    return records


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate source content while retaining all provenance."""
    grouped: dict[str, dict[str, Any]] = {}
    provenance: defaultdict[str, set[str]] = defaultdict(set)
    contracts: defaultdict[str, set[str]] = defaultdict(set)
    labels: defaultdict[str, set[str]] = defaultdict(set)
    types: defaultdict[str, set[str]] = defaultdict(set)
    annotations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[str, str] = {}

    for record in records:
        key = record["source_hash"]
        sources[key] = record["source"]
        provenance[key].add(record["provenance"])
        if record["contract_id"]:
            contracts[key].add(record["contract_id"])
        if record["source_label"]:
            labels[key].add(record["source_label"])
        types[key].update(record["vulnerability_types"])
        annotations[key].extend(record["line_annotations"])

    for key in sources:
        grouped[key] = {
            "source_hash": key,
            "source_file": f"sources/{key}.sol",
            "contract_ids": sorted(contracts[key]),
            "provenance": sorted(provenance[key]),
            "source_labels": sorted(labels[key]),
            "label_status": "vulnerable" if types[key] else "clean",
            "vulnerability_types": sorted(types[key]),
            "line_annotations": annotations[key],
            "source": sources[key],
        }
    return [grouped[key] for key in sorted(grouped)]


def build_normalized_corpus(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Create normalized source files, manifest, and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    records = deduplicate_records(load_records(raw_dir))

    rows = []
    for record in records:
        (source_dir / f"{record['source_hash']}.sol").write_text(
            record.pop("source"), encoding="utf-8"
        )
        row = dict(record)
        row["contract_ids"] = json.dumps(row["contract_ids"])
        row["provenance"] = json.dumps(row["provenance"])
        row["source_labels"] = json.dumps(row["source_labels"])
        row["vulnerability_types"] = json.dumps(row["vulnerability_types"])
        row["line_annotations"] = json.dumps(row["line_annotations"])
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    summary = {
        "raw_records": len(load_records(raw_dir)),
        "unique_sources": len(manifest),
        "vulnerable_sources": int((manifest.label_status == "vulnerable").sum()),
        "clean_sources": int((manifest.label_status == "clean").sum()),
        "line_annotated_sources": int(
            manifest.line_annotations.map(lambda value: value not in ("[]", "")).sum()
        ),
        "provenance": sorted({
            item
            for values in manifest.provenance
            for item in json.loads(values)
        }),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    result = build_normalized_corpus()
    print(f"normalized {len(result)} unique sources to {OUTPUT_DIR}")
