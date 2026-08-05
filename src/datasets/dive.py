"""Normalize the DIVE multi-label source corpus for SCAFE evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.empirical_analysis import normalize_source, source_hash
from src.datasets.lisa_bench import _existing_source_hashes
from src.paths import DATA_DIR, DERIVED_DIR


RAW_DIR = DATA_DIR / "raw" / "DIVE"
EXTRACTED_DIR = RAW_DIR / "extracted"
SOURCE_DIR = EXTRACTED_DIR / "Raw" / "PRE" / "Source codes"
LABELS_PATH = EXTRACTED_DIR / "Labels" / "DIVE_Labels.csv"
OUTPUT_DIR = DERIVED_DIR / "dive"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

LABEL_COLUMNS = {
    "Reentrancy": "reentrancy",
    "Access Control": "access_control",
    "Arithmetic": "arithmetic",
    "Unchecked Return Values": "unchecked_low_calls",
    "DoS": "denial_service",
    "Bad Randomness": "bad_randomness",
    "Front Running": "front_running",
    "Time manipulation": "time_manipulation",
}


def _source_path(contract_id: Any, source_dir: Path = SOURCE_DIR) -> Path:
    return source_dir / f"{str(contract_id).strip()}.sol"


def _types(row: pd.Series) -> list[str]:
    return sorted(
        normalized
        for original, normalized in LABEL_COLUMNS.items()
        if float(row.get(original, 0) or 0) == 1
    )


def load_records(
    labels_path: Path = LABELS_PATH,
    source_dir: Path = SOURCE_DIR,
) -> list[dict[str, Any]]:
    """Load only DIVE records for which complete Solidity source is present."""
    if not labels_path.exists() or not source_dir.exists():
        return []
    labels = pd.read_csv(labels_path)
    records = []
    for _, row in labels.iterrows():
        path = _source_path(row["contractID"], source_dir)
        if not path.exists():
            continue
        source = normalize_source(path.read_text(encoding="utf-8", errors="ignore"))
        if not source:
            continue
        types = _types(row)
        records.append({
            "contract_id": str(row["contractID"]),
            "source": source,
            "vulnerability_types": types,
            "label_status": "vulnerable" if types else "clean",
            "source_path": str(path),
            "label_origin": "multi_tool_consensus",
        })
    return records


def deduplicate_records(
    records: list[dict[str, Any]],
    existing_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = source_hash(record["source"])
        if key not in grouped:
            grouped[key] = {
                "source_hash": key,
                "contract_ids": [],
                "vulnerability_types": set(),
                "label_status": "clean",
                "source_path": record["source_path"],
                "label_origin": record["label_origin"],
                "source": record["source"],
            }
        item = grouped[key]
        item["contract_ids"].append(record["contract_id"])
        item["vulnerability_types"].update(record["vulnerability_types"])
        if record["label_status"] == "vulnerable":
            item["label_status"] = "vulnerable"

    output = []
    for key in sorted(grouped):
        item = grouped[key]
        output.append({
            "source_hash": key,
            "source_file": f"sources/{key}.sol",
            "contract_ids": sorted(set(item["contract_ids"])),
            "vulnerability_types": sorted(item["vulnerability_types"]),
            "label_status": item["label_status"],
            "label_origin": item["label_origin"],
            "overlaps_existing_source": bool(existing_hashes and key in existing_hashes),
            "source": item["source"],
        })
    return output


def build_normalized_corpus(
    labels_path: Path = LABELS_PATH,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_output = output_dir / "sources"
    source_output.mkdir(parents=True, exist_ok=True)
    raw_records = load_records(labels_path, source_dir)
    records = deduplicate_records(raw_records, _existing_source_hashes())

    rows = []
    for record in records:
        (source_output / f"{record['source_hash']}.sol").write_text(
            record.pop("source"), encoding="utf-8"
        )
        row = dict(record)
        row["contract_ids"] = json.dumps(row["contract_ids"])
        row["vulnerability_types"] = json.dumps(row["vulnerability_types"])
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    summary = {
        "raw_label_rows": int(len(pd.read_csv(labels_path))) if labels_path.exists() else 0,
        "source_records": len(raw_records),
        "unique_sources": len(manifest),
        "vulnerable_sources": int((manifest.label_status == "vulnerable").sum()),
        "clean_sources": int((manifest.label_status == "clean").sum()),
        "overlaps_existing_source": int(manifest.overlaps_existing_source.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    result = build_normalized_corpus()
    print(f"normalized {len(result)} unique sources to {OUTPUT_DIR}")
