"""Normalize LISA-Bench audit findings for source-based SCAFE evaluation."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.empirical_analysis import normalize_source, source_hash
from src.paths import DATA_DIR, DERIVED_DIR


RAW_DIR = DATA_DIR / "raw" / "LISA-Bench"
OUTPUT_DIR = DERIVED_DIR / "lisa_bench"
SOURCE_DIR = OUTPUT_DIR / "sources"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

_TYPE_PATTERNS = {
    "access_control": r"\b(access control|authorization|privilege|owner|admin)\b",
    "arithmetic": r"\b(overflow|underflow|integer arithmetic|rounding)\b",
    "bad_randomness": r"\b(random|randomness|predictable seed)\b",
    "denial_service": r"\b(denial of service|dos|griefing|gas limit)\b",
    "front_running": r"\b(front[- ]running|sandwich)\b",
    "reentrancy": r"\breentranc",
    "time_manipulation": r"\b(timestamp|time manipulation|block\.timestamp)\b",
    "unchecked_low_calls": r"\b(unchecked call|low[- ]level call|delegatecall|call\()\b",
}


def _infer_types(title: Any, content: Any) -> list[str]:
    text = f"{title or ''}\n{content or ''}".lower()
    return sorted(name for name, pattern in _TYPE_PATTERNS.items()
                  if re.search(pattern, text))


def _join_code(code_entries: Any) -> tuple[str, list[str]]:
    if not isinstance(code_entries, list):
        return "", []
    parts: list[str] = []
    filenames: list[str] = []
    for entry in code_entries:
        if not isinstance(entry, dict):
            continue
        content = normalize_source(entry.get("content"))
        if not content:
            continue
        filename = str(entry.get("filename") or "source.sol")
        filenames.append(filename)
        parts.append(f"// ===== FILE: {filename} =====\n{content}")
    return normalize_source("\n".join(parts)), filenames


def _existing_source_hashes() -> set[str]:
    """Build an exact-source overlap index from the current unified corpus."""
    xlsx = DERIVED_DIR / "unified_contract_eval_dataset.xlsx"
    if not xlsx.exists():
        return set()
    df = pd.read_excel(xlsx, sheet_name="contracts", usecols=["local_source_path"])
    paths = {Path(str(value)) for value in df["local_source_path"].dropna() if str(value)}
    hashes: set[str] = set()
    scanned: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".sol":
            candidates = [path]
        elif path.is_dir() and path not in scanned:
            scanned.add(path)
            candidates = list(path.rglob("*.sol"))
        else:
            candidates = []
        for candidate in candidates:
            try:
                hashes.add(source_hash(candidate.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return hashes


def load_records(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    records = []
    for path in sorted((raw_dir / "dataset").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source, filenames = _join_code(item.get("Code"))
        if not source:
            continue
        title = str(item.get("Title") or "")
        content = str(item.get("Content") or "")
        records.append({
            "record_id": path.stem,
            "source": source,
            "title": title,
            "audit_url": str(item.get("Source") or ""),
            "impact": str(item.get("Impact") or "").strip().lower(),
            "vulnerability_types": _infer_types(title, content),
            "filenames": filenames,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    return records


def deduplicate_records(
    records: list[dict[str, Any]],
    existing_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    ids: defaultdict[str, list[str]] = defaultdict(list)
    titles: defaultdict[str, list[str]] = defaultdict(list)
    urls: defaultdict[str, set[str]] = defaultdict(set)
    impacts: defaultdict[str, set[str]] = defaultdict(set)
    types: defaultdict[str, set[str]] = defaultdict(set)
    filenames: defaultdict[str, set[str]] = defaultdict(set)
    sources: dict[str, str] = {}

    for record in records:
        key = source_hash(record["source"])
        sources[key] = record["source"]
        ids[key].append(record["record_id"])
        titles[key].append(record["title"])
        if record["audit_url"]:
            urls[key].add(record["audit_url"])
        if record["impact"]:
            impacts[key].add(record["impact"])
        types[key].update(record["vulnerability_types"])
        filenames[key].update(record["filenames"])

    for key in sorted(sources):
        grouped[key] = {
            "source_hash": key,
            "source_file": f"sources/{key}.sol",
            "record_ids": sorted(ids[key]),
            "titles": sorted(set(titles[key])),
            "audit_urls": sorted(urls[key]),
            "impacts": sorted(impacts[key]),
            "vulnerability_types": sorted(types[key]),
            "filenames": sorted(filenames[key]),
            "label_status": "vulnerable",
            "label_confidence": "silver",
            "label_origin": "professional_audit_finding",
            "overlaps_existing_source": bool(existing_hashes and key in existing_hashes),
            "source": sources[key],
        }
    return list(grouped.values())


def build_normalized_corpus(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_records(raw_dir)
    records = deduplicate_records(raw_records, _existing_source_hashes())

    rows = []
    for record in records:
        (source_dir / f"{record['source_hash']}.sol").write_text(
            record.pop("source"), encoding="utf-8"
        )
        row = dict(record)
        for key in (
            "record_ids", "titles", "audit_urls", "impacts",
            "vulnerability_types", "filenames",
        ):
            row[key] = json.dumps(row[key])
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    summary = {
        "raw_records": len(raw_records),
        "unique_sources": len(manifest),
        "vulnerable_sources": len(manifest),
        "overlaps_existing_source": int(manifest.overlaps_existing_source.sum()),
        "label_status": "vulnerable_only",
        "negative_labels_created": 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    result = build_normalized_corpus()
    print(f"normalized {len(result)} unique sources to {OUTPUT_DIR}")
