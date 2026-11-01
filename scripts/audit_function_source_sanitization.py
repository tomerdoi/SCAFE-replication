#!/usr/bin/env python3
"""Audit annotation exposure in legacy function inputs; no model or target scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_function_semantic_transfer import _normalise, extract_functions
from src.source_sanitization import SOURCE_SANITIZATION_VERSION, sanitize_source_text


def inspect_source(raw: str) -> dict[str, int]:
    """Compare legacy and canonical inputs, extracting only affected sources."""

    clean = sanitize_source_text(raw, "DAppSCAN")
    result = {
        "source_changed": int(clean != raw),
        "legacy_functions_in_changed_source": 0,
        "sanitized_functions_in_changed_source": 0,
        "legacy_functions_with_annotation_comments": 0,
        "sanitized_functions_with_annotation_comments": 0,
        "function_sequence_changed": 0,
        "matched_functions_text_changed": 0,
        "matched_functions_normalized_identity_changed": 0,
    }
    if raw == clean:
        return result
    before = extract_functions(raw)
    after = extract_functions(clean)
    result["legacy_functions_in_changed_source"] = len(before)
    result["sanitized_functions_in_changed_source"] = len(after)
    result["legacy_functions_with_annotation_comments"] = sum(
        sanitize_source_text(text, "DAppSCAN") != text for _, text in before
    )
    result["sanitized_functions_with_annotation_comments"] = sum(
        sanitize_source_text(text, "DAppSCAN") != text for _, text in after
    )
    result["function_sequence_changed"] = int(
        [name for name, _ in before] != [name for name, _ in after]
    )
    # Only compare ordinal-matched functions when the complete name sequence
    # agrees. Extraction changes are a separate failure mode, not forced pairs.
    if not result["function_sequence_changed"]:
        for (_, raw_function), (_, clean_function) in zip(before, after, strict=True):
            result["matched_functions_text_changed"] += int(raw_function != clean_function)
            result["matched_functions_normalized_identity_changed"] += int(
                _normalise(raw_function) != _normalise(clean_function)
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite a prior experiment, even after a partial failure.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame = pd.read_csv(
        args.input, usecols=["source_dataset", "contract_id", "local_source_path"],
    )
    development = frame.loc[frame["source_dataset"].eq("DAppSCAN")].sort_values(
        ["contract_id", "local_source_path"], kind="stable",
    )
    if development.empty:
        raise ValueError("No DAppSCAN development records in registry")
    rows = []
    data_root = (REPO / "data").resolve()
    for index, record in enumerate(development.itertuples(index=False), start=1):
        row = {"contract_id": str(record.contract_id), "read_status": "ok"}
        value = record.local_source_path
        if not isinstance(value, str) or not value.strip():
            rows.append({**row, "read_status": "missing_path"})
            continue
        path = Path(value).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Registry source path escapes the local data directory")
        try:
            if path.is_file() and path.suffix.lower() == ".sol":
                raw = path.read_text(encoding="utf-8", errors="replace")
            elif path.is_dir():
                sources = sorted(path.rglob("*.sol"))
                if any(not item.resolve().is_relative_to(data_root) for item in sources):
                    raise ValueError("Source symlink escapes the local data directory")
                raw = "\n".join(item.read_text(encoding="utf-8", errors="replace") for item in sources)
            else:
                rows.append({**row, "read_status": "missing_or_non_solidity"})
                continue
        except OSError:
            rows.append({**row, "read_status": "read_error"})
            continue
        if not raw:
            rows.append({**row, "read_status": "empty"})
            continue
        rows.append({**row, **inspect_source(raw)})
        if index % 2000 == 0:
            print(f"Audited {index}/{len(development)} DAppSCAN rows", flush=True)
    details = pd.DataFrame(rows)
    details.to_csv(args.output_dir / "record_audit.csv", index=False)
    numeric = list(inspect_source(""))
    payload = {
        "protocol": "dappscan-function-annotation-exposure-audit-v1",
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "function_pipeline_sha256": hashlib.sha256(
            (REPO / "scripts/run_function_semantic_transfer.py").read_bytes()
        ).hexdigest(),
        "sanitizer_sha256": hashlib.sha256(
            (REPO / "src/source_sanitization.py").read_bytes()
        ).hexdigest(),
        "dataset": "DAppSCAN",
        "registry_rows": len(development),
        "read_status_counts": details.read_status.value_counts().to_dict(),
        "totals": {column: int(details.get(column, pd.Series(dtype=int)).sum()) for column in numeric},
        "records_with_legacy_function_annotation_exposure": int(
            details.get("legacy_functions_with_annotation_comments", pd.Series(dtype=int)).gt(0).sum()
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Counts are registry rows and extracted function occurrences, not independent projects.",
            "Function comparisons are restricted to sources changed by canonical sanitization.",
            "No transformer tokenization or inference: exposure in extracted text does not establish that annotations survived the 512-token window.",
            "No detector metrics, retraining, fusion selection, external source reads, or exploit validation.",
            "The legacy extractor remains approximate and omits constructors, receive, fallback, and enclosing context.",
            "Sanitization does not remove ordinary comments, near-clone overlap, or provenance confounding.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["totals"]["sanitized_functions_with_annotation_comments"]:
        raise RuntimeError("Sanitized extracted functions still expose annotation comments")


if __name__ == "__main__":
    main()
