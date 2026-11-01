#!/usr/bin/env python3
"""Measure legacy SWC annotation exposure in saved CodeBERT token windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_function_semantic_transfer import extract_functions
from src.source_sanitization import SOURCE_SANITIZATION_VERSION, sanitize_source_text

RESULTS = REPO / "data/derived/sanitized_v1/ml_results"
MARKER = re.compile(r"(?i)\bSWC[-_ ]?\d+\b")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_spans(source: str) -> list[tuple[int, int]]:
    """Find removable comments outside strings, using canonical recognition."""

    spans = []
    index = 0
    while index < len(source):
        if source[index] in ("'", '"'):
            quote = source[index]
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = len(source) if end < 0 else end
        elif source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            end = len(source) if closing < 0 else closing + 2
        else:
            index += 1
            continue
        comment = source[index:end]
        if sanitize_source_text(comment, "DAppSCAN") != comment:
            spans.append((index, end))
        index = end
    return spans


def span_visible(span: tuple[int, int], offsets: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(left < end and right > start and right > left for left, right in offsets)


def span_fully_visible(span: tuple[int, int], offsets: list[tuple[int, int]]) -> bool:
    """Require token-offset coverage of every character, ignoring special tokens."""

    start, end = span
    cursor = start
    for left, right in sorted(offsets):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def window_counts(source: str, offsets: list[tuple[int, int]]) -> dict[str, int]:
    comments = annotation_spans(source)
    markers = [
        (start + marker.start(), start + marker.end())
        for start, end in comments
        for marker in MARKER.finditer(source[start:end])
    ]
    return {
        "annotation_comments": len(comments),
        "annotation_comments_with_tokens_in_window": sum(span_visible(span, offsets) for span in comments),
        "annotation_markers": len(markers),
        "markers_with_tokens_in_window": sum(span_visible(span, offsets) for span in markers),
        "complete_markers_in_window": sum(span_fully_visible(span, offsets) for span in markers),
    }


def read_raw_source(path: Path) -> str:
    data_root = (REPO / "data").resolve()
    path = path.resolve()
    if not path.is_relative_to(data_root):
        raise ValueError("Source path escapes the local data directory")
    sources = sorted(path.rglob("*.sol")) if path.is_dir() else [path]
    if any(item.suffix.lower() != ".sol" or not item.resolve().is_relative_to(data_root) for item in sources):
        raise ValueError("Unexpected source extension or source symlink escape")
    return "\n".join(item.read_text(encoding="utf-8", errors="replace") for item in sources)


def main() -> None:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior_dir = RESULTS / "function_source_sanitization_audit_20261101"
    prior = json.loads((prior_dir / "summary.json").read_text())
    registry_path = REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
    registry_sha = file_sha256(registry_path)
    if registry_sha != prior["input_sha256"]:
        raise ValueError("Registry changed since the prior exposure audit")
    if file_sha256(REPO / "src/source_sanitization.py") != prior["sanitizer_sha256"]:
        raise ValueError("Canonical sanitizer changed since the prior audit")
    registry = pd.read_csv(registry_path, usecols=["source_dataset", "contract_id", "local_source_path"])
    registry = registry.loc[registry.source_dataset.eq("DAppSCAN")]
    if registry.contract_id.duplicated().any():
        raise ValueError("DAppSCAN registry IDs are not unique")
    paths = registry.set_index("contract_id").local_source_path.to_dict()
    affected = pd.read_csv(prior_dir / "record_audit.csv")
    affected = affected.loc[affected.legacy_functions_with_annotation_comments.gt(0)]
    checkpoint_dir = RESULTS / "function_semantic_transfer/codebert"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    training = json.loads((checkpoint_dir / "protocol.json").read_text())
    max_length = training["training"]["max_length"]
    if training["model_name"] != "microsoft/codebert-base" or max_length != 512:
        raise ValueError("Unexpected historical model or token-window configuration")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=max_length, direction="right")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if hf_tokenizer.model_max_length != max_length:
        raise ValueError("Tokenizer and training max-length metadata disagree")
    development_path = RESULTS / "primary_development/scafe_within_corpus_predictions.csv"
    development = pd.read_csv(development_path, usecols=["dataset", "contract_id", "model"])
    development_ids = set(development.loc[
        development.dataset.eq("DAppSCAN") & development.model.eq("et"), "contract_id"
    ])
    rows = []
    for contract_id in sorted(affected.contract_id):
        raw = read_raw_source(Path(paths[contract_id]))
        clean = sanitize_source_text(raw, "DAppSCAN")
        before, after = extract_functions(raw), extract_functions(clean)
        if [name for name, _ in before] != [name for name, _ in after]:
            raise ValueError("Function sequence changed; no ordinal pairing attempted")
        for ordinal, ((name, raw_function), (_, clean_function)) in enumerate(zip(before, after, strict=True)):
            if raw_function == clean_function:
                continue
            raw_encoding = tokenizer.encode(raw_function)
            clean_encoding = tokenizer.encode(clean_function)
            for text, encoding in [(raw_function, raw_encoding), (clean_function, clean_encoding)]:
                hf = hf_tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
                if encoding.ids != hf["input_ids"] or encoding.offsets != hf["offset_mapping"]:
                    raise ValueError("Native tokenizer does not match historical AutoTokenizer input behavior")
            row = {
                "contract_id": contract_id,
                "function_ordinal": ordinal,
                "function_name": name,
                "in_primary_development": int(contract_id in development_ids),
                "raw_function_sha256": hashlib.sha256(raw_function.encode()).hexdigest(),
                "sanitized_function_sha256": hashlib.sha256(clean_function.encode()).hexdigest(),
                "raw_source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_window_tokens": len(raw_encoding.ids),
                "sanitized_window_tokens": len(clean_encoding.ids),
                "input_ids_changed": int(raw_encoding.ids != clean_encoding.ids),
                **{"raw_" + key: value for key, value in window_counts(raw_function, raw_encoding.offsets).items()},
                **{"sanitized_" + key: value for key, value in window_counts(clean_function, clean_encoding.offsets).items()},
            }
            if row["sanitized_annotation_comments"]:
                raise ValueError("Sanitized function still contains removable annotation comments")
            rows.append(row)
    details = pd.DataFrame(rows)
    expected = prior["totals"]["matched_functions_text_changed"]
    if len(details) != expected:
        raise ValueError(f"Changed-function count differs from previous audit: {len(details)} != {expected}")
    if file_sha256(registry_path) != registry_sha:
        raise ValueError("Registry changed during the audit")

    def summarize(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "functions_audited": len(frame),
            "records_audited": int(frame.contract_id.nunique()),
            "functions_with_annotation_tokens_in_window": int(frame.raw_annotation_comments_with_tokens_in_window.gt(0).sum()),
            "functions_with_marker_tokens_in_window": int(frame.raw_markers_with_tokens_in_window.gt(0).sum()),
            "functions_with_complete_marker_in_window": int(frame.raw_complete_markers_in_window.gt(0).sum()),
            "records_with_complete_marker_in_window": int(frame.loc[frame.raw_complete_markers_in_window.gt(0), "contract_id"].nunique()),
            "functions_with_changed_input_ids": int(frame.input_ids_changed.sum()),
            "sanitized_functions_with_annotation_tokens_in_window": int(frame.sanitized_annotation_comments_with_tokens_in_window.gt(0).sum()),
            "total_annotation_markers": int(frame.raw_annotation_markers.sum()),
            "complete_markers_in_window": int(frame.raw_complete_markers_in_window.sum()),
        }

    legacy_scoring_path = RESULTS / "function_semantic_transfer/codebert_scored/scoring_protocol.json"
    legacy_scoring = json.loads(legacy_scoring_path.read_text())
    overlap = json.loads((checkpoint_dir / "overlap_audit.json").read_text())
    payload = {
        "protocol": "saved-codebert-annotation-window-audit-v1",
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "max_length": max_length,
        "tokenization": "saved local tokenizer, right truncation, special tokens enabled; IDs and offsets verified against local AutoTokenizer for every raw/sanitized pair",
        "source_scope": "Previously identified annotation-bearing DAppSCAN development sources only",
        "all_affected_records": summarize(details),
        "primary_development_subset": summarize(details.loc[details.in_primary_development.eq(1)]),
        "input_hashes": {
            "registry": registry_sha,
            "prior_audit_summary": file_sha256(prior_dir / "summary.json"),
            "prior_audit_details": file_sha256(prior_dir / "record_audit.csv"),
            "development_predictions": file_sha256(development_path),
            "tokenizer": file_sha256(tokenizer_dir / "tokenizer.json"),
            "tokenizer_config": file_sha256(tokenizer_dir / "tokenizer_config.json"),
            "training_protocol": file_sha256(checkpoint_dir / "protocol.json"),
            "legacy_scoring_protocol": file_sha256(legacy_scoring_path),
            "checkpoint_file_now": file_sha256(checkpoint_dir / "checkpoint.pt"),
            "audit_script": file_sha256(Path(__file__)),
        },
        "lineage": {
            "legacy_scoring_metadata_contains_checkpoint_hash": "checkpoint_sha256" in legacy_scoring,
            "auxiliary_training_input_available_at_recorded_path": Path(overlap["auxiliary"]).is_file(),
            "checkpoint_loaded": False,
            "prediction_effects_measured": False,
            "rescoring_decision": "deferred: historical scores lack a checkpoint checksum binding; original auxiliary input is unavailable; no model loading in this bounded CPU audit",
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Exposure and changed input IDs do not measure prediction changes or detector performance.",
            "No training, model inference, model/fusion selection, external source reads, or private uploads.",
            "DAppSCAN was already used for development; this is a forensic audit, not a fresh evaluation cohort.",
            "Counts are function/record occurrences, not independent projects; label quality and template/provenance confounding remain unresolved.",
            "The checkpoint hash establishes current identity only, not historical score provenance or full training reproducibility.",
        ],
    }
    details.to_csv(args.output_dir / "function_window_audit.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
