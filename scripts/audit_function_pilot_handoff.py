#!/usr/bin/env python3
"""Read-only handoff audit for the unreviewed public-source function pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_delivery(root: Path) -> dict[str, str]:
    root = root.resolve()
    manifest = json.loads((root / "DELIVERY_SHA256.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing delivered-file identity manifest")
    for name, digest in manifest.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Malformed delivered-file identity")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Delivered-file path escapes the pilot directory")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Delivered file is missing or changed; stop dependent processing")
    return manifest


def inspect_staging_rows(rows) -> dict[str, object]:
    source = Counter()
    context = Counter()
    review = Counter()
    ids = set()
    projects = set()
    token_sources = defaultdict(set)
    missing_projects = 0
    unknown_cells = 0
    count = 0
    for row in rows:
        identity = row.get("record_id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Missing or duplicate staging record ID")
        ids.add(identity)
        labels = row.get("pattern_labels")
        if not isinstance(labels, dict) or set(labels) != {"reentrancy", "unchecked_external_call", "authorization"}:
            raise ValueError("Unexpected unified pattern schema; explicit mapping review required")
        if any(value is not None for value in labels.values()):
            raise ValueError("Unified pattern labels changed; review/adjudication lineage must be checked")
        if row.get("training_eligible") is not False or row.get("split") is not None or row.get("human_review_status") != "unreviewed":
            raise ValueError("Pilot promotion state changed; explicit review required")
        if not isinstance(row.get("source_revision"), str) or not row["source_revision"]:
            raise ValueError("Missing pinned source revision")
        source[row.get("source", "missing")] += 1
        context[row.get("context_status", "missing")] += 1
        review[row["human_review_status"]] += 1
        project = row.get("project_id")
        if not isinstance(project, str) or not project.strip():
            missing_projects += 1
        else:
            projects.add((row["source"], project))
        if row.get("token_sha256") and row.get("model_code"):
            token_sources[row["token_sha256"]].add(row["source"])
        unknown_cells += len(labels)
        count += 1
    return {
        "records": count,
        "source_counts": dict(source), "context_status_counts": dict(context),
        "human_review_status_counts": dict(review), "unknown_pattern_cells": unknown_cells,
        "records_without_project_id": missing_projects, "named_source_project_pairs": len(projects),
        "reported_token_groups_spanning_sources": sum(len(sources) > 1 for sources in token_sources.values()),
        "training_eligible_records": 0, "assigned_split_records": 0,
    }


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected JSON objects in pilot export")
                yield row


def audit(root: Path) -> dict[str, object]:
    before = verify_delivery(root)
    overall = inspect_staging_rows(jsonl_rows(root / "public_output/candidate_records.jsonl"))
    functions = inspect_staging_rows(jsonl_rows(root / "public_output/function_candidates.jsonl"))
    reported = json.loads((root / "public_output/summary.json").read_text())
    if overall["records"] != reported["records"] or functions["records"] != reported["standalone_function_candidates"]:
        raise ValueError("Producer summary and exported record counts disagree")
    if verify_delivery(root) != before:
        raise ValueError("Pilot manifest changed during handoff audit")
    return {
        "protocol": "unreviewed-function-pilot-handoff-v1", "delivery_files_verified": len(before),
        "delivery_manifest_sha256": sha256(root / "DELIVERY_SHA256.json"),
        "source_manifest_sha256": sha256(root / "source_manifest.json"),
        "delivered_file_hashes": before,
        "staging": overall, "standalone_candidates": functions,
        "status": "verified_unreviewed_staging_not_training_ready",
        "limitations": [
            "Delivery hashes and state checks verify handoff consistency, not annotation truth, legal redistribution rights or provenance completeness.",
            "Token groups and context-status fields are producer metadata, not independently re-derived token/semantic checks.",
            "No model inputs or source claims were converted into training labels; no original file was modified.",
            "The three unified categories are not automatically mapped to the existing eleven pattern outputs.",
            "Independent review, context/localization, project/clone grouping against SCAFE and a frozen split definition remain required.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.pilot_dir.resolve()):
        raise ValueError("Write reports outside the read-only pilot")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = audit(args.pilot_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "delivered_file_hashes"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
