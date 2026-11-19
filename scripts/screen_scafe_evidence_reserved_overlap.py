"""Exclusion-only overlap screen against reserved DAppSCAN source.

No labels or findings are loaded.  A non-match remains evidence of no detected
overlap under this screen, not proof of project independence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time

from function_reserved_overlap import fingerprints, named_function_fragments


MAX_FILES = 30_000
MAX_BYTES = 350 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 240


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _decode_source(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="strict"), "cp1252"


def screen(
    repository_root: Path,
    config_path: Path,
    reserved_root: Path,
    inventory_path: Path,
    oracle_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Output path must be new")
    started = time.monotonic()
    inventory: dict[str, str] = {}
    project_prefixes: set[str] = set()
    with inventory_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            inventory[record["path"]] = record["sha256"]
            project_prefixes.add(record["path"].split("/", 1)[0])
    if len(inventory) > MAX_FILES:
        raise ValueError("Reserved inventory exceeds frozen file bound")
    reserved_exact_files = set(inventory.values())
    signature_to_occurrences: dict[str, Counter[str]] = {
        kind: Counter() for kind in ("exact", "tokens", "template")
    }
    files = bytes_seen = function_count = 0
    encoding_counts: Counter[str] = Counter()
    for relative, expected_hash in sorted(inventory.items()):
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("Reserved overlap screen exceeded time bound")
        path = (reserved_root / relative).resolve(strict=True)
        if not path.is_relative_to(reserved_root) or not path.is_file():
            raise ValueError("Reserved inventory path escapes root")
        raw = path.read_bytes()
        files += 1
        bytes_seen += len(raw)
        if len(raw) > MAX_FILE_BYTES or bytes_seen > MAX_BYTES:
            raise ValueError("Reserved source exceeds frozen byte bound")
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Reserved inventory hash mismatch: {relative}")
        text, encoding = _decode_source(raw)
        encoding_counts[encoding] += 1
        for fragment in named_function_fragments(text):
            function_count += 1
            for kind, digest in fingerprints(fragment["code"]).items():
                if kind != "token_count":
                    signature_to_occurrences[kind][digest] += 1
    config = json.loads(config_path.read_text())
    results: list[dict[str, object]] = []
    for project in config["projects"]:
        root = (repository_root / project["local_root"]).resolve(strict=True)
        candidate_paths: set[Path] = set()
        for pattern in project["entry_globs"]:
            candidate_paths.update(path for path in root.glob(pattern) if path.is_file())
        body_matches: list[dict[str, object]] = []
        exact_file_matches: list[str] = []
        source_bytes: dict[str, bytes] = {}
        for path in sorted(candidate_paths):
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("Candidate path escapes root")
            raw = resolved.read_bytes()
            relative = resolved.relative_to(root).as_posix()
            source_bytes[relative] = raw
            if hashlib.sha256(raw).hexdigest() in reserved_exact_files:
                exact_file_matches.append(relative)
        oracle = json.loads((oracle_dir / project["id"] / "oracle.json").read_text())
        query_functions: dict[tuple[str, int, int], dict[str, object]] = {}
        for row in oracle["owned_rows"]:
            span = row.get("query_function_span")
            if span is None:
                raise ValueError("Oracle query lacks enclosing function")
            query_functions[
                (row["query_source_path"], int(span[1]), int(span[2]))
            ] = row
        for (relative, start, length), row in sorted(query_functions.items()):
            raw = source_bytes.get(relative)
            if raw is None or start < 0 or length <= 0 or start + length > len(raw):
                raise ValueError("Oracle function span is outside candidate source")
            code = raw[start : start + length].decode("utf-8", errors="strict")
            signature = fingerprints(code)
            reasons = {
                kind: signature_to_occurrences[kind][signature[kind]]
                for kind in ("exact", "tokens", "template")
                if signature_to_occurrences[kind][signature[kind]]
            }
            if reasons:
                body_matches.append(
                    {
                        "source_path": relative,
                        "function": row["query_function_name"],
                        "span": [start, length],
                        "token_count": signature["token_count"],
                        "reserved_occurrence_counts": reasons,
                    }
                )
        repository_name = project["canonical_repository"].rstrip("/").split("/")[-1]
        normalized_repository = _normalize_name(repository_name)
        alias_hits = sorted(
            prefix
            for prefix in project_prefixes
            if normalized_repository
            and normalized_repository in _normalize_name(prefix)
        )
        direct = [
            match
            for match in body_matches
            if "exact" in match["reserved_occurrence_counts"]
            or "tokens" in match["reserved_occurrence_counts"]
        ]
        template_only = [
            match
            for match in body_matches
            if set(match["reserved_occurrence_counts"]) == {"template"}
        ]
        if exact_file_matches or direct:
            status = "exclude_direct_reserved_content_match"
        elif template_only or alias_hits:
            status = "quarantine_overlap_or_family_signal"
        else:
            status = "no_detected_match_not_independence_clearance"
        results.append(
            {
                "id": project["id"],
                "status": status,
                "candidate_source_files": len(candidate_paths),
                "candidate_query_functions": len(query_functions),
                "exact_file_matches": exact_file_matches,
                "direct_function_matches": direct,
                "template_only_matches": template_only,
                "reserved_project_alias_hits": alias_hits,
            }
        )
    result = {
        "protocol": "scafe-evidence-reserved-exclusion-v1",
        "role": "exclusion-only; nonmatch is not independence clearance",
        "reserved_labels_or_findings_loaded": False,
        "reserved_files": files,
        "reserved_bytes": bytes_seen,
        "reserved_named_function_occurrences": function_count,
        "reserved_encoding_counts": dict(encoding_counts),
        "projects": results,
        "status_counts": dict(Counter(item["status"] for item in results)),
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reserved-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(
        args.repository_root.resolve(),
        args.config.resolve(),
        args.reserved_root.resolve(),
        args.inventory.resolve(),
        args.oracle_dir.resolve(),
        args.output.resolve(),
    )
    compact = {
        "reserved_files": result["reserved_files"],
        "reserved_named_function_occurrences": result[
            "reserved_named_function_occurrences"
        ],
        "status_counts": result["status_counts"],
        "projects": [
            {
                "id": item["id"],
                "status": item["status"],
                "direct_function_matches": len(item["direct_function_matches"]),
                "template_only_matches": len(item["template_only_matches"]),
                "reserved_project_alias_hits": len(
                    item["reserved_project_alias_hits"]
                ),
            }
            for item in result["projects"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
