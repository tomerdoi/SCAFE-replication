"""Build compiler-reference modifier oracles for a frozen project manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from audit_modifier_reference_scope import audit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.solidity_modifier_resolution import SourceRepository  # noqa: E402


MAX_PROJECT_FILES = 250
MAX_PROJECT_SOURCE_BYTES = 25 * 1024 * 1024
MAX_COMPILER_SECONDS = 2 * 60 * 60


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_remappings(project: dict[str, object], root: Path) -> tuple[tuple[str, str], ...]:
    values = project.get("remappings", [])
    if "remappings_file" in project:
        remapping_file = root / str(project["remappings_file"])
        for line in remapping_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" not in line:
                    raise ValueError(f"Malformed remapping in {remapping_file}")
                prefix, target = line.split("=", 1)
                values.append([prefix, target])
    parsed: list[tuple[str, str]] = []
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise ValueError("Malformed remapping entry")
        prefix, target = value
        if "\\" in prefix or "\\" in target or target.startswith("/"):
            raise ValueError("Unsafe remapping")
        if not prefix.endswith("/"):
            prefix += "/"
        if not target.endswith("/"):
            target += "/"
        parsed.append((prefix, target))
    return tuple(parsed)


def _entry_sources(root: Path, patterns: Iterable[str]) -> tuple[str, ...]:
    paths: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("Entry source escapes project root")
            paths.add(resolved.relative_to(root).as_posix())
    if not paths:
        raise ValueError("Project has no entry sources")
    return tuple(sorted(paths))


def _source_closure(
    root: Path,
    entries: tuple[str, ...],
    remappings: tuple[tuple[str, str], ...],
) -> tuple[dict[str, str], tuple[str, ...]]:
    repository = SourceRepository(root, remappings)
    pending = list(entries)
    visited: set[str] = set()
    unresolved: set[str] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        parsed = repository.parse(path)
        for directive in parsed.imports:
            target = repository.resolve_import(path, directive.path)
            if target is None:
                unresolved.add(f"{path} -> {directive.path}")
            else:
                pending.append(target)
    if unresolved:
        raise ValueError("Unresolved imports: " + "; ".join(sorted(unresolved)))
    if len(visited) > MAX_PROJECT_FILES:
        raise ValueError("Project source closure exceeds file bound")
    sources = {path: repository.read(path) for path in sorted(visited)}
    byte_count = sum(len(text.encode("utf-8")) for text in sources.values())
    if byte_count > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("Project source closure exceeds byte bound")
    return sources, tuple(sorted(visited))


def _git_head(root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return process.stdout.strip()


def _compile(
    compiler: Path,
    sources: dict[str, str],
    remappings: tuple[tuple[str, str], ...],
) -> tuple[bytes, dict[str, object], dict[str, object]]:
    standard_input = {
        "language": "Solidity",
        "sources": {path: {"content": text} for path, text in sources.items()},
        "settings": {
            "remappings": [f"{prefix}={target}" for prefix, target in remappings],
            "outputSelection": {"*": {"": ["ast"]}},
        },
    }
    input_raw = json.dumps(standard_input, separators=(",", ":")).encode()
    process = subprocess.run(
        [str(compiler), "--standard-json"],
        input=input_raw,
        capture_output=True,
        timeout=MAX_COMPILER_SECONDS,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"Compiler process failed ({process.returncode}): "
            + process.stderr.decode(errors="replace")[-2000:]
        )
    payload = json.loads(process.stdout)
    errors = [
        item
        for item in payload.get("errors", [])
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    if errors:
        messages = [str(item.get("formattedMessage", item)) for item in errors]
        raise ValueError("Compiler semantic errors:\n" + "\n".join(messages[:20]))
    input_manifest = {
        "standard_json_sha256": _sha256_bytes(input_raw),
        "source_count": len(sources),
        "source_bytes": sum(len(text.encode()) for text in sources.values()),
        "source_sha256": {
            path: _sha256_bytes(text.encode()) for path, text in sources.items()
        },
        "remappings": standard_input["settings"]["remappings"],
    }
    return process.stdout, payload, input_manifest


def build(
    repository_root: Path,
    config_path: Path,
    compiler_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Oracle output directory must be new")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    projects = config.get("projects")
    if (
        not isinstance(projects, list)
        or not projects
        or len(projects) > 7
    ):
        raise ValueError("Oracle manifest must contain one to seven projects")
    output_dir.mkdir(parents=True)
    results: list[dict[str, object]] = []
    for project in projects:
        project_id = str(project["id"])
        root = (repository_root / str(project["local_root"])).resolve(strict=True)
        if _git_head(root) != project["revision"]:
            raise ValueError(f"Revision mismatch for {project_id}")
        license_path = (root / str(project["license_file"])).resolve(strict=True)
        if not license_path.is_relative_to(root) or not license_path.is_file():
            raise ValueError(f"Missing license for {project_id}")
        remappings = _parse_remappings(project, root)
        entries = _entry_sources(root, project["entry_globs"])
        sources, closure = _source_closure(root, entries, remappings)
        compiler = compiler_dir / f"solc-{project['compiler']}"
        qualification = json.loads((compiler_dir / "qualification.json").read_text())
        qualified = {
            item["version"]: item["sha256"] for item in qualification["compilers"]
        }
        if project["compiler"] not in qualified:
            raise ValueError(f"Compiler is not qualified for {project_id}")
        if _sha256_bytes(compiler.read_bytes()) != qualified[project["compiler"]]:
            raise ValueError(f"Compiler hash mismatch for {project_id}")
        output_raw, payload, input_manifest = _compile(
            compiler, sources, remappings
        )
        project_dir = output_dir / project_id
        project_dir.mkdir()
        (project_dir / "compiler_output.json").write_bytes(output_raw)
        oracle = audit(payload)
        entry_set = set(entries)
        owned_rows = [
            row for row in oracle["rows"] if row["query_source_path"] in entry_set
        ]
        owned_cross_source = sum(
            row["query_source_path"] != row["target_source_path"]
            for row in owned_rows
        )
        project_result = {
            "id": project_id,
            "canonical_repository": project["canonical_repository"],
            "revision": project["revision"],
            "split": project["split"],
            "compiler": project["compiler"],
            "compiler_sha256": qualified[project["compiler"]],
            "license_file": project["license_file"],
            "license_sha256": _sha256_bytes(license_path.read_bytes()),
            "entry_sources": entries,
            "closure_sources": closure,
            "input_manifest": input_manifest,
            "compiler_output_sha256": _sha256_bytes(output_raw),
            "owned_query_count": len(owned_rows),
            "owned_cross_source_query_count": owned_cross_source,
            "owned_rows": owned_rows,
            "full_closure_oracle_counts": {
                key: value
                for key, value in oracle.items()
                if key != "rows"
            },
        }
        (project_dir / "oracle.json").write_text(
            json.dumps(project_result, indent=2) + "\n"
        )
        results.append(project_result)
    summary = {
        "status": "compiler_oracles_built_not_method_evaluated",
        "protocol": config["protocol"],
        "config_sha256": _sha256_bytes(config_raw),
        "project_count": len(results),
        "development_projects": sum(
            result["split"] == "development" for result in results
        ),
        "confirmation_projects": sum(
            result["split"] == "confirmation" for result in results
        ),
        "owned_query_count": sum(result["owned_query_count"] for result in results),
        "owned_cross_source_query_count": sum(
            result["owned_cross_source_query_count"] for result in results
        ),
        "projects": [
            {
                key: result[key]
                for key in (
                    "id",
                    "revision",
                    "split",
                    "compiler",
                    "owned_query_count",
                    "owned_cross_source_query_count",
                )
            }
            for result in results
        ],
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compiler-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with Path("/tmp/paper-research-compute.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Shared paper compute lock is held") from error
        result = build(
            args.repository_root.resolve(),
            args.config.resolve(),
            args.compiler_dir.resolve(),
            args.output_dir.resolve(),
        )
        fcntl.flock(lease, fcntl.LOCK_UN)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
