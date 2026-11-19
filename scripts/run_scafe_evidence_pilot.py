"""Run the frozen paired SCAFE-Evidence modifier-resolution pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.solidity_modifier_resolution import (
    Resolution,
    replay_certificate,
    resolve_modifier,
    resolve_modifier_file_only,
    resolve_modifier_strong_traversal,
)


BOOTSTRAP_SEED = 20261119
BOOTSTRAP_DRAWS = 10_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remappings(project: dict[str, object], root: Path) -> tuple[tuple[str, str], ...]:
    values = list(project.get("remappings", []))
    if "remappings_file" in project:
        for line in (root / str(project["remappings_file"])).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prefix, target = line.split("=", 1)
                values.append([prefix, target])
    return tuple((str(prefix), str(target)) for prefix, target in values)


def _outcome(resolution: Resolution, oracle: dict[str, object]) -> str:
    if resolution.status == "unknown":
        return "unknown"
    return (
        "correct"
        if resolution.source_path == oracle["target_source_path"]
        and resolution.declaration_start == oracle["declaration_span"][1]
        else "wrong"
    )


def _run_method(
    method: Callable[[], Resolution],
    root: Path,
    oracle: dict[str, object],
) -> dict[str, object]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    resolution = method()
    cpu_ns = time.process_time_ns() - cpu_start
    wall_ns = time.perf_counter_ns() - wall_start
    return {
        "outcome": _outcome(resolution, oracle),
        "reason": resolution.reason,
        "source_path": resolution.source_path,
        "declaration_start": resolution.declaration_start,
        "files_read": list(resolution.files_read),
        "bytes_read": resolution.bytes_read,
        "wall_ns": wall_ns,
        "cpu_ns": cpu_ns,
        "certificate_replay": replay_certificate(root, resolution),
    }


def _aggregate(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    outcomes = Counter(row["methods"][method]["outcome"] for row in rows)
    answered = outcomes["correct"] + outcomes["wrong"]
    return {
        "correct": outcomes["correct"],
        "wrong": outcomes["wrong"],
        "unknown": outcomes["unknown"],
        "coverage": answered / len(rows) if rows else None,
        "selective_risk": outcomes["wrong"] / answered if answered else None,
        "median_bytes": statistics.median(
            row["methods"][method]["bytes_read"] for row in rows
        )
        if rows
        else None,
        "total_bytes": sum(row["methods"][method]["bytes_read"] for row in rows),
        "total_wall_ns": sum(row["methods"][method]["wall_ns"] for row in rows),
        "total_cpu_ns": sum(row["methods"][method]["cpu_ns"] for row in rows),
        "all_certificates_replay": all(
            row["methods"][method]["certificate_replay"] for row in rows
        ),
    }


def _project_bootstrap_saving(
    project_rows: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    savings: list[float] = []
    for rows in project_rows.values():
        selected = statistics.median(
            row["methods"]["scafe_evidence"]["bytes_read"] for row in rows
        )
        strong = statistics.median(
            row["methods"]["strong_traversal"]["bytes_read"] for row in rows
        )
        if strong:
            savings.append((strong - selected) / strong)
    if not savings:
        return {"project_median_savings": [], "ci95": None}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        statistics.mean(rng.choices(savings, k=len(savings)))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    low = draws[math.floor(0.025 * (BOOTSTRAP_DRAWS - 1))]
    high = draws[math.ceil(0.975 * (BOOTSTRAP_DRAWS - 1))]
    return {
        "project_median_savings": savings,
        "mean_project_saving": statistics.mean(savings),
        "ci95": [low, high],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def run(
    repository_root: Path,
    config_path: Path,
    oracle_dir: Path,
    exclusion_screen: Path,
    output_path: Path,
    split: str,
    policy_freeze: Path | None,
) -> dict[str, object]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Pilot output must be new")
    if split == "confirmation":
        if policy_freeze is None or not policy_freeze.is_file():
            raise ValueError("Confirmation requires an existing development policy freeze")
        freeze = json.loads(policy_freeze.read_text())
        if freeze.get("decision") != "continue_to_confirmation":
            raise ValueError("Development gate did not authorize confirmation")
        bindings = {
            "protocol_sha256": repository_root
            / "docs/SCAFE_EVIDENCE_PROTOCOL_20261119.md",
            "project_config_sha256": config_path,
            "oracle_summary_sha256": oracle_dir / "summary.json",
            "reserved_screen_sha256": exclusion_screen,
            "resolver_sha256": repository_root
            / "src/evidence/solidity_modifier_resolution.py",
            "pilot_runner_sha256": Path(__file__).resolve(),
            "development_result_sha256": repository_root
            / str(freeze["development_result_path"]),
        }
        mismatches = [
            name
            for name, path in bindings.items()
            if _sha256(path) != freeze.get(name)
        ]
        if mismatches:
            raise ValueError(
                "Frozen confirmation bindings changed: " + ", ".join(mismatches)
            )
    config = json.loads(config_path.read_text())
    exclusion = json.loads(exclusion_screen.read_text())
    exclusions_by_project: dict[str, dict[str, set[tuple[object, ...]]]] = {}
    for item in exclusion["projects"]:
        excluded_functions = {
            (
                match["source_path"],
                int(match["span"][0]),
                int(match["span"][1]),
            )
            for match in (
                item["direct_function_matches"] + item["template_only_matches"]
            )
        }
        exclusions_by_project[item["id"]] = {
            "functions": excluded_functions,
            "files": set(item["exact_file_matches"]),
        }
    projects = [
        project for project in config["projects"] if project["split"] == split
    ]
    all_rows: list[dict[str, object]] = []
    project_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for project in projects:
        root = (repository_root / project["local_root"]).resolve(strict=True)
        remappings = _remappings(project, root)
        oracle_path = oracle_dir / project["id"] / "oracle.json"
        oracle_payload = json.loads(oracle_path.read_text())
        for oracle in oracle_payload["owned_rows"]:
            source_path = oracle["query_source_path"]
            function_span = oracle["query_function_span"]
            excluded = exclusions_by_project[project["id"]]
            if source_path in excluded["files"] or (
                source_path,
                int(function_span[1]),
                int(function_span[2]),
            ) in excluded["functions"]:
                continue
            offset = oracle["query_span"][1]
            name = oracle["query_name"]
            methods = {
                "file_name_lookup": _run_method(
                    lambda root=root, source_path=source_path, offset=offset, name=name: resolve_modifier_file_only(
                        root,
                        source_path,
                        offset,
                        name,
                        same_contract_only=False,
                    ),
                    root,
                    oracle,
                ),
                "sound_file_only": _run_method(
                    lambda root=root, source_path=source_path, offset=offset, name=name: resolve_modifier_file_only(
                        root,
                        source_path,
                        offset,
                        name,
                        same_contract_only=True,
                    ),
                    root,
                    oracle,
                ),
                "scafe_evidence": _run_method(
                    lambda root=root, source_path=source_path, offset=offset, name=name, remappings=remappings: resolve_modifier(
                        root, source_path, offset, name, remappings
                    ),
                    root,
                    oracle,
                ),
                "strong_traversal": _run_method(
                    lambda root=root, source_path=source_path, offset=offset, name=name, remappings=remappings: resolve_modifier_strong_traversal(
                        root, source_path, offset, name, remappings
                    ),
                    root,
                    oracle,
                ),
            }
            row = {
                "project": project["id"],
                "split": split,
                "query_source_path": source_path,
                "query_span": oracle["query_span"],
                "query_name": name,
                "target_source_path": oracle["target_source_path"],
                "target_declaration_span": oracle["declaration_span"],
                "target_is_common_library": oracle["target_source_path"].startswith(
                    "lib/"
                ),
                "methods": methods,
            }
            all_rows.append(row)
            project_rows[project["id"]].append(row)
    methods = (
        "file_name_lookup",
        "sound_file_only",
        "scafe_evidence",
        "strong_traversal",
    )
    aggregate = {method: _aggregate(all_rows, method) for method in methods}
    per_project = {
        project: {method: _aggregate(rows, method) for method in methods}
        for project, rows in project_rows.items()
    }
    non_library = [row for row in all_rows if not row["target_is_common_library"]]
    non_library_aggregate = {
        method: _aggregate(non_library, method) for method in methods
    }
    bootstrap = _project_bootstrap_saving(project_rows)
    selected = aggregate["scafe_evidence"]
    strong = aggregate["strong_traversal"]
    median_saving = (
        (strong["median_bytes"] - selected["median_bytes"]) / strong["median_bytes"]
        if strong["median_bytes"]
        else None
    )
    outcomes_match = all(
        row["methods"]["scafe_evidence"]["outcome"]
        == row["methods"]["strong_traversal"]["outcome"]
        for row in all_rows
    )
    project_coverage = [
        metrics["scafe_evidence"]["coverage"] for metrics in per_project.values()
    ]
    result = {
        "status": f"{split}_paired_pilot_complete",
        "split": split,
        "protocol_sha256": _sha256(
            repository_root / "docs/SCAFE_EVIDENCE_PROTOCOL_20261119.md"
        ),
        "config_sha256": _sha256(config_path),
        "oracle_summary_sha256": _sha256(oracle_dir / "summary.json"),
        "exclusion_screen_sha256": _sha256(exclusion_screen),
        "queries_excluded_before_method_evaluation": sum(
            json.loads((oracle_dir / project["id"] / "oracle.json").read_text())[
                "owned_query_count"
            ]
            for project in projects
        )
        - len(all_rows),
        "query_count": len(all_rows),
        "cross_source_query_count": sum(
            row["query_source_path"] != row["target_source_path"] for row in all_rows
        ),
        "cross_source_fraction": (
            sum(
                row["query_source_path"] != row["target_source_path"]
                for row in all_rows
            )
            / len(all_rows)
            if all_rows
            else None
        ),
        "aggregate": aggregate,
        "per_project": per_project,
        "non_common_library_query_count": len(non_library),
        "non_common_library_aggregate": non_library_aggregate,
        "paired_project_bootstrap": bootstrap,
        "selected_vs_strong_outcomes_match": outcomes_match,
        "selected_median_byte_saving": median_saving,
        "gate_components": {
            "zero_selected_wrong": selected["wrong"] == 0,
            "coverage_at_least_90_percent": selected["coverage"] is not None
            and selected["coverage"] >= 0.90,
            "every_project_at_least_80_percent": bool(project_coverage)
            and min(project_coverage) >= 0.80,
            "median_byte_saving_at_least_20_percent": median_saving is not None
            and median_saving >= 0.20,
            "project_bootstrap_ci_excludes_zero": bootstrap["ci95"] is not None
            and bootstrap["ci95"][0] > 0,
            "cross_source_at_least_20_percent": len(all_rows) > 0
            and sum(
                row["query_source_path"] != row["target_source_path"]
                for row in all_rows
            )
            / len(all_rows)
            >= 0.20,
            "all_selected_certificates_replay": selected[
                "all_certificates_replay"
            ],
            "gain_persists_without_common_libraries": (
                non_library_aggregate["strong_traversal"]["median_bytes"] is not None
                and non_library_aggregate["strong_traversal"]["median_bytes"] > 0
                and (
                    non_library_aggregate["strong_traversal"]["median_bytes"]
                    - non_library_aggregate["scafe_evidence"]["median_bytes"]
                )
                / non_library_aggregate["strong_traversal"]["median_bytes"]
                >= 0.20
            ),
            "matched_selected_and_strong_outcomes": outcomes_match,
        },
        "rows": all_rows,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    result["all_gate_components_pass"] = all(result["gate_components"].values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--exclusion-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--policy-freeze", type=Path)
    args = parser.parse_args()
    result = run(
        args.repository_root.resolve(),
        args.config.resolve(),
        args.oracle_dir.resolve(),
        args.exclusion_screen.resolve(),
        args.output.resolve(),
        args.split,
        args.policy_freeze.resolve() if args.policy_freeze else None,
    )
    compact = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
