"""Combine the retained valid and corrective replacement confirmation cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_scafe_evidence_pilot import _aggregate, _project_bootstrap_saving


METHODS = (
    "file_name_lookup",
    "sound_file_only",
    "scafe_evidence",
    "strong_traversal",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    original: Path,
    replacement: Path,
    corrective_freeze: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Combined output must be new")
    first = json.loads(original.read_text())
    second = json.loads(replacement.read_text())
    retained = [row for row in first["rows"] if row["project"] == "multibridge"]
    replaced = [row for row in second["rows"] if row["project"] == "webauthn_owner_plugin"]
    if (
        not retained
        or not replaced
        or any(row["project"] == "worldcoin_token" for row in retained + replaced)
    ):
        raise ValueError("Unexpected confirmation cohort")
    rows = retained + replaced
    project_rows = {
        "multibridge": retained,
        "webauthn_owner_plugin": replaced,
    }
    aggregate = {method: _aggregate(rows, method) for method in METHODS}
    per_project = {
        project: {method: _aggregate(values, method) for method in METHODS}
        for project, values in project_rows.items()
    }
    non_library = [row for row in rows if not row["target_is_common_library"]]
    non_library_aggregate = {
        method: _aggregate(non_library, method) for method in METHODS
    }
    selected = aggregate["scafe_evidence"]
    strong = aggregate["strong_traversal"]
    median_saving = (
        strong["median_bytes"] - selected["median_bytes"]
    ) / strong["median_bytes"]
    non_library_saving = (
        non_library_aggregate["strong_traversal"]["median_bytes"]
        - non_library_aggregate["scafe_evidence"]["median_bytes"]
    ) / non_library_aggregate["strong_traversal"]["median_bytes"]
    bootstrap = _project_bootstrap_saving(project_rows)
    gate = {
        "zero_selected_wrong": selected["wrong"] == 0,
        "coverage_at_least_90_percent": selected["coverage"] >= 0.90,
        "every_project_at_least_80_percent": min(
            values["scafe_evidence"]["coverage"] for values in per_project.values()
        )
        >= 0.80,
        "median_byte_saving_at_least_20_percent": median_saving >= 0.20,
        "project_bootstrap_ci_excludes_zero": bootstrap["ci95"][0] > 0,
        "cross_source_at_least_20_percent": sum(
            row["query_source_path"] != row["target_source_path"] for row in rows
        )
        / len(rows)
        >= 0.20,
        "all_selected_certificates_replay": selected["all_certificates_replay"],
        "gain_persists_without_common_libraries": non_library_saving >= 0.20,
    }
    result = {
        "status": "qualified_confirmation_summary",
        "lineage": {
            "original_confirmation_sha256": _sha256(original),
            "corrective_replacement_sha256": _sha256(replacement),
            "corrective_freeze_sha256": _sha256(corrective_freeze),
            "worldcoin_status": "retained diagnostic but excluded from qualified confirmation because a first-party source file carries an UNLICENSED notice",
            "rerun_disclosure": "Multibridge was not rerun; WebAuthn Owner Plugin was evaluated once after the corrective freeze",
        },
        "project_count": len(project_rows),
        "query_count": len(rows),
        "cross_source_query_count": sum(
            row["query_source_path"] != row["target_source_path"] for row in rows
        ),
        "aggregate": aggregate,
        "per_project": per_project,
        "non_common_library_query_count": len(non_library),
        "non_common_library_aggregate": non_library_aggregate,
        "selected_median_byte_saving": median_saving,
        "non_common_library_median_byte_saving": non_library_saving,
        "paired_project_bootstrap": bootstrap,
        "gate_components": gate,
        "all_gate_components_pass": all(gate.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--corrective-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.original.resolve(),
        args.replacement.resolve(),
        args.corrective_freeze.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
