#!/usr/bin/env python3
"""Rebuild unified evaluation artifacts with model-visible source sanitization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.unified_contract_eval_risk_score_eval import (  # noqa: E402
    UnifiedContractEvalRiskScoreEvaluation,
)
from src.evaluation.unified_dataset_excel import (  # noqa: E402
    build_unified_df,
    write_excel,
)
from src.source_sanitization import SOURCE_SANITIZATION_VERSION  # noqa: E402


DEFAULT_OUTPUT = REPO / "data/derived/sanitized_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity_sha256(frame: pd.DataFrame) -> str:
    identity = frame[
        ["contract_id", "source_dataset", "local_source_path"]
    ].fillna("").astype(str)
    payload = identity.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel local-source extraction workers (default: up to 8).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_xlsx = output / "unified_contract_eval_dataset.xlsx"
    scored_xlsx = output / "unified_contract_eval_scored.xlsx"
    scored_csv = output / "unified_contract_eval_scored.csv"
    local_checkpoint = output / "local_source_features.checkpoint.csv"
    rebuild_state = output / "rebuild_state.json"

    dataset = build_unified_df()
    write_excel(dataset, dataset_xlsx)
    fingerprint = {
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "source_feature_extractor_sha256": _sha256(
            REPO / "src/risk/contract_risk.py"
        ),
        "source_sanitizer_sha256": _sha256(
            REPO / "src/source_sanitization.py"
        ),
        "dataset_identity_sha256": _dataset_identity_sha256(dataset),
    }
    if local_checkpoint.exists() and rebuild_state.exists():
        previous_state = json.loads(rebuild_state.read_text(encoding="utf-8"))
        previous_fingerprint = previous_state.get("fingerprint")
        if previous_fingerprint != fingerprint:
            local_checkpoint.unlink()
    elif local_checkpoint.exists():
        # Never resume an unversioned checkpoint.
        local_checkpoint.unlink()
    rebuild_state.write_text(
        json.dumps(
            {"status": "in_progress", "fingerprint": fingerprint},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = UnifiedContractEvalRiskScoreEvaluation()
    evaluator.derived_dir = output
    evaluator.run(
        in_xlsx=dataset_xlsx,
        out_registry_csv=output / "unified_contract_eval_registry.csv",
        out_risk_csv=output / "unified_contract_eval_contract_risk.csv",
        out_scored_xlsx=scored_xlsx,
        out_scored_csv=scored_csv,
        resume=False,
        fetch_sources=False,
        fetch_activity=False,
        local_workers=args.workers,
        local_checkpoint_csv=local_checkpoint,
        resume_local_checkpoint=True,
        local_checkpoint_every=2000,
    )

    scored = pd.read_csv(scored_csv, low_memory=False)
    dappscan = scored[scored["source_dataset"] == "DAppSCAN"]
    duplicate_ids = int(dappscan["contract_id"].duplicated().sum())
    if duplicate_ids:
        raise RuntimeError(
            f"sanitized rebuild produced {duplicate_ids} duplicate DAppSCAN IDs"
        )

    manifest = {
        "version": 2,
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "dappscan_swc_annotation_comments_removed": True,
        "source_feature_extraction": "uniform_full_rebuild",
        **fingerprint,
        "workers": int(args.workers),
        "rows": int(len(scored)),
        "dappscan_rows": int(len(dappscan)),
        "local_source_rows_recomputed": int(
            dataset["local_source_path"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "dappscan_duplicate_contract_ids": duplicate_ids,
        "scored_csv": str(scored_csv),
        "scored_csv_sha256": _sha256(scored_csv),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebuild_state.write_text(
        json.dumps(
            {"status": "complete", "fingerprint": fingerprint},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
