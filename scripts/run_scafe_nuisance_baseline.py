#!/usr/bin/env python3
"""Run a size/formatting-only LODO baseline under the primary protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.baseline_comparison import (  # noqa: E402
    DatasetSpec,
    load_scored_frame,
    run_repeated_leave_one_dataset_out,
    summarize_repeated_lodo,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "sanitized_v1"
    / "unified_contract_eval_scored.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "sanitized_v1"
    / "ml_results"
    / "nuisance_baseline_v1"
)
NUISANCE_FEATURES = (
    "n_files",
    "loc",
    "comment_ratio",
    "n_total_tokens",
    "n_unique_tokens",
    "n_duplicated_lines",
)
DATASET_SPECS = (
    DatasetSpec("DAppSCAN", group_by_project=True),
    DatasetSpec("DIVE"),
    DatasetSpec("Empirical-Analysis"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="et")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    missing = sorted(set(NUISANCE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing nuisance features: {missing}")

    metrics, predictions = run_repeated_leave_one_dataset_out(
        frame,
        DATASET_SPECS,
        list(NUISANCE_FEATURES),
        seeds=args.seeds,
        model_names=[args.model],
        calibrate_threshold=False,
    )
    metrics_frame = pd.DataFrame(metrics)
    summary = summarize_repeated_lodo(metrics_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "input_sha256": _sha256(args.input),
                "model": args.model,
                "seeds": args.seeds,
                "features": list(NUISANCE_FEATURES),
                "purpose": (
                    "Size/formatting-only negative control; features may still "
                    "correlate with source semantics and are not provenance labels."
                ),
                "canonicalize_within_dataset": True,
                "exclude_target_source_overlap": True,
                "exclude_cross_training_label_conflicts": True,
                "collapse_cross_training_same_label_duplicates": True,
                "threshold": "fixed_0.5",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote nuisance baseline artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
