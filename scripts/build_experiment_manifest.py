#!/usr/bin/env python3
"""Write the authoritative SCAFE dataset and feature manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.experiment_manifest import (
    build_dataset_manifest,
    build_feature_manifest,
)
from src.ml.baseline_comparison import load_scored_frame


DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "ml_results"
    / "baseline_comparison"
    / "scafe_experiment_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scored_frame(str(args.input))
    manifest = {
        "input": str(args.input),
        "dataset_manifest": build_dataset_manifest(frame),
        "feature_manifest": build_feature_manifest(frame),
        "protocol": {
            "primary_generalization": "leave_one_dataset_out",
            "source_duplicate_exclusion": True,
            "threshold_calibration": "training_only",
            "unknown_labels_excluded_from_primary_binary_task": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
