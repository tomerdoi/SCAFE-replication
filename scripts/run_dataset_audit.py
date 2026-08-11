#!/usr/bin/env python3
"""Audit source duplication and label quality in the unified SCAFE table."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.dataset_audit import DatasetAudit
from src.ml.baseline_comparison import load_scored_frame

INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
OUTPUT = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"


def main() -> None:
    frame = load_scored_frame(str(INPUT))
    summary, duplicates = DatasetAudit().run(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "dataset_label_audit.csv"
    duplicates_path = OUTPUT / "dataset_duplicate_audit.csv"
    summary.to_csv(summary_path, index=False)
    duplicates.to_csv(duplicates_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {duplicates_path}")
    print(
        f"Duplicate source groups: {len(duplicates)}; "
        f"cross-dataset groups: {int(duplicates['cross_dataset'].sum()) if not duplicates.empty else 0}; "
        f"label conflicts: {int(duplicates['label_conflict'].sum()) if not duplicates.empty else 0}"
    )


if __name__ == "__main__":
    main()
