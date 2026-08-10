#!/usr/bin/env python3
"""Score exact public paper benchmark slices with SCAFE only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.paper_benchmarks import BenchmarkRecord, load_exact_benchmarks
from src.ml.baseline_comparison import (
    DatasetSpec,
    load_scored_frame,
    run_transfer,
    source_feature_columns,
)
from src.risk.contract_risk import ContractRiskUtil

SCORED_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
BENCHMARK_ROOT = REPO_ROOT / "data" / "raw" / "paper_benchmarks"
OUTPUT_DIR = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"


def score_records(records: list[BenchmarkRecord]) -> pd.DataFrame:
    """Extract the same local source features used by the SCAFE pipeline."""

    risk_util = ContractRiskUtil()
    rows: list[dict] = []
    for record in records:
        features = risk_util.extract_features_from_local_file(
            contract_id=record.record_id,
            sol_path=record.source_path,
            address="",
        )
        rows.append(
            {
                **features,
                "source_dataset": record.benchmark,
                "source_subdataset": record.vuln_type,
                "contract_id": f"{record.benchmark}:{record.record_id}",
                "label_status": record.label_status,
                "local_source_path": record.source_path,
                "vuln_types": record.vuln_type,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    records = load_exact_benchmarks(BENCHMARK_ROOT)
    if not records:
        raise RuntimeError("No exact paper benchmark records were found")

    exact = score_records(records)
    existing = load_scored_frame(str(SCORED_INPUT))
    combined = pd.concat([existing, exact], ignore_index=True, sort=False)
    features = source_feature_columns(combined)
    source = DatasetSpec("DAppSCAN", group_by_project=True)
    rows: list[dict] = []
    for benchmark in sorted(exact["source_dataset"].unique()):
        labels = (
            ("clean", "vulnerable")
            if benchmark == "LLM-SmartAudit-common"
            else ("vulnerable",)
        )
        rows.extend(
            run_transfer(
                combined,
                source,
                DatasetSpec(benchmark, accepted_labels=labels),
                features,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact.to_csv(OUTPUT_DIR / "exact_paper_benchmark_scored.csv", index=False)
    (OUTPUT_DIR / "exact_paper_benchmark_metrics.json").write_text(
        json.dumps(
            {
                "record_counts": exact["source_dataset"].value_counts().to_dict(),
                "feature_count": len(features),
                "protocol": {
                    "train_on": "DAppSCAN",
                    "source_only": True,
                    "third_party_tools_run": False,
                    "threshold": 0.5,
                    "positive_only_metrics": "recall and coverage",
                },
                "metrics": rows,
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Scored {len(exact)} exact paper records")
    print(f"Wrote {OUTPUT_DIR / 'exact_paper_benchmark_metrics.json'}")


if __name__ == "__main__":
    main()
