#!/usr/bin/env python3
"""Run SCAFE on the SmartBugs Wild reentrancy slice."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.paper_benchmarks import BenchmarkRecord, smartbugs_wild_records
from src.ml.baseline_comparison import (
    DatasetSpec,
    load_scored_frame,
    run_transfer,
    source_feature_columns,
)
from src.risk.contract_risk import ContractRiskUtil

SCORED_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
WILD_ROOT = REPO_ROOT / "data" / "raw" / "paper_benchmarks" / "smartbugs-wild"
RESULTS_ROOT = REPO_ROOT / "data" / "raw" / "paper_benchmarks" / "smartbugs-results"
OUTPUT_DIR = REPO_ROOT / "data" / "derived" / "ml_results" / "baseline_comparison"


def _score_one(record: BenchmarkRecord) -> dict:
    risk_util = ContractRiskUtil()
    features = risk_util.extract_features_from_local_file(
        contract_id=record.record_id,
        sol_path=record.source_path,
        address="",
    )
    return {
        **features,
        "source_dataset": record.benchmark,
        "source_subdataset": record.vuln_type,
        "contract_id": f"{record.benchmark}:{record.record_id}",
        "label_status": record.label_status,
        "local_source_path": record.source_path,
        "vuln_types": record.vuln_type,
    }


def score_records(
    records: list[BenchmarkRecord], workers: int
) -> pd.DataFrame:
    if workers <= 1:
        rows = [_score_one(record) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_score_one, records, chunksize=100))
    return pd.DataFrame(rows)


def main() -> None:
    records = smartbugs_wild_records(WILD_ROOT, RESULTS_ROOT)
    print(f"Preparing {len(records)} SmartBugs Wild records", flush=True)
    exact = score_records(records, workers=8)
    existing = load_scored_frame(str(SCORED_INPUT))
    combined = pd.concat([existing, exact], ignore_index=True, sort=False)
    features = source_feature_columns(combined)
    metrics = run_transfer(
        combined,
        DatasetSpec("DAppSCAN", group_by_project=True),
        DatasetSpec(
            "SmartBugs-Wild-reentrancy",
            accepted_labels=("clean", "vulnerable"),
        ),
        features,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact.to_csv(OUTPUT_DIR / "smartbugs_wild_reentrancy_scored.csv", index=False)
    (OUTPUT_DIR / "smartbugs_wild_reentrancy_metrics.json").write_text(
        json.dumps(
            {
                "record_count": len(exact),
                "positive_count": int((exact["label_status"] == "vulnerable").sum()),
                "negative_count": int((exact["label_status"] == "clean").sum()),
                "feature_count": len(features),
                "label_provenance": "any SmartBugs Wild tool reported reentrancy",
                "third_party_tools_run": False,
                "metrics": metrics,
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Wrote {OUTPUT_DIR / 'smartbugs_wild_reentrancy_metrics.json'}")


if __name__ == "__main__":
    main()
