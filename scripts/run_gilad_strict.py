#!/usr/bin/env python3 -u
"""
STRICT-label variant of run_gilad_experiments.py.

Runs (under strict protocol):
  - Experiment 2: TF-IDF + Logistic Regression baseline
  - Experiment 3: Identifier obfuscation robustness

(Experiment 1 'semantic decomposition' is covered by run_scafe_sem_strict.py.
 Experiment 4 'Slither head-to-head' is dataset-independent and stays as-is.)

Outputs:
  data/derived/ml_results/gilad_experiments_strict.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from scripts.run_gilad_experiments import (
    run_tfidf_lr_baseline, run_obfuscation_experiment,
)

STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"


def load_data_strict(dataset: str = "merged"):
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    if dataset == "merged":
        dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
        dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
        forge = df[df["source_dataset"] == "FORGE"].copy()
        forge = forge[forge["label_status"] == "vulnerable"]
        data = pd.concat([dappscan, forge], ignore_index=True)
    else:
        data = df[
            (df["source_dataset"] == "DAppSCAN") &
            (df["label_status"].isin(["clean", "vulnerable"]))
        ].copy()

    data["y"] = (data["label_status"] == "vulnerable").astype(int)
    y = data["y"].values.astype(int)
    contract_ids = data["contract_id"].fillna("").astype(str).values
    source_paths = data["local_source_path"].fillna("").astype(str).values

    group_key = pd.Series(contract_ids)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)

    feat_cols = sorted([
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and c not in _EXCLUDE
        and c not in _ONCHAIN_FEATURES
    ])
    X_struct = (
        data[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float64)
    )
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"  {dataset} (STRICT): {len(y)} contracts (pos={y.sum()}, neg={(y==0).sum()}), "
        f"{len(feat_cols)} structural features",
        flush=True,
    )
    return X_struct, y, groups, feat_cols, contract_ids, source_paths


def main():
    print("=" * 100, flush=True)
    print("STRICT GILAD EXPERIMENTS — TF-IDF+LR + Obfuscation", flush=True)
    print("=" * 100, flush=True)

    print("\n--- Loading STRICT merged data ---", flush=True)
    X_struct, y, groups, sfeat, cids, spaths = load_data_strict("merged")
    all_results = {}

    print("\n" + "=" * 100)
    print("EXPERIMENT 2 (STRICT): TF-IDF + Logistic Regression Baseline")
    print("=" * 100)
    tfidf_lr_result = run_tfidf_lr_baseline(spaths, y, groups)
    all_results["tfidf_lr_baseline_strict"] = tfidf_lr_result

    print("\n" + "=" * 100)
    print("EXPERIMENT 3 (STRICT): Identifier Obfuscation Robustness")
    print("=" * 100)
    obfusc_result = run_obfuscation_experiment(X_struct, spaths, y, groups, sfeat)
    all_results["obfuscation_strict"] = obfusc_result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gilad_experiments_strict.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nAll STRICT results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
