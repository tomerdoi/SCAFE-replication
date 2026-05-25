#!/usr/bin/env python3 -u
"""
Run UniXcoder and CodeBERT on the STRICT-label corpus
(DAppSCAN clean+vulnerable + FORGE-vulnerable only; FORGE-unknown/suspicious excluded).

This fixes the protocol mismatch in the original submission where ML models
were evaluated on strict labels but DL baselines were evaluated on legacy labels.

Results are saved to:
  data/derived/dl_results/dl_strict_results.json
OOF predictions to:
  data/derived/oof_predictions/dl_{model}_strict_merged.csv

Runtime estimate (Apple M-series MPS):
  UniXcoder strict merged: ~6.5 h
  CodeBERT  strict merged: ~6.0 h
  Total (sequential):      ~12.5 h
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.dl_baselines import evaluate_codebert, _OUT_DIR, _OOF_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
_OUT_JSON   = _OUT_DIR / "dl_strict_results.json"

CODEBERT_HF  = "microsoft/codebert-base"
UNIXCODER_HF = "microsoft/unixcoder-base"

DATASET_LABEL = "strict_merged"   # used for OOF filenames: dl_{model}_strict_merged.csv


def load_strict_merged() -> tuple:
    """Load DAppSCAN + FORGE-vulnerable (strict scheme) from the strict CSV."""
    df = pd.read_csv(_STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)

    # DAppSCAN: keep clean + vulnerable
    dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
    dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])].copy()

    # FORGE strict: only verified-vulnerable (unknown and suspicious already excluded
    # in the strict CSV, but be explicit)
    forge = df[df["source_dataset"] == "FORGE"].copy()
    forge = forge[forge["label_status"] == "vulnerable"].copy()

    merged = pd.concat([dappscan, forge], ignore_index=True)
    merged["y"] = (merged["label_status"] == "vulnerable").astype(int)

    y = merged["y"].values.astype(int)

    group_key = merged["contract_id"].fillna("").astype(str)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)

    merged = merged.reset_index(drop=True)
    logger.info(
        f"Strict merged: {len(merged)} contracts  "
        f"pos={int(y.sum())}  neg={int((y == 0).sum())}  "
        f"pos_rate={y.mean():.1%}"
    )
    return merged, y, groups


def run_model(model_hf: str, model_label: str, df, y, groups, existing: dict) -> dict:
    """Run a single model, skipping if results already exist."""
    key = f"{model_label}_{DATASET_LABEL}"
    if key in existing:
        r = existing[key]
        print(
            f"  [SKIP] {model_label} strict_merged already done  "
            f"ROC-AUC={r['roc_auc_mean']:.4f}  PR-AUC={r['pr_auc_mean']:.4f}  "
            f"F1={r['f1_mean']:.4f}",
            flush=True,
        )
        return r

    print(f"\n{'='*80}", flush=True)
    print(f"Running {model_label} on STRICT MERGED  ({len(df)} contracts)", flush=True)
    print(f"  model_hf : {model_hf}", flush=True)
    print(f"  protocol : 5-fold GroupKFold, freeze 8 layers, AdamW lr=2e-5, bs=16, max_len=512", flush=True)
    print(f"  dataset  : {DATASET_LABEL}  (OOF → dl_{model_label}_{DATASET_LABEL}.csv)", flush=True)
    print(f"{'='*80}", flush=True)

    t0 = time.time()
    result = evaluate_codebert(
        df=df, y=y, groups=groups,
        model_name_hf=model_hf,
        model_label=model_label,
        max_len=512,
        n_folds=5,
        n_epochs=5,
        batch_size=16,
        lr=2e-5,
        freeze_layers=8,
        dataset=DATASET_LABEL,
    )
    elapsed = time.time() - t0
    result["model"]        = model_label
    result["dataset"]      = DATASET_LABEL
    result["elapsed_s"]    = round(elapsed, 1)

    print(
        f"\n  {model_label} strict_merged DONE in {elapsed/3600:.1f} h  "
        f"ROC-AUC={result['roc_auc_mean']:.4f}±{result['roc_auc_std']:.4f}  "
        f"PR-AUC={result['pr_auc_mean']:.4f}  F1={result['f1_mean']:.4f}  "
        f"P={result['precision_mean']:.4f} R={result['recall_mean']:.4f}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run DL baselines on strict-label corpus")
    parser.add_argument(
        "--models", default="unixcoder,codebert",
        help="Comma-separated list of models to run: unixcoder, codebert (default: both)",
    )
    args = parser.parse_args()
    models_to_run = [m.strip().lower() for m in args.models.split(",")]

    print("\n" + "="*80, flush=True)
    print("DL BASELINES — STRICT LABEL PROTOCOL", flush=True)
    print("Models:", ", ".join(models_to_run), flush=True)
    print("="*80 + "\n", flush=True)

    # Load checkpoint if partial results exist
    existing = {}
    if _OUT_JSON.exists():
        existing = json.loads(_OUT_JSON.read_text())
        print(f"Loaded existing results from {_OUT_JSON}: {list(existing.keys())}", flush=True)

    # Load strict merged data
    df, y, groups = load_strict_merged()

    all_results = dict(existing)

    model_map = {
        "unixcoder": (UNIXCODER_HF, "unixcoder"),
        "codebert":  (CODEBERT_HF,  "codebert"),
    }

    total_t0 = time.time()
    for model_key in models_to_run:
        if model_key not in model_map:
            print(f"Unknown model '{model_key}', skipping.", flush=True)
            continue
        hf, label = model_map[model_key]
        r = run_model(hf, label, df, y, groups, all_results)
        all_results[f"{label}_{DATASET_LABEL}"] = r

        # Save after every model so progress is not lost
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        _OUT_JSON.write_text(json.dumps(all_results, indent=2))
        print(f"  Checkpoint saved → {_OUT_JSON}", flush=True)

    total_elapsed = time.time() - total_t0

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*80, flush=True)
    print("STRICT-LABEL DL RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    for key, r in all_results.items():
        print(
            f"  {key:35s}: ROC-AUC={r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}"
            f"  PR-AUC={r['pr_auc_mean']:.4f}  F1={r['f1_mean']:.4f}"
            f"  P={r['precision_mean']:.4f}  R={r['recall_mean']:.4f}",
            flush=True,
        )

    print(f"\nTotal wall time: {total_elapsed/3600:.1f} h", flush=True)
    print(f"OOF files:       {_OOF_DIR}/dl_{{model}}_strict_merged.csv", flush=True)
    print(f"Results JSON:    {_OUT_JSON}", flush=True)
    print("="*80 + "\n", flush=True)

    # ── Compare with legacy results ────────────────────────────────────────────
    legacy_json = _OUT_DIR / "dl_baseline_results.json"
    if legacy_json.exists():
        legacy = json.loads(legacy_json.read_text())
        print("COMPARISON: strict vs legacy protocol", flush=True)
        print(f"{'Model':<15} {'Metric':<10} {'Strict':>8} {'Legacy':>8} {'Delta':>8}", flush=True)
        print("-" * 55, flush=True)
        for lbl in ["unixcoder", "codebert"]:
            skey = f"{lbl}_{DATASET_LABEL}"
            if skey not in all_results:
                continue
            sr = all_results[skey]
            # Find legacy merged result
            lr = next((x for x in legacy if x.get("model") == lbl
                       and "merged" in x.get("dataset", "").lower()), None)
            if lr is None:
                continue
            for metric, sk, lk in [
                ("ROC-AUC", "roc_auc_mean", "roc_auc_mean"),
                ("PR-AUC",  "pr_auc_mean",  "pr_auc_mean"),
                ("F1",      "f1_mean",       "f1_mean"),
            ]:
                sv, lv = sr[sk], lr[lk]
                print(
                    f"  {lbl:<13} {metric:<10} {sv:>8.4f} {lv:>8.4f} {sv-lv:>+8.4f}",
                    flush=True,
                )
        print("", flush=True)


if __name__ == "__main__":
    main()
