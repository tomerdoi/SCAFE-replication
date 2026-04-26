#!/usr/bin/env python3 -u
"""
Run UniXcoder fine-tuning with the exact same protocol as CodeBERT.

Uses the existing evaluate_codebert() infrastructure with
model_name_hf="microsoft/unixcoder-base".
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.dl_baselines import (
    evaluate_codebert, load_dappscan_data, load_merged_data,
    _OUT_DIR, _OOF_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UNIXCODER_HF = "microsoft/unixcoder-base"

RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"


def main():
    print("=" * 90, flush=True)
    print("UniXcoder Fine-Tuning Experiment", flush=True)
    print(f"Model: {UNIXCODER_HF}", flush=True)
    print("Same protocol as CodeBERT: 5-fold GroupKFold, freeze 8 layers,", flush=True)
    print("AdamW lr=2e-5, batch=16, max_len=512, early stop patience=2", flush=True)
    print("=" * 90, flush=True)

    all_results = {}
    out_path = RESULTS_DIR / "unixcoder_results.json"

    # ── Merged dataset (skip if already done) ──
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if "merged" in prev:
            print("--- Merged results already exist, skipping ---", flush=True)
            all_results["merged"] = prev["merged"]
            r_merged = prev["merged"]
            print(f"  (loaded) ROC-AUC={r_merged['roc_auc_mean']:.4f}±{r_merged['roc_auc_std']:.4f}  "
                  f"PR-AUC={r_merged['pr_auc_mean']:.4f}  F1={r_merged['f1_mean']:.4f}", flush=True)

    if "merged" not in all_results:
        print("\n--- Loading merged data ---", flush=True)
        df_m, y_m, g_m = load_merged_data()

        print("\n--- Running UniXcoder on MERGED ---", flush=True)
        t0 = time.time()
        r_merged = evaluate_codebert(
            df=df_m, y=y_m, groups=g_m,
            model_name_hf=UNIXCODER_HF,
            model_label="unixcoder",
            max_len=512,
            n_folds=5,
            n_epochs=5,
            batch_size=16,
            lr=2e-5,
            freeze_layers=8,
            dataset="Merged",
        )
        merged_time = time.time() - t0
        r_merged["display_name"] = "UniXcoder (fine-tuned)"
        r_merged["model"] = "unixcoder"
        all_results["merged"] = r_merged
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nMerged results: ROC-AUC={r_merged['roc_auc_mean']:.4f}±{r_merged['roc_auc_std']:.4f}  "
              f"PR-AUC={r_merged['pr_auc_mean']:.4f}  F1={r_merged['f1_mean']:.4f}  "
              f"Time={merged_time:.0f}s", flush=True)

    # ── DAppSCAN dataset ──
    print("\n--- Loading DAppSCAN data ---", flush=True)
    df_d, y_d, g_d = load_dappscan_data()

    print("\n--- Running UniXcoder on DAPPSCAN ---", flush=True)
    t0 = time.time()
    r_dappscan = evaluate_codebert(
        df=df_d, y=y_d, groups=g_d,
        model_name_hf=UNIXCODER_HF,
        model_label="unixcoder",
        max_len=512,
        n_folds=5,
        n_epochs=5,
        batch_size=16,
        lr=2e-5,
        freeze_layers=8,
        dataset="DAppSCAN",
    )
    dappscan_time = time.time() - t0
    r_dappscan["display_name"] = "UniXcoder (fine-tuned)"
    r_dappscan["model"] = "unixcoder"
    all_results["dappscan"] = r_dappscan
    print(f"\nDAppSCAN results: ROC-AUC={r_dappscan['roc_auc_mean']:.4f}±{r_dappscan['roc_auc_std']:.4f}  "
          f"PR-AUC={r_dappscan['pr_auc_mean']:.4f}  F1={r_dappscan['f1_mean']:.4f}  "
          f"Time={dappscan_time:.0f}s", flush=True)

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "unixcoder_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)

    # ── Final summary ──
    print("\n" + "=" * 90, flush=True)
    print("UNIXCODER RESULTS SUMMARY", flush=True)
    print("=" * 90, flush=True)
    for ds, r in all_results.items():
        print(f"  {ds:12s}: ROC-AUC={r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
              f"PR-AUC={r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f}  "
              f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
              f"P={r['precision_mean']:.4f} R={r['recall_mean']:.4f}", flush=True)

    # Compare with CodeBERT
    print("\n  Comparison with CodeBERT (from existing results):", flush=True)
    dl_json = _REPO / "data" / "derived" / "dl_results" / "dl_baseline_results.json"
    if dl_json.exists():
        dl_results = json.loads(dl_json.read_text())
        for r in dl_results:
            if r["model"] == "codebert":
                print(f"  CodeBERT {r['dataset']:12s}: ROC-AUC={r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
                      f"PR-AUC={r['pr_auc_mean']:.4f}  F1={r['f1_mean']:.4f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("DONE", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
