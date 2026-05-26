#!/usr/bin/env python3 -u
"""
STRICT-label variant of run_final_experiments.py.

Re-uses the existing experiment functions (precompute_ranking, run_cv,
run_vote_ensemble, etc.) from `run_final_experiments` but loads data
from `unified_contract_eval_scored_strict.csv` and excludes FORGE-unknown
contracts (keeps only FORGE-vulnerable).

Outputs:
  - data/derived/oof_predictions/ml_{Model}_strict_merged.csv          (OOF per model)
  - data/derived/oof_predictions/ml_Vote_RF_GBC_XGB_strict_merged.csv  (Vote OOF)
  - data/derived/oof_predictions/fold_assignments_strict_merged.csv    (fold map)
  - data/derived/ml_results/final_results_strict.json                  (metrics + trigram ablation)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Import all the experiment infrastructure
from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from scripts.run_final_experiments import (
    precompute_ranking, run_cv, run_vote_ensemble, get_models,
    optimal_f1_threshold, _save_fold_assignments, TRIGRAM_FEATURES,
)


# Strict CSV path
STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"


def load_data_strict(dataset: str = "merged"):
    """Load STRICT data: DAppSCAN clean+vulnerable + FORGE-vulnerable only."""
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    if dataset == "merged":
        dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
        dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
        forge = df[df["source_dataset"] == "FORGE"].copy()
        # STRICT: only FORGE-vulnerable (unknown and suspicious dropped)
        forge = forge[forge["label_status"] == "vulnerable"]
        data = pd.concat([dappscan, forge], ignore_index=True)
    elif dataset == "dappscan":
        data = df[
            (df["source_dataset"] == "DAppSCAN") &
            (df["label_status"].isin(["clean", "vulnerable"]))
        ].copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    data["y"] = (data["label_status"] == "vulnerable").astype(int)
    y = data["y"].values.astype(int)

    contract_ids = data["contract_id"].fillna("").astype(str).values
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

    X = (
        data[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float64)
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"  {dataset} (STRICT): {len(y)} contracts (pos={y.sum()}, neg={(y == 0).sum()}), "
        f"{len(feat_cols)} features",
        flush=True,
    )
    return X, y, groups, feat_cols, contract_ids


def main():
    print("=" * 80, flush=True)
    print("STRICT-LABEL EXPERIMENTS — full 292 source-text features", flush=True)
    print("Reads: unified_contract_eval_scored_strict.csv (22,701 contracts)", flush=True)
    print("=" * 80, flush=True)

    all_results = {}

    # === STRICT MERGED ==========================================================
    print("\n--- Loading STRICT merged data ---", flush=True)
    X_m, y_m, g_m, feats_m, cids_m = load_data_strict("merged")

    print("--- Saving STRICT fold assignments (merged) ---", flush=True)
    _save_fold_assignments(y_m, g_m, cids_m, "strict_merged")

    print("--- STRICT merged feature ranking ---", flush=True)
    ranked_m, ranked_names_m, _ = precompute_ranking(X_m, y_m, feats_m)

    print("\n--- STRICT merged k=80 (individual models) ---", flush=True)
    all_results["strict_merged_k80"] = run_cv(
        X_m, y_m, g_m, ranked_m, k=80,
        contract_ids=cids_m, dataset_name="strict_merged",
    )

    print("\n--- STRICT merged k=80 (Vote ensemble) ---", flush=True)
    all_results["strict_merged_k80"]["Vote(RF+GBC+XGB)"] = run_vote_ensemble(
        X_m, y_m, g_m, ranked_m, k=80,
        contract_ids=cids_m, dataset_name="strict_merged",
    )

    # Also run with full feature set (no MI+RF cut) for headline parity
    print("\n--- STRICT merged FULL feature set (k=all) ---", flush=True)
    all_results["strict_merged_kall"] = run_cv(
        X_m, y_m, g_m, list(range(len(feats_m))), k=len(feats_m),
        contract_ids=cids_m, dataset_name="strict_merged_kall",
    )

    # === STRICT DAPPSCAN ========================================================
    print("\n--- Loading STRICT DAppSCAN data ---", flush=True)
    X_d, y_d, g_d, feats_d, cids_d = load_data_strict("dappscan")

    print("--- Saving STRICT fold assignments (dappscan) ---", flush=True)
    _save_fold_assignments(y_d, g_d, cids_d, "strict_dappscan")

    print("--- STRICT DAppSCAN feature ranking ---", flush=True)
    ranked_d, ranked_names_d, _ = precompute_ranking(X_d, y_d, feats_d)

    print("\n--- STRICT DAppSCAN k=80 (individual models) ---", flush=True)
    all_results["strict_dappscan_k80"] = run_cv(
        X_d, y_d, g_d, ranked_d, k=80,
        contract_ids=cids_d, dataset_name="strict_dappscan",
    )

    # === TRIGRAM ABLATION (STRICT merged) ======================================
    print("\n--- STRICT Trigram ablation (merged, no trigrams) ---", flush=True)
    ranked_nt, ranked_names_nt, keep_nt = precompute_ranking(
        X_m, y_m, feats_m, exclude_feats=TRIGRAM_FEATURES
    )

    print("--- Ablation: RF + GBC without trigrams (STRICT merged) ---", flush=True)
    gkf_ab = GroupKFold(n_splits=5)
    pos_w = (y_m == 0).sum() / max((y_m == 1).sum(), 1)

    ab_results = {}
    X_nt = X_m[:, keep_nt]
    selected_nt = sorted(ranked_nt[:80])

    for mname in ["RandomForest", "GradientBoosting"]:
        t0 = time.time()
        fold_m = []
        for train_idx, test_idx in gkf_ab.split(X_nt, y_m, g_m):
            Xtr = X_nt[train_idx][:, selected_nt]
            Xte = X_nt[test_idx][:, selected_nt]
            ytr, yte = y_m[train_idx], y_m[test_idx]
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
            mfn = get_models(pos_w)[mname]
            m = mfn()
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            roc = roc_auc_score(yte, p)
            pr = average_precision_score(yte, p)
            _, f1 = optimal_f1_threshold(yte, p)
            fold_m.append({"roc": roc, "pr": pr, "f1": f1})
        rocs = [f["roc"] for f in fold_m]
        prs = [f["pr"] for f in fold_m]
        f1s = [f["f1"] for f in fold_m]
        ab_results[mname] = {
            "roc_auc_mean": round(np.mean(rocs), 4),
            "roc_auc_std": round(np.std(rocs), 4),
            "pr_auc_mean": round(np.mean(prs), 4),
            "f1_mean": round(np.mean(f1s), 4),
        }
        print(
            f"  {mname + ' (no trigrams, strict)':40s} ROC={np.mean(rocs):.4f}  "
            f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    all_results["trigram_ablation_strict_merged"] = ab_results

    # === SENSITIVITY TO k (STRICT merged) ======================================
    print("\n--- STRICT k-sensitivity sweep (RF only) ---", flush=True)
    k_results = {}
    from sklearn.ensemble import RandomForestClassifier
    for k in [20, 40, 60, 80, 100, 150, 200, 250, 292]:
        if k > len(ranked_m):
            continue
        sel = sorted(ranked_m[:k])
        fold_rocs = []
        gkf_k = GroupKFold(n_splits=5)
        for train_idx, test_idx in gkf_k.split(X_m, y_m, g_m):
            Xtr = X_m[train_idx][:, sel]
            Xte = X_m[test_idx][:, sel]
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
            rf = RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )
            rf.fit(Xtr, y_m[train_idx])
            p = rf.predict_proba(Xte)[:, 1]
            fold_rocs.append(roc_auc_score(y_m[test_idx], p))
        k_results[k] = round(np.mean(fold_rocs), 4)
        print(f"    k={k:4d}: ROC-AUC = {k_results[k]:.4f}", flush=True)
    all_results["k_sensitivity_strict"] = k_results

    # === Save ==================================================================
    all_results["ranked_top40_strict_merged"] = ranked_names_m[:40]

    out_path = _REPO / "data" / "derived" / "ml_results" / "final_results_strict.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}", flush=True)

    # === SUMMARY ===============================================================
    print("\n" + "=" * 80, flush=True)
    print("STRICT-LABEL FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for config in ["strict_dappscan_k80", "strict_merged_k80", "strict_merged_kall"]:
        if config not in all_results:
            continue
        print(f"\n  {config}:", flush=True)
        for mname, r in all_results[config].items():
            print(
                f"    {mname:25s} ROC={r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}  "
                f"PR={r['pr_auc_mean']:.3f}  F1={r['f1_mean']:.3f}  "
                f"P={r['precision_mean']:.3f} R={r['recall_mean']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
