#!/usr/bin/env python3 -u
"""
STRICT-label variant of run_scafe_sem_experiments.py.

Produces the Table 8 (Semantic Decomposition) numbers under the STRICT protocol.

Variants run:
  1. struct_only       : SCAFE structural features only (k=80)
  2. struct_plus_ident : structural + identifier profiling
  3. struct_plus_lsa   : structural + 50-component code LSA
  4. struct_plus_both  : structural + identifier + LSA   (full SCAFE-Sem)

For each variant, runs all 6 base models + Vote ensemble under
5-fold GroupKFold on the STRICT merged corpus, and saves:

  - OOF: data/derived/oof_predictions/sem_{Model}_strict_{variant}.csv
  - JSON: data/derived/ml_results/scafe_sem_strict_results.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.semantic_features import (
    read_source, compute_identifier_features, tokenize_solidity, CodeLSA,
)
from scripts.run_scafe_sem_experiments import (
    precompute_ranking, optimal_f1_threshold, get_models,
)

STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

N_LSA_COMPONENTS = 50


def load_data_strict(dataset: str = "merged"):
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    if dataset == "merged":
        dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
        dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
        forge = df[df["source_dataset"] == "FORGE"].copy()
        # STRICT: only vulnerable (drop unknown + suspicious)
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


def extract_semantic_features(source_paths, label=""):
    n = len(source_paths)
    print(
        f"  Extracting semantic features for {n} contracts"
        f"{' (' + label + ')' if label else ''}...",
        flush=True,
    )
    ident_rows = []
    token_docs = []
    t0 = time.time()
    for i, sp in enumerate(source_paths):
        src = read_source(sp)
        ident_rows.append(compute_identifier_features(src))
        token_docs.append(tokenize_solidity(src))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)...", flush=True)
    ident_df = pd.DataFrame(ident_rows).fillna(0.0)
    ident_cols = sorted(ident_df.columns.tolist())
    X_ident = ident_df[ident_cols].values.astype(np.float64)
    print(
        f"  Done: {len(ident_cols)} identifier features, "
        f"{time.time()-t0:.0f}s total",
        flush=True,
    )
    return X_ident, ident_cols, token_docs


def _save_oof(sample_indices, fold_indices, y_true, y_prob, contract_ids,
              model_name, variant):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sample_idx": sample_indices,
        "contract_id": contract_ids,
        "fold": fold_indices,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    safe_name = model_name.replace("(", "_").replace(")", "").replace("+", "_")
    fname = f"sem_{safe_name}_strict_{variant}.csv"
    df.to_csv(OOF_DIR / fname, index=False)


def run_variant(variant_name, X_struct, X_ident, token_docs, y, groups,
                struct_selected, contract_ids, n_lsa=N_LSA_COMPONENTS, n_folds=5):
    """variant_name ∈ {struct_only, struct_plus_ident, struct_plus_lsa, struct_plus_both}."""
    include_ident = "ident" in variant_name or "both" in variant_name
    include_lsa = "lsa" in variant_name or "both" in variant_name

    print(
        f"\n  >>> Variant: {variant_name} "
        f"(ident={include_ident}, lsa={include_lsa})", flush=True,
    )
    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_models(pos_weight)
    results = {}

    for model_name, model_fn in models.items():
        t0 = time.time()
        fold_metrics = []
        oof_sample_idx, oof_fold, oof_y_true, oof_y_prob = [], [], [], []
        for fold_idx, (tr, te) in enumerate(gkf.split(X_struct, y, groups)):
            blocks_tr = [X_struct[tr][:, struct_selected]]
            blocks_te = [X_struct[te][:, struct_selected]]
            if include_ident:
                blocks_tr.append(X_ident[tr])
                blocks_te.append(X_ident[te])
            if include_lsa:
                lsa = CodeLSA(n_components=n_lsa)
                Xtr_l = lsa.fit_transform([token_docs[i] for i in tr])
                Xte_l = lsa.transform([token_docs[i] for i in te])
                blocks_tr.append(Xtr_l)
                blocks_te.append(Xte_l)

            Xtr = np.hstack(blocks_tr)
            Xte = np.hstack(blocks_te)
            ytr, yte = y[tr], y[te]
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            m = model_fn()
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            roc = roc_auc_score(yte, p)
            pr = average_precision_score(yte, p)
            opt_t, opt_f1 = optimal_f1_threshold(yte, p)
            y_pred = (p >= opt_t).astype(int)
            prec = precision_score(yte, y_pred, zero_division=0.0)
            rec = recall_score(yte, y_pred, zero_division=0.0)
            fold_metrics.append(
                {"roc": roc, "pr": pr, "f1": opt_f1, "prec": prec, "rec": rec}
            )
            oof_sample_idx.extend(te.tolist())
            oof_fold.extend([fold_idx] * len(te))
            oof_y_true.extend(yte.tolist())
            oof_y_prob.extend(p.tolist())

        elapsed = time.time() - t0
        rocs = [f["roc"] for f in fold_metrics]
        prs = [f["pr"] for f in fold_metrics]
        f1s = [f["f1"] for f in fold_metrics]
        precs = [f["prec"] for f in fold_metrics]
        recs = [f["rec"] for f in fold_metrics]
        results[model_name] = {
            "roc_auc_mean": round(np.mean(rocs), 4),
            "roc_auc_std": round(np.std(rocs), 4),
            "roc_auc_folds": [round(x, 4) for x in rocs],
            "pr_auc_mean": round(np.mean(prs), 4),
            "pr_auc_std": round(np.std(prs), 4),
            "f1_mean": round(np.mean(f1s), 4),
            "f1_std": round(np.std(f1s), 4),
            "precision_mean": round(np.mean(precs), 3),
            "recall_mean": round(np.mean(recs), 3),
            "time_s": round(elapsed, 1),
        }
        print(
            f"    {model_name:22s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
            f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
            f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        if contract_ids is not None:
            oof_cids = [contract_ids[i] for i in oof_sample_idx]
            _save_oof(oof_sample_idx, oof_fold, oof_y_true, oof_y_prob,
                      oof_cids, model_name, variant_name)

    # Vote ensemble for this variant
    print(f"    --- Vote(RF+GBC+XGB) [{variant_name}] ---", flush=True)
    t0 = time.time()
    fold_metrics = []
    oof_sample_idx, oof_fold, oof_y_true, oof_y_prob = [], [], [], []
    for fold_idx, (tr, te) in enumerate(gkf.split(X_struct, y, groups)):
        blocks_tr = [X_struct[tr][:, struct_selected]]
        blocks_te = [X_struct[te][:, struct_selected]]
        if include_ident:
            blocks_tr.append(X_ident[tr])
            blocks_te.append(X_ident[te])
        if include_lsa:
            lsa = CodeLSA(n_components=n_lsa)
            Xtr_l = lsa.fit_transform([token_docs[i] for i in tr])
            Xte_l = lsa.transform([token_docs[i] for i in te])
            blocks_tr.append(Xtr_l)
            blocks_te.append(Xte_l)

        Xtr = np.hstack(blocks_tr)
        Xte = np.hstack(blocks_te)
        ytr, yte = y[tr], y[te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        vote = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42, n_jobs=-1)),
                ("gbc", GradientBoostingClassifier(
                    n_estimators=400, learning_rate=0.03, max_depth=5,
                    min_samples_leaf=15, subsample=0.8, random_state=42)),
                ("xgb", XGBClassifier(
                    n_estimators=500, learning_rate=0.05, max_depth=6,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=pos_weight,
                    eval_metric="logloss", random_state=42,
                    n_jobs=-1, verbosity=0)),
            ],
            voting="soft",
        )
        vote.fit(Xtr, ytr)
        p = vote.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, p)
        pr = average_precision_score(yte, p)
        opt_t, opt_f1 = optimal_f1_threshold(yte, p)
        y_pred = (p >= opt_t).astype(int)
        prec = precision_score(yte, y_pred, zero_division=0.0)
        rec = recall_score(yte, y_pred, zero_division=0.0)
        fold_metrics.append(
            {"roc": roc, "pr": pr, "f1": opt_f1, "prec": prec, "rec": rec}
        )
        oof_sample_idx.extend(te.tolist())
        oof_fold.extend([fold_idx] * len(te))
        oof_y_true.extend(yte.tolist())
        oof_y_prob.extend(p.tolist())
    elapsed = time.time() - t0
    rocs = [f["roc"] for f in fold_metrics]
    prs = [f["pr"] for f in fold_metrics]
    f1s = [f["f1"] for f in fold_metrics]
    precs = [f["prec"] for f in fold_metrics]
    recs = [f["rec"] for f in fold_metrics]
    results["Vote(RF+GBC+XGB)"] = {
        "roc_auc_mean": round(np.mean(rocs), 4),
        "roc_auc_std": round(np.std(rocs), 4),
        "roc_auc_folds": [round(x, 4) for x in rocs],
        "pr_auc_mean": round(np.mean(prs), 4),
        "pr_auc_std": round(np.std(prs), 4),
        "f1_mean": round(np.mean(f1s), 4),
        "f1_std": round(np.std(f1s), 4),
        "precision_mean": round(np.mean(precs), 3),
        "recall_mean": round(np.mean(recs), 3),
        "time_s": round(elapsed, 1),
    }
    print(
        f"    {'Vote(RF+GBC+XGB)':22s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
        f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
        f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    if contract_ids is not None:
        oof_cids = [contract_ids[i] for i in oof_sample_idx]
        _save_oof(oof_sample_idx, oof_fold, oof_y_true, oof_y_prob,
                  oof_cids, "Vote_RF_GBC_XGB", variant_name)

    return results


def main():
    print("=" * 80, flush=True)
    print("STRICT SCAFE-Sem decomposition — Table 8", flush=True)
    print("=" * 80, flush=True)

    X_struct, y, groups, struct_names, cids, src_paths = load_data_strict("merged")

    # Pre-select structural features
    ranked, _ = precompute_ranking(X_struct, y, struct_names)
    struct_selected = sorted(ranked[:80])
    print(f"  Selected structural features: k=80 / {len(struct_names)}", flush=True)

    X_ident, ident_cols, token_docs = extract_semantic_features(
        src_paths, label="strict merged"
    )

    all_results = {}
    for variant in [
        "struct_only", "struct_plus_ident", "struct_plus_lsa", "struct_plus_both",
    ]:
        all_results[variant] = run_variant(
            variant, X_struct, X_ident, token_docs, y, groups,
            struct_selected, cids,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "scafe_sem_strict_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
