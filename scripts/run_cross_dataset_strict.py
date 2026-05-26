#!/usr/bin/env python3 -u
"""
STRICT cross-dataset evaluation (full dual-stream pipeline).

Under the STRICT protocol, FORGE contains only known-vulnerable contracts
(unknown + suspicious are dropped), so the cross-dataset story changes:

  - Experiment A (DAppSCAN → FORGE-vuln):  train on DAppSCAN(clean+vuln),
    measure recall + mean probability on FORGE-vulnerable (all positive).
  - Experiment C (Merged GroupKFold CV):   DAppSCAN(clean+vuln) +
    FORGE-vuln, 5-fold GroupKFold under STRICT.
  - Experiment D (DAppSCAN-only CV):       sanity check.

Two feature families per experiment:
  (1) SCAFE structural only (~292 features, k=80 selected)
  (2) Full SCAFE-Sem dual-stream (structural + identifier + LSA)

Outputs:
  data/derived/ml_results/cross_dataset_strict.csv
  data/derived/ml_results/cross_dataset_strict.json
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.semantic_features import (
    read_source, compute_identifier_features, tokenize_solidity, CodeLSA,
)
from scripts.run_scafe_sem_experiments import precompute_ranking, get_models

STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

N_LSA = 50
K_STRUCT = 80

logger = logging.getLogger("cross_dataset_strict")


def _optimal_f1(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 199):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def _metrics(y_true, y_prob, name=""):
    roc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1 else float("nan")
    )
    pr = (
        average_precision_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1 else float("nan")
    )
    opt_t, opt_f1 = _optimal_f1(y_true, y_prob)
    y_pred = (y_prob >= opt_t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "dataset": name,
        "n_samples": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4),
        "f1": round(float(opt_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "threshold": round(float(opt_t), 4),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
    }


def _load_strict():
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)
    return df


def _feat_cols(df: pd.DataFrame) -> List[str]:
    return sorted([
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in _EXCLUDE
        and c not in _ONCHAIN_FEATURES
    ])


def _prep(df, feat_cols):
    X = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float64)
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_semantic(source_paths):
    ident_rows = [
        compute_identifier_features(read_source(sp)) for sp in source_paths
    ]
    token_docs = [tokenize_solidity(read_source(sp)) for sp in source_paths]
    ident_df = pd.DataFrame(ident_rows).fillna(0.0)
    ident_cols = sorted(ident_df.columns.tolist())
    return ident_df[ident_cols].values.astype(np.float64), ident_cols, token_docs


def _build_vote(pos_weight):
    return VotingClassifier(
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
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0)),
        ],
        voting="soft",
    )


def _run_a_strict_struct(df, feat_cols):
    """Experiment A (structural-only): train DAppSCAN → predict FORGE-vuln."""
    train_df = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    train_df["y"] = (train_df["label_status"] == "vulnerable").astype(int)
    test_df = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"] == "vulnerable")
    ].copy()
    test_df["y"] = 1

    X_train = _prep(train_df, feat_cols)
    X_test = _prep(test_df, feat_cols)
    y_train = train_df["y"].values
    y_test = test_df["y"].values

    # Select features via MI+RF on training only
    ranked, _ = precompute_ranking(X_train, y_train, feat_cols)
    sel = sorted(ranked[:K_STRUCT])

    rows = []
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    for mname, mfn in {**get_models(pos_w), "Vote": lambda: _build_vote(pos_w)}.items():
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train[:, sel])
        Xte = sc.transform(X_test[:, sel])
        m = mfn()
        m.fit(Xtr, y_train)
        p = m.predict_proba(Xte)[:, 1]
        # Calibrate threshold on training data
        Xtr_dev_prob = m.predict_proba(Xtr)[:, 1]
        opt_t, _ = _optimal_f1(y_train, Xtr_dev_prob)
        recall_at_t = (p >= opt_t).mean()
        rows.append({
            "experiment": "A_strict_struct",
            "model": mname,
            "train_on": "DAppSCAN",
            "dataset": "FORGE-vuln",
            "n_test": len(p),
            "mean_prob": round(float(p.mean()), 4),
            "median_prob": round(float(np.median(p)), 4),
            "threshold": round(float(opt_t), 4),
            "recall_at_threshold": round(float(recall_at_t), 4),
        })
        logger.info(
            f"  A struct {mname:8s}: "
            f"mean_prob={p.mean():.3f}  median={np.median(p):.3f}  "
            f"recall@t={recall_at_t:.3f}",
        )
    return rows


def _run_c_strict(df, feat_cols, use_semantic: bool):
    """Experiment C: merged GroupKFold CV under STRICT."""
    dapp = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    dapp["y"] = (dapp["label_status"] == "vulnerable").astype(int)
    forge = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"] == "vulnerable")
    ].copy()
    forge["y"] = 1
    merged = pd.concat([dapp, forge], ignore_index=True)

    X_struct = _prep(merged, feat_cols)
    y = merged["y"].values
    cid = merged["contract_id"].fillna("").astype(str)
    group_map = {g: i for i, g in enumerate(sorted(cid.unique()))}
    groups = cid.map(group_map).values

    if use_semantic:
        src = merged["local_source_path"].fillna("").astype(str).values
        logger.info(f"  C-sem: extracting semantic features for {len(merged)}…")
        X_ident, _, token_docs = _extract_semantic(src)
    else:
        X_ident = None
        token_docs = None

    logger.info(
        f"  C ({'dual-stream' if use_semantic else 'struct'}): "
        f"{len(merged)} contracts  (DAppSCAN={len(dapp)}, FORGE={len(forge)}, pos={y.sum()})"
    )

    ranked, _ = precompute_ranking(X_struct, y, feat_cols)
    sel = sorted(ranked[:K_STRUCT])
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)

    rows = []
    for mname, mfn in {**get_models(pos_w), "Vote": lambda: _build_vote(pos_w)}.items():
        gkf = GroupKFold(n_splits=5)
        per_fold = []
        for fi, (tr, te) in enumerate(gkf.split(X_struct, y, groups)):
            parts_tr = [X_struct[tr][:, sel]]
            parts_te = [X_struct[te][:, sel]]
            if use_semantic:
                parts_tr.append(X_ident[tr])
                parts_te.append(X_ident[te])
                lsa = CodeLSA(n_components=N_LSA)
                parts_tr.append(lsa.fit_transform([token_docs[i] for i in tr]))
                parts_te.append(lsa.transform([token_docs[i] for i in te]))
            Xtr = np.hstack(parts_tr)
            Xte = np.hstack(parts_te)
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
            m = mfn()
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            fm = _metrics(y[te], p, f"merged_strict_fold{fi}")
            per_fold.append(fm)
        roc = [f["roc_auc"] for f in per_fold]
        pr = [f["pr_auc"] for f in per_fold]
        f1s = [f["f1"] for f in per_fold]
        precs = [f["precision"] for f in per_fold]
        recs = [f["recall"] for f in per_fold]
        rows.append({
            "experiment": "C_merged_cv_strict" + ("_dualstream" if use_semantic else "_struct"),
            "model": mname,
            "train_on": "DAppSCAN+FORGE (strict)",
            "dataset": "merged_strict",
            "n_samples": len(merged),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "roc_auc": round(float(np.mean(roc)), 4),
            "roc_auc_std": round(float(np.std(roc)), 4),
            "pr_auc": round(float(np.mean(pr)), 4),
            "pr_auc_std": round(float(np.std(pr)), 4),
            "f1": round(float(np.mean(f1s)), 4),
            "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4),
        })
        logger.info(
            f"  C ({'dual' if use_semantic else 'struct'}) {mname:8s}: "
            f"ROC={np.mean(roc):.4f}±{np.std(roc):.4f}  "
            f"PR={np.mean(pr):.4f}  F1={np.mean(f1s):.4f}",
        )
    return rows


def _run_d_strict(df, feat_cols):
    """Experiment D: DAppSCAN-only 5-fold CV under STRICT (sanity check)."""
    sub = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    sub["y"] = (sub["label_status"] == "vulnerable").astype(int)
    X = _prep(sub, feat_cols)
    y = sub["y"].values
    cid = sub["contract_id"].fillna("").astype(str)
    gmap = {g: i for i, g in enumerate(sorted(cid.unique()))}
    groups = cid.map(gmap).values

    ranked, _ = precompute_ranking(X, y, feat_cols)
    sel = sorted(ranked[:K_STRUCT])
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)

    rows = []
    for mname, mfn in {**get_models(pos_w), "Vote": lambda: _build_vote(pos_w)}.items():
        per_fold = []
        for fi, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr][:, sel])
            Xte = sc.transform(X[te][:, sel])
            m = mfn()
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            per_fold.append(_metrics(y[te], p, f"dapp_strict_fold{fi}"))
        roc = [f["roc_auc"] for f in per_fold]
        rows.append({
            "experiment": "D_dappscan_cv_strict",
            "model": mname,
            "train_on": "DAppSCAN",
            "dataset": "DAppSCAN_strict",
            "roc_auc": round(float(np.mean(roc)), 4),
            "roc_auc_std": round(float(np.std(roc)), 4),
            "f1": round(float(np.mean([f["f1"] for f in per_fold])), 4),
        })
        logger.info(
            f"  D {mname:8s}: ROC={np.mean(roc):.4f}±{np.std(roc):.4f}",
        )
    return rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_strict()
    feat_cols = _feat_cols(df)
    logger.info(f"STRICT data: {len(df)}, structural features: {len(feat_cols)}")

    all_rows = []
    logger.info("\n=== Experiment A (struct): DAppSCAN → FORGE-vuln ===")
    all_rows.extend(_run_a_strict_struct(df, feat_cols))

    logger.info("\n=== Experiment C (struct): Merged GroupKFold CV ===")
    all_rows.extend(_run_c_strict(df, feat_cols, use_semantic=False))

    logger.info("\n=== Experiment C (dual-stream): Merged GroupKFold CV ===")
    all_rows.extend(_run_c_strict(df, feat_cols, use_semantic=True))

    logger.info("\n=== Experiment D: DAppSCAN-only CV ===")
    all_rows.extend(_run_d_strict(df, feat_cols))

    out_csv = RESULTS_DIR / "cross_dataset_strict.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    out_json = RESULTS_DIR / "cross_dataset_strict.json"
    out_json.write_text(json.dumps(all_rows, indent=2, default=str))
    logger.info(f"\nWrote {out_csv} ({len(all_rows)} rows) and {out_json}")
    logger.info(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
