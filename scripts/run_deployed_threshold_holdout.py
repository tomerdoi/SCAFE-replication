#!/usr/bin/env python3 -u
"""
Deployed-threshold held-out evaluation (true 80/20 project-grouped split).

Concept (GAP-15):
  The deployed threshold in the paper was calibrated using OOF predictions
  on the full corpus, which mixes calibration and evaluation samples. To
  give an honest deployment estimate, this script performs a single
  80/20 project-grouped split on the STRICT corpus:

    - Split 80% / 20% by project_id (NOT by contract_id) so no project
      appears in both folds.
    - Train the best SCAFE-Sem (dual-stream) model on 80%.
    - Calibrate the operating threshold on the same 80% via inner 5-fold
      GroupKFold cross-validation (target F1 = 0.745).
    - Apply that fixed threshold to the held-out 20% and report
      precision / recall / F1 / queue-reduction.

This is the "real" deployment number that a regulator could quote.

Output:
  data/derived/ml_results/deployed_threshold_holdout.json
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
)
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.semantic_features import (
    read_source, compute_identifier_features, tokenize_solidity, CodeLSA,
)
from scripts.run_scafe_sem_experiments import precompute_ranking

STRICT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

K_STRUCT = 80
N_LSA = 50
HOLDOUT_FRAC = 0.20
RANDOM_STATE = 42

logger = logging.getLogger("deployed_threshold")


def _optimal_f1(y_true, y_prob):
    best, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 199):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, t
    return best_t, best


def _project_id(contract_id: str) -> str:
    """Extract a coarser group than contract_id (the 'project').

    DAppSCAN paths look like `<audit>__<project>__<contract>` so we take
    the first two tokens as the project.  FORGE has no obvious project
    grouping so we fall back to the contract_id (most conservative).
    """
    if not contract_id:
        return ""
    tokens = contract_id.split("/")
    if len(tokens) >= 2:
        return "/".join(tokens[:2])
    return contract_id


def _load_strict():
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)
    dapp = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    forge = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"] == "vulnerable")
    ].copy()
    merged = pd.concat([dapp, forge], ignore_index=True)
    merged["y"] = (merged["label_status"] == "vulnerable").astype(int)
    merged["project_id"] = merged["contract_id"].fillna("").astype(str).map(_project_id)
    return merged


def _feat_cols(df):
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


def _build_features(struct_X, ident_X, token_docs, train_idx, test_idx, struct_sel):
    parts_tr = [struct_X[train_idx][:, struct_sel], ident_X[train_idx]]
    parts_te = [struct_X[test_idx][:, struct_sel], ident_X[test_idx]]
    lsa = CodeLSA(n_components=N_LSA)
    parts_tr.append(lsa.fit_transform([token_docs[i] for i in train_idx]))
    parts_te.append(lsa.transform([token_docs[i] for i in test_idx]))
    return np.hstack(parts_tr), np.hstack(parts_te)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_strict()
    feat_cols = _feat_cols(df)
    logger.info(
        f"STRICT corpus: {len(df)} contracts  "
        f"({df['y'].sum()} pos, {(df['y']==0).sum()} neg)  "
        f"projects: {df['project_id'].nunique()}"
    )

    # ── 80/20 project-grouped split ───────────────────────────────────────
    gss = GroupShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(df, df["y"], groups=df["project_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    logger.info(
        f"Held-out 20%: {len(test_df)} contracts  "
        f"(pos={test_df['y'].sum()}, neg={(test_df['y']==0).sum()})"
    )

    # ── Feature extraction ────────────────────────────────────────────────
    X_struct = _prep(df, feat_cols)
    y = df["y"].values
    src = df["local_source_path"].fillna("").astype(str).values
    logger.info("Extracting semantic features for full corpus...")
    ident_rows = [compute_identifier_features(read_source(sp)) for sp in src]
    token_docs = [tokenize_solidity(read_source(sp)) for sp in src]
    ident_df = pd.DataFrame(ident_rows).fillna(0.0)
    ident_cols = sorted(ident_df.columns.tolist())
    X_ident = ident_df[ident_cols].values.astype(np.float64)
    logger.info(f"  Identifier features: {len(ident_cols)}")

    # Feature selection on training only
    ranked, _ = precompute_ranking(X_struct[train_idx], y[train_idx], feat_cols)
    struct_sel = sorted(ranked[:K_STRUCT])

    # ── Inner 5-fold CV on training only → calibrate threshold ───────────
    pos_w = (y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)
    inner_groups = train_df["project_id"].values
    logger.info("Calibrating threshold via inner 5-fold GroupKFold on TRAINING fold...")
    inner_probs = np.zeros(len(train_idx))
    for fi, (itr, ite) in enumerate(
        GroupKFold(n_splits=5).split(X_struct[train_idx], y[train_idx], inner_groups)
    ):
        absolute_itr = train_idx[itr]
        absolute_ite = train_idx[ite]
        Xtr_inner, Xte_inner = _build_features(
            X_struct, X_ident, token_docs,
            absolute_itr, absolute_ite, struct_sel,
        )
        sc = StandardScaler()
        Xtr_inner = sc.fit_transform(Xtr_inner)
        Xte_inner = sc.transform(Xte_inner)
        m = _build_vote(pos_w)
        m.fit(Xtr_inner, y[absolute_itr])
        inner_probs[ite] = m.predict_proba(Xte_inner)[:, 1]

    y_train = y[train_idx]
    opt_t, opt_f1_val = _optimal_f1(y_train, inner_probs)
    logger.info(
        f"  Inner-CV calibration: threshold={opt_t:.4f}  inner-F1={opt_f1_val:.4f}"
    )

    # ── Final model: train on ALL 80%, evaluate on 20% with fixed threshold ─
    logger.info("Training final model on full 80% training fold...")
    Xtr_final, Xte_final = _build_features(
        X_struct, X_ident, token_docs, train_idx, test_idx, struct_sel,
    )
    sc = StandardScaler()
    Xtr_final = sc.fit_transform(Xtr_final)
    Xte_final = sc.transform(Xte_final)
    final_model = _build_vote(pos_w)
    final_model.fit(Xtr_final, y_train)
    test_probs = final_model.predict_proba(Xte_final)[:, 1]

    y_test = y[test_idx]
    y_pred = (test_probs >= opt_t).astype(int)
    test_roc = (
        roc_auc_score(y_test, test_probs)
        if len(np.unique(y_test)) > 1 else float("nan")
    )
    test_pr = (
        average_precision_score(y_test, test_probs)
        if len(np.unique(y_test)) > 1 else float("nan")
    )
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    queue_reduction = 1.0 - y_pred.mean()

    result = {
        "protocol": "strict_holdout",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "test_pos_rate": round(float(y_test.mean()), 4),
        "n_projects_train": int(train_df["project_id"].nunique()),
        "n_projects_test": int(test_df["project_id"].nunique()),
        "calibrated_threshold": round(float(opt_t), 4),
        "inner_cv_f1": round(float(opt_f1_val), 4),
        "test_roc_auc": round(float(test_roc), 4),
        "test_pr_auc": round(float(test_pr), 4),
        "test_f1": round(float(test_f1), 4),
        "test_precision": round(float(test_prec), 4),
        "test_recall": round(float(test_rec), 4),
        "test_tp": int(tp), "test_fn": int(fn),
        "test_fp": int(fp), "test_tn": int(tn),
        "queue_reduction": round(float(queue_reduction), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = RESULTS_DIR / "deployed_threshold_holdout.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("\nDeployed-threshold held-out results:")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
