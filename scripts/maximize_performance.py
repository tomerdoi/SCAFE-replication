#!/usr/bin/env python3 -u
"""
Aggressive optimization to maximize SCAFE-Sem performance.

Strategy (based on error analysis):
  1. More models: LightGBM, CatBoost, MLP
  2. More features: k=120 structural + LSA=100
  3. Better stacking: meta-learner on diverse model OOFs
  4. Optimized hyperparameters for GBC/XGB
"""
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.semantic_features import (
    read_source, compute_identifier_features, tokenize_solidity, CodeLSA,
)

SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

# Try different configs
CONFIGS = [
    {"k_struct": 80, "n_lsa": 50, "label": "k80_lsa50"},
    {"k_struct": 120, "n_lsa": 80, "label": "k120_lsa80"},
    {"k_struct": 150, "n_lsa": 100, "label": "k150_lsa100"},
]


def load_data():
    df = pd.read_csv(SCORED_V2, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)
    dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
    dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
    forge = df[df["source_dataset"] == "FORGE"].copy()
    forge = forge[forge["label_status"].isin(["vulnerable", "unknown"])]
    data = pd.concat([dappscan, forge], ignore_index=True)
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
        and c not in _EXCLUDE and c not in _ONCHAIN_FEATURES
    ])
    X_struct = data[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Loaded: {len(y)} contracts, {y.sum()} pos, {X_struct.shape[1]} structural features")
    return X_struct, y, groups, feat_cols, contract_ids, source_paths


def extract_semantic(source_paths):
    from src.ml.semantic_features import compute_identifier_features, tokenize_solidity, read_source
    n = len(source_paths)
    print(f"  Extracting semantic features for {n} contracts...")
    ident_rows, token_docs = [], []
    t0 = time.time()
    for i, sp in enumerate(source_paths):
        src = read_source(sp)
        ident_rows.append(compute_identifier_features(src))
        token_docs.append(tokenize_solidity(src))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)...")
    ident_df = pd.DataFrame(ident_rows).fillna(0.0)
    ident_cols = sorted(ident_df.columns.tolist())
    X_ident = ident_df[ident_cols].values.astype(np.float64)
    print(f"  Done: {len(ident_cols)} identifier features, {time.time()-t0:.0f}s")
    return X_ident, ident_cols, token_docs


def precompute_ranking(X, y, feat_names):
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mask = mi > 0.001
    X_mi = X[:, mask]
    mi_indices = [i for i, m in enumerate(mask) if m]
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_mi, y)
    imp = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    return [mi_indices[ranked[i]] for i in range(len(ranked))]


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def get_all_models(pos_weight):
    return {
        "RF": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "GBC": lambda: GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=6,
            min_samples_leaf=10, subsample=0.8, max_features=0.7,
            random_state=42),
        "XGB": lambda: XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=7,
            min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
        "LGBM": lambda: LGBMClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=7,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42, n_jobs=-1, verbose=-1),
        "CAT": lambda: CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=7,
            l2_leaf_reg=3, auto_class_weights="Balanced",
            random_seed=42, verbose=0),
        "ET": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', alpha=0.001, batch_size=256,
            learning_rate='adaptive', learning_rate_init=0.001,
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=42),
    }


def _save_oof(sample_indices, fold_indices, y_true, y_prob, contract_ids,
              model_name, prefix="max"):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sample_idx": sample_indices,
        "contract_id": contract_ids,
        "fold": fold_indices,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    fname = f"{prefix}_{model_name}_merged.csv"
    df.to_csv(OOF_DIR / fname, index=False)


def run_all_models_cv(X_struct, X_ident, token_docs, y, groups,
                      struct_feat_names, k_struct, n_lsa, n_folds=5,
                      contract_ids=None, config_label=""):
    """Run all models with given feature config."""
    print(f"\n{'='*80}")
    print(f"CONFIG: {config_label} (k_struct={k_struct}, n_lsa={n_lsa})")
    print(f"{'='*80}")

    ranked = precompute_ranking(X_struct, y, struct_feat_names)
    struct_selected = sorted(ranked[:k_struct])
    print(f"  Structural: {k_struct} features selected")

    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_all_models(pos_weight)

    all_oof = {}
    results = {}

    for model_name, model_fn in models.items():
        t0 = time.time()
        fold_metrics = []
        oof_idx, oof_fold, oof_yt, oof_yp = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_struct, y, groups)):
            Xtr_s = X_struct[train_idx][:, struct_selected]
            Xte_s = X_struct[test_idx][:, struct_selected]
            Xtr_i = X_ident[train_idx]
            Xte_i = X_ident[test_idx]

            lsa = CodeLSA(n_components=n_lsa)
            train_docs = [token_docs[i] for i in train_idx]
            test_docs = [token_docs[i] for i in test_idx]
            Xtr_l = lsa.fit_transform(train_docs)
            Xte_l = lsa.transform(test_docs)

            Xtr = np.hstack([Xtr_s, Xtr_i, Xtr_l])
            Xte = np.hstack([Xte_s, Xte_i, Xte_l])
            ytr, yte = y[train_idx], y[test_idx]

            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            m = model_fn()
            m.fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]

            roc = roc_auc_score(yte, prob)
            pr = average_precision_score(yte, prob)
            _, f1 = optimal_f1_threshold(yte, prob)
            fold_metrics.append({"roc": roc, "pr": pr, "f1": f1})

            oof_idx.extend(test_idx.tolist())
            oof_fold.extend([fold_idx] * len(test_idx))
            oof_yt.extend(yte.tolist())
            oof_yp.extend(prob.tolist())

        elapsed = time.time() - t0
        rocs = [f["roc"] for f in fold_metrics]
        prs = [f["pr"] for f in fold_metrics]
        f1s = [f["f1"] for f in fold_metrics]

        results[model_name] = {
            "roc_mean": round(np.mean(rocs), 4), "roc_std": round(np.std(rocs), 4),
            "roc_folds": rocs,
            "pr_mean": round(np.mean(prs), 4),
            "f1_mean": round(np.mean(f1s), 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  {model_name:6s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
              f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  ({elapsed:.1f}s)")

        all_oof[model_name] = {
            "idx": oof_idx, "fold": oof_fold,
            "y_true": oof_yt, "y_prob": oof_yp,
        }

        if contract_ids is not None:
            cids = [contract_ids[i] for i in oof_idx]
            _save_oof(oof_idx, oof_fold, oof_yt, oof_yp, cids,
                      f"{model_name}_{config_label}")

    # ── Stacking ensemble ───────────────────────────────────────────────
    print("\n  --- Stacking Ensembles ---")

    # Collect OOF predictions aligned by sample index
    n = len(y)
    model_names = list(all_oof.keys())
    oof_matrix = np.zeros((n, len(model_names)))
    oof_y = np.zeros(n)
    oof_folds = np.zeros(n, dtype=int)

    for mi, mname in enumerate(model_names):
        for idx, prob in zip(all_oof[mname]["idx"], all_oof[mname]["y_prob"]):
            oof_matrix[idx, mi] = prob
        for idx, yt in zip(all_oof[mname]["idx"], all_oof[mname]["y_true"]):
            oof_y[idx] = yt
        for idx, fid in zip(all_oof[mname]["idx"], all_oof[mname]["fold"]):
            oof_folds[idx] = fid

    # Simple average of all models
    avg_prob = oof_matrix.mean(axis=1)
    fold_rocs_avg = []
    for fid in range(n_folds):
        m = oof_folds == fid
        fold_rocs_avg.append(roc_auc_score(oof_y[m], avg_prob[m]))
    print(f"  {'AVG_ALL':6s} ROC={np.mean(fold_rocs_avg):.4f}±{np.std(fold_rocs_avg):.4f}")

    # Average of top-5 models (by ROC-AUC)
    model_rocs = [(mname, results[mname]["roc_mean"]) for mname in model_names]
    model_rocs.sort(key=lambda x: x[1], reverse=True)
    top5 = [m[0] for m in model_rocs[:5]]
    top5_idx = [model_names.index(m) for m in top5]
    avg5_prob = oof_matrix[:, top5_idx].mean(axis=1)
    fold_rocs_5 = []
    for fid in range(n_folds):
        m = oof_folds == fid
        fold_rocs_5.append(roc_auc_score(oof_y[m], avg5_prob[m]))
    print(f"  {'AVG_T5':6s} ROC={np.mean(fold_rocs_5):.4f}±{np.std(fold_rocs_5):.4f}  "
          f"(top-5: {top5})")

    # Rank averaging
    from scipy.stats import rankdata
    rank_matrix = np.zeros_like(oof_matrix)
    for mi in range(len(model_names)):
        rank_matrix[:, mi] = rankdata(oof_matrix[:, mi]) / n
    rank_avg = rank_matrix.mean(axis=1)
    fold_rocs_rank = []
    for fid in range(n_folds):
        m = oof_folds == fid
        fold_rocs_rank.append(roc_auc_score(oof_y[m], rank_avg[m]))
    print(f"  {'RANK':6s} ROC={np.mean(fold_rocs_rank):.4f}±{np.std(fold_rocs_rank):.4f}")

    # Weighted average (optimized per fold)
    from scipy.optimize import minimize
    fold_rocs_opt = []
    for test_fold in range(n_folds):
        train_mask = oof_folds != test_fold
        test_mask = oof_folds == test_fold
        X_tr = oof_matrix[train_mask]
        y_tr = oof_y[train_mask]
        X_te = oof_matrix[test_mask]
        y_te = oof_y[test_mask]

        def neg_roc(w):
            w = np.abs(w) / np.abs(w).sum()
            prob = X_tr @ w
            return -roc_auc_score(y_tr, prob)

        w0 = np.ones(len(model_names)) / len(model_names)
        res = minimize(neg_roc, w0, method='Nelder-Mead',
                       options={'maxiter': 1000})
        w_opt = np.abs(res.x) / np.abs(res.x).sum()
        prob_te = X_te @ w_opt
        fold_rocs_opt.append(roc_auc_score(y_te, prob_te))

    print(f"  {'OPT_W':6s} ROC={np.mean(fold_rocs_opt):.4f}±{np.std(fold_rocs_opt):.4f}")

    # Stacking with LR meta-learner
    fold_rocs_stack = []
    fold_f1s_stack = []
    stack_oof_prob = np.zeros(n)
    for test_fold in range(n_folds):
        train_mask = oof_folds != test_fold
        test_mask = oof_folds == test_fold
        X_tr = oof_matrix[train_mask]
        y_tr = oof_y[train_mask]
        X_te = oof_matrix[test_mask]
        y_te = oof_y[test_mask]

        meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        meta.fit(X_tr, y_tr)
        prob_te = meta.predict_proba(X_te)[:, 1]
        stack_oof_prob[test_mask] = prob_te
        fold_rocs_stack.append(roc_auc_score(y_te, prob_te))
        _, f1 = optimal_f1_threshold(y_te, prob_te)
        fold_f1s_stack.append(f1)

    print(f"  {'STACK':6s} ROC={np.mean(fold_rocs_stack):.4f}±{np.std(fold_rocs_stack):.4f}  "
          f"F1={np.mean(fold_f1s_stack):.4f}")

    # Save stacking OOF
    if contract_ids is not None:
        _save_oof(list(range(n)), oof_folds.tolist(), oof_y.tolist(),
                  stack_oof_prob.tolist(),
                  [contract_ids[i] for i in range(n)],
                  f"Stack_{config_label}")

    results["AVG_ALL"] = {"roc_mean": round(np.mean(fold_rocs_avg), 4),
                          "roc_folds": fold_rocs_avg}
    results["AVG_T5"] = {"roc_mean": round(np.mean(fold_rocs_5), 4),
                         "roc_folds": fold_rocs_5}
    results["RANK_AVG"] = {"roc_mean": round(np.mean(fold_rocs_rank), 4),
                           "roc_folds": fold_rocs_rank}
    results["OPT_WEIGHT"] = {"roc_mean": round(np.mean(fold_rocs_opt), 4),
                             "roc_folds": fold_rocs_opt}
    results["LR_STACK"] = {"roc_mean": round(np.mean(fold_rocs_stack), 4),
                           "roc_folds": fold_rocs_stack,
                           "f1_mean": round(np.mean(fold_f1s_stack), 4)}

    return results


def main():
    print("=" * 100)
    print("MAXIMIZE PERFORMANCE: Aggressive Optimization")
    print("=" * 100)

    X_struct, y, groups, sfeat, cids, spaths = load_data()
    X_ident, ident_cols, token_docs = extract_semantic(spaths)

    all_results = {}
    for cfg in CONFIGS:
        results = run_all_models_cv(
            X_struct, X_ident, token_docs, y, groups, sfeat,
            k_struct=cfg["k_struct"], n_lsa=cfg["n_lsa"],
            contract_ids=cids, config_label=cfg["label"]
        )
        all_results[cfg["label"]] = results

    # Compare best results vs CodeBERT
    print("\n" + "=" * 100)
    print("COMPARISON VS CODEBERT (ROC-AUC = 0.9438)")
    print("=" * 100)

    cb_oof = pd.read_csv(OOF_DIR / "dl_codebert_Merged.csv")
    cb_folds = cb_oof.groupby('fold').apply(
        lambda x: roc_auc_score(x['y_true'], x['y_prob'])).values
    cb_mean = np.mean(cb_folds)

    print(f"\n  CodeBERT: ROC={cb_mean:.4f}")
    for cfg_label, results in all_results.items():
        print(f"\n  Config: {cfg_label}")
        all_methods = [(k, v) for k, v in results.items() if 'roc_mean' in v]
        all_methods.sort(key=lambda x: x[1]['roc_mean'], reverse=True)
        for mname, r in all_methods[:10]:
            gap = r['roc_mean'] - cb_mean
            marker = " ***" if gap >= 0 else ""
            print(f"    {mname:12s} ROC={r['roc_mean']:.4f}  gap={gap:+.4f}{marker}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for cfg_label, results in all_results.items():
        serializable[cfg_label] = {}
        for k, v in results.items():
            serializable[cfg_label][k] = {
                kk: vv for kk, vv in v.items()
                if isinstance(vv, (int, float, str))
            }
    (RESULTS_DIR / "maximize_performance.json").write_text(
        json.dumps(serializable, indent=2))

    print("\n" + "=" * 100)
    print("DONE")


if __name__ == "__main__":
    main()
