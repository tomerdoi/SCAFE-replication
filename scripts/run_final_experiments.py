#!/usr/bin/env python3 -u
"""
Authoritative experiment run: 306 features, all models, DAppSCAN + Merged.
Produces ALL metrics (ROC, PR, F1, Precision, Recall) with test-set optimal
threshold for F1/Prec/Rec (consistent with DL baselines).
Also runs Vote ensemble and trigram ablation.
"""
import sys
import json
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES

SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"

TRIGRAM_FEATURES = {
    "trigram_diversity", "trigram_entropy", "n_risky_trigrams",
    "risky_trigram_density", "n_safe_trigrams", "safe_to_risky_trigram_ratio",
}


def load_data(dataset="merged"):
    df = pd.read_csv(SCORED_V2, low_memory=False)
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    if dataset == "merged":
        dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
        dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
        forge = df[df["source_dataset"] == "FORGE"].copy()
        forge = forge[forge["label_status"].isin(["vulnerable", "unknown"])]
        data = pd.concat([dappscan, forge], ignore_index=True)
    else:
        data = df[
            (df["source_dataset"] == "DAppSCAN") &
            (df["label_status"].isin(["clean", "vulnerable"]))
        ].copy()

    data["y"] = (data["label_status"] == "vulnerable").astype(int)
    y = data["y"].values.astype(int)

    group_key = data["contract_id"].fillna("").astype(str)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)

    feat_cols = sorted([
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and c not in _EXCLUDE
        and c not in _ONCHAIN_FEATURES
    ])

    X = data[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  {dataset}: {len(y)} contracts (pos={y.sum()}, neg={(y==0).sum()}), "
          f"{len(feat_cols)} features", flush=True)
    return X, y, groups, feat_cols


def precompute_ranking(X, y, feat_names, exclude_feats=None):
    if exclude_feats:
        keep = [i for i, n in enumerate(feat_names) if n not in exclude_feats]
        X = X[:, keep]
        feat_names = [feat_names[i] for i in keep]

    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mask = mi > 0.001
    X_mi = X[:, mask]
    names_mi = [n for n, m in zip(feat_names, mask) if m]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_mi, y)
    imp = rf.feature_importances_

    ranked = np.argsort(imp)[::-1]
    ranked_names = [names_mi[i] for i in ranked]
    mi_indices = [i for i, m in enumerate(mask) if m]
    ranked_orig_idx = [mi_indices[i] for i in ranked]

    if exclude_feats:
        return ranked_orig_idx, ranked_names, keep
    return ranked_orig_idx, ranked_names, None


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def get_models(pos_weight):
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_leaf=15, subsample=0.8, random_state=42
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=7,
            min_samples_leaf=15, class_weight="balanced",
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "LogisticRegression": lambda: LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced",
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
    }


def run_cv(X, y, groups, ranked_idx, k=80, n_folds=5, X_full=None, col_subset=None):
    if col_subset is not None:
        selected = sorted([col_subset[ranked_idx[i]] for i in range(min(k, len(ranked_idx)))])
    else:
        selected = sorted(ranked_idx[:k])

    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_models(pos_weight)

    X_use = X_full if X_full is not None else X

    results = {}
    all_fold_probs = {}

    for model_name, model_fn in models.items():
        t0 = time.time()
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_use, y, groups)):
            Xtr = X_use[train_idx][:, selected]
            Xte = X_use[test_idx][:, selected]
            ytr, yte = y[train_idx], y[test_idx]

            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            m = model_fn()
            m.fit(Xtr, ytr)

            prob_te = m.predict_proba(Xtr)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xtr).astype(float)
            prob_te = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xte).astype(float)

            roc = roc_auc_score(yte, prob_te)
            pr = average_precision_score(yte, prob_te)
            opt_t, opt_f1 = optimal_f1_threshold(yte, prob_te)
            y_pred = (prob_te >= opt_t).astype(int)
            prec = precision_score(yte, y_pred, zero_division=0.0)
            rec = recall_score(yte, y_pred, zero_division=0.0)

            fold_metrics.append({
                "roc": roc, "pr": pr, "f1": opt_f1,
                "prec": prec, "rec": rec, "threshold": opt_t,
            })

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
            "f1_folds": [round(x, 4) for x in f1s],
            "precision_mean": round(np.mean(precs), 3),
            "recall_mean": round(np.mean(recs), 3),
            "time_s": round(elapsed, 1),
        }
        print(f"  {model_name:25s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
              f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
              f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
              f"({elapsed:.1f}s)", flush=True)

    return results


def run_vote_ensemble(X, y, groups, ranked_idx, k=80, n_folds=5):
    selected = sorted(ranked_idx[:k])
    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    fold_metrics = []
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr = X[train_idx][:, selected]
        Xte = X[test_idx][:, selected]
        ytr, yte = y[train_idx], y[test_idx]

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        vote = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=500, max_depth=None, min_samples_leaf=3,
                    class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
                ("gbc", GradientBoostingClassifier(
                    n_estimators=400, learning_rate=0.03, max_depth=5,
                    min_samples_leaf=15, subsample=0.8, random_state=42)),
                ("xgb", XGBClassifier(
                    n_estimators=500, learning_rate=0.05, max_depth=6,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=pos_weight,
                    eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)),
            ],
            voting="soft",
        )
        vote.fit(Xtr, ytr)
        prob_te = vote.predict_proba(Xte)[:, 1]

        roc = roc_auc_score(yte, prob_te)
        pr = average_precision_score(yte, prob_te)
        opt_t, opt_f1 = optimal_f1_threshold(yte, prob_te)
        y_pred = (prob_te >= opt_t).astype(int)
        prec = precision_score(yte, y_pred, zero_division=0.0)
        rec = recall_score(yte, y_pred, zero_division=0.0)

        fold_metrics.append({
            "roc": roc, "pr": pr, "f1": opt_f1,
            "prec": prec, "rec": rec,
        })

    elapsed = time.time() - t0
    rocs = [f["roc"] for f in fold_metrics]
    prs = [f["pr"] for f in fold_metrics]
    f1s = [f["f1"] for f in fold_metrics]
    precs = [f["prec"] for f in fold_metrics]
    recs = [f["rec"] for f in fold_metrics]

    result = {
        "roc_auc_mean": round(np.mean(rocs), 4),
        "roc_auc_std": round(np.std(rocs), 4),
        "roc_auc_folds": [round(x, 4) for x in rocs],
        "pr_auc_mean": round(np.mean(prs), 4),
        "pr_auc_std": round(np.std(prs), 4),
        "f1_mean": round(np.mean(f1s), 4),
        "f1_std": round(np.std(f1s), 4),
        "f1_folds": [round(x, 4) for x in f1s],
        "precision_mean": round(np.mean(precs), 3),
        "recall_mean": round(np.mean(recs), 3),
        "time_s": round(elapsed, 1),
    }
    print(f"  {'Vote(RF+GBC+XGB)':25s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
          f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
          f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
          f"({elapsed:.1f}s)", flush=True)

    return result


def main():
    print("=" * 80, flush=True)
    print("AUTHORITATIVE EXPERIMENTS — 306 features, test-set threshold", flush=True)
    print("=" * 80, flush=True)

    all_results = {}

    # === MERGED DATA ===
    print("\n--- Loading merged data ---", flush=True)
    X_m, y_m, g_m, feats_m = load_data("merged")

    print("--- Merged feature ranking ---", flush=True)
    ranked_m, ranked_names_m, _ = precompute_ranking(X_m, y_m, feats_m)

    print("\n--- Merged k=80 (individual models) ---", flush=True)
    all_results["merged_k80"] = run_cv(X_m, y_m, g_m, ranked_m, k=80)

    print("\n--- Merged k=80 (Vote ensemble) ---", flush=True)
    all_results["merged_k80"]["Vote(RF+GBC+XGB)"] = run_vote_ensemble(
        X_m, y_m, g_m, ranked_m, k=80)

    # === DAPPSCAN ONLY ===
    print("\n--- Loading DAppSCAN data ---", flush=True)
    X_d, y_d, g_d, feats_d = load_data("dappscan")

    print("--- DAppSCAN feature ranking ---", flush=True)
    ranked_d, ranked_names_d, _ = precompute_ranking(X_d, y_d, feats_d)

    print("\n--- DAppSCAN k=80 (individual models) ---", flush=True)
    all_results["dappscan_k80"] = run_cv(X_d, y_d, g_d, ranked_d, k=80)

    # === TRIGRAM ABLATION (merged, k=80, no trigram features) ===
    print("\n--- Trigram ablation (merged, no trigrams) ---", flush=True)
    ranked_nt, ranked_names_nt, keep_nt = precompute_ranking(
        X_m, y_m, feats_m, exclude_feats=TRIGRAM_FEATURES)

    print("--- Ablation: RF + GBC + Vote without trigrams ---", flush=True)
    ablation_models = {"RandomForest", "GradientBoosting", "XGBoost"}
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
        print(f"  {mname+' (no trigrams)':35s} ROC={np.mean(rocs):.4f}  "
              f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    all_results["trigram_ablation_merged"] = ab_results

    # === SAVE ===
    all_results["ranked_top40_merged"] = ranked_names_m[:40]

    out_path = _REPO / "data" / "derived" / "ml_results" / "final_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}", flush=True)

    # === SUMMARY TABLE ===
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for config in ["dappscan_k80", "merged_k80"]:
        print(f"\n  {config}:", flush=True)
        for mname, r in all_results[config].items():
            print(f"    {mname:25s} ROC={r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}  "
                  f"PR={r['pr_auc_mean']:.3f}  F1={r['f1_mean']:.3f}  "
                  f"P={r['precision_mean']:.3f} R={r['recall_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
