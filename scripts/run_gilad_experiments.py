#!/usr/bin/env python3 -u
"""
Experiments addressing Gilad's review comments.

1. Semantic stream decomposition ablation:
   - structural-only (already exists in final_results.json, re-verified here)
   - structural + identifier only (NEW)
   - structural + LSA only (NEW)
   - full (already exists in scafe_sem_results.json, re-verified here)

2. TF-IDF + Logistic Regression baseline over raw source tokens.

3. Identifier obfuscation robustness experiment.

4. Slither head-to-head on the compilable subset.
"""
import sys
import json
import time
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier,
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
from src.ml.semantic_features import (
    read_source, compute_identifier_features, tokenize_solidity, CodeLSA,
    _extract_identifiers, _SOL_KEYWORDS,
)

SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"
SLITHER_DIR = _REPO / "data" / "derived" / "slither_results"

N_LSA_COMPONENTS = 50


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
    X_struct = data[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X_struct = np.nan_to_num(X_struct, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  {dataset}: {len(y)} contracts (pos={y.sum()}, neg={(y==0).sum()}), "
          f"{len(feat_cols)} structural features", flush=True)
    return X_struct, y, groups, feat_cols, contract_ids, source_paths


def extract_semantic_features(source_paths, label=""):
    n = len(source_paths)
    print(f"  Extracting semantic features for {n} contracts...", flush=True)
    ident_rows = []
    token_docs = []
    t0 = time.time()
    for i, sp in enumerate(source_paths):
        src = read_source(sp)
        ident_rows.append(compute_identifier_features(src))
        token_docs.append(tokenize_solidity(src))
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)...", flush=True)
    elapsed = time.time() - t0
    ident_df = pd.DataFrame(ident_rows).fillna(0.0)
    ident_cols = sorted(ident_df.columns.tolist())
    X_ident = ident_df[ident_cols].values.astype(np.float64)
    print(f"  Done: {len(ident_cols)} identifier features, {elapsed:.0f}s total", flush=True)
    return X_ident, ident_cols, token_docs


def precompute_ranking(X, y, feat_names):
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mask = mi > 0.001
    X_mi = X[:, mask]
    mi_indices = [i for i, m in enumerate(mask) if m]
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1)
    rf.fit(X_mi, y)
    imp = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    ranked_orig_idx = [mi_indices[ranked[i]] for i in range(len(ranked))]
    return ranked_orig_idx


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def get_models(pos_weight):
    return {
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_leaf=15, subsample=0.8, random_state=42),
        "Vote(RF+GBC+XGB)": lambda: VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=3,
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
            voting="soft"),
    }


def run_ablation_cv(X_struct, X_ident, token_docs, y, groups,
                    struct_feat_names, k_struct=80, n_lsa=50, n_folds=5,
                    streams="full"):
    """Run CV with configurable stream combinations.

    streams: "struct_only" | "struct_ident" | "struct_lsa" | "full"
    """
    ranked_struct = precompute_ranking(X_struct, y, struct_feat_names)
    struct_selected = sorted(ranked_struct[:k_struct])

    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_models(pos_weight)
    results = {}

    for model_name, model_fn in models.items():
        t0 = time.time()
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_struct, y, groups)):
            Xtr_s = X_struct[train_idx][:, struct_selected]
            Xte_s = X_struct[test_idx][:, struct_selected]

            parts_tr = [Xtr_s]
            parts_te = [Xte_s]

            if streams in ("struct_ident", "full"):
                parts_tr.append(X_ident[train_idx])
                parts_te.append(X_ident[test_idx])

            if streams in ("struct_lsa", "full"):
                lsa = CodeLSA(n_components=n_lsa)
                train_docs = [token_docs[i] for i in train_idx]
                test_docs = [token_docs[i] for i in test_idx]
                parts_tr.append(lsa.fit_transform(train_docs))
                parts_te.append(lsa.transform(test_docs))

            Xtr = np.hstack(parts_tr)
            Xte = np.hstack(parts_te)
            ytr, yte = y[train_idx], y[test_idx]

            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            m = model_fn()
            m.fit(Xtr, ytr)
            prob_te = m.predict_proba(Xte)[:, 1]

            roc = roc_auc_score(yte, prob_te)
            pr = average_precision_score(yte, prob_te)
            _, opt_f1 = optimal_f1_threshold(yte, prob_te)
            fold_metrics.append({"roc": roc, "pr": pr, "f1": opt_f1})

        elapsed = time.time() - t0
        rocs = [f["roc"] for f in fold_metrics]
        prs = [f["pr"] for f in fold_metrics]
        f1s = [f["f1"] for f in fold_metrics]

        results[model_name] = {
            "roc_auc_mean": round(np.mean(rocs), 4),
            "roc_auc_std": round(np.std(rocs), 4),
            "pr_auc_mean": round(np.mean(prs), 4),
            "pr_auc_std": round(np.std(prs), 4),
            "f1_mean": round(np.mean(f1s), 4),
            "f1_std": round(np.std(f1s), 4),
            "time_s": round(elapsed, 1),
        }
        print(f"    {streams:15s} {model_name:25s} "
              f"ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
              f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  ({elapsed:.1f}s)",
              flush=True)

    return results


# ── Experiment 2: TF-IDF + LR Baseline ──────────────────────────────────────

def run_tfidf_lr_baseline(source_paths, y, groups, n_folds=5):
    """TF-IDF over raw source tokens + Logistic Regression."""
    print("\n  Tokenizing all contracts for TF-IDF...", flush=True)
    t0 = time.time()
    docs = []
    for sp in source_paths:
        src = read_source(sp)
        docs.append(tokenize_solidity(src))
    print(f"  Tokenized {len(docs)} contracts in {time.time()-t0:.0f}s", flush=True)

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(y)), y, groups)):
        train_docs = [docs[i] for i in train_idx]
        test_docs = [docs[i] for i in test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        tfidf = TfidfVectorizer(
            max_features=5000, min_df=5, max_df=0.95,
            sublinear_tf=True, norm='l2')
        Xtr = tfidf.fit_transform(train_docs)
        Xte = tfidf.transform(test_docs)

        lr = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(Xtr, ytr)
        prob_te = lr.predict_proba(Xte)[:, 1]

        roc = roc_auc_score(yte, prob_te)
        pr = average_precision_score(yte, prob_te)
        _, opt_f1 = optimal_f1_threshold(yte, prob_te)
        fold_metrics.append({"roc": roc, "pr": pr, "f1": opt_f1})
        print(f"    Fold {fold_idx}: ROC={roc:.4f} PR={pr:.4f} F1={opt_f1:.4f}", flush=True)

    rocs = [f["roc"] for f in fold_metrics]
    prs = [f["pr"] for f in fold_metrics]
    f1s = [f["f1"] for f in fold_metrics]

    result = {
        "roc_auc_mean": round(np.mean(rocs), 4),
        "roc_auc_std": round(np.std(rocs), 4),
        "pr_auc_mean": round(np.mean(prs), 4),
        "pr_auc_std": round(np.std(prs), 4),
        "f1_mean": round(np.mean(f1s), 4),
        "f1_std": round(np.std(f1s), 4),
    }
    print(f"  TF-IDF+LR: ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
          f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}", flush=True)
    return result


# ── Experiment 3: Identifier Obfuscation ─────────────────────────────────────

def obfuscate_identifiers(source: str) -> str:
    """Replace all user-defined identifiers with generic names."""
    idents = set(_extract_identifiers(source))
    if not idents:
        return source

    sorted_idents = sorted(idents, key=len, reverse=True)
    result = source
    for i, ident in enumerate(sorted_idents):
        pattern = re.compile(r'\b' + re.escape(ident) + r'\b')
        result = pattern.sub(f'var_{i}', result)
    return result


def run_obfuscation_experiment(X_struct, source_paths, y, groups,
                               struct_feat_names, k_struct=80, n_folds=5):
    """Run full SCAFE-Sem pipeline with obfuscated identifiers on test fold."""
    print("\n  Running identifier obfuscation experiment...", flush=True)

    ranked_struct = precompute_ranking(X_struct, y, struct_feat_names)
    struct_selected = sorted(ranked_struct[:k_struct])

    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    fold_metrics_normal = []
    fold_metrics_obfusc = []
    fold_metrics_struct_only_obfusc = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_struct, y, groups)):
        print(f"    Fold {fold_idx}...", flush=True)
        Xtr_s = X_struct[train_idx][:, struct_selected]
        Xte_s = X_struct[test_idx][:, struct_selected]
        ytr, yte = y[train_idx], y[test_idx]

        # Read sources
        train_sources = [read_source(source_paths[i]) for i in train_idx]
        test_sources = [read_source(source_paths[i]) for i in test_idx]

        # Normal identifier features (training)
        train_ident = pd.DataFrame([compute_identifier_features(s) for s in train_sources]).fillna(0.0)
        ident_cols = sorted(train_ident.columns.tolist())
        Xtr_i = train_ident[ident_cols].values.astype(np.float64)

        # Normal identifier features (test)
        test_ident = pd.DataFrame([compute_identifier_features(s) for s in test_sources]).fillna(0.0)
        for c in ident_cols:
            if c not in test_ident.columns:
                test_ident[c] = 0.0
        Xte_i_normal = test_ident[ident_cols].values.astype(np.float64)

        # Obfuscated identifier features (test only)
        test_obfusc = [obfuscate_identifiers(s) for s in test_sources]
        test_ident_obf = pd.DataFrame([compute_identifier_features(s) for s in test_obfusc]).fillna(0.0)
        for c in ident_cols:
            if c not in test_ident_obf.columns:
                test_ident_obf[c] = 0.0
        Xte_i_obfusc = test_ident_obf[ident_cols].values.astype(np.float64)

        # LSA (normal train, normal test, obfuscated test)
        lsa = CodeLSA(n_components=N_LSA_COMPONENTS)
        train_docs = [tokenize_solidity(s) for s in train_sources]
        test_docs_normal = [tokenize_solidity(s) for s in test_sources]
        test_docs_obfusc = [tokenize_solidity(s) for s in test_obfusc]

        Xtr_l = lsa.fit_transform(train_docs)
        Xte_l_normal = lsa.transform(test_docs_normal)
        Xte_l_obfusc = lsa.transform(test_docs_obfusc)

        # Build combined features
        Xtr_full = np.hstack([Xtr_s, Xtr_i, Xtr_l])
        Xte_normal = np.hstack([Xte_s, Xte_i_normal, Xte_l_normal])
        Xte_obfusc = np.hstack([Xte_s, Xte_i_obfusc, Xte_l_obfusc])

        sc = StandardScaler()
        Xtr_full = sc.fit_transform(Xtr_full)
        Xte_normal = sc.transform(Xte_normal)
        Xte_obfusc = sc.transform(Xte_obfusc)

        # Train model (GBC for consistency)
        m = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_leaf=15, subsample=0.8, random_state=42)
        m.fit(Xtr_full, ytr)

        # Evaluate normal
        prob_normal = m.predict_proba(Xte_normal)[:, 1]
        roc_n = roc_auc_score(yte, prob_normal)
        pr_n = average_precision_score(yte, prob_normal)
        _, f1_n = optimal_f1_threshold(yte, prob_normal)
        fold_metrics_normal.append({"roc": roc_n, "pr": pr_n, "f1": f1_n})

        # Evaluate obfuscated
        prob_obfusc = m.predict_proba(Xte_obfusc)[:, 1]
        roc_o = roc_auc_score(yte, prob_obfusc)
        pr_o = average_precision_score(yte, prob_obfusc)
        _, f1_o = optimal_f1_threshold(yte, prob_obfusc)
        fold_metrics_obfusc.append({"roc": roc_o, "pr": pr_o, "f1": f1_o})

        # Also test: structural-only under obfuscation (as a control)
        sc2 = StandardScaler()
        Xtr_struct = sc2.fit_transform(Xtr_s)
        Xte_struct = sc2.transform(Xte_s)
        m2 = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_leaf=15, subsample=0.8, random_state=42)
        m2.fit(Xtr_struct, ytr)
        prob_struct = m2.predict_proba(Xte_struct)[:, 1]
        roc_s = roc_auc_score(yte, prob_struct)
        pr_s = average_precision_score(yte, prob_struct)
        _, f1_s = optimal_f1_threshold(yte, prob_struct)
        fold_metrics_struct_only_obfusc.append({"roc": roc_s, "pr": pr_s, "f1": f1_s})

        print(f"      Normal: ROC={roc_n:.4f}  Obfusc: ROC={roc_o:.4f}  "
              f"Struct-only: ROC={roc_s:.4f}", flush=True)

    def summarize(metrics):
        return {
            "roc_auc_mean": round(np.mean([m["roc"] for m in metrics]), 4),
            "roc_auc_std": round(np.std([m["roc"] for m in metrics]), 4),
            "pr_auc_mean": round(np.mean([m["pr"] for m in metrics]), 4),
            "f1_mean": round(np.mean([m["f1"] for m in metrics]), 4),
        }

    result = {
        "normal": summarize(fold_metrics_normal),
        "obfuscated": summarize(fold_metrics_obfusc),
        "structural_only": summarize(fold_metrics_struct_only_obfusc),
    }
    print(f"\n  OBFUSCATION RESULTS (GradientBoosting):")
    print(f"    Normal (full):     ROC={result['normal']['roc_auc_mean']:.4f}")
    print(f"    Obfuscated (test): ROC={result['obfuscated']['roc_auc_mean']:.4f}")
    print(f"    Struct-only:       ROC={result['structural_only']['roc_auc_mean']:.4f}")
    print(f"    Drop from obfusc:  {result['normal']['roc_auc_mean'] - result['obfuscated']['roc_auc_mean']:.4f}")
    return result


# ── Experiment 4: Slither Head-to-Head ───────────────────────────────────────

def run_slither_headtohead():
    """Compare SCAFE-Sem vs Slither on the Slither-compilable subset."""
    print("\n  Running Slither head-to-head analysis...", flush=True)

    slither_csv = SLITHER_DIR / "slither_vs_ground_truth.csv"
    if not slither_csv.exists():
        print("  Slither results not found, skipping.", flush=True)
        return None

    slither_df = pd.read_csv(slither_csv)
    slither_success = slither_df[slither_df["slither_success"] == True].copy()
    print(f"  Slither compilable: {len(slither_success)} contracts", flush=True)

    results = {}
    for oof_name, label in [
        ("sem_Vote_RF_GBC_XGB_merged.csv", "SCAFE-Sem Vote (merged)"),
        ("sem_GradientBoosting_merged.csv", "SCAFE-Sem GBC (merged)"),
        ("ml_Vote_RF_GBC_XGB_merged.csv", "SCAFE Vote (merged)"),
    ]:
        oof_path = OOF_DIR / oof_name
        if not oof_path.exists():
            continue

        oof_df = pd.read_csv(oof_path)
        merged = slither_success.merge(oof_df, on="contract_id", how="inner",
                                        suffixes=("_slither", "_oof"))

        if len(merged) < 50:
            print(f"  {label}: only {len(merged)} matched, skipping.", flush=True)
            continue

        y_true = merged["y_true"].values if "y_true" in merged.columns else \
                 (merged["label_status_slither"] == "vulnerable").astype(int).values
        y_prob = merged["y_prob"].values

        slither_pred = merged["slither_label_medium"].values if "slither_label_medium" in merged.columns else None

        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        _, f1_val = optimal_f1_threshold(y_true, y_prob)

        result_entry = {
            "n_contracts": len(merged),
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "f1": round(f1_val, 4),
        }

        if slither_pred is not None:
            slither_f1 = f1_score(y_true, slither_pred, zero_division=0)
            slither_prec = precision_score(y_true, slither_pred, zero_division=0)
            slither_rec = recall_score(y_true, slither_pred, zero_division=0)
            result_entry["slither_f1"] = round(slither_f1, 4)
            result_entry["slither_precision"] = round(slither_prec, 4)
            result_entry["slither_recall"] = round(slither_rec, 4)

        results[label] = result_entry
        print(f"  {label}: {len(merged)} contracts, ROC={roc:.4f}, PR={pr:.4f}, F1={f1_val:.4f}", flush=True)
        if slither_pred is not None:
            print(f"    Slither on same subset: F1={slither_f1:.4f}, P={slither_prec:.4f}, R={slither_rec:.4f}", flush=True)

    return results


def main():
    print("=" * 100, flush=True)
    print("EXPERIMENTS FOR GILAD'S REVIEW COMMENTS", flush=True)
    print("=" * 100, flush=True)

    all_results = {}

    # Load data
    print("\n--- Loading merged data ---", flush=True)
    X_struct, y, groups, sfeat, cids, spaths = load_data("merged")

    print("\n--- Extracting semantic features ---", flush=True)
    X_ident, ident_cols, token_docs = extract_semantic_features(spaths, "merged")

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 1: Semantic Stream Decomposition Ablation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: Semantic Stream Decomposition Ablation (merged)")
    print("=" * 100)

    ablation_results = {}
    for streams in ["struct_only", "struct_ident", "struct_lsa", "full"]:
        print(f"\n  --- {streams} ---", flush=True)
        ablation_results[streams] = run_ablation_cv(
            X_struct, X_ident, token_docs, y, groups, sfeat, streams=streams)

    all_results["semantic_ablation"] = ablation_results

    print("\n  ABLATION SUMMARY (Vote ensemble, merged):")
    print(f"  {'Configuration':20s} {'ROC-AUC':>12s} {'PR-AUC':>12s} {'F1':>12s}")
    print(f"  {'-'*56}")
    for streams in ["struct_only", "struct_ident", "struct_lsa", "full"]:
        r = ablation_results[streams].get("Vote(RF+GBC+XGB)",
            ablation_results[streams].get("GradientBoosting", {}))
        print(f"  {streams:20s} {r.get('roc_auc_mean',0):>8.4f}±{r.get('roc_auc_std',0):.4f} "
              f"{r.get('pr_auc_mean',0):>8.4f}±{r.get('pr_auc_std',0):.4f} "
              f"{r.get('f1_mean',0):>8.4f}±{r.get('f1_std',0):.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 2: TF-IDF + LR Baseline
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: TF-IDF + Logistic Regression Baseline (merged)")
    print("=" * 100)

    tfidf_lr_result = run_tfidf_lr_baseline(spaths, y, groups)
    all_results["tfidf_lr_baseline"] = tfidf_lr_result

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 3: Identifier Obfuscation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXPERIMENT 3: Identifier Obfuscation Robustness (merged)")
    print("=" * 100)

    obfusc_result = run_obfuscation_experiment(
        X_struct, spaths, y, groups, sfeat)
    all_results["obfuscation"] = obfusc_result

    # ═══════════════════════════════════════════════════════════════════════
    # Experiment 4: Slither Head-to-Head
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXPERIMENT 4: Slither Head-to-Head on Compilable Subset")
    print("=" * 100)

    slither_result = run_slither_headtohead()
    if slither_result:
        all_results["slither_headtohead"] = slither_result

    # ═══════════════════════════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gilad_experiments.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nAll results saved to {out_path}", flush=True)

    print("\n" + "=" * 100)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
