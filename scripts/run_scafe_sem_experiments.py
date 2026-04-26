#!/usr/bin/env python3 -u
"""
SCAFE-Sem experiment: dual-stream structure-semantic feature fusion.

Stream 1 (structural):  Original SCAFE features (306 static features)
Stream 2 (semantic):    Identifier profiling + Code Token LSA

Runs 5-fold GroupKFold CV on merged and DAppSCAN data, saves OOF
predictions, and compares against DL baselines.
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
)

SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"

N_LSA_COMPONENTS = 50


def load_data(dataset="merged"):
    """Load structural features, labels, groups, and source paths."""
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

    # Structural features
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
    """Pre-compute identifier features and tokenized docs for all contracts."""
    n = len(source_paths)
    print(f"  Extracting semantic features for {n} contracts{' ('+label+')' if label else ''}...",
          flush=True)

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
    """MI filter + RF importance ranking for feature selection."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mask = mi > 0.001
    X_mi = X[:, mask]
    mi_indices = [i for i, m in enumerate(mask) if m]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_mi, y)
    imp = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    ranked_orig_idx = [mi_indices[ranked[i]] for i in range(len(ranked))]
    ranked_names = [feat_names[idx] for idx in ranked_orig_idx]
    return ranked_orig_idx, ranked_names


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
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
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }


def _save_oof(sample_indices, fold_indices, y_true, y_prob, contract_ids,
              model_name, dataset):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sample_idx": sample_indices,
        "contract_id": contract_ids,
        "fold": fold_indices,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    fname = f"sem_{model_name}_{dataset}.csv"
    df.to_csv(OOF_DIR / fname, index=False)


def run_scafe_sem_cv(X_struct, X_ident, token_docs, y, groups,
                     struct_feat_names, ident_feat_names,
                     k_struct=80, n_lsa=N_LSA_COMPONENTS, n_folds=5,
                     contract_ids=None, dataset_name="merged"):
    """Full SCAFE-Sem cross-validation with per-fold LSA fitting."""

    print(f"\n  Running SCAFE-Sem CV (k_struct={k_struct}, n_lsa={n_lsa})...",
          flush=True)

    # Pre-compute structural feature ranking on full data for selection
    ranked_struct, ranked_struct_names = precompute_ranking(
        X_struct, y, struct_feat_names)
    struct_selected = sorted(ranked_struct[:k_struct])
    print(f"  Structural: {k_struct} selected from {X_struct.shape[1]}")

    gkf = GroupKFold(n_splits=n_folds)
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_models(pos_weight)

    results = {}

    # Individual models
    for model_name, model_fn in models.items():
        t0 = time.time()
        fold_metrics = []
        oof_sample_idx, oof_fold, oof_y_true, oof_y_prob = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_struct, y, groups)):
            # Structural features (pre-selected)
            Xtr_s = X_struct[train_idx][:, struct_selected]
            Xte_s = X_struct[test_idx][:, struct_selected]

            # Identifier features (all, static)
            Xtr_i = X_ident[train_idx]
            Xte_i = X_ident[test_idx]

            # LSA features (fit on training fold only)
            lsa = CodeLSA(n_components=n_lsa)
            train_docs = [token_docs[i] for i in train_idx]
            test_docs = [token_docs[i] for i in test_idx]
            Xtr_l = lsa.fit_transform(train_docs)
            Xte_l = lsa.transform(test_docs)

            # Concatenate all streams
            Xtr = np.hstack([Xtr_s, Xtr_i, Xtr_l])
            Xte = np.hstack([Xte_s, Xte_i, Xte_l])
            ytr, yte = y[train_idx], y[test_idx]

            # Scale
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            # Train & predict
            m = model_fn()
            m.fit(Xtr, ytr)
            prob_te = m.predict_proba(Xte)[:, 1]

            roc = roc_auc_score(yte, prob_te)
            pr = average_precision_score(yte, prob_te)
            opt_t, opt_f1 = optimal_f1_threshold(yte, prob_te)
            y_pred = (prob_te >= opt_t).astype(int)
            prec = precision_score(yte, y_pred, zero_division=0.0)
            rec = recall_score(yte, y_pred, zero_division=0.0)
            fold_metrics.append({"roc": roc, "pr": pr, "f1": opt_f1,
                                 "prec": prec, "rec": rec})

            oof_sample_idx.extend(test_idx.tolist())
            oof_fold.extend([fold_idx] * len(test_idx))
            oof_y_true.extend(yte.tolist())
            oof_y_prob.extend(prob_te.tolist())

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
        print(f"  {model_name:25s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
              f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
              f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
              f"({elapsed:.1f}s)", flush=True)

        if contract_ids is not None:
            oof_cids = [contract_ids[i] for i in oof_sample_idx]
            _save_oof(oof_sample_idx, oof_fold, oof_y_true, oof_y_prob,
                      oof_cids, model_name, dataset_name)

    # Vote ensemble
    print("\n  --- Vote(RF+GBC+XGB) ---", flush=True)
    t0 = time.time()
    fold_metrics = []
    oof_sample_idx, oof_fold, oof_y_true, oof_y_prob = [], [], [], []

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

        vote = VotingClassifier(
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
        fold_metrics.append({"roc": roc, "pr": pr, "f1": opt_f1,
                             "prec": prec, "rec": rec})

        oof_sample_idx.extend(test_idx.tolist())
        oof_fold.extend([fold_idx] * len(test_idx))
        oof_y_true.extend(yte.tolist())
        oof_y_prob.extend(prob_te.tolist())

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
    print(f"  {'Vote(RF+GBC+XGB)':25s} ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
          f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}  "
          f"P={np.mean(precs):.3f} R={np.mean(recs):.3f}  "
          f"({elapsed:.1f}s)", flush=True)

    if contract_ids is not None:
        oof_cids = [contract_ids[i] for i in oof_sample_idx]
        _save_oof(oof_sample_idx, oof_fold, oof_y_true, oof_y_prob,
                  oof_cids, "Vote_RF_GBC_XGB", dataset_name)

    return results


def load_dl_baselines(dataset_label):
    """Load DL baseline OOF predictions for comparison."""
    baselines = {}
    for name, fname in [('TextCNN', f'dl_textcnn_{dataset_label}.csv'),
                        ('BiLSTM', f'dl_bilstm_{dataset_label}.csv'),
                        ('CodeBERT', f'dl_codebert_{dataset_label}.csv')]:
        p = OOF_DIR / fname
        if p.exists():
            baselines[name] = pd.read_csv(p)['y_prob'].values
    return baselines


def compare_with_baselines(results, y, folds_file, dl_baselines, dataset_name):
    """Print comparison table."""
    print(f"\n{'='*100}")
    print(f"COMPARISON: SCAFE-Sem vs baselines ({dataset_name})")
    print(f"{'='*100}")

    # Load original SCAFE results for reference
    orig_vote_path = OOF_DIR / f"ml_Vote_RF_GBC_XGB_{dataset_name}.csv"
    if orig_vote_path.exists():
        orig_probs = pd.read_csv(orig_vote_path)['y_prob'].values
        roc = roc_auc_score(y, orig_probs)
        print(f"  {'SCAFE (original Vote)':30s} ROC-AUC={roc:.4f}")

    # SCAFE-Sem
    for mname, r in results.items():
        print(f"  {'SCAFE-Sem '+mname:30s} ROC-AUC={r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
              f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
              f"P={r['precision_mean']:.3f} R={r['recall_mean']:.3f}")

    # DL baselines
    if dl_baselines:
        folds_df = pd.read_csv(folds_file) if folds_file else None
        folds = folds_df['fold'].values if folds_df is not None else None

        print(f"\n  DL Baselines:")
        for name, probs in dl_baselines.items():
            if folds is not None:
                fold_rocs = []
                fold_f1s = []
                for fid in sorted(np.unique(folds)):
                    mask = folds == fid
                    fold_rocs.append(roc_auc_score(y[mask], probs[mask]))
                    _, f1 = optimal_f1_threshold(y[mask], probs[mask])
                    fold_f1s.append(f1)
                print(f"  {name:30s} ROC-AUC={np.mean(fold_rocs):.4f}±{np.std(fold_rocs):.4f}  "
                      f"F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}")
            else:
                roc = roc_auc_score(y, probs)
                print(f"  {name:30s} ROC-AUC={roc:.4f}")


def main():
    print("=" * 100, flush=True)
    print("SCAFE-Sem: Dual-Stream Structure-Semantic Feature Fusion", flush=True)
    print("=" * 100, flush=True)

    all_results = {}

    # === MERGED DATA ===
    print("\n--- Loading merged data ---", flush=True)
    X_struct_m, y_m, g_m, sfeat_m, cids_m, spaths_m = load_data("merged")

    print("\n--- Extracting semantic features (merged) ---", flush=True)
    X_ident_m, ident_cols_m, token_docs_m = extract_semantic_features(spaths_m, "merged")

    print("\n--- Running SCAFE-Sem CV (merged) ---", flush=True)
    results_m = run_scafe_sem_cv(
        X_struct_m, X_ident_m, token_docs_m, y_m, g_m,
        sfeat_m, ident_cols_m,
        contract_ids=cids_m, dataset_name="merged"
    )
    all_results["merged"] = results_m

    dl_m = load_dl_baselines("Merged")
    folds_file_m = OOF_DIR / "fold_assignments_merged.csv"
    compare_with_baselines(results_m, y_m, folds_file_m, dl_m, "merged")

    # === DAPPSCAN DATA ===
    print("\n\n--- Loading DAppSCAN data ---", flush=True)
    X_struct_d, y_d, g_d, sfeat_d, cids_d, spaths_d = load_data("dappscan")

    print("\n--- Extracting semantic features (DAppSCAN) ---", flush=True)
    X_ident_d, ident_cols_d, token_docs_d = extract_semantic_features(spaths_d, "dappscan")

    print("\n--- Running SCAFE-Sem CV (DAppSCAN) ---", flush=True)
    results_d = run_scafe_sem_cv(
        X_struct_d, X_ident_d, token_docs_d, y_d, g_d,
        sfeat_d, ident_cols_d,
        contract_ids=cids_d, dataset_name="dappscan"
    )
    all_results["dappscan"] = results_d

    dl_d = load_dl_baselines("DAppSCAN")
    folds_file_d = OOF_DIR / "fold_assignments_dappscan.csv"
    compare_with_baselines(results_d, y_d, folds_file_d, dl_d, "dappscan")

    # === SAVE RESULTS ===
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "scafe_sem_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)

    # === FINAL SUMMARY ===
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    for config in ["merged", "dappscan"]:
        print(f"\n  {config}:")
        for mname, r in all_results[config].items():
            print(f"    {mname:25s} ROC={r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}  "
                  f"PR={r['pr_auc_mean']:.4f}  F1={r['f1_mean']:.4f}  "
                  f"P={r['precision_mean']:.3f} R={r['recall_mean']:.3f}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
