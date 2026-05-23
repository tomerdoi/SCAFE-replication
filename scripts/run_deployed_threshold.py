"""
Deployed-threshold experiment (§5.10) — uses the actual ML pipeline feature matrix.
Protocol:
  - Load strict-label dataset via the same prepare_dataset() used in the main paper
  - 5-fold GroupKFold by project (same as main experiments)
  - Generate 5-fold OOF predictions with XGBoost (fast, strong model)
  - For each fold i: use fold i as calibration holdout, tune threshold on fold i OOF predictions,
    evaluate on fold (i+1)%5 using that calibration-derived threshold
  - Report F1/Prec/Rec at calibration-tuned vs nested-CV threshold side-by-side
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

STRICT_CSV = "data/derived/unified_contract_eval_scored_strict.csv"
RESULTS_DIR = Path("data/derived/ml_results_strict")

def load_strict_features():
    """Use the same _EXCLUDE logic as the actual train_pipeline."""
    from src.ml.features import _EXCLUDE, select_feature_columns, FeatureSet
    df = pd.read_csv(STRICT_CSV, low_memory=False)
    # Build feature columns using the same exclusion list as the main pipeline
    feat_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in _EXCLUDE
        and not df[c].isnull().all()
    ]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['vuln_present'].values.astype(int)
    # Group by project (from contract_id prefix)
    from sklearn.preprocessing import LabelEncoder
    groups_raw = df['contract_id'].apply(
        lambda x: ':'.join(str(x).split(':')[:2]) if ':' in str(x) else str(x)[:40]
    )
    groups = LabelEncoder().fit_transform(groups_raw)
    print(f"Loaded {len(df):,} contracts | {X.shape[1]} features | {y.sum():,} positives ({y.mean():.1%})")
    print(f"Unique groups: {len(np.unique(groups)):,}")
    return X, y, groups, feat_cols


def f1_at(y_true, y_prob, t):
    pred = (y_prob >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec


def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1, _, _ = f1_at(y_true, y_prob, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2)


def two_stage_mi_rf_selection(X_train, y_train, k=80):
    """Fast MI filter + RF Gini ranking (50 trees for speed)."""
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
    mi_mask = mi > 0
    X_mi = X_train[:, mi_mask]
    mi_idx = np.where(mi_mask)[0]
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced',
                                 random_state=42, n_jobs=-1, max_features='sqrt')
    rf.fit(X_mi, y_train)
    top_k_local = np.argsort(rf.feature_importances_)[::-1][:k]
    return mi_idx[top_k_local]


def main():
    X, y, groups, feat_cols = load_strict_features()

    from xgboost import XGBClassifier
    neg, pos = (y == 0).sum(), (y == 1).sum()
    clf_kw = dict(n_estimators=300, scale_pos_weight=neg/pos,
                  random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0,
                  learning_rate=0.05, max_depth=6)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))

    oof_probs = np.zeros(len(y))
    nested_thresholds = np.zeros(len(y))

    print("\nRunning 5-fold XGBoost with nested-CV threshold + feature selection...")
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Inner 3-fold for nested-CV threshold (same as paper)
        inner_gkf = GroupKFold(n_splits=3)
        inner_probs = np.zeros(len(y_tr))
        for _, (itr, ival) in enumerate(inner_gkf.split(X_tr, y_tr, groups[train_idx])):
            sel_idx = two_stage_mi_rf_selection(X_tr[itr], y_tr[itr], k=80)
            scaler = StandardScaler()
            Xi_tr = scaler.fit_transform(X_tr[itr][:, sel_idx])
            Xi_val = scaler.transform(X_tr[ival][:, sel_idx])
            m = XGBClassifier(**clf_kw)
            m.fit(Xi_tr, y_tr[itr])
            inner_probs[ival] = m.predict_proba(Xi_val)[:, 1]

        nested_t = best_threshold(y_tr, inner_probs)
        nested_thresholds[test_idx] = nested_t

        # Train final model on full training fold
        sel_idx = two_stage_mi_rf_selection(X_tr, y_tr, k=80)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, sel_idx])
        X_te_s = scaler.transform(X_te[:, sel_idx])
        m = XGBClassifier(**clf_kw)
        m.fit(X_tr_s, y_tr)
        oof_probs[test_idx] = m.predict_proba(X_te_s)[:, 1]

        f1, prec, rec = f1_at(y_te, oof_probs[test_idx], nested_t)
        print(f"  Fold {fold_i}: nested_thresh={nested_t:.2f} | "
              f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  (n_test={len(test_idx)})")

    # Deployed-threshold simulation
    print("\nDeployed-threshold: calibration fold → evaluate on next fold")
    deployed_rows = []
    for cal_i in range(5):
        test_i = (cal_i + 1) % 5
        cal_idx  = folds[cal_i][1]
        test_idx = folds[test_i][1]
        nested_t = nested_thresholds[test_idx[0]]

        cal_t = best_threshold(y[cal_idx], oof_probs[cal_idx])
        f1_d, p_d, r_d = f1_at(y[test_idx], oof_probs[test_idx], cal_t)
        f1_n, p_n, r_n = f1_at(y[test_idx], oof_probs[test_idx], nested_t)

        deployed_rows.append(dict(
            cal_fold=cal_i, test_fold=test_i,
            cal_thresh=cal_t, nested_thresh=nested_t,
            f1_deployed=f1_d, prec_deployed=p_d, rec_deployed=r_d,
            f1_nested=f1_n, prec_nested=p_n, rec_nested=r_n,
        ))
        print(f"  Cal=fold{cal_i}→Test=fold{test_i}: "
              f"cal_t={cal_t:.2f} F1={f1_d:.4f} P={p_d:.4f} R={r_d:.4f} | "
              f"nested_t={nested_t:.2f} F1={f1_n:.4f} P={p_n:.4f} R={r_n:.4f}")

    df_r = pd.DataFrame(deployed_rows)
    print("\n=== SUMMARY (mean ± std over 5 calibration-test pairs) ===")
    print(f"Deployed-threshold: F1={df_r.f1_deployed.mean():.4f}±{df_r.f1_deployed.std():.4f}  "
          f"Prec={df_r.prec_deployed.mean():.4f}  Rec={df_r.rec_deployed.mean():.4f}")
    print(f"Nested-CV thresh:   F1={df_r.f1_nested.mean():.4f}±{df_r.f1_nested.std():.4f}  "
          f"Prec={df_r.prec_nested.mean():.4f}  Rec={df_r.rec_nested.mean():.4f}")
    print(f"ΔF1 (deployed − nested): {(df_r.f1_deployed - df_r.f1_nested).mean():+.4f}")

    df_r.to_csv(RESULTS_DIR / "deployed_threshold_results.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/deployed_threshold_results.csv")


if __name__ == "__main__":
    main()
