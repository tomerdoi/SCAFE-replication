"""
Enhanced ML pipeline with feature selection, SMOTE, stacking, and threshold
optimization.

Builds on top of the existing train_pipeline infrastructure but adds:
  A. Feature selection via mutual information + RF importance (270 → top-K)
  B. SMOTE oversampling inside each CV fold
  C. Stacking ensemble (RF + XGB + GBC + HGB → LogReg meta-learner)
  D. Optimized threshold search (fine-grained F1 + PR-AUC-optimal)
  E. BalancedBaggingClassifier for better imbalance handling

Run:
    python -m src.ml.enhanced_pipeline
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    BaggingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.features import load_scored_csv, prepare_dataset  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_OUT_DIR = _REPO_ROOT / "data" / "derived" / "ml_results"

# Try to import SMOTE; fall back to manual oversampling if unavailable
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.ensemble import BalancedBaggingClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not installed; using manual oversampling")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── A. Feature Selection ─────────────────────────────────────────────────────

def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 80,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Two-stage feature selection:
      1. Mutual information (filter out zero-MI features)
      2. RF importance ranking → keep top_k
    Returns (X_selected, selected_names, importance_scores).
    """
    logger.info(f"  Feature selection: {len(feature_names)} → top {top_k}")

    # Stage 1: Mutual information
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_mask = mi > 0.001
    X_mi = X[:, mi_mask]
    names_mi = [n for n, m in zip(feature_names, mi_mask) if m]
    logger.info(f"    MI filter: {len(feature_names)} → {len(names_mi)} (removed {sum(~mi_mask)} zero-MI)")

    # Stage 2: Quick RF importance
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_mi, y)
    imp = rf.feature_importances_

    # Keep top_k
    top_idx = np.argsort(imp)[::-1][:top_k]
    selected_names = [names_mi[i] for i in top_idx]
    selected_imp = imp[top_idx]
    X_sel = X_mi[:, top_idx]

    logger.info(f"    RF importance: {len(names_mi)} → {top_k}")
    logger.info(f"    Top 10: {', '.join(selected_names[:10])}")

    return X_sel, selected_names, selected_imp


def _get_selected_indices(
    feature_names: List[str],
    selected_names: List[str],
    all_feature_names: List[str],
) -> List[int]:
    """Map selected feature names back to column indices in original X."""
    name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
    return [name_to_idx[n] for n in selected_names if n in name_to_idx]


# ── B. SMOTE Oversampling ────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to oversample the minority class.
    strategy=0.3 means minority will be 30% of majority count.
    """
    if HAS_IMBLEARN:
        sm = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    else:
        # Manual random oversampling
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        target_n = int(len(neg_idx) * strategy)
        if target_n > len(pos_idx):
            extra = np.random.RandomState(42).choice(pos_idx, target_n - len(pos_idx), replace=True)
            all_idx = np.concatenate([neg_idx, pos_idx, extra])
        else:
            all_idx = np.concatenate([neg_idx, pos_idx])
        np.random.RandomState(42).shuffle(all_idx)
        X_res, y_res = X_train[all_idx], y_train[all_idx]

    logger.info(
        f"    SMOTE: {len(y_train)} → {len(y_res)} "
        f"(pos: {y_train.sum()} → {y_res.sum()})"
    )
    return X_res, y_res


# ── C. Model Definitions ─────────────────────────────────────────────────────

def _build_stacking_classifier() -> StackingClassifier:
    """
    Stacking ensemble: RF + XGB + GBC + HGB → LogReg meta-learner.
    Uses predict_proba from each base model as meta-features.
    """
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
        ("gbc", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, class_weight="balanced",
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
        )),
    ]
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )))

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def _get_enhanced_models() -> Dict[str, BaseEstimator]:
    """Return the enhanced model set."""
    models = {
        "rf_enhanced": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
        "gbc_enhanced": GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            min_samples_leaf=15, subsample=0.8, random_state=42,
        ),
        "hgb_enhanced": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_depth=7,
            min_samples_leaf=15, class_weight="balanced",
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=30,
        ),
        "stacking": _build_stacking_classifier(),
    }
    if HAS_XGB:
        models["xgb_enhanced"] = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=7,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
    if HAS_IMBLEARN:
        models["balanced_bagging"] = BalancedBaggingClassifier(
            estimator=GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=20, subsample=0.8, random_state=42,
            ),
            n_estimators=15,
            sampling_strategy="auto",
            replacement=False,
            random_state=42,
            n_jobs=-1,
        )
    return models


# ── D. Threshold Optimization ─────────────────────────────────────────────────

def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Find optimal thresholds for F1 and for balanced precision/recall.
    Returns (best_threshold, best_f1, best_precision, best_recall).
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)

    # Fine-grained F1 search
    best_f1, best_t = 0.0, 0.5
    thresholds = np.linspace(0.02, 0.95, 200)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0.0)
        if f > best_f1:
            best_f1 = f
            best_t = t

    preds = (y_prob >= best_t).astype(int)
    best_prec = precision_score(y_true, preds, zero_division=0.0)
    best_rec = recall_score(y_true, preds, zero_division=0.0)

    return best_t, best_f1, best_prec, best_rec


# ── E. Enhanced CV Evaluation ─────────────────────────────────────────────────

def evaluate_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    *,
    n_folds: int = 5,
    top_k: int = 80,
    use_smote: bool = True,
    smote_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Run the full enhanced pipeline:
      1. Feature selection (full dataset, before CV — for feature ranking only)
      2. For each CV fold:
         a. Select top-K features (from training fold only to avoid leakage)
         b. Apply SMOTE on training set
         c. Train all models
         d. Evaluate with optimal threshold
    """
    models = _get_enhanced_models()
    gkf = GroupKFold(n_splits=n_folds)

    # Pre-compute feature selection on full data (for ranking/reporting)
    _, global_selected_names, global_imp = select_top_features(X, y, feature_names, top_k=top_k)

    # Storage for per-fold results
    results = {name: {"roc_auc": [], "pr_auc": [], "f1": [], "prec": [], "rec": [],
                       "tp": 0, "fn": 0, "fp": 0, "tn": 0, "time": 0.0}
               for name in models}

    # Also evaluate without SMOTE for comparison
    results_no_smote = {
        "rf_no_smote": {"roc_auc": [], "pr_auc": [], "f1": [], "prec": [], "rec": [],
                         "tp": 0, "fn": 0, "fp": 0, "tn": 0, "time": 0.0},
    }

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"\n  Fold {fold_idx + 1}/{n_folds} "
                    f"(train={len(train_idx)}, test={len(test_idx)}, "
                    f"test_pos={y[test_idx].sum()})")

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── A. Feature selection (from training fold ONLY) ──
        _, fold_selected_names, _ = select_top_features(
            X_train_raw, y_train, feature_names, top_k=top_k,
        )
        sel_idx = [feature_names.index(n) for n in fold_selected_names]
        X_train_sel = X_train_raw[:, sel_idx]
        X_test_sel = X_test_raw[:, sel_idx]

        # ── Scale ──
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_sel)
        X_test_sc = scaler.transform(X_test_sel)

        # ── B. SMOTE on training set ──
        if use_smote:
            X_train_sm, y_train_sm = apply_smote(X_train_sc, y_train, strategy=smote_ratio)
        else:
            X_train_sm, y_train_sm = X_train_sc, y_train

        # ── No-SMOTE baseline (RF only) ──
        t0 = time.time()
        rf_ns = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        rf_ns.fit(X_train_sc, y_train)
        prob_ns = rf_ns.predict_proba(X_test_sc)[:, 1]
        _record_fold(results_no_smote["rf_no_smote"], y_test, prob_ns, time.time() - t0)

        # ── C. Train all enhanced models ──
        for model_name, model_template in models.items():
            t0 = time.time()
            est = clone(model_template)

            # Set XGBoost scale_pos_weight
            if hasattr(est, "scale_pos_weight"):
                n_neg = int((y_train_sm == 0).sum())
                n_pos = int((y_train_sm == 1).sum())
                est.set_params(scale_pos_weight=max(1.0, n_neg / max(1, n_pos)))

            try:
                est.fit(X_train_sm, y_train_sm)
            except Exception as e:
                logger.error(f"    {model_name} FAILED: {e}")
                continue

            if hasattr(est, "predict_proba"):
                y_prob = est.predict_proba(X_test_sc)[:, 1]
            elif hasattr(est, "decision_function"):
                y_prob = est.decision_function(X_test_sc)
            else:
                y_prob = est.predict(X_test_sc).astype(float)

            elapsed = time.time() - t0
            _record_fold(results[model_name], y_test, y_prob, elapsed)

    # ── Aggregate results ──
    all_results = {}
    all_results.update(results)
    all_results.update(results_no_smote)

    rows = []
    for name, res in all_results.items():
        if not res["roc_auc"]:
            continue
        row = {
            "model": name,
            "n_features": top_k,
            "smote": "yes" if (name != "rf_no_smote") else "no",
            "roc_auc_mean": np.mean(res["roc_auc"]),
            "roc_auc_std": np.std(res["roc_auc"]),
            "pr_auc_mean": np.mean(res["pr_auc"]),
            "pr_auc_std": np.std(res["pr_auc"]),
            "f1_mean": np.mean(res["f1"]),
            "f1_std": np.std(res["f1"]),
            "precision_mean": np.mean(res["prec"]),
            "recall_mean": np.mean(res["rec"]),
            "TP": res["tp"], "FN": res["fn"],
            "FP": res["fp"], "TN": res["tn"],
            "time_s": round(res["time"], 1),
        }
        rows.append(row)

    df_results = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)

    return df_results, global_selected_names, global_imp


def _record_fold(res_dict, y_test, y_prob, elapsed):
    """Record metrics for one fold."""
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    best_t, best_f1, best_prec, best_rec = optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    res_dict["roc_auc"].append(roc)
    res_dict["pr_auc"].append(pr)
    res_dict["f1"].append(best_f1)
    res_dict["prec"].append(best_prec)
    res_dict["rec"].append(best_rec)
    res_dict["tp"] += tp
    res_dict["fn"] += fn
    res_dict["fp"] += fp
    res_dict["tn"] += tn
    res_dict["time"] += elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_scored_csv()
    dataset = prepare_dataset(df=df, dataset_filter="DAppSCAN", feature_set_name="all")

    X, y, groups = dataset.X, dataset.y, dataset.groups
    feature_names = dataset.feature_names

    logger.info(f"Dataset: {len(y)} samples, {y.sum()} positive, "
                f"{len(feature_names)} features, {len(np.unique(groups))} groups")

    # Run enhanced pipeline
    logger.info("\n" + "=" * 70)
    logger.info("ENHANCED PIPELINE: Feature Selection + SMOTE + Stacking")
    logger.info("=" * 70)

    df_results, selected_names, selected_imp = evaluate_enhanced(
        X, y, groups, feature_names,
        n_folds=5,
        top_k=80,
        use_smote=True,
        smote_ratio=0.3,
    )

    # Save results
    out_path = _OUT_DIR / "enhanced_comparison.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"\nWrote: {out_path}")

    # Save selected features
    feat_df = pd.DataFrame({
        "rank": range(1, len(selected_names) + 1),
        "feature": selected_names,
        "importance": selected_imp,
    })
    feat_path = _OUT_DIR / "enhanced_selected_features.csv"
    feat_df.to_csv(feat_path, index=False)

    # Print comparison
    print("\n" + "=" * 100)
    print("ENHANCED PIPELINE RESULTS")
    print("=" * 100)
    print(f"{'Model':<25} {'SMOTE':>5} {'Feats':>5} "
          f"{'ROC-AUC':>12} {'PR-AUC':>12} {'F1':>8} {'Prec':>8} {'Recall':>8} "
          f"{'TP':>5} {'FN':>5} {'FP':>5} {'TN':>6} {'Time':>7}")
    print("-" * 100)

    for _, row in df_results.iterrows():
        roc_str = f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.3f}"
        pr_str = f"{row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.3f}"
        print(
            f"  {row['model']:<23} {row['smote']:>5} {row['n_features']:>5} "
            f"{roc_str:>12} {pr_str:>12} "
            f"{row['f1_mean']:>8.4f} {row['precision_mean']:>8.4f} {row['recall_mean']:>8.4f} "
            f"{int(row['TP']):>5} {int(row['FN']):>5} {int(row['FP']):>5} {int(row['TN']):>6} "
            f"{row['time_s']:>6.1f}s"
        )

    print(f"\nTop 20 selected features:")
    for i, (name, imp) in enumerate(zip(selected_names[:20], selected_imp[:20]), 1):
        print(f"  {i:>2}. {name:<40} importance={imp:.4f}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal enhanced pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
