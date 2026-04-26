"""
Cross-validated evaluation for vulnerability classification models.

Provides:
- GroupKFold CV (leakage-safe: no contract appears in both train and test)
- Metrics: ROC-AUC, PR-AUC, F1 (optimal threshold), precision, recall
- Feature importance extraction (model-agnostic via permutation + native)
- Comparison table generation
- Baseline comparison (risk_z rule-based score)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .features import PreparedDataset
from .models import ModelConfig

logger = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics from a single CV fold."""
    fold_idx: int
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    n_pos_test: int
    threshold: float          # optimal F1 threshold
    y_true: np.ndarray        # test labels
    y_prob: np.ndarray        # predicted probabilities


@dataclass
class ModelResult:
    """Aggregated results for one model across all CV folds."""
    model_name: str
    display_name: str
    family: str
    feature_set: str
    dataset_filter: str
    n_features: int
    n_folds: int
    fold_results: List[FoldResult]
    train_time_seconds: float
    # Aggregated metrics (mean ± std)
    roc_auc_mean: float = 0.0
    roc_auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    precision_mean: float = 0.0
    recall_mean: float = 0.0
    # Feature importance (if available)
    feature_importance: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.fold_results:
            self.roc_auc_mean = float(np.mean([f.roc_auc for f in self.fold_results]))
            self.roc_auc_std = float(np.std([f.roc_auc for f in self.fold_results]))
            self.pr_auc_mean = float(np.mean([f.pr_auc for f in self.fold_results]))
            self.pr_auc_std = float(np.std([f.pr_auc for f in self.fold_results]))
            self.f1_mean = float(np.mean([f.f1 for f in self.fold_results]))
            self.f1_std = float(np.std([f.f1 for f in self.fold_results]))
            self.precision_mean = float(np.mean([f.precision for f in self.fold_results]))
            self.recall_mean = float(np.mean([f.recall for f in self.fold_results]))


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find the threshold that maximizes F1 score."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0.0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


# ── Core CV evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    model_config: ModelConfig,
    dataset: PreparedDataset,
    *,
    n_folds: int = 5,
    scale_features: bool = True,
) -> ModelResult:
    """
    Run GroupKFold cross-validation for a single model.

    Parameters
    ----------
    model_config : the model to evaluate
    dataset : prepared dataset (X, y, groups, feature_names)
    n_folds : number of CV folds
    scale_features : whether to StandardScale features (important for linear models)
    """
    X, y, groups = dataset.X, dataset.y, dataset.groups
    gkf = GroupKFold(n_splits=n_folds)

    fold_results: List[FoldResult] = []
    fitted_models: List[BaseEstimator] = []

    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Optional scaling
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Handle XGBoost scale_pos_weight
        est = model_config.build()
        if hasattr(est, "scale_pos_weight"):
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            est.set_params(scale_pos_weight=max(1.0, n_neg / max(1, n_pos)))

        # Fit
        est.fit(X_train, y_train)
        fitted_models.append(est)

        # Predict
        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X_test)[:, 1]
        elif hasattr(est, "decision_function"):
            y_prob = est.decision_function(X_test)
        else:
            y_prob = est.predict(X_test).astype(float)

        # Metrics
        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        opt_t, opt_f1 = _optimal_f1_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0.0)
        rec = recall_score(y_test, y_pred, zero_division=0.0)

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            roc_auc=roc,
            pr_auc=pr,
            f1=opt_f1,
            precision=prec,
            recall=rec,
            n_train=len(y_train),
            n_test=len(y_test),
            n_pos_test=int(y_test.sum()),
            threshold=opt_t,
            y_true=y_test,
            y_prob=y_prob,
        ))

    train_time = time.time() - t0

    # ── Feature importance ──
    feature_imp = _extract_feature_importance(fitted_models, dataset.feature_names)

    result = ModelResult(
        model_name=model_config.name,
        display_name=model_config.display_name,
        family=model_config.family,
        feature_set=dataset.feature_set_name,
        dataset_filter=dataset.dataset_filter,
        n_features=len(dataset.feature_names),
        n_folds=n_folds,
        fold_results=fold_results,
        train_time_seconds=train_time,
        feature_importance=feature_imp,
    )

    logger.info(
        f"  {model_config.display_name:30s}: "
        f"ROC-AUC={result.roc_auc_mean:.4f}±{result.roc_auc_std:.3f}  "
        f"PR-AUC={result.pr_auc_mean:.4f}±{result.pr_auc_std:.3f}  "
        f"F1={result.f1_mean:.4f}  "
        f"({train_time:.1f}s)"
    )
    return result


def _extract_feature_importance(
    models: List[BaseEstimator],
    feature_names: List[str],
) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from fitted models.
    Averages across CV folds. Supports tree-based (feature_importances_)
    and linear (coef_) models.
    """
    importances = []
    for m in models:
        if hasattr(m, "feature_importances_"):
            importances.append(m.feature_importances_)
        elif hasattr(m, "coef_"):
            importances.append(np.abs(m.coef_).ravel())

    if not importances:
        return None

    # Average across folds
    imp_matrix = np.array(importances)
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": mean_imp,
        "importance_std": std_imp,
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ── Baseline evaluation ──────────────────────────────────────────────────────

def evaluate_baseline(
    dataset: PreparedDataset,
    score_column: str = "risk_z",
) -> ModelResult:
    """
    Evaluate the rule-based risk score as a baseline "model".
    No training needed — just computes metrics using the pre-computed score.
    """
    df = dataset.df
    y = dataset.y
    groups = dataset.groups

    scores = pd.to_numeric(df[score_column], errors="coerce").fillna(0.0).values

    gkf = GroupKFold(n_splits=5)
    fold_results: List[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dataset.X, y, groups)):
        y_test = y[test_idx]
        y_prob = scores[test_idx]

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        opt_t, opt_f1 = _optimal_f1_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0.0)
        rec = recall_score(y_test, y_pred, zero_division=0.0)

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            roc_auc=roc,
            pr_auc=pr,
            f1=opt_f1,
            precision=prec,
            recall=rec,
            n_train=len(train_idx),
            n_test=len(test_idx),
            n_pos_test=int(y_test.sum()),
            threshold=opt_t,
            y_true=y_test,
            y_prob=y_prob,
        ))

    result = ModelResult(
        model_name=f"baseline_{score_column}",
        display_name=f"Baseline ({score_column})",
        family="baseline",
        feature_set="N/A",
        dataset_filter=dataset.dataset_filter,
        n_features=0,
        n_folds=5,
        fold_results=fold_results,
        train_time_seconds=0.0,
    )

    logger.info(
        f"  {'Baseline (' + score_column + ')':30s}: "
        f"ROC-AUC={result.roc_auc_mean:.4f}±{result.roc_auc_std:.3f}  "
        f"PR-AUC={result.pr_auc_mean:.4f}±{result.pr_auc_std:.3f}  "
        f"F1={result.f1_mean:.4f}"
    )
    return result


# ── Comparison table ─────────────────────────────────────────────────────────

def build_comparison_table(results: List[ModelResult]) -> pd.DataFrame:
    """
    Build a tidy comparison DataFrame sorted by ROC-AUC descending.
    """
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "display_name": r.display_name,
            "family": r.family,
            "feature_set": r.feature_set,
            "dataset": r.dataset_filter,
            "n_features": r.n_features,
            "n_folds": r.n_folds,
            "roc_auc_mean": round(r.roc_auc_mean, 4),
            "roc_auc_std": round(r.roc_auc_std, 4),
            "pr_auc_mean": round(r.pr_auc_mean, 4),
            "pr_auc_std": round(r.pr_auc_std, 4),
            "f1_mean": round(r.f1_mean, 4),
            "f1_std": round(r.f1_std, 4),
            "precision_mean": round(r.precision_mean, 4),
            "recall_mean": round(r.recall_mean, 4),
            "train_time_s": round(r.train_time_seconds, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return df
