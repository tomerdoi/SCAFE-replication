"""
Model registry for vulnerability classification.

Each entry defines a named model configuration that returns a scikit-learn-
compatible estimator.  Models are grouped by family (linear, tree, ensemble)
for easy comparison in the paper.

Usage:
    from src.ml.models import get_all_models, get_model
    models = get_all_models()          # dict[str, ModelConfig]
    est = get_model("xgb").build()     # instantiate estimator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Named model configuration.

    Attributes
    ----------
    name        : short ID used as dict key / in result tables
    display_name: human-readable name for plots / paper
    family      : grouping (linear, tree, ensemble, baseline)
    build_fn    : callable that returns a fresh sklearn estimator
    params      : dict of key hyperparameters (for documentation / logging)
    description : one-liner for the paper methods section
    """
    name: str
    display_name: str
    family: str
    build_fn: Callable[[], BaseEstimator]
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def build(self) -> BaseEstimator:
        """Instantiate a fresh copy of the estimator."""
        return self.build_fn()


# ── Model factory functions ─────────────────────────────────────────────────

def _build_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )


def _build_logistic_regression_l1():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=0.5,
        penalty="l1",
        class_weight="balanced",
        solver="saga",
        max_iter=3000,
        random_state=42,
    )


def _build_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _build_extra_trees():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _build_hist_gradient_boosting():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )


def _build_xgboost() -> BaseEstimator:
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,   # will be overridden at train time if needed
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _build_gradient_boosting():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )


# ── Registry ────────────────────────────────────────────────────────────────

def _all_model_configs() -> List[ModelConfig]:
    """Return the full catalogue of model configurations."""
    configs = [
        ModelConfig(
            name="logreg",
            display_name="Logistic Regression",
            family="linear",
            build_fn=_build_logistic_regression,
            params={"C": 1.0, "class_weight": "balanced"},
            description="L2-regularized logistic regression with balanced class weights.",
        ),
        ModelConfig(
            name="logreg_l1",
            display_name="Logistic Regression (L1)",
            family="linear",
            build_fn=_build_logistic_regression_l1,
            params={"C": 0.5, "penalty": "l1", "class_weight": "balanced"},
            description="L1-regularized (sparse) logistic regression.",
        ),
        ModelConfig(
            name="rf",
            display_name="Random Forest",
            family="ensemble",
            build_fn=_build_random_forest,
            params={"n_estimators": 500, "class_weight": "balanced_subsample"},
            description="Random forest (500 trees, balanced subsampling).",
        ),
        ModelConfig(
            name="et",
            display_name="Extra Trees",
            family="ensemble",
            build_fn=_build_extra_trees,
            params={"n_estimators": 500, "class_weight": "balanced_subsample"},
            description="Extremely randomized trees (500 trees, balanced).",
        ),
        ModelConfig(
            name="hgb",
            display_name="HistGradientBoosting",
            family="boosting",
            build_fn=_build_hist_gradient_boosting,
            params={"max_iter": 500, "lr": 0.05, "class_weight": "balanced"},
            description="Histogram-based gradient boosting (sklearn) with early stopping.",
        ),
        ModelConfig(
            name="gbc",
            display_name="GradientBoosting",
            family="boosting",
            build_fn=_build_gradient_boosting,
            params={"n_estimators": 300, "lr": 0.05, "subsample": 0.8},
            description="Classic gradient boosting (sklearn).",
        ),
    ]

    # XGBoost: only include if importable
    try:
        import xgboost  # noqa: F401
        configs.append(ModelConfig(
            name="xgb",
            display_name="XGBoost",
            family="boosting",
            build_fn=_build_xgboost,
            params={"n_estimators": 500, "lr": 0.05, "max_depth": 6},
            description="XGBoost gradient boosting with early stopping.",
        ))
    except ImportError:
        logger.warning("XGBoost not installed; skipping xgb model.")

    return configs


def get_all_models() -> Dict[str, ModelConfig]:
    """Return {name: ModelConfig} for all available models."""
    return {m.name: m for m in _all_model_configs()}


def get_model(name: str) -> ModelConfig:
    """Get a single model config by name."""
    models = get_all_models()
    if name not in models:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(models)}")
    return models[name]


def list_models() -> List[str]:
    """Return sorted list of model names."""
    return sorted(get_all_models().keys())
