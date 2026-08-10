"""Leakage-safe SCAFE evaluation on external benchmark datasets.

This module intentionally evaluates SCAFE only.  It does not invoke Slither,
LLMs, hosted APIs, or any third-party vulnerability detector.

The comparison protocol is deliberately stricter than the historical merged
CV experiment:

* source-derived numeric features only (no bytecode or activity features);
* dataset-specific GroupKFold for within-dataset estimates;
* train-on-one-dataset/test-on-another transfer estimates;
* fixed 0.5 decision threshold, avoiding test-label threshold tuning;
* positive-only datasets report recall/coverage only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.models import get_all_models


@dataclass(frozen=True)
class DatasetSpec:
    """Label and grouping policy for one benchmark dataset."""

    name: str
    accepted_labels: tuple[str, ...] = ("clean", "vulnerable")
    positive_label: str = "vulnerable"
    group_by_project: bool = False


@dataclass
class FittedModel:
    """A fitted SCAFE estimator and its training scaler."""

    estimator: ClassifierMixin
    scaler: StandardScaler
    feature_columns: list[str]

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_columns)
        matrix = self.scaler.transform(matrix)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(matrix)[:, 1]
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(matrix)
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
        return self.estimator.predict(matrix).astype(float)


def source_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric, source-derived columns without known leakage channels."""

    excluded = _EXCLUDE | _ONCHAIN_FEATURES
    return sorted(
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column not in excluded
    )


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def load_scored_frame(path: str) -> pd.DataFrame:
    """Load and validate the unified scored SCAFE table."""

    frame = pd.read_csv(path, low_memory=False)
    required = {"source_dataset", "label_status", "contract_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Scored CSV is missing required columns: {sorted(missing)}")
    for column in ("source_dataset", "label_status", "contract_id"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def select_dataset(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Select binary-labelled rows and reject unknown labels explicitly."""

    selected = frame[
        (frame["source_dataset"] == spec.name)
        & frame["label_status"].isin(spec.accepted_labels)
    ].copy()
    if selected.empty:
        raise ValueError(f"No usable labelled rows found for {spec.name!r}")
    selected["y"] = (selected["label_status"] == spec.positive_label).astype(int)
    return selected


def _groups(frame: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    if spec.group_by_project and "source_subdataset" in frame:
        keys = frame["source_subdataset"].replace("", pd.NA).fillna(frame["contract_id"])
    else:
        keys = frame["contract_id"]
    encoded = {value: index for index, value in enumerate(sorted(keys.unique()))}
    return keys.map(encoded).to_numpy(dtype=int)


def _fit(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    model_name: str,
) -> FittedModel:
    x_train = _feature_matrix(train, feature_columns)
    y_train = train["y"].to_numpy(dtype=int)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    config = get_all_models()[model_name]
    estimator = config.build()
    if "scale_pos_weight" in estimator.get_params():
        negatives = max(1, int((y_train == 0).sum()))
        positives = max(1, int((y_train == 1).sum()))
        estimator.set_params(scale_pos_weight=negatives / positives)
    estimator.fit(x_train, y_train)
    return FittedModel(estimator, scaler, list(feature_columns))


def _metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    dataset: str,
    model: str,
    experiment: str,
    threshold: float = 0.5,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    positive_only = not np.any(y_true == 0)
    if positive_only:
        roc_auc = float("nan")
        pr_auc = float("nan")
        f1 = precision = float("nan")
        tn = fp = 0
        tp = int(predictions.sum())
        fn = int(len(predictions) - tp)
    else:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        precision = float(precision_score(y_true, predictions, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "n_samples": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "coverage": float(predictions.mean()),
        "threshold": threshold,
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "positive_only": positive_only,
    }


def run_dataset_cv(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_columns: Sequence[str],
    *,
    model_names: Iterable[str] = ("rf", "et", "hgb", "xgb"),
    n_splits: int = 5,
) -> list[dict]:
    """Run source-only GroupKFold for one dataset."""

    selected = select_dataset(frame, spec)
    y = selected["y"].to_numpy(dtype=int)
    groups = _groups(selected, spec)
    if np.unique(y).size < 2:
        return []
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return []
    rows: list[dict] = []
    for model_name in model_names:
        fold_rows = []
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(selected, y, groups):
            fitted = _fit(selected.iloc[train_idx], feature_columns, model_name)
            probabilities = fitted.predict_probability(selected.iloc[test_idx])
            fold_rows.append(
                _metrics(
                    y[test_idx],
                    probabilities,
                    dataset=spec.name,
                    model=model_name,
                    experiment="within_dataset_group_cv",
                )
            )
        aggregate = dict(fold_rows[0])
        aggregate["n_samples"] = int(len(selected))
        aggregate["n_pos"] = int(y.sum())
        aggregate["n_neg"] = int((y == 0).sum())
        aggregate["prevalence"] = float(y.mean())
        for count in ("TP", "FN", "FP", "TN"):
            aggregate[count] = int(sum(row[count] for row in fold_rows))
        for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall", "coverage"):
            values = [row[metric] for row in fold_rows if not np.isnan(row[metric])]
            aggregate[metric] = float(np.mean(values)) if values else float("nan")
            aggregate[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
        aggregate["folds"] = n_splits
        rows.append(aggregate)
    return rows


def run_transfer(
    frame: pd.DataFrame,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec,
    feature_columns: Sequence[str],
    *,
    model_names: Iterable[str] = ("rf", "et", "hgb", "xgb"),
) -> list[dict]:
    """Train on one labelled dataset and evaluate on another."""

    train = select_dataset(frame, train_spec)
    test = select_dataset(frame, test_spec)
    if train["y"].nunique() < 2:
        raise ValueError(f"Training dataset {train_spec.name!r} needs both classes")
    rows: list[dict] = []
    for model_name in model_names:
        fitted = _fit(train, feature_columns, model_name)
        probabilities = fitted.predict_probability(test)
        rows.append(
            _metrics(
                test["y"].to_numpy(dtype=int),
                probabilities,
                dataset=test_spec.name,
                model=model_name,
                experiment=f"transfer_{train_spec.name}_to_{test_spec.name}",
            )
        )
    return rows
