"""Leakage-safe SCAFE evaluation on external benchmark datasets.

This module intentionally evaluates SCAFE only.  It does not invoke Slither,
LLMs, hosted APIs, or any third-party vulnerability detector.

The comparison protocol is deliberately stricter than the historical merged
CV experiment:

* source-derived numeric features only (no bytecode or activity features);
* dataset-specific GroupKFold for within-dataset estimates;
* train-on-one-dataset/test-on-another transfer estimates;
* fixed or train-only calibrated decision thresholds, never test-label tuning;
* positive-only datasets report recall/coverage only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES
from src.ml.models import get_all_models


CANONICAL_VULNERABILITY_TYPES = (
    "reentrancy",
    "access_control",
    "arithmetic",
    "unchecked_calls",
    "denial_service",
    "time_manipulation",
    "bad_randomness",
    "front_running",
    "delegatecall",
    "other",
)


def canonicalize_vulnerability_type(value: object) -> str:
    """Map SWC/CWE/DASP labels to SCAFE's shared taxonomy."""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "reentr" in text:
        return "reentrancy"
    if any(token in text for token in ("access", "authoriz", "tx.origin", "cwe-269", "cwe-284", "cwe-285")):
        return "access_control"
    if any(token in text for token in ("arithmetic", "overflow", "underflow", "cwe-682")):
        return "arithmetic"
    if any(token in text for token in ("unchecked", "low-level", "low_level", "failed call", "cwe-252", "cwe-703")):
        return "unchecked_calls"
    if any(token in text for token in ("denial", "dos", "gas limit", "gasless", "cwe-691", "cwe-754", "cwe-755")):
        return "denial_service"
    if any(
        token in text
        for token in (
            "timestamp",
            "time manipulation",
            "time_manipulation",
            "block values",
            "cwe-829",
        )
    ):
        return "time_manipulation"
    if any(token in text for token in ("random", "weak sources")):
        return "bad_randomness"
    if any(token in text for token in ("front-running", "front_running", "transaction order", "cwe-362")):
        return "front_running"
    if "delegatecall" in text:
        return "delegatecall"
    return "other"


def canonicalize_vulnerability_types(value: object) -> str:
    """Canonicalize a semicolon-separated multi-label field."""

    labels = {
        canonicalize_vulnerability_type(token)
        for token in str(value or "").split(";")
        if canonicalize_vulnerability_type(token)
    }
    return ";".join(sorted(labels))


def add_canonical_vulnerability_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized multi-label taxonomy values."""

    result = frame.copy()
    result["taxonomy_types"] = result.get(
        "vuln_types", pd.Series("", index=result.index)
    ).map(canonicalize_vulnerability_types)
    return result


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


@lru_cache(maxsize=100_000)
def source_identity_hash(path_value: str) -> str:
    """Return a whitespace/comment-insensitive source identity hash."""

    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    try:
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="ignore")
        elif path.is_dir():
            source = "\n".join(
                item.read_text(encoding="utf-8", errors="ignore")
                for item in sorted(path.rglob("*.sol"))
            )
        else:
            return ""
    except OSError:
        return ""
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    source = re.sub(r"//.*?$", " ", source, flags=re.MULTILINE)
    source = re.sub(r"\s+", " ", source).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def exclude_test_source_duplicates(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove train rows whose source is also present in held-out test."""

    if "local_source_path" not in train or "local_source_path" not in test:
        return train, 0
    test_hashes = {
        source_identity_hash(value)
        for value in test["local_source_path"].fillna("").astype(str)
    }
    test_hashes.discard("")
    if not test_hashes:
        return train, 0
    train_hashes = train["local_source_path"].fillna("").astype(str).map(
        source_identity_hash
    )
    duplicate_mask = train_hashes.isin(test_hashes)
    return train.loc[~duplicate_mask].copy(), int(duplicate_mask.sum())


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
    *,
    random_state: int = 42,
) -> FittedModel:
    x_train = _feature_matrix(train, feature_columns)
    y_train = train["y"].to_numpy(dtype=int)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    config = get_all_models()[model_name]
    estimator = config.build()
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)
    if "scale_pos_weight" in estimator.get_params():
        negatives = max(1, int((y_train == 0).sum()))
        positives = max(1, int((y_train == 1).sum()))
        estimator.set_params(scale_pos_weight=negatives / positives)
    estimator.fit(x_train, y_train)
    return FittedModel(estimator, scaler, list(feature_columns))


def _optimal_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a threshold using labels from a training-only validation split."""

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _fit_with_train_calibration(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    model_name: str,
    *,
    random_state: int = 42,
) -> tuple[FittedModel, float]:
    """Fit on all training data and calibrate a threshold without test labels."""

    if train["y"].nunique() < 2 or len(train) < 10:
        return _fit(
            train, feature_columns, model_name, random_state=random_state
        ), 0.5
    calibration_train, calibration_valid = train_test_split(
        train,
        test_size=0.2,
        random_state=random_state,
        stratify=train["y"],
    )
    calibration_model = _fit(
        calibration_train.reset_index(drop=True),
        feature_columns,
        model_name,
        random_state=random_state,
    )
    calibration_probabilities = calibration_model.predict_probability(
        calibration_valid
    )
    threshold = _optimal_f1_threshold(
        calibration_valid["y"].to_numpy(dtype=int),
        calibration_probabilities,
    )
    return _fit(
        train, feature_columns, model_name, random_state=random_state
    ), threshold


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
    exclude_duplicate_sources: bool = True,
) -> list[dict]:
    """Train on one labelled dataset and evaluate on another."""

    train = select_dataset(frame, train_spec)
    test = select_dataset(frame, test_spec)
    duplicate_count = 0
    if exclude_duplicate_sources:
        train, duplicate_count = exclude_test_source_duplicates(train, test)
    if train["y"].nunique() < 2:
        raise ValueError(f"Training dataset {train_spec.name!r} needs both classes")
    rows: list[dict] = []
    for model_name in model_names:
        fitted = _fit(train, feature_columns, model_name)
        probabilities = fitted.predict_probability(test)
        result = _metrics(
            test["y"].to_numpy(dtype=int),
            probabilities,
            dataset=test_spec.name,
            model=model_name,
            experiment=f"transfer_{train_spec.name}_to_{test_spec.name}",
        )
        result["n_train"] = int(len(train))
        result["n_duplicate_train_excluded"] = duplicate_count
        rows.append(result)
    return rows


def run_leave_one_dataset_out(
    frame: pd.DataFrame,
    specs: Sequence[DatasetSpec],
    feature_columns: Sequence[str],
    *,
    model_names: Iterable[str] = ("rf", "et", "hgb", "xgb"),
    calibrate_threshold: bool = False,
    exclude_duplicate_sources: bool = True,
    random_state: int = 42,
) -> tuple[list[dict], pd.DataFrame]:
    """Train on every dataset except one and test on the held-out dataset."""

    selected = {spec.name: select_dataset(frame, spec) for spec in specs}
    rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for target in specs:
        test = selected[target.name]
        train_parts = [
            selected[spec.name] for spec in specs if spec.name != target.name
        ]
        train = pd.concat(train_parts, ignore_index=True)
        duplicate_count = 0
        if exclude_duplicate_sources:
            train, duplicate_count = exclude_test_source_duplicates(train, test)
        if train["y"].nunique() < 2:
            continue
        for model_name in model_names:
            if calibrate_threshold:
                fitted, threshold = _fit_with_train_calibration(
                    train,
                    feature_columns,
                    model_name,
                    random_state=random_state,
                )
            else:
                fitted, threshold = (
                    _fit(
                        train,
                        feature_columns,
                        model_name,
                        random_state=random_state,
                    ),
                    0.5,
                )
            probabilities = fitted.predict_probability(test)
            result = _metrics(
                test["y"].to_numpy(dtype=int),
                probabilities,
                dataset=target.name,
                model=model_name,
                experiment="leave_one_dataset_out",
                threshold=threshold,
            )
            result["train_datasets"] = ",".join(
                spec.name for spec in specs if spec.name != target.name
            )
            result["n_train"] = int(len(train))
            result["n_duplicate_train_excluded"] = duplicate_count
            result["random_state"] = random_state
            rows.append(result)
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "dataset": target.name,
                        "model": model_name,
                        "contract_id": test["contract_id"].to_numpy(),
                        "label_status": test["label_status"].to_numpy(),
                        "vuln_types": test.get(
                            "vuln_types", pd.Series("", index=test.index)
                        ).to_numpy(),
                        "y_true": test["y"].to_numpy(dtype=int),
                        "y_prob": probabilities,
                        "y_pred": (probabilities >= threshold).astype(int),
                        "random_state": random_state,
                    }
                )
            )
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return rows, predictions


def run_repeated_leave_one_dataset_out(
    frame: pd.DataFrame,
    specs: Sequence[DatasetSpec],
    feature_columns: Sequence[str],
    *,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_names: Iterable[str] = ("rf", "et", "hgb", "xgb"),
    calibrate_threshold: bool = True,
    exclude_duplicate_sources: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """Repeat duplicate-safe LODO across independent model seeds."""

    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for seed in seeds:
        rows, predictions = run_leave_one_dataset_out(
            frame,
            specs,
            feature_columns,
            model_names=model_names,
            calibrate_threshold=calibrate_threshold,
            exclude_duplicate_sources=exclude_duplicate_sources,
            random_state=int(seed),
        )
        all_rows.extend(rows)
        if not predictions.empty:
            all_predictions.append(predictions)
    return all_rows, (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )


def summarize_repeated_lodo(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute seed means and percentile confidence intervals."""

    if metrics.empty:
        return pd.DataFrame()
    metric_names = ("roc_auc", "pr_auc", "f1", "precision", "recall", "coverage")
    rows: list[dict] = []
    for (dataset, model), group in metrics.groupby(["dataset", "model"]):
        result: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "n_seeds": int(group["random_state"].nunique()),
        }
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                result[f"{metric}_mean"] = float("nan")
                result[f"{metric}_std"] = float("nan")
                result[f"{metric}_ci95_low"] = float("nan")
                result[f"{metric}_ci95_high"] = float("nan")
                continue
            rng = np.random.default_rng(42)
            samples = rng.choice(
                values,
                size=(2000, len(values)),
                replace=True,
            ).mean(axis=1)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=0))
            result[f"{metric}_ci95_low"] = float(np.percentile(samples, 2.5))
            result[f"{metric}_ci95_high"] = float(np.percentile(samples, 97.5))
        rows.append(result)
    return pd.DataFrame(rows)


def vulnerability_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize recall and false-positive rate by vulnerability type."""

    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (dataset, model), group in predictions.groupby(["dataset", "model"]):
        positive_rows = group[group["y_true"] == 1]
        types: set[str] = set()
        for value in positive_rows["vuln_types"].fillna(""):
            types.update(
                token.strip()
                for token in str(value).split(";")
                if token.strip()
            )
        for vulnerability_type in sorted(types):
            mask = positive_rows["vuln_types"].fillna("").map(
                lambda value: vulnerability_type in {
                    token.strip() for token in str(value).split(";") if token.strip()
                }
            )
            subset = positive_rows[mask]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "analysis_level": "vulnerability_type",
                    "vulnerability_type": vulnerability_type,
                    "n": int(len(subset)),
                    "recall": float(subset["y_pred"].mean()) if len(subset) else float("nan"),
                    "mean_probability": float(subset["y_prob"].mean()) if len(subset) else float("nan"),
                    "false_negative_count": int((subset["y_pred"] == 0).sum()),
                }
            )
        clean = group[group["y_true"] == 0]
        if not clean.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "analysis_level": "clean_control",
                    "vulnerability_type": "clean",
                    "n": int(len(clean)),
                    "recall": float("nan"),
                    "mean_probability": float(clean["y_prob"].mean()),
                    "false_negative_count": int((clean["y_pred"] == 1).sum()),
                    "false_positive_rate": float(clean["y_pred"].mean()),
                }
            )
    return pd.DataFrame(rows)


def run_leave_one_dataset_out_ovr(
    frame: pd.DataFrame,
    specs: Sequence[DatasetSpec],
    feature_columns: Sequence[str],
    *,
    vulnerability_types: Sequence[str] = CANONICAL_VULNERABILITY_TYPES,
    model_names: Iterable[str] = ("rf", "et", "hgb", "xgb"),
    calibrate_threshold: bool = False,
    exclude_duplicate_sources: bool = True,
    random_state: int = 42,
) -> tuple[list[dict], pd.DataFrame]:
    """Run held-out-dataset one-vs-rest models for each taxonomy type."""

    normalized = add_canonical_vulnerability_types(frame)
    selected = {spec.name: select_dataset(normalized, spec) for spec in specs}
    rows: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    for vulnerability_type in vulnerability_types:
        for target in specs:
            test = selected[target.name].copy()
            test["y"] = test["taxonomy_types"].map(
                lambda value: vulnerability_type in str(value).split(";")
            ).astype(int)
            train = pd.concat(
                [
                    selected[spec.name]
                    for spec in specs
                    if spec.name != target.name
                ],
                ignore_index=True,
            )
            duplicate_count = 0
            if exclude_duplicate_sources:
                train, duplicate_count = exclude_test_source_duplicates(
                    train, test
                )
            train["y"] = train["taxonomy_types"].map(
                lambda value: vulnerability_type in str(value).split(";")
            ).astype(int)
            if train["y"].nunique() < 2 or test["y"].nunique() < 2:
                continue
            for model_name in model_names:
                if calibrate_threshold:
                    fitted, threshold = _fit_with_train_calibration(
                        train,
                        feature_columns,
                        model_name,
                        random_state=random_state,
                    )
                else:
                    fitted, threshold = (
                        _fit(
                            train,
                            feature_columns,
                            model_name,
                            random_state=random_state,
                        ),
                        0.5,
                    )
                probabilities = fitted.predict_probability(test)
                result = _metrics(
                    test["y"].to_numpy(dtype=int),
                    probabilities,
                    dataset=target.name,
                    model=model_name,
                    experiment="leave_one_dataset_out_ovr",
                    threshold=threshold,
                )
                result["vulnerability_type"] = vulnerability_type
                result["train_datasets"] = ",".join(
                    spec.name for spec in specs if spec.name != target.name
                )
                result["n_train"] = int(len(train))
                result["n_duplicate_train_excluded"] = duplicate_count
                result["random_state"] = random_state
                rows.append(result)
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": target.name,
                            "model": model_name,
                            "vulnerability_type": vulnerability_type,
                            "contract_id": test["contract_id"].to_numpy(),
                            "y_true": test["y"].to_numpy(dtype=int),
                            "y_prob": probabilities,
                            "y_pred": (probabilities >= threshold).astype(int),
                        }
                    )
                )
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return rows, predictions
