#!/usr/bin/env python3
"""Create fold-consistent local SHAP case studies for the primary LODO task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ml.baseline_comparison import (  # noqa: E402
    _feature_matrix,
    _fit,
    canonicalize_training_sources,
    exclude_test_source_duplicates,
    load_scored_frame,
    select_dataset,
    source_identity_hash,
)
from src.ml.baseline_comparison import DatasetSpec  # noqa: E402


INPUT = REPO / "data/derived/unified_contract_eval_scored.csv"
OUTPUT = REPO / "data/derived/shap_results/fold_consistent_primary"
PAPER_FIGURES = REPO.parent / "sc_risk_score_overleaf/figures"
ALL_DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")
TARGETS = ("DIVE", "Empirical-Analysis")


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_source_hash"] = result["local_source_path"].fillna("").astype(str).map(
        source_identity_hash
    )
    result["_source_hash"] = result["_source_hash"].where(
        result["_source_hash"].ne(""),
        "contract:" + result["contract_id"].astype(str),
    )
    conflict_keys = (
        result.groupby(["source_dataset", "_source_hash"])["label_status"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if len(conflict_keys):
        indexed = result.set_index(["source_dataset", "_source_hash"])
        result = indexed.loc[~indexed.index.isin(conflict_keys)].reset_index()
    result = result.drop_duplicates(["source_dataset", "_source_hash"], keep="first")
    return result.drop(columns="_source_hash")


def _case_indices(y_true: np.ndarray, probabilities: np.ndarray) -> list[tuple[str, int]]:
    predictions = (probabilities >= 0.5).astype(int)
    selections: list[tuple[str, int]] = []
    for name, mask, order in (
        ("TP", (y_true == 1) & (predictions == 1), np.argsort(-probabilities)),
        ("FP", (y_true == 0) & (predictions == 1), np.argsort(-probabilities)),
        ("FN", (y_true == 1) & (predictions == 0), np.argsort(probabilities)),
    ):
        candidates = [int(i) for i in order if mask[i]]
        if candidates:
            selections.append((name, candidates[0]))
    return selections


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    frame = _canonicalize(load_scored_frame(str(INPUT)))
    feature_columns = sorted(
        column
        for column in frame.columns
        if column not in {"y"} and column in __import__(
            "src.ml.baseline_comparison", fromlist=["source_feature_columns"]
        ).source_feature_columns(frame)
    )
    cases: list[dict[str, object]] = []
    for held_out in TARGETS:
        spec = DatasetSpec(name=held_out, group_by_project=True)
        test = select_dataset(frame, spec)
        train = pd.concat(
            [select_dataset(frame, DatasetSpec(name=name, group_by_project=True))
             for name in ALL_DATASETS if name != held_out],
            ignore_index=True,
        )
        train, removed = exclude_test_source_duplicates(train, test)
        train, collapsed, conflicts = canonicalize_training_sources(train)
        fitted = _fit(train, feature_columns, "rf", random_state=42)
        x_test = fitted.scaler.transform(_feature_matrix(test, feature_columns))
        probabilities = fitted.estimator.predict_proba(x_test)[:, 1]
        y_true = test["y"].to_numpy(dtype=int)
        for category, position in _case_indices(y_true, probabilities):
            row = test.iloc[position]
            background = fitted.scaler.transform(
                _feature_matrix(train.sample(min(100, len(train)), random_state=42), feature_columns)
            )
            explainer = shap.TreeExplainer(fitted.estimator, background)
            values = np.asarray(explainer.shap_values(x_test[position:position + 1]))
            if values.ndim == 3:
                values = values[:, :, 1]
            values = values.reshape(-1)
            case_id = f"{held_out}_{category}"
            top = np.argsort(np.abs(values))[::-1][:10]
            for rank, feature_index in enumerate(top, start=1):
                cases.append(
                    {
                        "case_id": case_id,
                        "held_out_dataset": held_out,
                        "category": category,
                        "contract_id": str(row["contract_id"]),
                        "source_identity_hash": source_identity_hash(
                            str(row.get("local_source_path", ""))
                        ),
                        "fold_protocol": "LODO",
                        "model": "rf",
                        "seed": 42,
                        "y_true": int(row["y"]),
                        "y_probability": float(probabilities[position]),
                        "train_duplicate_rows_removed": int(removed),
                        "cross_training_same_label_copies_collapsed": int(collapsed),
                        "cross_training_conflict_rows_excluded": int(conflicts),
                        "rank": rank,
                        "feature": feature_columns[feature_index],
                        "shap_value": float(values[feature_index]),
                        "abs_shap_value": float(abs(values[feature_index])),
                    }
                )

    cases_frame = pd.DataFrame(cases)
    cases_frame.to_csv(OUTPUT / "local_shap_cases.csv", index=False)
    metadata = {
        "protocol": "canonicalized source-only external-target transfer",
        "datasets": list(TARGETS),
        "model": "rf",
        "seed": 42,
        "threshold": 0.5,
        "case_categories": ["TP", "FP", "FN"],
        "cases": sorted(cases_frame["case_id"].unique().tolist()),
        "feature_count": len(feature_columns),
        "explanation_scope": "held-out fold only; fold-specific scaler/model/background",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), squeeze=False)
    for ax, case_id in zip(axes.ravel(), metadata["cases"]):
        group = cases_frame[cases_frame["case_id"] == case_id].sort_values("shap_value")
        ax.barh(group["feature"], group["shap_value"], color=np.where(group["shap_value"] >= 0, "#b2182b", "#2166ac"))
        probability = float(group["y_probability"].iloc[0])
        ax.set_title(f"{case_id}  p={probability:.3f}", fontsize=9)
        ax.tick_params(axis="y", labelsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
    for ax in axes.ravel()[len(metadata["cases"]):]:
        ax.axis("off")
    fig.suptitle("Fold-consistent local SHAP case studies (held-out LODO folds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_shap_cases.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
