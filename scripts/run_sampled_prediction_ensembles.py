#!/usr/bin/env python3
"""Legacy ensemble utilities retained for artifact compatibility.

Do not execute this module for publication results: its original development
protocol used DAppSCAN LODO scores whose base models had seen the external
corpora. Use :mod:`scripts.run_sampled_prediction_ensembles_v2`, which separates
DAppSCAN within-corpus development from external evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCAFE = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1"
    / "scafe/scafe_repeated_lodo_predictions.csv"
)
DEFAULT_DL = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1"
    / "dl/predictions.csv"
)
DEFAULT_OUTPUT = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_ensemble_v1"
)
DEVELOPMENT_DATASET = "DAppSCAN"
TARGET_DATASETS = ("DIVE", "Empirical-Analysis")
NEURAL_MODELS = ("textcnn", "bilstm", "codebert", "unixcoder")
CASCADE_DELEGATION_FRACTIONS = (0.10, 0.25, 0.50, 0.75)
METRICS = ("roc_auc", "pr_auc", "f1", "precision", "recall")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def align_predictions(scafe: pd.DataFrame, neural: pd.DataFrame) -> pd.DataFrame:
    """Return one strictly aligned row per corpus/contract with all scores."""

    required_scafe = {"dataset", "contract_id", "y_true", "y_prob", "random_state"}
    required_dl = required_scafe | {"model", "source_identity_hash"}
    if not required_scafe.issubset(scafe.columns):
        raise ValueError(f"SCAFE columns missing: {sorted(required_scafe - set(scafe))}")
    if not required_dl.issubset(neural.columns):
        raise ValueError(f"DL columns missing: {sorted(required_dl - set(neural))}")
    scafe = scafe.loc[
        scafe["dataset"].isin((DEVELOPMENT_DATASET, *TARGET_DATASETS))
        & scafe["random_state"].eq(42)
        & scafe.get("model", "rf").eq("rf")
    ].copy()
    if scafe.duplicated(["dataset", "contract_id"]).any():
        raise ValueError("Duplicate SCAFE dataset/contract predictions")
    base = scafe[
        ["dataset", "contract_id", "y_true", "y_prob"]
    ].rename(columns={"y_prob": "scafe_prob"})
    identities: pd.DataFrame | None = None
    for model in NEURAL_MODELS:
        part = neural.loc[
            neural["model"].eq(model) & neural["random_state"].eq(42),
            ["dataset", "contract_id", "source_identity_hash", "y_true", "y_prob"],
        ].copy()
        if part.duplicated(["dataset", "contract_id"]).any():
            raise ValueError(f"Duplicate {model} dataset/contract predictions")
        if identities is None:
            identities = part[
                ["dataset", "contract_id", "source_identity_hash"]
            ].copy()
        base = base.merge(
            part[["dataset", "contract_id", "y_true", "y_prob"]].rename(
                columns={"y_true": f"{model}_y_true", "y_prob": f"{model}_prob"}
            ),
            on=["dataset", "contract_id"],
            how="inner",
            validate="one_to_one",
        )
        if not np.array_equal(base["y_true"], base.pop(f"{model}_y_true")):
            raise ValueError(f"Label mismatch between SCAFE and {model}")
    if identities is None:
        raise ValueError("No neural predictions found")
    base = base.merge(
        identities,
        on=["dataset", "contract_id"],
        how="left",
        validate="one_to_one",
    )
    expected = {
        dataset: len(scafe[scafe["dataset"].eq(dataset)])
        for dataset in (DEVELOPMENT_DATASET, *TARGET_DATASETS)
    }
    actual = base.groupby("dataset").size().to_dict()
    if actual != expected or any(count != 500 for count in actual.values()):
        raise ValueError(f"Prediction alignment incomplete: {actual}, expected {expected}")
    return base.sort_values(["dataset", "contract_id"]).reset_index(drop=True)


def choose_neural_partner(development: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Select one neural model by development ROC-AUC, then PR-AUC."""

    scores = {
        model: metric_values(
            development["y_true"].to_numpy(),
            development[f"{model}_prob"].to_numpy(),
        )
        for model in NEURAL_MODELS
    }
    selected = max(
        NEURAL_MODELS,
        key=lambda model: (scores[model]["roc_auc"], scores[model]["pr_auc"]),
    )
    return selected, scores[selected]


def choose_cascade_gate(
    development: pd.DataFrame,
    neural_model: str,
) -> tuple[float, float, dict[str, float]]:
    """Select a SCAFE distance-to-cutoff gate on development records only."""

    distance = np.abs(development["scafe_prob"].to_numpy() - 0.5)
    candidates = []
    for fraction in CASCADE_DELEGATION_FRACTIONS:
        threshold = float(np.quantile(distance, fraction, method="higher"))
        delegate = distance <= threshold
        probability = np.where(
            delegate,
            development[f"{neural_model}_prob"].to_numpy(),
            development["scafe_prob"].to_numpy(),
        )
        scores = metric_values(development["y_true"].to_numpy(), probability)
        candidates.append((scores["roc_auc"], scores["pr_auc"], -fraction,
                           threshold, float(delegate.mean()), scores))
    _, _, _, threshold, actual_fraction, scores = max(candidates)
    return threshold, actual_fraction, scores


def build_method_predictions(aligned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    development = aligned[aligned["dataset"].eq(DEVELOPMENT_DATASET)].copy()
    neural_model, neural_development_metrics = choose_neural_partner(development)
    gate, development_delegation, gate_metrics = choose_cascade_gate(
        development, neural_model
    )

    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    feature_columns = ["scafe_prob", f"{neural_model}_prob"]
    stacker.fit(development[feature_columns], development["y_true"])

    rows: list[dict] = []
    for _, row in aligned.iterrows():
        scafe_prob = float(row["scafe_prob"])
        neural_prob = float(row[f"{neural_model}_prob"])
        delegate = abs(scafe_prob - 0.5) <= gate
        stack_prob = float(
            stacker.predict_proba(
                pd.DataFrame([[scafe_prob, neural_prob]], columns=feature_columns)
            )[0, 1]
        )
        probabilities = {
            "scafe": scafe_prob,
            **{
                model: float(row[f"{model}_prob"])
                for model in NEURAL_MODELS
            },
            "soft_vote": (scafe_prob + neural_prob) / 2.0,
            "stacking": stack_prob,
            "confidence_cascade": neural_prob if delegate else scafe_prob,
        }
        for method, probability in probabilities.items():
            rows.append({
                "dataset": row["dataset"],
                "role": (
                    "development"
                    if row["dataset"] == DEVELOPMENT_DATASET
                    else "external_target"
                ),
                "contract_id": row["contract_id"],
                "source_identity_hash": row["source_identity_hash"],
                "method": method,
                "neural_partner": neural_model,
                "y_true": int(row["y_true"]),
                "y_prob": probability,
                "y_pred": int(probability >= 0.5),
                "delegated_to_neural": bool(
                    delegate if method == "confidence_cascade" else False
                ),
            })
    details = {
        "selected_neural_partner": neural_model,
        "neural_selection_metric": "DAppSCAN sampled-LODO ROC-AUC; PR-AUC tie-break",
        "selected_neural_development_metrics": neural_development_metrics,
        "cascade_gate": {
            "definition": "delegate when abs(SCAFE probability - 0.5) <= threshold",
            "candidate_delegation_fractions": list(CASCADE_DELEGATION_FRACTIONS),
            "selected_distance_threshold": gate,
            "development_delegation_fraction": development_delegation,
            "development_metrics": gate_metrics,
        },
        "soft_vote": "unweighted arithmetic mean of SCAFE and selected-neural probabilities",
        "stacking": {
            "features": feature_columns,
            "estimator": "StandardScaler + class-weight-balanced logistic regression",
            "fit_dataset": DEVELOPMENT_DATASET,
            "random_state": 42,
        },
    }
    return pd.DataFrame(rows), details


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, role, method), group in predictions.groupby(
        ["dataset", "role", "method"], sort=False
    ):
        rows.append({
            "dataset": dataset,
            "role": role,
            "method": method,
            "n_contracts": int(len(group)),
            "positive_prevalence": float(group["y_true"].mean()),
            "delegation_fraction": (
                float(group["delegated_to_neural"].mean())
                if method == "confidence_cascade"
                else 0.0
            ),
            **metric_values(group["y_true"].to_numpy(), group["y_prob"].to_numpy()),
        })
    return pd.DataFrame(rows)


def bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Record-bootstrap target metrics and paired deltas versus SCAFE."""

    rng = np.random.default_rng(seed)
    rows = []
    for dataset in TARGET_DATASETS:
        subset = predictions[predictions["dataset"].eq(dataset)]
        pivot = subset.pivot(
            index="contract_id", columns="method", values=["y_true", "y_prob"]
        )
        y_true = pivot["y_true"]["scafe"].to_numpy(dtype=int)
        method_probs = {
            method: pivot["y_prob"][method].to_numpy(dtype=float)
            for method in pivot["y_prob"].columns
        }
        n = len(y_true)
        estimates = {method: {metric: [] for metric in METRICS}
                     for method in method_probs}
        deltas = {method: {metric: [] for metric in METRICS}
                  for method in method_probs if method != "scafe"}
        for _ in range(repetitions):
            indices = rng.integers(0, n, size=n)
            sampled_y = y_true[indices]
            if np.unique(sampled_y).size < 2:
                continue
            sampled_metrics = {
                method: metric_values(sampled_y, probability[indices])
                for method, probability in method_probs.items()
            }
            for method, values in sampled_metrics.items():
                for metric in METRICS:
                    estimates[method][metric].append(values[metric])
                    if method != "scafe":
                        deltas[method][metric].append(
                            values[metric] - sampled_metrics["scafe"][metric]
                        )
        for method in method_probs:
            point = metric_values(y_true, method_probs[method])
            for metric in METRICS:
                values = np.asarray(estimates[method][metric])
                row = {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_repetitions": int(len(values)),
                }
                if method != "scafe":
                    delta_values = np.asarray(deltas[method][metric])
                    row.update({
                        "delta_vs_scafe": float(
                            point[metric]
                            - metric_values(y_true, method_probs["scafe"])[metric]
                        ),
                        "delta_ci_lower": float(np.quantile(delta_values, 0.025)),
                        "delta_ci_upper": float(np.quantile(delta_values, 0.975)),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scafe-predictions", type=Path, default=DEFAULT_SCAFE)
    parser.add_argument("--dl-predictions", type=Path, default=DEFAULT_DL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _legacy_main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    scafe = pd.read_csv(args.scafe_predictions, low_memory=False)
    neural = pd.read_csv(args.dl_predictions, low_memory=False)
    aligned = align_predictions(scafe, neural)
    predictions, details = build_method_predictions(aligned)
    metrics = summarize(predictions)
    intervals = bootstrap_intervals(
        predictions,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output / "aligned_base_predictions.csv", index=False)
    predictions.to_csv(output / "ensemble_predictions.csv", index=False)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    intervals.to_csv(output / "ensemble_bootstrap_intervals.csv", index=False)
    protocol = {
        "version": 1,
        "scope": "post-hoc sampled LODO ensemble diagnostic",
        "publication_scope": "sampled estimates; not full-corpus results",
        "development_dataset": DEVELOPMENT_DATASET,
        "external_targets": list(TARGET_DATASETS),
        "target_labels_used_for_selection_or_fitting": False,
        "base_models_retrained": False,
        "threshold": 0.5,
        "bootstrap": {
            "unit": "record",
            "paired_against_scafe": True,
            "repetitions": int(args.bootstrap_repetitions),
            "seed": int(args.seed),
            "limitation": "does not model project, corpus, annotation, or meta-model uncertainty",
        },
        "inputs": {
            "scafe_predictions": str(args.scafe_predictions.resolve()),
            "scafe_sha256": _sha256(args.scafe_predictions.resolve()),
            "dl_predictions": str(args.dl_predictions.resolve()),
            "dl_sha256": _sha256(args.dl_predictions.resolve()),
        },
        "alignment": {
            "keys": ["dataset", "contract_id"],
            "records_per_dataset": {
                str(dataset): int(count)
                for dataset, count in aligned.groupby("dataset").size().items()
            },
        },
        **details,
    }
    (output / "ensemble_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(details, indent=2))


def main() -> None:
    raise RuntimeError(
        "Deprecated unsafe development protocol; run "
        "scripts/run_sampled_prediction_ensembles_v2.py develop, then evaluate"
    )


if __name__ == "__main__":
    main()
