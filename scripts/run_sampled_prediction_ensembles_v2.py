#!/usr/bin/env python3
"""Develop on DAppSCAN OOF predictions, then evaluate frozen ensembles externally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_sampled_prediction_ensembles import (  # noqa: E402
    CASCADE_DELEGATION_FRACTIONS,
    DEVELOPMENT_DATASET,
    NEURAL_MODELS,
    TARGET_DATASETS,
    align_predictions as align_external_predictions,
    bootstrap_intervals,
    metric_values,
    summarize,
)
from scripts.summarize_sampled_within_models import align as align_development  # noqa: E402


DEFAULT_WITHIN_SCAFE = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1/"
    "scafe/scafe_within_corpus_predictions.csv"
)
DEFAULT_WITHIN_DL = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_campaign_v1/"
    "dl/dl_within_corpus_oof_predictions.csv"
)
DEFAULT_EXTERNAL_SCAFE = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/"
    "scafe/scafe_repeated_lodo_predictions.csv"
)
DEFAULT_EXTERNAL_DL = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/"
    "dl/predictions.csv"
)
DEFAULT_OUTPUT = (
    REPO / "data/derived/sanitized_v1/ml_results/sampled_ensemble_v2"
)
EPSILON = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logits(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    probabilities = np.clip(frame[columns].to_numpy(dtype=float), EPSILON, 1 - EPSILON)
    return np.log(probabilities / (1 - probabilities))


def develop(
    within_scafe_path: Path,
    within_dl_path: Path,
    output: Path,
    *,
    scafe_model: str = "rf",
) -> dict:
    """Select and fit every ensemble component without reading external artifacts."""

    aligned = align_development(
        pd.read_csv(within_scafe_path, low_memory=False),
        pd.read_csv(within_dl_path, low_memory=False),
        scafe_model=scafe_model,
    )
    model_metrics = {
        model: metric_values(
            aligned["y_true"].to_numpy(),
            aligned[f"{model}_prob"].to_numpy(),
        )
        for model in NEURAL_MODELS
    }
    partner = max(
        NEURAL_MODELS,
        key=lambda model: (
            model_metrics[model]["roc_auc"],
            model_metrics[model]["pr_auc"],
        ),
    )

    distance = np.abs(aligned["scafe_prob"].to_numpy() - 0.5)
    gate_candidates = []
    for fraction in CASCADE_DELEGATION_FRACTIONS:
        threshold = float(np.quantile(distance, fraction, method="higher"))
        delegate = distance <= threshold
        probability = np.where(
            delegate,
            aligned[f"{partner}_prob"].to_numpy(),
            aligned["scafe_prob"].to_numpy(),
        )
        scores = metric_values(aligned["y_true"].to_numpy(), probability)
        gate_candidates.append({
            "candidate_fraction": float(fraction),
            "threshold": threshold,
            "actual_fraction": float(delegate.mean()),
            **scores,
        })
    selected_gate = max(
        gate_candidates,
        key=lambda row: (row["roc_auc"], row["pr_auc"], -row["actual_fraction"]),
    )

    feature_columns = ["scafe_prob", f"{partner}_prob"]
    x = _logits(aligned, feature_columns)
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    stacker = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
        solver="lbfgs",
        C=1.0,
    ).fit(standardized, aligned["y_true"])

    output.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output / "development_aligned_predictions.csv", index=False)
    pd.DataFrame([
        {"model": model, **scores} for model, scores in model_metrics.items()
    ]).to_csv(output / "development_selection.csv", index=False)
    pd.DataFrame(gate_candidates).to_csv(
        output / "development_cascade_candidates.csv", index=False
    )
    lock = {
        "version": 2,
        "development_dataset": DEVELOPMENT_DATASET,
        "scafe_model": scafe_model,
        "external_targets": list(TARGET_DATASETS),
        "development_protocol": "sampled five-fold project-group-disjoint OOF",
        "development_records": int(len(aligned)),
        "development_positive": int(aligned["y_true"].sum()),
        "selection_metric": "ROC-AUC; PR-AUC tie-break",
        "selected_neural_partner": partner,
        "selected_neural_metrics": model_metrics[partner],
        "cascade": {
            "definition": "delegate when abs(SCAFE probability - 0.5) <= threshold",
            "candidate_delegation_fractions": list(CASCADE_DELEGATION_FRACTIONS),
            "selected": selected_gate,
        },
        "soft_vote": "unweighted arithmetic mean",
        "stacking": {
            "input_columns": feature_columns,
            "probability_transform": f"clipped logit with epsilon={EPSILON}",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficient": stacker.coef_[0].tolist(),
            "intercept": float(stacker.intercept_[0]),
            "class_weight": "balanced",
            "C": 1.0,
            "solver": "lbfgs",
            "random_state": 42,
        },
        "development_inputs": {
            "scafe": str(within_scafe_path.resolve()),
            "scafe_sha256": _sha256(within_scafe_path.resolve()),
            "dl": str(within_dl_path.resolve()),
            "dl_sha256": _sha256(within_dl_path.resolve()),
        },
        "target_artifacts_read_during_development": False,
    }
    (output / "ensemble_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def _stack_probability(
    frame: pd.DataFrame,
    partner: str,
    stacking: dict,
) -> np.ndarray:
    x = _logits(frame, ["scafe_prob", f"{partner}_prob"])
    mean = np.asarray(stacking["scaler_mean"], dtype=float)
    scale = np.asarray(stacking["scaler_scale"], dtype=float)
    coefficient = np.asarray(stacking["coefficient"], dtype=float)
    linear = ((x - mean) / scale) @ coefficient + float(stacking["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -50, 50)))


def evaluate(
    lock_path: Path,
    external_scafe_path: Path,
    external_dl_path: Path,
    output: Path,
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("version") != 2
        or lock.get("target_artifacts_read_during_development") is not False
    ):
        raise ValueError("Invalid or unsealed ensemble-development lock")
    partner = str(lock["selected_neural_partner"])
    aligned = align_external_predictions(
        pd.read_csv(external_scafe_path, low_memory=False),
        pd.read_csv(external_dl_path, low_memory=False),
        scafe_model=str(lock.get("scafe_model", "rf")),
    )
    external = aligned[aligned["dataset"].isin(TARGET_DATASETS)].copy()
    stack_probability = _stack_probability(external, partner, lock["stacking"])
    gate = float(lock["cascade"]["selected"]["threshold"])
    distance = np.abs(external["scafe_prob"].to_numpy() - 0.5)
    delegate = distance <= gate
    neural_probability = external[f"{partner}_prob"].to_numpy(dtype=float)
    scafe_probability = external["scafe_prob"].to_numpy(dtype=float)

    rows: list[dict] = []
    for position, (_, row) in enumerate(external.iterrows()):
        probabilities = {
            "scafe": scafe_probability[position],
            **{
                model: float(row[f"{model}_prob"])
                for model in NEURAL_MODELS
            },
            "soft_vote": (
                scafe_probability[position] + neural_probability[position]
            ) / 2.0,
            "stacking": stack_probability[position],
            "confidence_cascade": (
                neural_probability[position]
                if delegate[position]
                else scafe_probability[position]
            ),
        }
        for method, probability in probabilities.items():
            rows.append({
                "dataset": str(row["dataset"]),
                "role": "external_target",
                "contract_id": str(row["contract_id"]),
                "source_identity_hash": str(row["source_identity_hash"]),
                "method": method,
                "neural_partner": partner,
                "y_true": int(row["y_true"]),
                "y_prob": float(probability),
                "y_pred": int(probability >= 0.5),
                "delegated_to_neural": bool(
                    delegate[position] if method == "confidence_cascade" else False
                ),
            })
    predictions = pd.DataFrame(rows)
    metrics = summarize(predictions)
    intervals = bootstrap_intervals(
        predictions,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )

    output.mkdir(parents=True, exist_ok=True)
    external.to_csv(output / "external_aligned_predictions.csv", index=False)
    predictions.to_csv(output / "ensemble_predictions.csv", index=False)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    intervals.to_csv(output / "ensemble_bootstrap_intervals.csv", index=False)
    protocol = {
        "version": 2,
        "scope": "post-hoc sampled LODO ensemble diagnostic",
        "publication_scope": "sampled estimates; not full-corpus results",
        "lock_sha256": _sha256(lock_path),
        "development_dataset": DEVELOPMENT_DATASET,
        "external_targets": list(TARGET_DATASETS),
        "target_labels_used_for_selection_or_fitting": False,
        "base_models_retrained": False,
        "selected_neural_partner": partner,
        "cascade_gate": gate,
        "bootstrap": {
            "unit": "record",
            "paired_against_scafe": True,
            "repetitions": bootstrap_repetitions,
            "seed": seed,
            "limitation": (
                "conditional on frozen samples/base predictions; does not model "
                "project, corpus, annotation, seed, or meta-model uncertainty"
            ),
        },
        "external_inputs": {
            "scafe": str(external_scafe_path.resolve()),
            "scafe_sha256": _sha256(external_scafe_path.resolve()),
            "dl": str(external_dl_path.resolve()),
            "dl_sha256": _sha256(external_dl_path.resolve()),
        },
        "records_per_target": {
            str(dataset): int(count)
            for dataset, count in external.groupby("dataset").size().items()
        },
    }
    (output / "ensemble_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics, intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop_parser = subparsers.add_parser("develop")
    develop_parser.add_argument("--within-scafe", type=Path, default=DEFAULT_WITHIN_SCAFE)
    develop_parser.add_argument("--within-dl", type=Path, default=DEFAULT_WITHIN_DL)
    develop_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    develop_parser.add_argument("--scafe-model", default="rf")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--lock", type=Path, default=DEFAULT_OUTPUT / "ensemble_lock.json"
    )
    evaluate_parser.add_argument(
        "--external-scafe", type=Path, default=DEFAULT_EXTERNAL_SCAFE
    )
    evaluate_parser.add_argument("--external-dl", type=Path, default=DEFAULT_EXTERNAL_DL)
    evaluate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "develop":
        lock = develop(
            args.within_scafe,
            args.within_dl,
            args.output_dir,
            scafe_model=args.scafe_model,
        )
        print(json.dumps({
            "selected_neural_partner": lock["selected_neural_partner"],
            "cascade": lock["cascade"]["selected"],
        }, indent=2))
        return
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    metrics, _ = evaluate(
        args.lock,
        args.external_scafe,
        args.external_dl,
        args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
