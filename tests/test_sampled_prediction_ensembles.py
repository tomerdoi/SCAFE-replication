import json

import numpy as np
import pandas as pd

import scripts.run_sampled_prediction_ensembles_v2 as ensemble_v2
from scripts.run_sampled_prediction_ensembles import (
    NEURAL_MODELS,
    align_predictions,
    build_method_predictions,
    summarize,
)


def _frames():
    scafe_rows = []
    neural_rows = []
    for dataset in ("DAppSCAN", "DIVE", "Empirical-Analysis"):
        for index in range(500):
            label = int(index % 5 == 0)
            contract_id = f"{dataset}-{index}"
            scafe_rows.append({
                "dataset": dataset,
                "model": "rf",
                "contract_id": contract_id,
                "y_true": label,
                "y_prob": 0.8 if label else 0.2,
                "random_state": 42,
            })
            for model_index, model in enumerate(NEURAL_MODELS):
                neural_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "contract_id": contract_id,
                    "source_identity_hash": f"hash-{dataset}-{index}",
                    "y_true": label,
                    "y_prob": (0.7 if label else 0.3) + model_index * 0.001,
                    "random_state": 42,
                })
    return pd.DataFrame(scafe_rows), pd.DataFrame(neural_rows)


def test_alignment_and_ensemble_outputs_are_complete():
    scafe, neural = _frames()
    aligned = align_predictions(scafe, neural)
    assert aligned.groupby("dataset").size().eq(500).all()
    predictions, details = build_method_predictions(aligned)
    assert details["selected_neural_partner"] in NEURAL_MODELS
    assert predictions.groupby(["dataset", "method"]).size().eq(500).all()
    metrics = summarize(predictions)
    assert len(metrics) == 24
    assert set(metrics["role"]) == {"development", "external_target"}


def test_v2_development_writes_target_isolated_lock(tmp_path, monkeypatch):
    labels = np.tile([0, 0, 0, 0, 1], 300)
    aligned = pd.DataFrame({
        "contract_id": [f"contract-{index}" for index in range(1_500)],
        "source_identity_hash": [f"hash-{index}" for index in range(1_500)],
        "group": [f"project-{index % 20}" for index in range(1_500)],
        "y_true": labels,
        "scafe_prob": np.where(labels, 0.8, 0.2),
        "textcnn_prob": np.where(labels, 0.75, 0.25),
        "bilstm_prob": np.where(labels, 0.7, 0.3),
        "codebert_prob": np.where(labels, 0.65, 0.35),
        "unixcoder_prob": np.where(labels, 0.6, 0.4),
    })
    monkeypatch.setattr(
        ensemble_v2,
        "align_development",
        lambda _scafe, _neural, *, scafe_model: aligned,
    )
    scafe_path = tmp_path / "within_scafe.csv"
    neural_path = tmp_path / "within_neural.csv"
    pd.DataFrame({"placeholder": [1]}).to_csv(scafe_path, index=False)
    pd.DataFrame({"placeholder": [1]}).to_csv(neural_path, index=False)

    lock = ensemble_v2.develop(scafe_path, neural_path, tmp_path / "output")
    written = json.loads(
        (tmp_path / "output" / "ensemble_lock.json").read_text(encoding="utf-8")
    )

    assert lock == written
    assert written["target_artifacts_read_during_development"] is False
    assert written["scafe_model"] == "rf"
    assert written["development_records"] == 1_500
    assert "external_inputs" not in written
