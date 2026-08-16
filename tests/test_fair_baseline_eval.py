import json
import sys
import types

import pandas as pd

from src.ml.fair_baseline_eval import (
    _checkpoint_path,
    _load_checkpoint,
    _prediction_artifact_path,
    _write_checkpoint,
    aggregate_chunk_probabilities,
    chunk_tokenizer_ids,
    chunk_tokens,
    exclude_cross_dataset_duplicates,
    load_contract_predictions,
    reevaluate_contract_predictions,
    run_fair_lodo,
    split_lodo_contracts,
)


def test_regex_chunking_is_complete_and_non_overlapping():
    tokens = list(range(2050))
    chunks = chunk_tokens(tokens, 1024)
    assert [len(chunk) for chunk in chunks] == [1024, 1024, 2]
    assert [item for chunk in chunks for item in chunk] == tokens


def test_tokenizer_chunking_has_deterministic_512_windows():
    chunks = chunk_tokenizer_ids(list(range(1025)), 512)
    assert [len(chunk) for chunk in chunks] == [512, 512, 1]
    assert chunks[1][0] == 512


def test_chunk_probabilities_aggregate_to_contract_predictions():
    result = aggregate_chunk_probabilities(
        ["a", "a", "b"], [0.2, 0.8, 0.9]
    )
    assert result["contract_id"].tolist() == ["a", "b"]
    assert result["n_chunks"].tolist() == [2, 1]
    assert result["y_prob"].tolist() == [0.5, 0.9]


def test_lodo_removes_cross_dataset_duplicate_before_chunking(tmp_path):
    duplicate = tmp_path / "duplicate.sol"
    duplicate.write_text("contract Different { uint value; }")
    train_copy = tmp_path / "train.sol"
    train_copy.write_text("contract C { }")
    heldout = tmp_path / "heldout.sol"
    heldout.write_text("contract C { }")
    frame = pd.DataFrame(
        {
            "source_dataset": ["A", "B", "A"],
            "label_status": ["clean", "vulnerable", "vulnerable"],
            "contract_id": ["train-duplicate", "test", "train-keep"],
            "local_source_path": [str(train_copy), str(heldout), str(duplicate)],
        }
    )
    train, test, hashes = split_lodo_contracts(frame, "B")
    assert test["contract_id"].tolist() == ["test"]
    assert train["contract_id"].tolist() == ["train-keep"]
    assert hashes


def test_duplicate_filter_is_contract_row_operation(tmp_path):
    source = tmp_path / "same.sol"
    source.write_text("contract C {}")
    train = pd.DataFrame(
        {
            "contract_id": ["one", "two"],
            "local_source_path": [str(source), str(source)],
        }
    )
    test = pd.DataFrame(
        {"contract_id": ["heldout"], "local_source_path": [str(source)]}
    )
    filtered, _ = exclude_cross_dataset_duplicates(train, test)
    assert filtered.empty


def test_checkpoint_save_and_load_round_trip(tmp_path):
    path = _checkpoint_path(tmp_path, 42, "A", "textcnn")
    rows = [{"dataset": "A", "model": "textcnn", "contract_id": "c",
             "random_state": 42}]
    _write_checkpoint(path, {
        "version": 1, "random_state": 42, "held_out_dataset": "A",
        "model": "textcnn", "predictions": rows,
    })
    assert _load_checkpoint(
        path, seed=42, held_out="A", model="textcnn",
        expected_contract_ids=["c"],
    ) == rows


def test_resume_skips_valid_completed_unit(tmp_path, monkeypatch):
    import src.ml.fair_baseline_eval as evaluator

    frame = pd.DataFrame({
        "source_dataset": ["A", "B"],
        "label_status": ["clean", "vulnerable"],
        "contract_id": ["a", "b"],
        "local_source_path": ["", ""],
    })
    calls = []
    baselines = types.SimpleNamespace(tokenize_solidity=lambda source: [])
    monkeypatch.setitem(sys.modules, "src.ml.dl_baselines", baselines)
    monkeypatch.setattr(evaluator, "_fit_predict",
                        lambda *args, **kwargs: calls.append(args) or [0.9])
    evaluator.run_fair_lodo(
        frame, ["B"], ["textcnn"], [42], checkpoint_dir=tmp_path,
    )
    evaluator.run_fair_lodo(
        frame, ["B"], ["textcnn"], [42], checkpoint_dir=tmp_path,
    )
    assert len(calls) == 1


def test_corrupt_checkpoint_is_recomputed_and_replaced(tmp_path, monkeypatch):
    import src.ml.fair_baseline_eval as evaluator

    frame = pd.DataFrame({
        "source_dataset": ["A", "B"],
        "label_status": ["clean", "vulnerable"],
        "contract_id": ["a", "b"],
        "local_source_path": ["", ""],
    })
    baselines = types.SimpleNamespace(tokenize_solidity=lambda source: [])
    monkeypatch.setitem(sys.modules, "src.ml.dl_baselines", baselines)
    monkeypatch.setattr(evaluator, "_fit_predict", lambda *args, **kwargs: [0.9])
    evaluator.run_fair_lodo(
        frame, ["B"], ["textcnn"], [42], checkpoint_dir=tmp_path,
    )
    checkpoint = _checkpoint_path(tmp_path, 42, "B", "textcnn")
    checkpoint.write_text("{", encoding="utf-8")
    predictions, metadata = evaluator.run_fair_lodo(
        frame, ["B"], ["textcnn"], [42], checkpoint_dir=tmp_path,
    )
    assert len(predictions) == 1
    assert metadata["completed_units"][0]["n_predictions"] == 1
    json.loads(checkpoint.read_text(encoding="utf-8"))


def test_progress_status_and_contract_artifact_are_written(tmp_path, monkeypatch):
    import src.ml.fair_baseline_eval as evaluator

    train_source = tmp_path / "train.sol"
    test_source = tmp_path / "test.sol"
    train_source.write_text("contract Train {}", encoding="utf-8")
    test_source.write_text("contract Test {}", encoding="utf-8")
    frame = pd.DataFrame({
        "source_dataset": ["A", "B"],
        "label_status": ["clean", "vulnerable"],
        "contract_id": ["a", "b"],
        "local_source_path": [str(train_source), str(test_source)],
    })
    baselines = types.SimpleNamespace(tokenize_solidity=lambda source: [])
    monkeypatch.setitem(sys.modules, "src.ml.dl_baselines", baselines)
    progress_events = []
    original_write_progress = evaluator._write_progress
    def capture_progress(path, payload):
        progress_events.append(dict(payload))
        original_write_progress(path, payload)
    monkeypatch.setattr(evaluator, "_write_progress", capture_progress)

    def fake_fit(*args, **kwargs):
        for epoch in (1, 2):
            kwargs["progress_callback"](epoch)
        return [0.9]
    monkeypatch.setattr(evaluator, "_fit_predict", fake_fit)
    predictions, _ = run_fair_lodo(
        frame, ["B"], ["textcnn"], [42], epochs=2, checkpoint_dir=tmp_path,
    )

    status = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert status["phase"] == "unit_complete"
    assert status["completed_units"] == status["total_units"] == 1
    assert status["percent_units"] == 100.0
    assert status["train_contracts"] == 1
    assert [event["phase"] for event in progress_events] == [
        "unit_start", "epoch", "epoch", "unit_complete",
    ]
    assert [event["epoch"] for event in progress_events[1:3]] == [1, 2]
    artifact = _prediction_artifact_path(tmp_path, 42, "B", "textcnn")
    loaded = load_contract_predictions(artifact)
    assert loaded["contract_id"].tolist() == predictions["contract_id"].tolist()
    assert {"dataset", "model", "held_out_dataset", "source_identity_hash",
            "n_chunks", "y_true", "y_prob", "y_pred"}.issubset(loaded.columns)


def test_threshold_reevaluation_uses_saved_probabilities():
    frame = pd.DataFrame({
        "y_true": [0, 0, 1, 1],
        "y_prob": [0.1, 0.4, 0.6, 0.9],
    })
    metrics = reevaluate_contract_predictions(frame, threshold=0.7)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 2 / 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
