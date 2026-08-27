"""Duplicate-safe, contract-level leave-one-dataset-out DL baselines.

This module is intentionally independent from :mod:`src.ml.dl_baselines`.
The legacy evaluator retains its historical GroupKFold and test-set tuning
semantics; this evaluator uses a fixed 0.5 threshold and never reads held-out
labels while fitting or selecting an epoch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOKEN_WINDOW = 1024
BERT_WINDOW = 512
FIXED_THRESHOLD = 0.5
DEFAULT_MODELS = ("textcnn", "bilstm", "codebert", "unixcoder")
CHECKPOINT_VERSION = 1
PREDICTION_ARTIFACT_VERSION = 1
PROGRESS_FILENAME = "progress.json"
TRAIN_CHECKPOINT_INTERVAL = 25


def chunk_tokens(tokens: Sequence[Any], window_size: int = TOKEN_WINDOW) -> list[list[Any]]:
    """Return deterministic, non-overlapping windows covering all tokens."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    return [list(tokens[start:start + window_size])
            for start in range(0, len(tokens), window_size)] or [[]]


def chunk_tokenizer_ids(
    token_ids: Sequence[int], window_size: int = BERT_WINDOW
) -> list[list[int]]:
    """Chunk tokenizer IDs without overlap; special tokens are not inferred."""
    return chunk_tokens(token_ids, window_size)


def aggregate_chunk_probabilities(
    contract_ids: Sequence[str], chunk_probabilities: Sequence[float]
) -> pd.DataFrame:
    """Mean chunk probabilities into one row per contract, preserving order."""
    if len(contract_ids) != len(chunk_probabilities):
        raise ValueError("contract_ids and chunk_probabilities must have equal length")
    frame = pd.DataFrame({"contract_id": list(contract_ids),
                          "y_prob": np.asarray(chunk_probabilities, dtype=float)})
    if frame.empty:
        return frame
    return (frame.groupby("contract_id", sort=False, as_index=False)
            .agg(y_prob=("y_prob", "mean"), n_chunks=("y_prob", "size")))


def _source_hash(source: str) -> str:
    """Hash source after removing comments/whitespace, matching source identity."""
    source = re.sub(r"/\*.*?\*/", " ", source or "", flags=re.DOTALL)
    source = re.sub(r"//.*?$", " ", source, flags=re.MULTILINE)
    return hashlib.sha256(re.sub(r"\s+", " ", source).strip().encode()).hexdigest()


def source_hash_for_path(path: object) -> str:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ""
    path_obj = Path(raw_path)
    try:
        if path_obj.is_file():
            source = path_obj.read_text(encoding="utf-8", errors="replace")
        elif path_obj.is_dir():
            source = "\n".join(
                item.read_text(encoding="utf-8", errors="replace")
                for item in sorted(path_obj.rglob("*.sol"))
            )
        else:
            source = ""
    except (OSError, TypeError):
        source = ""
    if not source:
        return ""
    return _source_hash(source)


def exclude_cross_dataset_duplicates(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, set[str]]:
    """Remove complete training contracts duplicated by held-out source.

    The operation is performed on contract rows and must happen before any
    chunking.  Empty/unreadable sources are not treated as duplicates.
    """
    required = {"local_source_path", "contract_id"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        return train.copy(), set()
    test_hashes = {
        source_hash_for_path(path) for path in test["local_source_path"]
    } - {_source_hash("")}
    train_hashes = train["local_source_path"].map(source_hash_for_path)
    duplicate_hashes = set(train_hashes[train_hashes.isin(test_hashes)])
    return train.loc[~train_hashes.isin(duplicate_hashes)].copy(), duplicate_hashes


def _validate_contract_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_dataset", "label_status", "contract_id", "local_source_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("source_dataset", "label_status", "contract_id"):
        result[column] = result[column].fillna("").astype(str)
    result = result[result["label_status"].isin(["clean", "vulnerable", "unknown"])]
    result = result[result["label_status"].isin(["clean", "vulnerable"]) |
                    (result["source_dataset"] == "FORGE")]
    result["y"] = (result["label_status"] == "vulnerable").astype(int)
    result = _deduplicate_within_dataset(result)
    return result.reset_index(drop=True)


def _deduplicate_within_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per source identity within a dataset.

    Exact source duplicates with conflicting labels are excluded because no
    deterministic label choice is scientifically justified. Unreadable
    sources use contract_id as a conservative fallback identity.
    """
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["_source_identity"] = result["local_source_path"].map(source_hash_for_path)
    result["_source_identity"] = result["_source_identity"].where(
        result["_source_identity"].ne(""), "id:" + result["contract_id"]
    )
    key = ["source_dataset", "_source_identity"]
    conflicting = (
        result.groupby(key, sort=False)["label_status"].nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if len(conflicting):
        result = result.set_index(key)
        result = result.loc[~result.index.isin(conflicting)].reset_index()
    result = result.drop_duplicates(key, keep="first")
    return result.drop(columns="_source_identity")


def split_lodo_contracts(
    frame: pd.DataFrame, held_out_dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Split contracts, then remove cross-dataset source duplicates from train."""
    frame = _validate_contract_frame(frame)
    test = frame[frame["source_dataset"] == held_out_dataset].copy()
    train = frame[frame["source_dataset"] != held_out_dataset].copy()
    train, duplicate_hashes = exclude_cross_dataset_duplicates(train, test)
    train_ids = set(train["contract_id"])
    test_ids = set(test["contract_id"])
    if train_ids & test_ids:
        raise ValueError("contract_id overlap between train and held-out contracts")
    return train.reset_index(drop=True), test.reset_index(drop=True), duplicate_hashes


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _read_sources(frame: pd.DataFrame) -> list[str]:
    values = []
    for path in frame["local_source_path"]:
        raw_path = str(path or "").strip()
        if not raw_path:
            values.append("")
            continue
        path_obj = Path(raw_path)
        try:
            if path_obj.is_file():
                values.append(path_obj.read_text(encoding="utf-8", errors="replace"))
            elif path_obj.is_dir():
                values.append("\n".join(
                    item.read_text(encoding="utf-8", errors="replace")
                    for item in sorted(path_obj.rglob("*.sol"))
                ))
            else:
                values.append("")
        except OSError:
            values.append("")
    return values


def _internal_split(frame: pd.DataFrame, seed: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(frame))
    if len(indices) < 4 or frame["y"].nunique() < 2:
        return indices, indices
    try:
        return train_test_split(indices, test_size=validation_fraction,
                                random_state=seed, stratify=frame["y"])
    except ValueError:
        return train_test_split(indices, test_size=validation_fraction,
                                random_state=seed)


def _fit_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    progress_callback: Any | None = None,
    batch_progress_callback: Any | None = None,
    training_checkpoint_path: Path | None = None,
) -> np.ndarray:
    """Fit one model using only train/internal-validation data."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.ml.dl_baselines import (
        BiLSTMClassifier, CodeBERTClassifier, SolidityBERTDataset,
        SolidityTokenDataset, TextCNN, build_vocab, predict, tokenize_solidity,
        train_epoch,
    )
    _seed_everything(seed)
    is_bert = model_name in {"codebert", "unixcoder"}
    train_sources, test_sources = _read_sources(train), _read_sources(test)
    if is_bert:
        from transformers import AutoTokenizer
        hf_name = "microsoft/unixcoder-base" if model_name == "unixcoder" else "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        def encode(sources: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            for source in sources:
                ids = tokenizer(
                    source, add_special_tokens=False, verbose=False
                )["input_ids"]
                for window in chunk_tokenizer_ids(ids):
                    cls_id = tokenizer.cls_token_id
                    sep_id = tokenizer.sep_token_id
                    special_tokens = (
                        [cls_id] if cls_id is not None else []
                    ) + list(window) + (
                        [sep_id] if sep_id is not None else []
                    )
                    input_ids = special_tokens[:BERT_WINDOW]
                    attention = [1] * len(input_ids)
                    pad_id = tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = 0
                    padding = BERT_WINDOW - len(input_ids)
                    input_ids.extend([pad_id] * max(0, padding))
                    attention.extend([0] * max(0, padding))
                    rows.append((np.asarray(input_ids, dtype=np.int64),
                                 np.asarray(attention, dtype=np.int64)))
            return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
        X_train, M_train = encode(train_sources)
        X_test, M_test = encode(test_sources)
        y_train = np.repeat(train["y"].to_numpy(), [
            len(chunk_tokenizer_ids(tokenizer(s, add_special_tokens=False)["input_ids"]))
            for s in train_sources])
        y_test = np.repeat(test["y"].to_numpy(), [
            len(chunk_tokenizer_ids(tokenizer(s, add_special_tokens=False)["input_ids"]))
            for s in test_sources])
        train_ds = SolidityBERTDataset(X_train, M_train, y_train)
        test_ds = SolidityBERTDataset(X_test, M_test, y_test)
        model = CodeBERTClassifier(model_name=hf_name, freeze_layers=8).to("cpu")
    else:
        tokenized_train = [chunk for s in train_sources for chunk in
                           chunk_tokens(tokenize_solidity(s))]
        tokenized_test = [chunk for s in test_sources for chunk in
                          chunk_tokens(tokenize_solidity(s))]
        vocab = build_vocab(tokenized_train, min_freq=1)
        X_train = np.stack([np.pad([vocab.get(t, 1) for t in x][:TOKEN_WINDOW],
                                    (0, max(0, TOKEN_WINDOW - len(x))),
                                    constant_values=0)[:TOKEN_WINDOW] for x in tokenized_train])
        X_test = np.stack([np.pad([vocab.get(t, 1) for t in x][:TOKEN_WINDOW],
                                  (0, max(0, TOKEN_WINDOW - len(x))),
                                  constant_values=0)[:TOKEN_WINDOW] for x in tokenized_test])
        y_train = np.repeat(train["y"].to_numpy(), [len(chunk_tokens(tokenize_solidity(s))) for s in train_sources])
        y_test = np.repeat(test["y"].to_numpy(), [len(chunk_tokens(tokenize_solidity(s))) for s in test_sources])
        train_ds, test_ds = SolidityTokenDataset(X_train, y_train), SolidityTokenDataset(X_test, y_test)
        model_cls = TextCNN if model_name == "textcnn" else BiLSTMClassifier
        model = model_cls(vocab_size=len(vocab)).to("cpu")
    internal_train, internal_val = _internal_split(train, seed, 0.2)
    # Chunk indices are derived from contract indices, so validation remains contract-disjoint.
    train_contracts = set(internal_train.tolist())
    val_contracts = set(internal_val.tolist())
    def chunk_membership(contract_indices: set[int]) -> np.ndarray:
        counts = [len(chunk_tokenizer_ids(tokenizer(s, add_special_tokens=False)["input_ids"]))
                  if is_bert else len(chunk_tokens(tokenize_solidity(s)))
                  for s in train_sources]
        return np.array([i in contract_indices for i, count in enumerate(counts)
                         for _ in range(count)])
    tr_mask, va_mask = chunk_membership(train_contracts), chunk_membership(val_contracts)
    # A deterministic order is required for exact batch-level resumption.
    tr_loader = DataLoader(Subset(train_ds, np.flatnonzero(tr_mask)), batch_size=batch_size, shuffle=False)
    va_loader = DataLoader(Subset(train_ds, np.flatnonzero(va_mask)), batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 if is_bert else 1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    start_epoch, start_batch = 1, 0
    if training_checkpoint_path is not None:
        state = _load_training_checkpoint(training_checkpoint_path)
        if state and state.get("model_name") == model_name and state.get("seed") == int(seed):
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            best_state = state.get("best_state", best_state)
            best_loss = float(state.get("best_loss", best_loss))
            stale = int(state.get("stale", stale))
            start_epoch = int(state.get("epoch", 1))
            start_batch = int(state.get("batch", 0))
            if state.get("torch_rng_state") is not None:
                torch.set_rng_state(state["torch_rng_state"])
            if state.get("numpy_rng_state") is not None:
                np.random.set_state(state["numpy_rng_state"])
            if state.get("python_rng_state") is not None:
                random.setstate(state["python_rng_state"])
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        last_batch = epoch_start_batch
        def save_batch(batch: int, total: int) -> None:
            nonlocal last_batch
            last_batch = batch
            if (training_checkpoint_path is not None
                    and (batch % TRAIN_CHECKPOINT_INTERVAL == 0 or batch == total)):
                _write_training_checkpoint(training_checkpoint_path, {
                    "version": 1, "model_name": model_name, "seed": int(seed),
                    "epoch": int(epoch), "batch": int(batch),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_state": best_state, "best_loss": best_loss, "stale": stale,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                })
        def report_train_batch(batch: int, total: int) -> None:
            save_batch(batch, total)
            if batch_progress_callback is not None:
                batch_progress_callback(epoch, batch, total)
        train_epoch(
            model, tr_loader, optimizer, criterion, "cpu", is_bert=is_bert,
            progress_callback=report_train_batch,
            start_batch=epoch_start_batch,
        )
        val_prob = predict(model, va_loader, "cpu", is_bert=is_bert)
        val_y = y_train[va_mask]
        val_loss = float(torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(val_prob).float(), torch.from_numpy(val_y).float()))
        if val_loss < best_loss:
            best_state, best_loss, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= patience:
                if training_checkpoint_path is not None:
                    _write_training_checkpoint(training_checkpoint_path, {
                        "version": 1, "model_name": model_name, "seed": int(seed),
                        "epoch": int(epoch + 1), "batch": 0,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_state": best_state, "best_loss": best_loss, "stale": stale,
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                    })
                if progress_callback is not None:
                    progress_callback(epoch)
                break
        if progress_callback is not None:
            progress_callback(epoch)
        if training_checkpoint_path is not None:
            _write_training_checkpoint(training_checkpoint_path, {
                "version": 1, "model_name": model_name, "seed": int(seed),
                "epoch": int(epoch + 1), "batch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state, "best_loss": best_loss, "stale": stale,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            })
        start_batch = 0
    model.load_state_dict(best_state)
    return predict(model, DataLoader(test_ds, batch_size=batch_size), "cpu", is_bert=is_bert)


def _checkpoint_path(checkpoint_dir: Path, seed: int, held_out: str, model: str) -> Path:
    unit = json.dumps([int(seed), str(held_out), str(model)], separators=(",", ":"))
    digest = hashlib.sha256(unit.encode("utf-8")).hexdigest()[:24]
    return checkpoint_dir / f"{digest}.json"


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a checkpoint atomically so readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _training_checkpoint_path(
    checkpoint_dir: Path, seed: int, held_out: str, model: str
) -> Path:
    return checkpoint_dir / (
        f"{_checkpoint_path(checkpoint_dir, seed, held_out, model).stem}.training.pt"
    )


def _write_training_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist tensor-bearing training state."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _load_training_checkpoint(path: Path) -> dict[str, Any] | None:
    import torch

    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, EOFError, RuntimeError, ValueError, TypeError):
        return None


def _prediction_artifact_path(
    checkpoint_dir: Path, seed: int, held_out: str, model: str
) -> Path:
    return checkpoint_dir / f"{_checkpoint_path(checkpoint_dir, seed, held_out, model).stem}.predictions.json"


def _incremental_prediction_artifact_path(
    checkpoint_dir: Path, seed: int, held_out: str, model: str
) -> Path:
    """Return the append-only contract prediction path for an active unit."""
    return checkpoint_dir / (
        f"{_checkpoint_path(checkpoint_dir, seed, held_out, model).stem}.predictions.jsonl"
    )


_PREDICTION_FIELDS = {
    "dataset", "model", "held_out_dataset", "random_state", "contract_id",
    "source_identity_hash", "n_chunks", "y_true", "y_prob", "y_pred",
}


def _prediction_identity(row: Mapping[str, Any]) -> str:
    source_hash = str(row.get("source_identity_hash", ""))
    return source_hash if source_hash else str(row.get("contract_id", ""))


def _load_incremental_predictions(
    path: Path,
    *,
    seed: int,
    held_out: str,
    model: str,
    expected_contract_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load valid completed contract rows, tolerating a truncated final line."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    expected_ids = {str(value) for value in expected_contract_ids}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed during a write can leave only this line
                    # incomplete; previously flushed records remain usable.
                    continue
                contract_id = str(row.get("contract_id", ""))
                if (
                    not _PREDICTION_FIELDS.issubset(row)
                    or _prediction_identity(row) not in expected_ids
                    or row.get("dataset") != held_out
                    or row.get("held_out_dataset") != held_out
                    or row.get("model") != model
                    or row.get("random_state") != int(seed)
                ):
                    continue
                rows[_prediction_identity(row)] = dict(row)
    except OSError:
        return {}
    return rows


def _append_prediction(path: Path, row: Mapping[str, Any]) -> None:
    """Durably append one contract result, flushing it before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(row), separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_prediction_artifact(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Persist contract-level predictions atomically for independent re-analysis."""
    _write_checkpoint(path, {
        "version": PREDICTION_ARTIFACT_VERSION,
        "predictions": [dict(row) for row in rows],
    })


def load_contract_predictions(path: Path | str) -> pd.DataFrame:
    """Load a durable per-contract prediction artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["predictions"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("prediction artifact must contain a list of predictions")
    if any(not _PREDICTION_FIELDS.issubset(row) for row in rows):
        raise ValueError("prediction artifact is missing required contract fields")
    return pd.DataFrame(rows)


def reevaluate_contract_predictions(
    predictions: pd.DataFrame | Path | str, threshold: float = FIXED_THRESHOLD
) -> dict[str, float]:
    """Recompute classification metrics without retraining."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    frame = load_contract_predictions(predictions) if isinstance(predictions, (Path, str)) else predictions.copy()
    required = {"y_true", "y_prob"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required.difference(frame.columns))}")
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    y_true = frame["y_true"].astype(int).to_numpy()
    y_prob = frame["y_prob"].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    _write_checkpoint(path, payload)


def _ensure_console_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)


def _load_checkpoint(
    path: Path,
    *,
    seed: int,
    held_out: str,
    model: str,
    expected_contract_ids: Sequence[str],
) -> list[dict[str, Any]] | None:
    """Return rows only for a complete, matching checkpoint."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("version") != CHECKPOINT_VERSION
                or payload.get("random_state") != int(seed)
                or payload.get("held_out_dataset") != str(held_out)
                or payload.get("model") != str(model)):
            return None
        rows = payload["predictions"]
        if not isinstance(rows, list):
            return None
        expected = [str(value) for value in expected_contract_ids]
        actual = [_prediction_identity(row) for row in rows]
        if actual != expected or any(
            row.get("dataset") != held_out
            or row.get("model") != model
            or row.get("random_state") != int(seed)
            for row in rows
        ):
            return None
        return rows
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def run_fair_lodo(
    frame: pd.DataFrame,
    datasets: Sequence[str],
    models: Sequence[str] = DEFAULT_MODELS,
    seeds: Sequence[int] = (42,),
    *,
    max_contracts: int | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    patience: int = 3,
    checkpoint_dir: Path | str | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run fair LODO and return contract-level predictions plus protocol metadata."""
    frame = _validate_contract_frame(frame)
    if max_contracts is not None:
        frame = frame.groupby("source_dataset", sort=False).head(max_contracts).reset_index(drop=True)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        _ensure_console_logging()
    rows: list[dict[str, Any]] = []
    completed_units: list[dict[str, Any]] = []
    total_units = len(seeds) * len(datasets) * len(models)
    progress_path = checkpoint_root / PROGRESS_FILENAME if checkpoint_root is not None else None
    completed_count = 0
    elapsed_start = time.monotonic()
    for seed in seeds:
        for held_out in datasets:
            train, test, duplicates = split_lodo_contracts(frame, held_out)
            for model in models:
                unit_number = completed_count + 1
                checkpoint = (
                    _checkpoint_path(checkpoint_root, seed, held_out, model)
                    if checkpoint_root is not None else None
                )
                artifact = (
                    _prediction_artifact_path(checkpoint_root, seed, held_out, model)
                    if checkpoint_root is not None else None
                )
                incremental_artifact = (
                    _incremental_prediction_artifact_path(
                        checkpoint_root, seed, held_out, model
                    )
                    if checkpoint_root is not None else None
                )
                training_artifact = (
                    _training_checkpoint_path(checkpoint_root, seed, held_out, model)
                    if checkpoint_root is not None else None
                )
                expected_contract_ids = [
                    _prediction_identity({
                        "contract_id": contract["contract_id"],
                        "source_identity_hash": source_hash_for_path(
                            contract["local_source_path"]
                        ),
                    })
                    for _, contract in test.iterrows()
                ]
                partial_rows = (
                    _load_incremental_predictions(
                        incremental_artifact,
                        seed=seed,
                        held_out=held_out,
                        model=model,
                        expected_contract_ids=expected_contract_ids,
                    )
                    if incremental_artifact is not None and resume
                    else {}
                )
                progress_base = {
                    "seed": int(seed), "held_out_dataset": str(held_out),
                    "model": str(model), "epochs": int(epochs),
                    "train_contracts": int(len(train)), "test_contracts": int(len(test)),
                    "completed_units": int(completed_count), "total_units": int(total_units),
                    "percent_units": float(100 * completed_count / total_units) if total_units else 100.0,
                }
                if progress_path is not None:
                    _write_progress(progress_path, {
                        **progress_base, "phase": "unit_start", "epoch": 0,
                        "elapsed_seconds": float(time.monotonic() - elapsed_start),
                    })
                logger.info(
                    "Starting unit %d/%d: seed=%s held_out=%s model=%s train=%d test=%d",
                    unit_number, total_units, seed, held_out, model, len(train), len(test),
                )
                unit_rows = (
                    _load_checkpoint(
                        checkpoint, seed=seed, held_out=held_out, model=model,
                        expected_contract_ids=test["contract_id"].tolist(),
                    )
                    if checkpoint is not None and resume and checkpoint.is_file()
                    else None
                )
                if unit_rows is not None:
                    # Backfill fields for checkpoints created before artifacts
                    # were introduced while preserving resume compatibility.
                    for row in unit_rows:
                        row.setdefault("held_out_dataset", str(held_out))
                if unit_rows is None:
                    def report_epoch(epoch: int) -> None:
                        elapsed = float(time.monotonic() - elapsed_start)
                        payload = {**progress_base, "phase": "epoch", "epoch": int(epoch),
                                   "elapsed_seconds": elapsed}
                        if progress_path is not None:
                            _write_progress(progress_path, payload)
                        logger.info("Unit %d/%d: epoch %d/%d", unit_number, total_units, epoch, epochs)

                    def report_batch(epoch: int, batch: int, total_batches: int) -> None:
                        if batch != 1 and batch != total_batches and batch % 25 != 0:
                            return
                        elapsed = float(time.monotonic() - elapsed_start)
                        payload = {
                            **progress_base, "phase": "batch", "epoch": int(epoch),
                            "batch": int(batch), "total_batches": int(total_batches),
                            "percent_batches": float(100 * batch / total_batches)
                            if total_batches else 100.0,
                            "elapsed_seconds": elapsed,
                        }
                        if progress_path is not None:
                            _write_progress(progress_path, payload)
                        logger.info(
                            "Unit %d/%d: epoch %d/%d batch %d/%d",
                            unit_number, total_units, epoch, epochs, batch, total_batches,
                        )

                    probs = _fit_predict(model, train, test, seed, epochs=epochs,
                                         batch_size=batch_size, patience=patience,
                                         progress_callback=report_epoch,
                                         batch_progress_callback=report_batch,
                                         training_checkpoint_path=training_artifact)
                    counts = [len(chunk_tokenizer_ids([]))] * len(test)
                    # Reconstruct counts without retaining model-specific intermediate arrays.
                    if model in {"codebert", "unixcoder"}:
                        from transformers import AutoTokenizer
                        name = "microsoft/unixcoder-base" if model == "unixcoder" else "microsoft/codebert-base"
                        tok = AutoTokenizer.from_pretrained(name)
                        counts = [len(chunk_tokenizer_ids(tok(s, add_special_tokens=False)["input_ids"])) for s in _read_sources(test)]
                    else:
                        from src.ml.dl_baselines import tokenize_solidity
                        counts = [len(chunk_tokens(tokenize_solidity(s))) for s in _read_sources(test)]
                    cursor = 0
                    unit_rows = []
                    for (_, contract), n_chunks in zip(test.iterrows(), counts):
                        contract_prob = float(np.mean(probs[cursor:cursor + n_chunks]))
                        cursor += n_chunks
                        contract_id = str(contract["contract_id"])
                        source_identity_hash = source_hash_for_path(
                            contract["local_source_path"]
                        )
                        row_identity = _prediction_identity({
                            "contract_id": contract_id,
                            "source_identity_hash": source_identity_hash,
                        })
                        row = partial_rows.get(row_identity)
                        if row is None:
                            row = {"dataset": held_out, "model": model,
                                   "contract_id": contract_id,
                                   "source_identity_hash": source_identity_hash,
                                   "n_chunks": n_chunks, "y_true": int(contract["y"]),
                                   "y_prob": contract_prob,
                                   "y_pred": int(contract_prob >= FIXED_THRESHOLD),
                                   "random_state": seed,
                                   "held_out_dataset": held_out}
                            if incremental_artifact is not None:
                                _append_prediction(incremental_artifact, row)
                        unit_rows.append(row)
                    if checkpoint is not None:
                        _write_checkpoint(checkpoint, {
                            "version": CHECKPOINT_VERSION,
                            "random_state": int(seed),
                            "held_out_dataset": str(held_out),
                            "model": str(model),
                            "predictions": unit_rows,
                        })
                    if artifact is not None:
                        _write_prediction_artifact(artifact, unit_rows)
                    if training_artifact is not None and training_artifact.is_file():
                        training_artifact.unlink()
                elif artifact is not None and not artifact.is_file():
                    _write_prediction_artifact(artifact, unit_rows)
                rows.extend(unit_rows)
                completed_count += 1
                completed_units.append({
                    "random_state": int(seed), "held_out_dataset": str(held_out),
                    "model": str(model), "n_predictions": len(unit_rows),
                })
                if progress_path is not None:
                    _write_progress(progress_path, {
                        **progress_base, "phase": "unit_complete", "epoch": int(epochs),
                        "elapsed_seconds": float(time.monotonic() - elapsed_start),
                        "completed_units": int(completed_count),
                        "percent_units": float(100 * completed_count / total_units) if total_units else 100.0,
                    })
                logger.info("Completed unit %d/%d: seed=%s held_out=%s model=%s",
                            completed_count, total_units, seed, held_out, model)
    predictions = pd.DataFrame(rows)
    completed_units = list({
        (unit["random_state"], unit["held_out_dataset"], unit["model"]): unit
        for unit in completed_units
    }.values())
    if not predictions.empty:
        predictions["_prediction_identity"] = predictions.apply(
            _prediction_identity, axis=1
        )
        unit_columns = ["random_state", "dataset", "model", "_prediction_identity"]
        predictions = predictions.drop_duplicates(unit_columns, keep="first").reset_index(drop=True)
        predictions = predictions.drop(columns="_prediction_identity")
    metadata = {"protocol": "duplicate_safe_contract_lodo_v1",
                "chunking": {"regex_tokens": TOKEN_WINDOW, "tokenizer_tokens": BERT_WINDOW,
                             "overlap": 0, "aggregation": "mean"},
                "threshold": FIXED_THRESHOLD, "held_out_threshold_tuning": False,
                "internal_validation_for_early_stopping": True,
                "datasets": list(datasets), "models": list(models), "seeds": list(seeds),
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_dir": str(checkpoint_root) if checkpoint_root is not None else None,
                "resume": bool(resume), "completed_units": completed_units}
    return predictions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("data/derived/fair_baseline_predictions.csv"))
    parser.add_argument("--metadata-json", type=Path, default=Path("data/derived/fair_baseline_protocol.json"))
    parser.add_argument("--datasets", default="DAppSCAN,FORGE")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-contracts", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("data/derived/fair_baseline_checkpoints"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    from src.ml.features import load_scored_csv
    frame = load_scored_csv(args.input_csv)
    predictions, metadata = run_fair_lodo(
        frame, [x.strip() for x in args.datasets.split(",") if x.strip()],
        [x.strip() for x in args.models.split(",") if x.strip()],
        [int(x) for x in args.seeds.split(",") if x.strip()],
        max_contracts=args.max_contracts, epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir, resume=args.resume)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
