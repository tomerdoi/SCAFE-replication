"""Strict, source-safe within-corpus DL baseline evaluation.

This evaluator intentionally does not import the LODO split/training protocol.
It shares only architecture and encoding primitives with the legacy baseline
implementation.  Every outer test contract is evaluated from all of its
non-overlapping source windows and is never used for model selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.ml.fair_baseline_eval import (
    BERT_WINDOW, FIXED_THRESHOLD, TOKEN_WINDOW, _read_sources, chunk_tokenizer_ids,
    chunk_tokens, source_hash_for_path,
)

DEFAULT_MODELS = ("textcnn", "bilstm", "codebert", "unixcoder")
CHECKPOINT_VERSION = 1
TRAIN_CHECKPOINT_INTERVAL = 200
PREPROCESSING_VERSION = 2


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
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


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize binary labels and exclude same-dataset source conflicts."""
    required = {"source_dataset", "label_status", "contract_id", "local_source_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for column in required:
        result[column] = result[column].fillna("").astype(str)
    result = result[result["label_status"].isin(("clean", "vulnerable"))].copy()
    result["_source_identity"] = result["local_source_path"].map(source_hash_for_path)
    result["_source_identity"] = result["_source_identity"].where(
        result["_source_identity"].ne(""), "contract:" + result["contract_id"]
    )
    key = ["source_dataset", "_source_identity"]
    conflict_keys = result.groupby(key)["label_status"].nunique().loc[lambda s: s > 1].index
    conflict_mask = result.set_index(key).index.isin(conflict_keys)
    conflicts = int(conflict_mask.sum())
    result = result.loc[~conflict_mask].copy()
    before = len(result)
    result = result.drop_duplicates(key, keep="first").reset_index(drop=True)
    result["y"] = (result["label_status"] == "vulnerable").astype(int)
    return result, {
        "binary_label_rows_retained": int(len(result)),
        "conflicting_label_rows_excluded": conflicts,
        "same_dataset_duplicate_rows_excluded": int(before - len(result)),
    }


def groups_for(frame: pd.DataFrame, dataset: str) -> np.ndarray:
    """Return project-disjoint DAppSCAN or source-disjoint groups."""
    source = frame["_source_identity"].astype(str)
    if dataset == "DAppSCAN" and "source_subdataset" in frame:
        project = frame["source_subdataset"].fillna("").astype(str).str.strip()
        key = project.where(project.ne(""), source)
    else:
        key = source
    return pd.factorize(key, sort=True)[0]


def group_disjoint_inner_split(
    frame: pd.DataFrame, groups: np.ndarray, seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Choose validation only from outer training, with no group overlap."""
    indices = np.arange(len(frame))
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("An outer training fold needs at least two groups for validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_index, validation_index = next(splitter.split(indices, frame["y"], groups))
    if set(groups[train_index]) & set(groups[validation_index]):
        raise AssertionError("inner train/validation groups overlap")
    return train_index, validation_index


def aggregate_contract_probabilities(
    contract_ids: Sequence[str], probabilities: Sequence[float]
) -> pd.DataFrame:
    from src.ml.fair_baseline_eval import aggregate_chunk_probabilities
    return aggregate_chunk_probabilities(contract_ids, probabilities)


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unit_stem(dataset: str, seed: int, model: str, fold: int) -> str:
    return _stable_hash([dataset, int(seed), model, int(fold)])[:24]


def _paths(root: Path, dataset: str, seed: int, model: str, fold: int) -> dict[str, Path]:
    stem = _unit_stem(dataset, seed, model, fold)
    return {name: root / f"{stem}.{suffix}" for name, suffix in {
        "complete": "complete.json", "training": "training.pt", "predictions": "predictions.jsonl",
    }.items()}


def _load_completion(path: Path, config: Mapping[str, Any], split_hash: str) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload["version"] != CHECKPOINT_VERSION or payload["config"] != dict(config)
                or payload["split_membership_hash"] != split_hash):
            return None
        rows = payload["predictions"]
        return rows if isinstance(rows, list) else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _save_training(path: Path, state: Mapping[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_training(path: Path, config: Mapping[str, Any], split_hash: str) -> dict[str, Any] | None:
    import torch
    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return state if state.get("config") == dict(config) and state.get("split_membership_hash") == split_hash else None
    except (OSError, RuntimeError, ValueError, TypeError, EOFError):
        return None


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= FIXED_THRESHOLD).astype(int)
    binary = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if binary else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if binary else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def transformer_payload_windows(
    token_ids: Sequence[int], special_token_count: int
) -> list[list[int]]:
    """Chunk payload IDs without dropping tokens reserved for special IDs."""
    payload_window = BERT_WINDOW - int(special_token_count)
    if payload_window <= 0:
        raise ValueError("Transformer special tokens exhaust the input window")
    return chunk_tokenizer_ids(token_ids, window_size=payload_window)


class SourcePreprocessor:
    """Run-scoped source/window cache; vocabulary fitting remains fold-local."""

    def __init__(self, frame: pd.DataFrame) -> None:
        sources = _read_sources(frame)
        identities = frame["_source_identity"].astype(str).tolist()
        self._source = dict(zip(identities, sources, strict=True))
        self._bytes = {
            identity: len(source.encode("utf-8"))
            for identity, source in self._source.items()
        }
        self._text_windows: dict[str, list[list[str]]] = {}
        self._tokenizer_windows: dict[tuple[str, int], dict[str, list[list[int]]]] = {}

    def source_bytes(self, frame: pd.DataFrame) -> list[int]:
        return [self._bytes[identity] for identity in frame["_source_identity"].astype(str)]

    def text_parts(self, frame: pd.DataFrame) -> tuple[list[list[str]], list[int]]:
        from src.ml.dl_baselines import tokenize_solidity
        parts: list[list[str]] = []
        counts: list[int] = []
        for identity in frame["_source_identity"].astype(str):
            windows = self._text_windows.get(identity)
            if windows is None:
                windows = chunk_tokens(tokenize_solidity(self._source[identity]))
                self._text_windows[identity] = windows
            parts.extend(windows)
            counts.append(len(windows))
        return parts, counts

    def tokenizer_parts(
        self, frame: pd.DataFrame, tokenizer: Any, tokenizer_name: str
    ) -> tuple[list[list[int]], list[int]]:
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        key = (tokenizer_name, special_count)
        cache = self._tokenizer_windows.setdefault(key, {})
        parts: list[list[int]] = []
        counts: list[int] = []
        for identity in frame["_source_identity"].astype(str):
            windows = cache.get(identity)
            if windows is None:
                token_ids = tokenizer(
                    self._source[identity], add_special_tokens=False
                )["input_ids"]
                windows = transformer_payload_windows(token_ids, special_count)
                cache[identity] = windows
            parts.extend(windows)
            counts.append(len(windows))
        return parts, counts


def _encode_and_fit(
    model_name: str, train: pd.DataFrame, test: pd.DataFrame, inner_train: np.ndarray,
    inner_val: np.ndarray, seed: int, config: Mapping[str, Any], split_hash: str,
    checkpoint: Path | None, preprocessor: SourcePreprocessor,
) -> tuple[np.ndarray, list[int]]:
    """Train CPU model and return test chunk probabilities and source coverage."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.ml.dl_baselines import BiLSTMClassifier, CodeBERTClassifier, SolidityBERTDataset, SolidityTokenDataset, TextCNN, build_vocab, predict, tokenize_solidity, train_epoch

    device = str(config.get("device", "cpu"))
    _seed(seed)
    is_bert = model_name in {"codebert", "unixcoder"}
    if is_bert:
        from transformers import AutoTokenizer
        hf_name = "microsoft/unixcoder-base" if model_name == "unixcoder" else "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        train_parts, train_counts = preprocessor.tokenizer_parts(train, tokenizer, hf_name)
        test_parts, test_counts = preprocessor.tokenizer_parts(test, tokenizer, hf_name)
        def padded(parts: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
            ids, masks = [], []
            for part in parts:
                row = ([tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []) + part + ([tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else [])
                if len(row) > BERT_WINDOW:
                    raise AssertionError("Transformer window exceeded configured input length")
                mask = [1] * len(row)
                row += [tokenizer.pad_token_id or 0] * (BERT_WINDOW - len(row))
                mask += [0] * (BERT_WINDOW - len(mask))
                ids.append(row); masks.append(mask)
            return np.asarray(ids, dtype=np.int64), np.asarray(masks, dtype=np.int64)
        x_train, m_train = padded(train_parts); x_test, m_test = padded(test_parts)
        labels = np.repeat(train["y"].to_numpy(), train_counts)
        train_ds = SolidityBERTDataset(x_train, m_train, labels)
        # Prediction has no label dependency: held-out labels are deliberately
        # not placed in the inference dataset.
        test_ds = SolidityBERTDataset(x_test, m_test, np.zeros(len(x_test), dtype=np.int64))
        model = CodeBERTClassifier(model_name=hf_name, freeze_layers=8).to(device)
    else:
        train_parts, train_counts = preprocessor.text_parts(train)
        test_parts, test_counts = preprocessor.text_parts(test)
        vocab = build_vocab(train_parts, min_freq=1)
        def padded(parts: list[list[str]]) -> np.ndarray:
            return np.asarray([[vocab.get(token, 1) for token in p[:TOKEN_WINDOW]] + [0] * (TOKEN_WINDOW - len(p[:TOKEN_WINDOW])) for p in parts], dtype=np.int64)
        labels = np.repeat(train["y"].to_numpy(), train_counts)
        train_ds = SolidityTokenDataset(padded(train_parts), labels)
        test_ds = SolidityTokenDataset(
            padded(test_parts), np.zeros(len(test_parts), dtype=np.int64)
        )
        model = (TextCNN if model_name == "textcnn" else BiLSTMClassifier)(vocab_size=len(vocab)).to(device)
    membership = np.repeat(np.arange(len(train)), train_counts)
    tr_loader = DataLoader(Subset(train_ds, np.flatnonzero(np.isin(membership, inner_train))), batch_size=int(config["batch_size"]), shuffle=False)
    va_mask = np.isin(membership, inner_val)
    va_loader = DataLoader(Subset(train_ds, np.flatnonzero(va_mask)), batch_size=int(config["batch_size"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 if is_bert else 1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state, best_loss, stale, start_epoch, start_batch = copy.deepcopy(model.state_dict()), float("inf"), 0, 1, 0
    state = _load_training(checkpoint, config, split_hash) if checkpoint else None
    if state:
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        best_state, best_loss, stale = state["best_state"], state["best_loss"], state["stale"]
        start_epoch, start_batch = state["epoch"], state["batch"]
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        def save_batch(batch: int, total: int) -> None:
            if checkpoint and (batch % TRAIN_CHECKPOINT_INTERVAL == 0 or batch == total):
                _save_training(checkpoint, {"config": dict(config), "split_membership_hash": split_hash, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_state": best_state, "best_loss": best_loss, "stale": stale, "epoch": epoch, "batch": batch, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})
        train_epoch(model, tr_loader, optimizer, criterion, device, is_bert=is_bert, progress_callback=save_batch, start_batch=start_batch if epoch == start_epoch else 0)
        val_prob = predict(model, va_loader, device, is_bert=is_bert)
        val_loss = float(torch.nn.functional.binary_cross_entropy(torch.from_numpy(val_prob).float(), torch.from_numpy(labels[va_mask]).float()))
        if val_loss < best_loss:
            best_state, best_loss, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
        start_batch = 0
    model.load_state_dict(best_state)
    return (
        predict(model, DataLoader(test_ds, batch_size=int(config["batch_size"])), device, is_bert=is_bert),
        train_counts + test_counts,
    )


def run_within_corpus(
    frame: pd.DataFrame, datasets: Sequence[str], models: Sequence[str] = DEFAULT_MODELS,
    seeds: Sequence[int] = (42, 43, 44, 45, 46), *, folds: int = 5, epochs: int = 10,
    patience: int = 3, batch_size: int = 16, validation_fraction: float = 0.2,
    checkpoint_dir: Path | str | None = None, resume: bool = True,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    canonical, audit = canonicalize(frame)
    root = Path(checkpoint_dir) if checkpoint_dir else None
    prediction_rows: list[dict[str, Any]] = []; metric_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}; split_memberships: dict[str, str] = {}
    for dataset in datasets:
        selected = canonical[canonical["source_dataset"] == dataset].reset_index(drop=True)
        preprocessor = SourcePreprocessor(selected)
        groups = groups_for(selected, dataset)
        n_folds = min(folds, len(np.unique(groups)))
        if n_folds < 2 or selected["y"].nunique() < 2:
            raise ValueError(f"{dataset} cannot support binary {n_folds}-fold group CV")
        for seed in seeds:
            outer = GroupKFold(n_splits=n_folds)
            for fold, (outer_train, outer_test) in enumerate(outer.split(selected, selected["y"], groups)):
                inner_train, inner_val = group_disjoint_inner_split(selected.iloc[outer_train], groups[outer_train], int(seed) + fold, validation_fraction)
                inner_train, inner_val = outer_train[inner_train], outer_train[inner_val]
                if set(groups[inner_train]) & set(groups[inner_val]) or set(groups[outer_train]) & set(groups[outer_test]):
                    raise AssertionError("group-disjoint split violated")
                membership = {"outer_train": selected.iloc[outer_train]["_source_identity"].tolist(), "outer_test": selected.iloc[outer_test]["_source_identity"].tolist(), "inner_train": selected.iloc[inner_train]["_source_identity"].tolist(), "inner_validation": selected.iloc[inner_val]["_source_identity"].tolist()}
                split_hash = _stable_hash(membership)
                split_memberships[f"{dataset}/{seed}/{fold}"] = split_hash
                outer_positions = {int(value): position for position, value in enumerate(outer_train)}
                for model in models:
                    if model == "textcnn":
                        # Preserve compatibility with the active v1 TextCNN campaign.
                        config = {"protocol": "within_corpus_dl_v1", "dataset": dataset, "seed": int(seed), "model": model, "fold": fold, "epochs": int(epochs), "patience": int(patience), "batch_size": int(batch_size), "validation_fraction": float(validation_fraction), "threshold": FIXED_THRESHOLD, "device": "cpu", "token_windows": {"text": TOKEN_WINDOW, "transformer": BERT_WINDOW}}
                    else:
                        config = {"protocol": "within_corpus_dl_v2", "preprocessing_version": PREPROCESSING_VERSION, "dataset": dataset, "seed": int(seed), "model": model, "fold": fold, "epochs": int(epochs), "patience": int(patience), "batch_size": int(batch_size), "validation_fraction": float(validation_fraction), "threshold": FIXED_THRESHOLD, "device": str(device), "token_windows": {"text": TOKEN_WINDOW, "transformer_input": BERT_WINDOW, "transformer_payload": BERT_WINDOW - 2}, "optimizer": {"name": "AdamW", "learning_rate": 2e-5 if model in {"codebert", "unixcoder"} else 1e-3}, "transformer_frozen_layers": 8}
                    paths = _paths(root, dataset, int(seed), model, fold) if root else {}
                    rows = _load_completion(paths["complete"], config, split_hash) if root and resume else None
                    if rows is None:
                        probs, chunks_ = _encode_and_fit(model, selected.iloc[outer_train], selected.iloc[outer_test], np.asarray([outer_positions[int(index)] for index in inner_train]), np.asarray([outer_positions[int(index)] for index in inner_val]), int(seed), config, split_hash, paths.get("training"), preprocessor)
                        counts = chunks_[-len(outer_test):]; cursor = 0; rows = []
                        for index, count in zip(outer_test, counts):
                            probability = float(np.mean(probs[cursor:cursor + count])); cursor += count
                            contract = selected.iloc[index]
                            rows.append({"dataset": dataset, "model": model, "random_state": int(seed), "fold": fold, "contract_id": contract["contract_id"], "source_identity_hash": contract["_source_identity"], "group": str(groups[index]), "n_chunks": int(count), "source_bytes": int(preprocessor.source_bytes(selected.iloc[[index]])[0]), "y_true": int(contract["y"]), "y_prob": probability, "y_pred": int(probability >= FIXED_THRESHOLD)})
                        if root:
                            _append_jsonl(paths["predictions"], rows)
                            _atomic_json(paths["complete"], {"version": CHECKPOINT_VERSION, "config": config, "split_membership_hash": split_hash, "predictions": rows})
                            if paths["training"].exists(): paths["training"].unlink()
                    fold_frame = pd.DataFrame(rows)
                    metric_rows.append({"dataset": dataset, "model": model, "random_state": int(seed), "fold": fold, "n_contracts": len(rows), "n_groups_test": len(np.unique(groups[outer_test])), "split_membership_hash": split_hash, **_metrics(fold_frame.y_true.to_numpy(), fold_frame.y_prob.to_numpy())})
                    prediction_rows.extend(rows)
        dataset_predictions = [row for row in prediction_rows if row["dataset"] == dataset]
        coverage[dataset] = {
            "contracts": int(len(selected)),
            "source_bytes": int(sum(preprocessor.source_bytes(selected))),
            "test_chunk_windows_by_model": {
                model: int(sum(row["n_chunks"] for row in dataset_predictions if row["model"] == model))
                for model in models
            },
        }
    predictions, metrics = pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)
    metadata = {"protocol": {"name": "within_corpus_full_source_dl_v2", "scope": "within-corpus only", "outer_split": "GroupKFold", "inner_validation": "group-disjoint outer-training-only", "threshold": FIXED_THRESHOLD, "no_test_selection_assertion": True, "source_execution": False, "device": str(device)}, "duplicate_audit": audit, "chunking": {"textcnn_bilstm_tokens": TOKEN_WINDOW, "codebert_unixcoder_input_tokens": BERT_WINDOW, "codebert_unixcoder_payload_tokens": BERT_WINDOW - 2, "overlap": 0, "aggregation": "contract mean"}, "preprocessing": {"version": PREPROCESSING_VERSION, "scope": "in-memory per dataset", "vocabulary_fit": "outer-training-only"}, "data_coverage": coverage, "split_membership_hashes": split_memberships, "models": list(models), "seeds": list(seeds)}
    return predictions, metrics, metadata
