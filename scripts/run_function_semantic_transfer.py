#!/usr/bin/env python3
"""Train external function-pattern models and evaluate frozen SCAFE fusion.

This script deliberately separates three stages:

1. train a multi-label CodeBERT/UniXcoder model on the auxiliary function corpus;
2. score functions extracted from SCAFE source records and aggregate to records;
3. select a fusion using DAppSCAN OOF predictions, then evaluate it unchanged on
   DIVE and Empirical-Analysis LODO predictions.

The auxiliary corpus is filtered for exact normalized-function overlap with all
primary SCAFE sources before training.  This does not remove near clones or
provenance dependence, so results remain an experimental transfer diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_AUXILIARY = Path.home() / "Downloads" / "codebert_dataset_expanded.csv"
DEFAULT_REGISTRY = REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = REPO / "data/derived/sanitized_v1/ml_results/function_semantic_transfer"
PRIMARY_DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")
LABELS = (
    "EXTERNAL_CALL_BEFORE_STATE_UPDATE",
    "MISSING_INPUT_VALIDATION",
    "MISSING_ZERO_ADDRESS_CHECK",
    "PRECISION_LOSS_IN_DIVISION",
    "ROLE_OR_OWNER_REASSIGNMENT",
    "SENSITIVE_STATE_READ_WITHOUT_LOCK",
    "STATE_UPDATE_ONLY_ON_SUCCESS",
    "UNCHECKED_ARITHMETIC_OPERATION",
    "UNGUARDED_SELFDESTRUCT",
    "UNSAFE_DELEGATECALL",
    "WEAK_CALLER_VALIDATION",
)
MODEL_NAMES = {
    "codebert": "microsoft/codebert-base",
    "unixcoder": "microsoft/unixcoder-base",
}
FUNCTION_RE = re.compile(
    r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)[^{;]*\{", re.DOTALL
)


def _normalise(source: str) -> str:
    """Whitespace/comment-insensitive exact-comparison representation."""

    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    without_line = re.sub(r"//.*?$", " ", without_block, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", without_line).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_functions(source: str) -> list[tuple[str, str]]:
    """Extract named Solidity functions using balanced braces.

    This is source-only extraction, not an AST.  Invalid/incomplete functions
    are skipped rather than guessed.
    """

    result: list[tuple[str, str]] = []
    for match in FUNCTION_RE.finditer(source):
        depth = 0
        end = None
        for position in range(match.end() - 1, len(source)):
            char = source[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if end is not None:
            result.append((match.group(1), source[match.start():end]))
    return result


def _source_text(path_value: str) -> str:
    path = Path(path_value)
    try:
        if path.is_file() and path.suffix.lower() == ".sol":
            return path.read_text(encoding="utf-8", errors="replace")
        if path.is_dir():
            return "\n".join(
                item.read_text(encoding="utf-8", errors="replace")
                for item in sorted(path.rglob("*.sol"))
            )
    except OSError:
        return ""
    return ""


def _registry_function_hashes(registry: pd.DataFrame) -> set[str]:
    """Return exact normalized function hashes from primary evaluation sources."""

    hashes: set[str] = set()
    seen_paths: set[str] = set()
    primary = registry[registry["source_dataset"].isin(PRIMARY_DATASETS)]
    for path_value in primary["local_source_path"].dropna().astype(str):
        if not path_value or path_value in seen_paths:
            continue
        seen_paths.add(path_value)
        for _, function in extract_functions(_source_text(path_value)):
            hashes.add(_sha256(_normalise(function)))
    return hashes


def _multihot(labels: pd.Series) -> np.ndarray:
    matrix = np.zeros((len(labels), len(LABELS)), dtype=np.float32)
    index = {label: position for position, label in enumerate(LABELS)}
    for row, value in enumerate(labels.astype(str)):
        for label in value.split(","):
            if label in index:
                matrix[row, index[label]] = 1.0
    return matrix


def _thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> list[float]:
    values: list[float] = []
    for column in range(probabilities.shape[1]):
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, column], probabilities[:, column]
        )
        f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
        # precision/recall include the no-threshold endpoint.
        position = int(np.nanargmax(f1[:-1]))
        values.append(float(thresholds[position]))
    return values


def _label_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: Sequence[float]
) -> list[dict[str, float | str]]:
    rows = []
    for position, label in enumerate(LABELS):
        target = y_true[:, position]
        probability = probabilities[:, position]
        rows.append(
            {
                "label": label,
                "prevalence": int(target.sum()),
                "pr_auc": float(average_precision_score(target, probability)),
                "roc_auc": float(roc_auc_score(target, probability)),
                "f1": float(
                    f1_score(target, probability >= thresholds[position], zero_division=0)
                ),
                "threshold": float(thresholds[position]),
            }
        )
    return rows


@dataclass(frozen=True)
class FunctionExample:
    text: str
    labels: np.ndarray


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> None:
    """Fine-tune a frozen-bottom transformer and save a reusable checkpoint."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this Python environment")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")

    output = args.output / args.model
    output.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry, low_memory=False)
    auxiliary = pd.read_csv(args.auxiliary, low_memory=False)
    required = {"id", "statement", "label", "split"}
    missing = required.difference(auxiliary.columns)
    if missing:
        raise ValueError(f"Auxiliary CSV misses columns: {sorted(missing)}")
    if not set(auxiliary["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Auxiliary split must be train, val, or test")

    overlap_hashes = _registry_function_hashes(registry)
    function_hashes = auxiliary["statement"].astype(str).map(
        lambda text: _sha256(_normalise(text))
    )
    keep = ~function_hashes.isin(overlap_hashes)
    filtered = auxiliary.loc[keep].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Exact-overlap filtering removed all auxiliary functions")
    split_counts = filtered["split"].value_counts().to_dict()
    if any(split not in split_counts for split in ("train", "val", "test")):
        raise ValueError(f"Filtering left an incomplete split: {split_counts}")
    overlap_audit = {
        "protocol": "exact-normalized-function-overlap-v1",
        "auxiliary": str(args.auxiliary.resolve()),
        "registry": str(args.registry.resolve()),
        "primary_datasets": list(PRIMARY_DATASETS),
        "registry_function_hashes": len(overlap_hashes),
        "auxiliary_rows_before_filter": int(len(auxiliary)),
        "auxiliary_rows_removed": int((~keep).sum()),
        "auxiliary_rows_after_filter": int(len(filtered)),
        "rows_by_split_after_filter": split_counts,
    }
    (output / "overlap_audit.json").write_text(
        json.dumps(overlap_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        return

    model_name = MODEL_NAMES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    class TensorDataset(Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.text = frame["statement"].astype(str).tolist()
            self.labels = _multihot(frame["label"])

        def __len__(self) -> int:
            return len(self.text)

        def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
            return self.text[index], self.labels[index]

    def collate(batch: list[tuple[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item[0] for item in batch],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            np.stack([item[1] for item in batch]), dtype=torch.float32
        )
        return encoded

    class MultiLabelTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
            for name, parameter in self.encoder.named_parameters():
                if "encoder.layer." in name:
                    layer = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer < args.freeze_layers:
                        parameter.requires_grad = False
                elif "embeddings" in name:
                    parameter.requires_grad = False
            self.dropout = nn.Dropout(args.dropout)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            representation = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(representation))

    _set_seed(args.seed)
    train_frame = filtered[filtered["split"] == "train"]
    validation_frame = filtered[filtered["split"] == "val"]
    test_frame = filtered[filtered["split"] == "test"]
    train_data, validation_data, test_data = (
        TensorDataset(train_frame),
        TensorDataset(validation_frame),
        TensorDataset(test_frame),
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    device = torch.device(args.device)
    model = MultiLabelTransformer().to(device)
    positives = _multihot(train_frame["label"]).sum(axis=0)
    pos_weight = torch.tensor(
        (len(train_frame) - positives) / np.maximum(positives, 1),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    state_path = output / "training_state.pt"

    def probabilities(loader: DataLoader) -> np.ndarray:
        model.eval()
        result: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                result.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(result)

    best_state = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    run_config = {
        "model_name": model_name,
        "max_length": args.max_length,
        "freeze_layers": args.freeze_layers,
        "seed": args.seed,
        "auxiliary_rows_after_filter": int(len(filtered)),
    }
    if args.resume and state_path.is_file():
        resumed = torch.load(state_path, map_location="cpu", weights_only=False)
        if resumed.get("run_config") != run_config:
            raise ValueError(
                "Existing resumable state belongs to a different model/data configuration"
            )
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if hasattr(value, "to"):
                    optimizer_state[key] = value.to(device)
        best_state = resumed["best_state"]
        best_score = float(resumed["best_score"])
        stale = int(resumed["stale"])
        history = list(resumed["history"])
        start_epoch = int(resumed["completed_epoch"]) + 1
        print(
            f"Resuming {args.model} after epoch {start_epoch - 1}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_probability = probabilities(validation_loader)
        validation_y = _multihot(validation_frame["label"])
        score = float(
            np.mean(
                [
                    average_precision_score(validation_y[:, column], validation_probability[:, column])
                    for column in range(len(LABELS))
                ]
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_pr_auc": score,
            }
        )
        if score > best_score:
            best_score = score
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                print(
                    f"Early stopping {args.model} after epoch {epoch}",
                    flush=True,
                )
        print(
            f"{args.model} epoch={epoch} train_loss={np.mean(losses):.5f} "
            f"validation_macro_pr_auc={score:.5f}",
            flush=True,
        )
        temporary = state_path.with_suffix(".tmp")
        torch.save(
            {
                "run_config": run_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "stale": stale,
                "history": history,
                "completed_epoch": epoch,
            },
            temporary,
        )
        temporary.replace(state_path)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probability = probabilities(validation_loader)
    thresholds = _thresholds(_multihot(validation_frame["label"]), validation_probability)
    test_probability = probabilities(test_loader)
    test_metrics = _label_metrics(
        _multihot(test_frame["label"]), test_probability, thresholds
    )
    tokenizer.save_pretrained(output / "tokenizer")
    torch.save(
        {
            "version": 1,
            "model_name": model_name,
            "labels": list(LABELS),
            "freeze_layers": args.freeze_layers,
            "max_length": args.max_length,
            "state_dict": model.state_dict(),
            "thresholds": thresholds,
        },
        output / "checkpoint.pt",
    )
    if state_path.exists():
        state_path.unlink()
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(test_metrics).to_csv(output / "internal_test_metrics.csv", index=False)
    metadata = {
        "protocol": "external-function-pattern-multilabel-v1",
        "model": args.model,
        "model_name": model_name,
        "labels": list(LABELS),
        "splits": "provided train/val/test; no project provenance was available",
        "source_overlap_filter": {
            "method": "exact normalized function hash against all primary SCAFE sources",
            "removed_rows": int((~keep).sum()),
            "remaining_rows": int(len(filtered)),
        },
        "rows_by_split_after_filter": split_counts,
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "early_stopping_metric": "validation macro PR-AUC",
            "best_validation_macro_pr_auc": best_score,
            "freeze_layers": args.freeze_layers,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
        },
        "limitations": [
            "The auxiliary corpus lacks project/repository provenance.",
            "Exact filtering does not remove near-clone or template overlap.",
            "Pattern labels are not equivalent to verified exploitability.",
        ],
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_model(checkpoint: Path, device_name: str):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    payload = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("labels") != list(LABELS):
        raise ValueError("Unexpected function-model checkpoint")
    model_name = str(payload["model_name"])

    class MultiLabelTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.dropout = nn.Dropout(0.0)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(hidden))

    device = torch.device(device_name)
    model = MultiLabelTransformer()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return (
        model,
        AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True),
        payload,
        device,
    )


def score(args: argparse.Namespace) -> None:
    """Score source-record functions, retaining complete per-function provenance."""

    import torch

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this Python environment")
    model, tokenizer, payload, device = _load_model(args.checkpoint, args.device)
    registry = pd.read_csv(args.registry, low_memory=False)
    registry = registry[
        registry["source_dataset"].isin(args.datasets)
        & registry["local_source_path"].notna()
    ].copy()
    rows: list[dict[str, object]] = []
    source_cache: dict[str, list[tuple[str, str]]] = {}
    source_identity_cache: dict[str, str] = {}

    def predict(texts: list[str]) -> np.ndarray:
        result = []
        with torch.no_grad():
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(
                    texts[start:start + args.batch_size],
                    truncation=True,
                    max_length=int(payload["max_length"]),
                    padding=True,
                    return_tensors="pt",
                )
                logits = model(
                    encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
                )
                result.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(result) if result else np.empty((0, len(LABELS)))

    pending: list[tuple[dict[str, object], str]] = []
    for _, record in registry.iterrows():
        path_value = str(record["local_source_path"])
        functions = source_cache.get(path_value)
        if functions is None:
            source = _source_text(path_value)
            functions = extract_functions(source)
            source_cache[path_value] = functions
            source_identity_cache[path_value] = _sha256(_normalise(source)) if source else ""
        for ordinal, (name, text) in enumerate(functions):
            pending.append(
                (
                    {
                        "dataset": record["source_dataset"],
                        "contract_id": record["contract_id"],
                        "source_identity_hash": source_identity_cache[path_value],
                        "function_ordinal": ordinal,
                        "function_name": name,
                        "function_hash": _sha256(_normalise(text)),
                        "source_path": path_value,
                    },
                    text,
                )
            )
        if len(pending) >= args.flush_functions:
            probability = predict([text for _, text in pending])
            for (metadata, _), values in zip(pending, probability, strict=True):
                rows.append({**metadata, **dict(zip(LABELS, values.tolist(), strict=True))})
            pending.clear()
    if pending:
        probability = predict([text for _, text in pending])
        for (metadata, _), values in zip(pending, probability, strict=True):
            rows.append({**metadata, **dict(zip(LABELS, values.tolist(), strict=True))})
    functions = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    functions.to_csv(args.output / "function_predictions.csv", index=False)
    if functions.empty:
        raise RuntimeError("No functions were extracted from requested SCAFE records")
    functions["semantic_any_max"] = functions[list(LABELS)].max(axis=1)
    aggregate = (
        functions.groupby(["dataset", "contract_id", "source_identity_hash"], as_index=False)
        .agg(
            n_functions=("function_hash", "size"),
            semantic_any_max=("semantic_any_max", "max"),
            **{f"semantic_{label.lower()}_max": (label, "max") for label in LABELS},
        )
    )
    # A less sparse, robust alternative: mean of the top three function maxima.
    top3 = (
        functions.groupby(["dataset", "contract_id", "source_identity_hash"])["semantic_any_max"]
        .apply(lambda values: float(values.nlargest(min(3, len(values))).mean()))
        .rename("semantic_any_top3_mean")
        .reset_index()
    )
    aggregate = aggregate.merge(
        top3, on=["dataset", "contract_id", "source_identity_hash"], how="left"
    )
    aggregate.to_csv(args.output / "contract_aggregates.csv", index=False)
    (args.output / "scoring_protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "datasets": list(args.datasets),
                "aggregation": {
                    "semantic_any_max": "max pattern probability across all functions",
                    "semantic_any_top3_mean": "mean of three highest function maxima",
                    "per_pattern": "max probability for each pattern across functions",
                },
                "records_requested": int(len(registry)),
                "records_scored": int(aggregate.shape[0]),
                "functions_scored": int(functions.shape[0]),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _metric(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
    }


def evaluate(args: argparse.Namespace) -> None:
    """Freeze development-selected semantic fusion and evaluate LODO targets."""

    aggregates = pd.read_csv(args.aggregates, low_memory=False)
    development = pd.read_csv(args.development_predictions, low_memory=False)
    lodo = pd.read_csv(args.lodo_predictions, low_memory=False)
    development = development[
        (development["dataset"] == "DAppSCAN") & (development["model"] == "et")
    ]
    lodo = lodo[
        lodo["dataset"].isin(("DIVE", "Empirical-Analysis")) & (lodo["model"] == "et")
    ]
    # Each record has one score per seed.  Averaging must occur before meta-model
    # fitting/evaluation, so repeated seed rows are never treated as independent.
    development = (
        development.groupby(["dataset", "contract_id", "source_identity_hash", "y_true"], as_index=False)["y_prob"]
        .mean()
        .rename(columns={"y_prob": "scafe_probability"})
    )
    lodo = (
        lodo.groupby(["dataset", "contract_id", "y_true"], as_index=False)["y_prob"]
        .mean()
        .rename(columns={"y_prob": "scafe_probability"})
    )
    semantic_columns = [
        "semantic_any_max",
        "semantic_any_top3_mean",
        *[f"semantic_{label.lower()}_max" for label in LABELS],
    ]

    def join(frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(
            aggregates[["dataset", "contract_id", *semantic_columns]],
            on=["dataset", "contract_id"],
            how="left",
            validate="one_to_one",
        )
        merged["semantic_coverage"] = merged[semantic_columns].notna().any(axis=1)
        merged[semantic_columns] = merged[semantic_columns].fillna(0.0)
        return merged

    development, external = join(development), join(lodo)
    candidate_rows = []
    for column in semantic_columns:
        candidate_rows.append(
            {"method": column, **_metric(development.y_true, development[column])}
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["roc_auc", "pr_auc"], ascending=False
    )
    selected_semantic = str(candidates.iloc[0]["method"])
    selected_alpha = None
    fusion_candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        probability = (
            alpha * development["scafe_probability"]
            + (1.0 - alpha) * development[selected_semantic]
        )
        fusion_candidates.append(
            {"alpha_scafe": float(alpha), **_metric(development.y_true, probability)}
        )
    fusion = pd.DataFrame(fusion_candidates).sort_values(
        ["roc_auc", "pr_auc", "alpha_scafe"], ascending=[False, False, False]
    )
    selected_alpha = float(fusion.iloc[0]["alpha_scafe"])
    distance = np.abs(development["scafe_probability"].to_numpy(dtype=float) - 0.5)
    cascade_candidates = []
    for fraction in np.arange(0.05, 1.01, 0.05):
        threshold = float(np.quantile(distance, fraction, method="higher"))
        delegate = distance <= threshold
        probability = np.where(
            delegate,
            development[selected_semantic].to_numpy(dtype=float),
            development["scafe_probability"].to_numpy(dtype=float),
        )
        cascade_candidates.append(
            {
                "delegation_fraction_requested": float(fraction),
                "delegation_fraction_actual": float(delegate.mean()),
                "distance_threshold": threshold,
                **_metric(development.y_true, probability),
            }
        )
    cascade = pd.DataFrame(cascade_candidates).sort_values(
        ["roc_auc", "pr_auc", "delegation_fraction_actual"],
        ascending=[False, False, True],
    )
    development_scafe_metrics = _metric(
        development.y_true, development["scafe_probability"].to_numpy(dtype=float)
    )
    cascade_improvements = cascade.loc[
        (cascade["roc_auc"] > development_scafe_metrics["roc_auc"])
        | (
            (cascade["roc_auc"] == development_scafe_metrics["roc_auc"])
            & (cascade["pr_auc"] > development_scafe_metrics["pr_auc"])
        )
    ]
    selected_cascade = (
        cascade_improvements.iloc[0].to_dict() if not cascade_improvements.empty else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output / "development_semantic_candidates.csv", index=False)
    fusion.to_csv(args.output / "development_fusion_candidates.csv", index=False)
    cascade.to_csv(args.output / "development_cascade_candidates.csv", index=False)

    output_rows = []
    for dataset, frame in external.groupby("dataset"):
        semantic = frame[selected_semantic].to_numpy(dtype=float)
        scafe = frame["scafe_probability"].to_numpy(dtype=float)
        methods = {
            "scafe": scafe,
            f"semantic:{selected_semantic}": semantic,
            "development_frozen_linear_fusion": selected_alpha * scafe
            + (1.0 - selected_alpha) * semantic,
        }
        external_delegate = None
        if selected_cascade is not None:
            external_delegate = (
                np.abs(scafe - 0.5) <= float(selected_cascade["distance_threshold"])
            )
            methods["development_frozen_uncertainty_cascade"] = np.where(
                external_delegate, semantic, scafe
            )
        for method, probability in methods.items():
            output_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": len(frame),
                    "semantic_coverage": float(frame["semantic_coverage"].mean()),
                    "delegation_fraction": float(external_delegate.mean())
                    if (
                        method == "development_frozen_uncertainty_cascade"
                        and external_delegate is not None
                    )
                    else 0.0,
                    **_metric(frame.y_true.to_numpy(), probability),
                }
            )
    pd.DataFrame(output_rows).to_csv(args.output / "external_metrics.csv", index=False)
    (args.output / "fusion_lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "development_dataset": "DAppSCAN",
                "external_targets": ["DIVE", "Empirical-Analysis"],
                "scafe_model": "Extra Trees",
                "semantic_feature_selected_on_development": selected_semantic,
                "fusion": {
                    "definition": "alpha * SCAFE + (1-alpha) * semantic feature",
                    "alpha_scafe_selected_on_development": selected_alpha,
                    "selection_metric": "ROC-AUC; PR-AUC and SCAFE weight tie-breaks",
                },
                "uncertainty_cascade": {
                    "definition": (
                        "delegate to the semantic function model when "
                        "abs(SCAFE probability - 0.5) is at or below the "
                        "DAppSCAN-selected threshold"
                    ),
                    "selected_on_development": selected_cascade,
                    "rejected": selected_cascade is None,
                    "development_scafe_metrics": development_scafe_metrics,
                },
                "target_labels_used_for_selection": False,
                "limitations": [
                    "Auxiliary training data lack project provenance.",
                    "Only exact normalized-function overlap was excluded.",
                    "Function extraction is source-only and approximate.",
                ],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(required=True, dest="command")
    training = commands.add_parser("train")
    training.add_argument("--auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    training.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    training.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    training.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--patience", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--max-length", type=int, default=512)
    training.add_argument("--freeze-layers", type=int, default=8)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume safely from the most recently completed epoch.",
    )
    training.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform and persist the overlap audit without loading or training a model.",
    )
    scoring = commands.add_parser("score")
    scoring.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    scoring.add_argument("--checkpoint", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--datasets", nargs="+", choices=PRIMARY_DATASETS, default=PRIMARY_DATASETS)
    scoring.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    scoring.add_argument("--batch-size", type=int, default=16)
    scoring.add_argument("--flush-functions", type=int, default=512)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--aggregates", type=Path, required=True)
    evaluation.add_argument(
        "--development-predictions",
        type=Path,
        default=REPO / "data/derived/sanitized_v1/ml_results/primary_development/scafe_within_corpus_predictions.csv",
    )
    evaluation.add_argument(
        "--lodo-predictions",
        type=Path,
        default=REPO / "data/derived/sanitized_v1/ml_results/primary_lodo/scafe_repeated_lodo_predictions.csv",
    )
    evaluation.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "score":
        score(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
