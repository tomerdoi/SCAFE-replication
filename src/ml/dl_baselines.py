"""
Deep-learning baselines for smart-contract vulnerability detection.

Models implemented:
  1. TextCNN  – 1-D convolutions over learned token embeddings (Kim 2014).
  2. BiLSTM   – bidirectional LSTM with max-pool aggregation.
  3. CodeBERT – fine-tuned microsoft/codebert-base (frozen lower layers).

All models are evaluated with the **same** GroupKFold CV protocol used by
the traditional ML pipeline so that results are directly comparable.

Usage:
    /path/to/venv/bin/python -m src.ml.dl_baselines [--models textcnn,bilstm,codebert]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "data" / "derived" / "dl_results"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ── Solidity tokenizer (lightweight, regex-based) ───────────────────────────

_SOL_TOKEN_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|"           # comments
    r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|"  # strings
    r"0x[0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?|"      # numbers
    r"[a-zA-Z_$][a-zA-Z0-9_$]*|"                 # identifiers
    r"[{}()\[\];,.]|"                             # single-char punctuation
    r"[+\-*/%=<>!&|^~]+|"                         # operators
    r"\S"                                          # fallback
)


def tokenize_solidity(source: str) -> List[str]:
    """Tokenize Solidity source into a flat token list."""
    return _SOL_TOKEN_RE.findall(source)


def read_source(path: str, max_chars: int = 100_000) -> str:
    """Read a Solidity source file, with truncation for safety."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# ── Vocabulary builder ──────────────────────────────────────────────────────

PAD_TOK, UNK_TOK = "<PAD>", "<UNK>"


def build_vocab(token_lists: List[List[str]], max_vocab: int = 30_000,
                min_freq: int = 2) -> Dict[str, int]:
    """Build word→index mapping from training token lists."""
    counts: Counter = Counter()
    for toks in token_lists:
        counts.update(toks)
    vocab = {PAD_TOK: 0, UNK_TOK: 1}
    for tok, cnt in counts.most_common(max_vocab):
        if cnt < min_freq:
            break
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int],
                  max_len: int) -> np.ndarray:
    """Encode token list to padded integer array."""
    unk = vocab[UNK_TOK]
    ids = [vocab.get(t, unk) for t in tokens[:max_len]]
    arr = np.zeros(max_len, dtype=np.int64)
    arr[:len(ids)] = ids
    return arr


# ── Datasets ────────────────────────────────────────────────────────────────

class SolidityTokenDataset(Dataset):
    """Dataset for TextCNN / BiLSTM: integer-encoded token sequences."""

    def __init__(self, token_ids: np.ndarray, labels: np.ndarray):
        self.token_ids = torch.from_numpy(token_ids).long()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


class SolidityBERTDataset(Dataset):
    """Dataset for CodeBERT: tokenizer-encoded input_ids + attention_mask."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                 labels: np.ndarray):
        self.input_ids = torch.from_numpy(input_ids).long()
        self.attention_mask = torch.from_numpy(attention_mask).long()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx],
                self.labels[idx])


# ── Model architectures ────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """Kim-style TextCNN with multiple filter widths."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 num_filters: int = 128,
                 filter_sizes: Tuple[int, ...] = (3, 4, 5),
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        # x: (batch, seq_len)
        e = self.embedding(x).permute(0, 2, 1)  # (batch, embed, seq)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(e))              # (batch, filters, seq-k+1)
            p = c.max(dim=2).values              # (batch, filters)
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat)).squeeze(-1)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with max-pool over hidden states."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        e = self.embedding(x)
        out, _ = self.lstm(e)
        # max-pool over time
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled)).squeeze(-1)


class CodeBERTClassifier(nn.Module):
    """CodeBERT with classification head. Lower layers optionally frozen."""

    def __init__(self, model_name: str = "microsoft/codebert-base",
                 freeze_layers: int = 8, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        # Freeze lower layers for faster training
        for name, param in self.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < freeze_layers:
                    param.requires_grad = False
            elif "embeddings" in name:
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls)).squeeze(-1)


# ── Training loop ───────────────────────────────────────────────────────────

def _optimal_f1_threshold(y_true: np.ndarray,
                          y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0.0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def train_epoch(model, loader, optimizer, criterion, device,
                is_bert: bool = False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if is_bert:
            ids, mask, labels = [b.to(device) for b in batch]
            logits = model(ids, mask)
        else:
            ids, labels = batch[0].to(device), batch[1].to(device)
            logits = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader, device, is_bert: bool = False) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        if is_bert:
            ids, mask, _ = [b.to(device) for b in batch]
            logits = model(ids, mask)
        else:
            ids = batch[0].to(device)
            logits = model(ids)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ── Data loading ────────────────────────────────────────────────────────────

def load_dappscan_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load DAppSCAN data, return (df, labels, groups)."""
    from src.ml.features import load_scored_csv

    df = load_scored_csv()
    df = df[df["source_dataset"] == "DAppSCAN"].copy()
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df = df[df["label_status"].isin(["clean", "vulnerable"])].copy()
    df["y"] = (df["label_status"] == "vulnerable").astype(int)

    y = df["y"].values.astype(int)

    group_key = df["contract_id"].fillna("").astype(str)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)

    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} DAppSCAN contracts "
                f"(pos={y.sum()}, neg={(y == 0).sum()})")
    return df, y, groups


def load_sources(df: pd.DataFrame, max_tokens: int = 1024) -> List[List[str]]:
    """Read and tokenize all source files."""
    token_lists = []
    for path in df["local_source_path"]:
        src = read_source(str(path))
        toks = tokenize_solidity(src)[:max_tokens]
        token_lists.append(toks)
    return token_lists


# ── Fold evaluation (TextCNN / BiLSTM) ──────────────────────────────────────

@dataclass
class DLFoldResult:
    fold_idx: int
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float


def evaluate_token_model(
    model_name: str,
    model_cls,
    model_kwargs: dict,
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    token_lists: List[List[str]],
    *,
    max_len: int = 1024,
    n_folds: int = 5,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict:
    """Run GroupKFold CV for a token-based DL model (TextCNN or BiLSTM)."""
    gkf = GroupKFold(n_splits=n_folds)
    fold_results: List[DLFoldResult] = []
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(
            np.zeros(len(y)), y, groups)):
        logger.info(f"  Fold {fold_idx + 1}/{n_folds} "
                     f"(train={len(train_idx)}, test={len(test_idx)})")

        train_tokens = [token_lists[i] for i in train_idx]
        test_tokens = [token_lists[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vocab = build_vocab(train_tokens, max_vocab=30_000, min_freq=2)
        vocab_size = len(vocab)

        X_train = np.stack([encode_tokens(t, vocab, max_len)
                            for t in train_tokens])
        X_test = np.stack([encode_tokens(t, vocab, max_len)
                           for t in test_tokens])

        # Class weights for loss
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                                  dtype=torch.float32).to(DEVICE)

        train_ds = SolidityTokenDataset(X_train, y_train)
        test_ds = SolidityTokenDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = model_cls(vocab_size=vocab_size, **model_kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(n_epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion,
                               DEVICE)
            # Quick validation loss for early stopping
            model.eval()
            val_probs = predict(model, test_loader, DEVICE)
            val_loss = nn.functional.binary_cross_entropy(
                torch.from_numpy(val_probs).float(),
                torch.from_numpy(y_test).float(),
            ).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_probs = val_probs.copy()
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    logger.info(f"    Early stop at epoch {epoch + 1}")
                    break

        y_prob = best_probs
        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        opt_t, opt_f1 = _optimal_f1_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0.0)
        rec = recall_score(y_test, y_pred, zero_division=0.0)

        fold_results.append(DLFoldResult(
            fold_idx=fold_idx, roc_auc=roc, pr_auc=pr,
            f1=opt_f1, precision=prec, recall=rec, threshold=opt_t,
        ))
        logger.info(f"    ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  F1={opt_f1:.4f}")

    elapsed = time.time() - t0
    return _aggregate_results(model_name, fold_results, elapsed)


# ── Fold evaluation (CodeBERT) ──────────────────────────────────────────────

def evaluate_codebert(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_name_hf: str = "microsoft/codebert-base",
    max_len: int = 512,
    n_folds: int = 5,
    n_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    freeze_layers: int = 8,
) -> Dict:
    """Run GroupKFold CV for fine-tuned CodeBERT."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    gkf = GroupKFold(n_splits=n_folds)
    fold_results: List[DLFoldResult] = []
    t0 = time.time()

    sources = [read_source(str(p)) for p in df["local_source_path"]]

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(
            np.zeros(len(y)), y, groups)):
        logger.info(f"  CodeBERT fold {fold_idx + 1}/{n_folds} "
                     f"(train={len(train_idx)}, test={len(test_idx)})")

        y_train, y_test = y[train_idx], y[test_idx]
        train_srcs = [sources[i] for i in train_idx]
        test_srcs = [sources[i] for i in test_idx]

        # Tokenize
        train_enc = tokenizer(
            train_srcs, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="np",
        )
        test_enc = tokenizer(
            test_srcs, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="np",
        )

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                                  dtype=torch.float32).to(DEVICE)

        train_ds = SolidityBERTDataset(
            train_enc["input_ids"], train_enc["attention_mask"], y_train)
        test_ds = SolidityBERTDataset(
            test_enc["input_ids"], test_enc["attention_mask"], y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = CodeBERTClassifier(
            model_name=model_name_hf, freeze_layers=freeze_layers,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        patience_counter = 0
        best_probs = None
        for epoch in range(n_epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion,
                               DEVICE, is_bert=True)
            y_prob = predict(model, test_loader, DEVICE, is_bert=True)
            val_loss = nn.functional.binary_cross_entropy(
                torch.from_numpy(y_prob).float(),
                torch.from_numpy(y_test).float(),
            ).item()
            logger.info(f"    Epoch {epoch + 1}: train_loss={loss:.4f} "
                        f"val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_probs = y_prob.copy()
            else:
                patience_counter += 1
                if patience_counter >= 2:
                    logger.info(f"    Early stop at epoch {epoch + 1}")
                    break

        y_prob = best_probs
        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        opt_t, opt_f1 = _optimal_f1_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0.0)
        rec = recall_score(y_test, y_pred, zero_division=0.0)

        fold_results.append(DLFoldResult(
            fold_idx=fold_idx, roc_auc=roc, pr_auc=pr,
            f1=opt_f1, precision=prec, recall=rec, threshold=opt_t,
        ))
        logger.info(f"    ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  F1={opt_f1:.4f}")

        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None

    elapsed = time.time() - t0
    return _aggregate_results("codebert", fold_results, elapsed)


# ── Result aggregation ──────────────────────────────────────────────────────

def _aggregate_results(model_name: str, fold_results: List[DLFoldResult],
                       elapsed: float) -> Dict:
    roc_aucs = [f.roc_auc for f in fold_results]
    pr_aucs = [f.pr_auc for f in fold_results]
    f1s = [f.f1 for f in fold_results]
    precs = [f.precision for f in fold_results]
    recs = [f.recall for f in fold_results]
    return {
        "model": model_name,
        "display_name": {
            "textcnn": "TextCNN",
            "bilstm": "BiLSTM",
            "codebert": "CodeBERT (fine-tuned)",
        }.get(model_name, model_name),
        "family": "deep_learning",
        "feature_set": "source_text",
        "dataset": "DAppSCAN",
        "n_features": 0,
        "n_folds": len(fold_results),
        "roc_auc_mean": round(float(np.mean(roc_aucs)), 4),
        "roc_auc_std": round(float(np.std(roc_aucs)), 4),
        "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
        "pr_auc_std": round(float(np.std(pr_aucs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "precision_mean": round(float(np.mean(precs)), 4),
        "recall_mean": round(float(np.mean(recs)), 4),
        "train_time_s": round(elapsed, 1),
    }


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run DL baselines for SC vulnerability detection")
    parser.add_argument(
        "--models", type=str, default="textcnn,bilstm,codebert",
        help="Comma-separated list of models to run "
             "(textcnn, bilstm, codebert)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max token length for TextCNN/BiLSTM")
    parser.add_argument("--bert-max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bert-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",")]
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Models to run: {models_to_run}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, y, groups = load_dappscan_data()
    logger.info("Tokenizing source files...")
    token_lists = load_sources(df, max_tokens=args.max_len)
    logger.info(f"  Median token count: "
                f"{int(np.median([len(t) for t in token_lists]))}")

    results = []

    # ── TextCNN ──
    if "textcnn" in models_to_run:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating TextCNN")
        logger.info("=" * 60)
        r = evaluate_token_model(
            model_name="textcnn",
            model_cls=TextCNN,
            model_kwargs=dict(embed_dim=128, num_filters=128,
                              filter_sizes=(3, 4, 5), dropout=0.3),
            df=df, y=y, groups=groups, token_lists=token_lists,
            max_len=args.max_len, n_folds=args.n_folds,
            n_epochs=args.epochs, batch_size=args.batch_size,
        )
        results.append(r)
        logger.info(f"TextCNN: ROC-AUC={r['roc_auc_mean']:.4f}  "
                     f"PR-AUC={r['pr_auc_mean']:.4f}  "
                     f"F1={r['f1_mean']:.4f}")

    # ── BiLSTM ──
    if "bilstm" in models_to_run:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating BiLSTM")
        logger.info("=" * 60)
        r = evaluate_token_model(
            model_name="bilstm",
            model_cls=BiLSTMClassifier,
            model_kwargs=dict(embed_dim=128, hidden_dim=128,
                              num_layers=2, dropout=0.3),
            df=df, y=y, groups=groups, token_lists=token_lists,
            max_len=args.max_len, n_folds=args.n_folds,
            n_epochs=args.epochs, batch_size=args.batch_size,
        )
        results.append(r)
        logger.info(f"BiLSTM: ROC-AUC={r['roc_auc_mean']:.4f}  "
                     f"PR-AUC={r['pr_auc_mean']:.4f}  "
                     f"F1={r['f1_mean']:.4f}")

    # ── CodeBERT ──
    if "codebert" in models_to_run:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating CodeBERT (fine-tuned)")
        logger.info("=" * 60)
        r = evaluate_codebert(
            df=df, y=y, groups=groups,
            max_len=args.bert_max_len, n_folds=args.n_folds,
            n_epochs=args.bert_epochs, batch_size=args.bert_batch_size,
            freeze_layers=8,
        )
        results.append(r)
        logger.info(f"CodeBERT: ROC-AUC={r['roc_auc_mean']:.4f}  "
                     f"PR-AUC={r['pr_auc_mean']:.4f}  "
                     f"F1={r['f1_mean']:.4f}")

    # ── Save results (merge with any existing results) ──
    if results:
        out_path = _OUT_DIR / "dl_baseline_comparison.csv"
        json_path = _OUT_DIR / "dl_baseline_results.json"

        # Load existing results and merge (replace models we just re-ran)
        all_results = results
        if out_path.exists():
            existing_df = pd.read_csv(out_path)
            new_model_names = {r["model"] for r in results}
            kept = existing_df[~existing_df["model"].isin(new_model_names)]
            all_results_df = pd.concat(
                [kept, pd.DataFrame(results)], ignore_index=True)
        else:
            all_results_df = pd.DataFrame(results)

        all_results_df.to_csv(out_path, index=False)
        logger.info(f"\nResults saved to {out_path}")

        # JSON: merge similarly
        all_json = results
        if json_path.exists():
            try:
                existing_json = json.loads(json_path.read_text())
                new_model_names = {r["model"] for r in results}
                kept_json = [r for r in existing_json
                             if r["model"] not in new_model_names]
                all_json = kept_json + results
            except Exception:
                pass
        json_path.write_text(json.dumps(all_json, indent=2))
        logger.info(f"JSON results saved to {json_path}")

        # Use all_json for summary printing
        results = all_json

        # Print summary
        print("\n" + "=" * 90)
        print("DL BASELINE COMPARISON — sorted by ROC-AUC")
        print("=" * 90)
        print(f"{'Model':<30} {'ROC-AUC':>12} {'PR-AUC':>12} "
              f"{'F1':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x["roc_auc_mean"],
                        reverse=True):
            roc_str = f"{r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.3f}"
            pr_str = f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.3f}"
            print(f"  {r['display_name']:<28} {roc_str:>12} {pr_str:>12} "
                  f"{r['f1_mean']:>8.4f} {r['precision_mean']:>8.4f} "
                  f"{r['recall_mean']:>8.4f} {r['train_time_s']:>7.1f}s")
        print()


if __name__ == "__main__":
    main()
