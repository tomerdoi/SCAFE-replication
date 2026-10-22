#!/usr/bin/env python3
"""Evaluate zero-shot Laya vulnerability ranking on a small DAppSCAN sample.

This is an exploratory record-level evaluation, not a trained detector.  The
sample is balanced and source-length matched, DAppSCAN SWC annotations are
removed before inference, and every completed prediction is appended to disk
so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.source_sanitization import (  # noqa: E402
    SOURCE_SANITIZATION_VERSION,
    read_source_record,
)


DEFAULT_INPUT = (
    REPO
    / "data/derived/sanitized_v1/sampled_campaign_v1/scored_sample.csv"
)
DEFAULT_OUTPUT = (
    REPO / "data/derived/sanitized_v1/ml_results/laya_dappscan_poc"
)
QUESTION = {
    "vulnerable": {
        "type": "noul",
        "instructions": (
            "Does the Solidity code contain a concrete, exploitable security "
            "vulnerability? Mark true only when code evidence supports a flaw "
            "such as reentrancy, an unchecked low-level call, broken access "
            "control or unprotected withdrawal, unsafe delegatecall or "
            "selfdestruct, exploitable arithmetic, timestamp dependence, "
            "denial of service, or an equivalent vulnerability. Do not mark "
            "code vulnerable merely because it is complex, makes an external "
            "call, uses old syntax, has suspicious names, or lacks context."
        ),
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-chars", type=int, default=16000)
    parser.add_argument("--model", default="typed-decisions")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected records (useful for a smoke test).",
    )
    return parser.parse_args()


def bounded_source(source: str, limit: int) -> str:
    """Retain deterministic head, middle, and tail slices of long source."""

    if limit <= 0:
        raise ValueError("--max-source-chars must be positive")
    if len(source) <= limit:
        return source
    marker = "\n\n/* ... deterministic source slice boundary ... */\n\n"
    available = limit - 2 * len(marker)
    if available < 300:
        return source[:limit]
    head = available // 3
    middle = available // 3
    tail = available - head - middle
    midpoint = len(source) // 2
    middle_start = max(head, midpoint - middle // 2)
    middle_end = min(len(source) - tail, middle_start + middle)
    return (
        source[:head]
        + marker
        + source[middle_start:middle_end]
        + marker
        + source[-tail:]
    )


def load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "contract_id",
        "source_dataset",
        "local_source_path",
        "_source_identity",
        "y",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    frame = frame[
        frame["source_dataset"].astype(str).str.casefold().eq("dappscan")
    ].copy()
    frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(int)
    if set(frame["y"].unique()) != {0, 1}:
        raise ValueError("DAppSCAN input must contain both binary labels")
    if frame["_source_identity"].duplicated().any():
        raise ValueError("Input contains duplicate source identities")

    sources: list[str] = []
    for row in frame.itertuples(index=False):
        sources.append(
            read_source_record(row.local_source_path, row.source_dataset)
        )
    frame["_sanitized_source"] = sources
    frame["_source_chars"] = [len(source) for source in sources]
    frame = frame[frame["_source_chars"].gt(0)].copy()
    return frame


def select_length_matched(
    frame: pd.DataFrame, *, per_class: int, seed: int
) -> pd.DataFrame:
    """Randomly choose positives and greedily match unused negatives by length."""

    positives = frame[frame["y"].eq(1)]
    negatives = frame[frame["y"].eq(0)]
    if per_class <= 0 or per_class > min(len(positives), len(negatives)):
        raise ValueError(
            f"--per-class must be in 1..{min(len(positives), len(negatives))}"
        )
    chosen_positive = positives.sample(n=per_class, random_state=seed)
    available = negatives.copy()
    selected: list[pd.Series] = []
    for (_, positive) in chosen_positive.sort_values(
        "_source_identity"
    ).iterrows():
        target = math.log1p(int(positive["_source_chars"]))
        distances = (
            np.log1p(available["_source_chars"].astype(float)) - target
        ).abs()
        nearest = available.assign(_distance=distances).sort_values(
            ["_distance", "_source_identity"]
        ).iloc[0]
        selected.extend([positive, nearest.drop(labels="_distance")])
        available = available.drop(index=nearest.name)
    result = pd.DataFrame(selected).reset_index(drop=True)
    return result.sort_values(
        ["_source_identity"], kind="stable"
    ).reset_index(drop=True)


def config_payload(args: argparse.Namespace, selected: pd.DataFrame) -> dict[str, Any]:
    return {
        "version": 1,
        "input": str(args.input.resolve()),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "per_class": args.per_class,
        "max_source_chars": args.max_source_chars,
        "window": args.window,
        "question": QUESTION,
        "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "selected_source_identities": sorted(
            selected["_source_identity"].astype(str).tolist()
        ),
    }


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            completed[str(record["source_identity"])] = record
    return completed


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def bootstrap_interval(
    y: np.ndarray,
    score: np.ndarray,
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if iterations <= 0:
        return None
    rng = np.random.default_rng(seed)
    by_label = [np.flatnonzero(y == label) for label in (0, 1)]
    values: list[float] = []
    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(items, size=len(items), replace=True) for items in by_label]
        )
        values.append(float(scorer(y[indices], score[indices])))
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def summarize(
    records: list[dict[str, Any]], *, bootstrap: int, seed: int
) -> dict[str, Any]:
    y = np.asarray([record["label"] for record in records], dtype=int)
    score = np.asarray(
        [record["vulnerability_probability"] for record in records], dtype=float
    )
    prediction = (score >= 0.5).astype(int)
    source_chars = np.asarray(
        [record["source_chars"] for record in records], dtype=float
    )
    windows = np.asarray([record["windows"] for record in records], dtype=float)
    length_auc = float(roc_auc_score(y, source_chars))
    rho_chars = spearmanr(score, source_chars).statistic
    rho_windows = spearmanr(score, windows).statistic
    return {
        "records": int(len(records)),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "bootstrap_iterations": int(bootstrap),
        "roc_auc": float(roc_auc_score(y, score)),
        "roc_auc_ci95_stratified_bootstrap": bootstrap_interval(
            y, score, metric="roc_auc", iterations=bootstrap, seed=seed
        ),
        "average_precision_balanced_sample": float(
            average_precision_score(y, score)
        ),
        "average_precision_ci95_stratified_bootstrap": bootstrap_interval(
            y, score, metric="average_precision", iterations=bootstrap, seed=seed + 1
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y, prediction)
        ),
        "precision_at_0_5": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "recall_at_0_5": float(recall_score(y, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, prediction, zero_division=0)),
        "score": {
            "min": float(score.min()),
            "median": float(np.median(score)),
            "max": float(score.max()),
            "positive_mean": float(score[y == 1].mean()),
            "negative_mean": float(score[y == 0].mean()),
        },
        "nuisance_checks": {
            "source_length_roc_auc": length_auc,
            "score_source_length_spearman": (
                None if np.isnan(rho_chars) else float(rho_chars)
            ),
            "score_window_count_spearman": (
                None if np.isnan(rho_windows) else float(rho_windows)
            ),
        },
        "interpretation_limits": [
            "Exploratory zero-shot record-level evaluation; no Laya fine-tuning.",
            "Balanced case-control sampling; average precision is not a natural-prevalence estimate.",
            "Confidence is uncalibrated for Solidity vulnerability detection.",
            "Long files are bounded with deterministic head/middle/tail slices.",
            "DAppSCAN labels are record-level and may not identify every vulnerable function.",
        ],
    }


def main() -> int:
    args = parse_args()
    selected = select_length_matched(
        load_candidates(args.input.resolve()),
        per_class=args.per_class,
        seed=args.seed,
    )
    config = config_payload(args, selected)
    run_id = stable_hash(config)[:12]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{run_id}.json"
    results_path = output_dir / f"predictions_{run_id}.jsonl"
    summary_path = output_dir / f"summary_{run_id}.json"
    manifest_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completed = read_completed(results_path)
    work = selected
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        work = work.iloc[: args.limit]
    pending = [
        row
        for _, row in work.iterrows()
        if str(row["_source_identity"]) not in completed
    ]
    if pending:
        from laya import Router

        router = Router(device=args.device, max_loaded=1)
        for position, row in enumerate(pending, start=1):
            source = bounded_source(
                str(row["_sanitized_source"]), args.max_source_chars
            )
            state = (
                "Task: assess the supplied Solidity source for a concrete "
                "exploitable vulnerability. The source has been stripped of "
                "dataset label annotations.\n\n"
                "SOLIDITY SOURCE\n"
                "===============\n"
                f"{source}"
            )
            started = time.perf_counter()
            result = router.predict_long(
                state,
                QUESTION,
                model=args.model,
                window=args.window,
                batch_size=args.batch_size,
            )
            elapsed = time.perf_counter() - started
            answer = result["answers"]["vulnerable"]
            record = {
                "run_id": run_id,
                "contract_id": str(row["contract_id"]),
                "source_identity": str(row["_source_identity"]),
                "source_path": str(row["local_source_path"]),
                "label": int(row["y"]),
                "vulnerability_probability": float(answer["noul"]),
                "answer_confidence": float(answer.get("answer_confidence", 0.0)),
                "source_chars": int(row["_source_chars"]),
                "model_visible_chars": len(source),
                "windows": int(result.get("usage", {}).get("windows", 1)),
                "deciding_window": answer.get("window"),
                "routing": result.get("routing"),
                "latency_seconds": elapsed,
            }
            append_record(results_path, record)
            completed[record["source_identity"]] = record
            print(
                f"[{position}/{len(pending)}] y={record['label']} "
                f"p={record['vulnerability_probability']:.4f} "
                f"windows={record['windows']} "
                f"{record['contract_id']}",
                flush=True,
            )

    selected_ids = set(work["_source_identity"].astype(str))
    records = [
        record
        for identity, record in completed.items()
        if identity in selected_ids
    ]
    if len(records) == len(work) and {record["label"] for record in records} == {0, 1}:
        summary = {
            "run_id": run_id,
            "manifest": str(manifest_path),
            "predictions": str(results_path),
            **summarize(records, bootstrap=args.bootstrap, seed=args.seed),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed": len(records),
                    "selected": len(work),
                    "predictions": str(results_path),
                    "resume": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
