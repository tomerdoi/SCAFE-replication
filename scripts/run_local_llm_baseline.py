"""
A-LLM: Crash-safe, resumable local LLM baseline for vulnerability detection.

Runs zero-shot binary classification on each contract using Ollama.
Supports three models: qwen2.5-coder:7b-instruct, codellama:latest, qwen2.5:7b

Features:
- Incremental JSONL output: each prediction written immediately after inference
- Crash/shutdown safe: on restart, already-processed contracts are skipped
- Deterministic: temperature=0, seed=42
- Five-fold GroupKFold evaluation aligned with ML pipeline

Usage:
    python scripts/run_local_llm_baseline.py --model qwen2.5-coder:7b-instruct
    python scripts/run_local_llm_baseline.py --model codellama:latest
    python scripts/run_local_llm_baseline.py --model qwen2.5:7b
    python scripts/run_local_llm_baseline.py --eval-only   # compute metrics from saved JSONL
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCORED = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
_LLM_DIR = _REPO / "data" / "derived" / "llm_results"
_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert smart contract security auditor with deep knowledge of "
    "Solidity and EVM vulnerabilities including reentrancy, access control issues, "
    "integer overflow/underflow, unchecked external calls, timestamp dependence, "
    "front-running, and proxy pattern misuse."
)

USER_TEMPLATE = """\
Analyze the following Solidity smart contract and determine whether it contains security vulnerabilities.

Respond on the FIRST LINE with exactly one word: VULNERABLE or CLEAN.
Then on a second line, provide a single sentence explaining the main reason.

Contract:
```solidity
{source_code}
```"""

COT_TEMPLATE = """\
Analyze the following Solidity smart contract for security vulnerabilities.

Think step-by-step about: reentrancy, access control, arithmetic safety, unchecked calls, \
and proxy/delegatecall patterns. Then give your final answer.

Respond on the FIRST LINE with exactly one word: VULNERABLE or CLEAN.
Then on a second line, provide a single sentence explaining the main reason.

Contract:
```solidity
{source_code}
```"""

MAX_CHARS = 24_000   # ~6,000 tokens at ~4 chars/token; well within 128K context


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", model.lower()).strip("_")


def _read_source(local_source_path: str) -> str:
    """Read all .sol files under local_source_path directory (or single file)."""
    p = Path(local_source_path)
    if not p.exists():
        return ""
    sol_files = sorted(p.rglob("*.sol")) if p.is_dir() else [p]
    parts = []
    for f in sol_files:
        try:
            content = f.read_text(errors="replace")
            parts.append(f"// ===== {f.name} =====\n{content}")
        except Exception:
            pass
    combined = "\n\n".join(parts)
    return combined[:MAX_CHARS]


def _project_group(contract_id: str) -> str:
    parts = contract_id.split(":")
    if contract_id.startswith("dappscan:") and len(parts) >= 3:
        return parts[1]
    return contract_id


def _call_ollama(model: str, prompt: str, system: str, timeout: int = 120) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
        },
    }
    resp = requests.post(_OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data


def _parse_prediction(response_text: str) -> tuple[int | None, str]:
    """
    Parse model output into binary prediction and explanation.
    Returns (prediction, explanation) where prediction is 1=VULNERABLE, 0=CLEAN, None=unparseable.

    Handles:
      - "VULNERABLE" / "CLEAN" on first line (ideal)
      - "VULNERABILITY" variants (model uses noun instead of adjective)
      - "YES" / "NO" responses
      - "NOT VULNERABLE" / "SAFE" / "NO VULNERABILITY"
      - Verbose responses: scan full text for dominant signal
    """
    import re
    text = response_text.strip()
    first_line = text.split("\n")[0].strip().upper()
    explanation = text[len(text.split("\n")[0]):].strip() if "\n" in text else ""

    # --- Pass 1: first line exact keywords ---
    # "NOT VULNERABLE" before "VULNERABLE" to avoid substring match
    if re.search(r'\bNOT\s+VULNERABLE\b|\bNO\s+VULNERABILIT|\bSAFE\b|\bCLEAN\b|\bNO\b', first_line):
        # make sure it's not "NOT CLEAN" == vulnerable
        if re.search(r'\bNOT\s+CLEAN\b', first_line):
            return 1, explanation
        return 0, explanation
    if re.search(r'\bVULNERABILIT|\bVULNERABLE\b|\bYES\b', first_line):
        return 1, explanation

    # --- Pass 2: scan first 200 chars ---
    head = text[:200].upper()
    if re.search(r'\bNOT\s+VULNERABLE\b|\bNO\s+VULNERABILIT|\bNO\s+SECURITY\b|\bSAFE\b|\bCLEAN\b', head):
        return 0, explanation
    if re.search(r'\bVULNERABILIT|\bVULNERABLE\b|\bYES\b', head):
        return 1, explanation

    # --- Pass 3: count signals across full text ---
    full = text.upper()
    vuln_hits = len(re.findall(r'\bVULNERABIL|\bEXPLOIT|\bATTACK\b|\bREENTRANC|\bOVERFLOW\b', full))
    safe_hits = len(re.findall(r'\bNO\s+VULNERABILIT|\bNOT\s+VULNERABLE\b|\bSAFE\b|\bCLEAN\b|\bNO\s+ISSUE|\bNO\s+RISK', full))
    if vuln_hits > safe_hits and vuln_hits > 0:
        return 1, explanation
    if safe_hits > vuln_hits and safe_hits > 0:
        return 0, explanation

    return None, explanation


def load_existing(jsonl_path: Path) -> set[str]:
    """Return set of already-processed contract_ids that have a definitive result.

    Contracts with prediction=None (parse failure, no error) are NOT included
    so they will be retried on resume — the improved parser may succeed this time.
    Contracts with a real error (e.g. timeout) ARE included to avoid infinite retries.
    """
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                # Skip only if we have a definitive answer OR a real network error
                if rec.get("prediction") is not None or rec.get("error"):
                    done.add(rec["contract_id"])
            except Exception:
                pass
    return done


def stratified_sample(df: pd.DataFrame, max_pos: int, max_neg: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample up to max_pos positives and max_neg negatives, preserving all positives if fewer
    than max_pos exist. Negatives are sampled uniformly. Returns a shuffled DataFrame.
    """
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]

    rng = np.random.default_rng(seed)

    if len(pos) > max_pos:
        pos_idx = rng.choice(len(pos), size=max_pos, replace=False)
        pos = pos.iloc[pos_idx]

    if len(neg) > max_neg:
        neg_idx = rng.choice(len(neg), size=max_neg, replace=False)
        neg = neg.iloc[neg_idx]

    combined = pd.concat([pos, neg], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Stratified sample: %d pos, %d neg, %d total", len(pos), len(neg), len(combined))
    return combined


def run_inference(model: str, variant: str = "zero_shot",
                  max_pos: int = 2200, max_neg: int = 2200) -> Path:
    """
    Run LLM inference; resume from existing JSONL if interrupted.

    Evaluates on a stratified sample (all positives + equal negatives) for tractability.
    At 6s/contract, 4,400 contracts ≈ 7.3 hours per model.

    Parameters
    ----------
    max_pos, max_neg : maximum positives/negatives to evaluate (0 = no limit)
    """
    _LLM_DIR.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(model)
    out_path = _LLM_DIR / f"{slug}_{variant}_predictions.jsonl"

    logger.info("Loading dataset: %s", _SCORED)
    df = pd.read_csv(_SCORED, low_memory=False)
    df_vs = df[df["has_verified_source"] == 1].reset_index(drop=True)
    logger.info("Contracts with verified source: %d", len(df_vs))

    if max_pos > 0 or max_neg > 0:
        df_vs = stratified_sample(df_vs, max_pos=max_pos, max_neg=max_neg)

    done = load_existing(out_path)
    remaining = df_vs[~df_vs["contract_id"].isin(done)]
    logger.info("Already done: %d  |  Remaining: %d", len(done), len(remaining))

    if len(remaining) == 0:
        logger.info("All contracts already processed. Skipping inference.")
        return out_path

    template = COT_TEMPLATE if variant == "cot" else USER_TEMPLATE

    with out_path.open("a") as fout:
        for i, (_, row) in enumerate(remaining.iterrows()):
            contract_id = row["contract_id"]
            t0 = time.time()

            source = _read_source(str(row.get("local_source_path", "")))
            truncated = len(source) >= MAX_CHARS

            if not source:
                record = {
                    "contract_id": contract_id,
                    "label_status": row["label_status"],
                    "y_true": int(row.get("y", -1)),
                    "prediction": None,
                    "explanation": "",
                    "raw_response": "",
                    "truncated": False,
                    "elapsed_s": 0.0,
                    "error": "no_source",
                    "model": model,
                    "variant": variant,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                continue

            prompt = template.format(source_code=source)

            try:
                data = _call_ollama(model, prompt, SYSTEM_PROMPT)
                raw = data.get("response", "").strip()
                pred, expl = _parse_prediction(raw)
                error = None
            except Exception as e:
                raw, pred, expl, error = "", None, "", str(e)
                logger.warning("Error on %s: %s", contract_id, e)

            elapsed = time.time() - t0

            record = {
                "contract_id": contract_id,
                "label_status": row["label_status"],
                "y_true": int(row.get("y", -1)),
                "prediction": pred,
                "explanation": expl,
                "raw_response": raw[:2000],
                "truncated": truncated,
                "elapsed_s": round(elapsed, 2),
                "error": error,
                "model": model,
                "variant": variant,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            total_done = len(done) + i + 1
            total = len(df_vs)
            pct = total_done / total * 100
            eta_s = elapsed * (total - total_done)
            eta_h = eta_s / 3600
            logger.info(
                "[%d/%d %.1f%%] %s  pred=%s  t=%.1fs  ETA=%.1fh",
                total_done, total, pct, contract_id[:60], pred, elapsed, eta_h,
            )

    logger.info("Inference complete. Output: %s", out_path)
    return out_path


def compute_metrics(jsonl_path: Path, model: str, variant: str) -> dict:
    """Compute 5-fold GroupKFold metrics from saved JSONL predictions."""
    logger.info("Computing metrics from %s", jsonl_path)

    records = []
    with jsonl_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    df_pred = pd.DataFrame(records)
    df_pred = df_pred[df_pred["error"].isna() & df_pred["prediction"].notna()].copy()
    df_pred["prediction"] = df_pred["prediction"].astype(int)

    # Load strict CSV for group info
    df_strict = pd.read_csv(_SCORED, low_memory=False)
    df_strict = df_strict[df_strict["has_verified_source"] == 1]

    df_merged = df_strict.merge(df_pred[["contract_id", "prediction", "y_true"]], on="contract_id", how="inner")
    logger.info("Merged predictions for %d contracts", len(df_merged))

    # GroupKFold — same protocol as ML pipeline
    group_key = df_merged["contract_id"].map(_project_group)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)
    y_true = df_merged["y_true"].values
    y_pred = df_merged["prediction"].values

    gkf = GroupKFold(n_splits=5)
    fold_metrics = []
    for fold_idx, (_, test_idx) in enumerate(gkf.split(df_merged, y_true, groups)):
        yt = y_true[test_idx]
        yp = y_pred[test_idx]
        fold_metrics.append({
            "fold": fold_idx,
            "f1": f1_score(yt, yp, zero_division=0.0),
            "precision": precision_score(yt, yp, zero_division=0.0),
            "recall": recall_score(yt, yp, zero_division=0.0),
            "n": len(yt),
            "n_pos": int(yt.sum()),
        })

    fm = pd.DataFrame(fold_metrics)
    results = {
        "model": model,
        "variant": variant,
        "n_contracts": len(df_merged),
        "n_parseable": int(df_pred["prediction"].notna().sum()),
        "truncation_rate": float(df_pred["truncated"].mean()),
        "f1_mean": float(fm["f1"].mean()),
        "f1_std": float(fm["f1"].std()),
        "precision_mean": float(fm["precision"].mean()),
        "recall_mean": float(fm["recall"].mean()),
        "fold_details": fold_metrics,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {model} ({variant})")
    print(f"{'='*60}")
    print(f"  Contracts evaluated: {results['n_contracts']:,}")
    print(f"  Parseable predictions: {results['n_parseable']:,} ({results['n_parseable']/max(1,results['n_contracts']):.1%})")
    print(f"  Truncation rate: {results['truncation_rate']:.1%}")
    print(f"  F1:        {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"  Precision: {results['precision_mean']:.4f}")
    print(f"  Recall:    {results['recall_mean']:.4f}")
    for row in fold_metrics:
        print(f"    Fold {row['fold']}: F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}  n={row['n']}  pos={row['n_pos']}")

    out_json = _LLM_DIR / f"{_model_slug(model)}_{variant}_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Metrics saved → %s", out_json)
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM baseline inference for SCAFE revision")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct",
                        choices=["qwen2.5-coder:7b-instruct", "codellama:latest", "qwen2.5:7b"],
                        help="Ollama model to use")
    parser.add_argument("--variant", default="zero_shot", choices=["zero_shot", "cot"],
                        help="Prompting variant")
    parser.add_argument("--max-pos", type=int, default=2200,
                        help="Max positives to evaluate (0=all)")
    parser.add_argument("--max-neg", type=int, default=2200,
                        help="Max negatives to evaluate (0=all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference; compute metrics from existing JSONL")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    jsonl_path = _LLM_DIR / f"{slug}_{args.variant}_predictions.jsonl"

    if not args.eval_only:
        jsonl_path = run_inference(args.model, args.variant,
                                   max_pos=args.max_pos, max_neg=args.max_neg)

    compute_metrics(jsonl_path, args.model, args.variant)


if __name__ == "__main__":
    main()
