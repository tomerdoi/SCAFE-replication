#!/usr/bin/env python3
"""Audit lightweight-parser execution on a reproducible corpus sample."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ml.baseline_comparison import (  # noqa: E402
    DatasetSpec,
    load_scored_frame,
    select_dataset,
)

DEFAULT_INPUT = REPO / "data/derived/unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = REPO / "data/derived/parser_compatibility_audit"
DATASETS = ("DAppSCAN", "DIVE", "Empirical-Analysis")
CONSTRUCT_PATTERNS = {
    "assembly": r"\bassembly\b",
    "custom_error": r"\berror\s+\w+\s*\(",
    "interface": r"\binterface\s+\w+",
    "proxy_or_delegatecall": r"\bproxy\b|\.delegatecall\b",
    "receive": r"\breceive\s*\(",
    "user_defined_value_type": r"\btype\s+\w+\s+is\s+",
}


def _read_source(path_value: object) -> str:
    path = Path(str(path_value or ""))
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        if path.is_dir():
            return "\n".join(
                item.read_text(encoding="utf-8", errors="replace")
                for item in sorted(path.rglob("*.sol"))
            )
    except OSError:
        pass
    return ""


def _audit_row(row: dict[str, object]) -> dict[str, object]:
    source = _read_source(row["local_source_path"])
    result = {
        "source_dataset": row["source_dataset"],
        "contract_id": row["contract_id"],
        "compiler_version": row.get("compiler_version", ""),
        "source_chars": len(source),
        "readable_nonempty": bool(source),
        "ast_parse_ok": False,
        "exception_class": "",
    }
    result.update(
        {
            f"has_{name}": bool(re.search(pattern, source, flags=re.IGNORECASE))
            for name, pattern in CONSTRUCT_PATTERNS.items()
        }
    )
    if not source:
        result["exception_class"] = "EmptyOrUnreadableSource"
        return result
    try:
        from solidity_parser import parser as solidity_parser

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            solidity_parser.parse(source, loc=True)
        result["ast_parse_ok"] = True
    except Exception as exc:  # audit records the parser's public failure surface
        result["exception_class"] = type(exc).__name__
    return result


def _sample(frame: pd.DataFrame, per_dataset: int, seed: int) -> pd.DataFrame:
    parts = []
    for offset, dataset in enumerate(DATASETS):
        selected = select_dataset(frame, DatasetSpec(dataset))
        parts.append(
            selected.sample(
                n=min(per_dataset, len(selected)),
                random_state=seed + offset,
            )
        )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-dataset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frame = load_scored_frame(str(args.input))
    sample = _sample(frame, args.per_dataset, args.seed)
    records = sample[
        ["source_dataset", "contract_id", "compiler_version", "local_source_path"]
    ].to_dict("records")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        audited = list(executor.map(_audit_row, records, chunksize=10))
    results = pd.DataFrame(audited)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "parser_compatibility_sample.csv", index=False)
    summary_rows = []
    for dataset, group in results.groupby("source_dataset", sort=False):
        summary_rows.append(
            {
                "dataset": dataset,
                "n": int(len(group)),
                "readable_nonempty": int(group["readable_nonempty"].sum()),
                "parse_successes": int(group["ast_parse_ok"].sum()),
                "parse_success_rate": float(group["ast_parse_ok"].mean()),
            }
        )
    metadata = {
        "sampling": "uniform without replacement within canonicalized dataset",
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "parser": "solidity-parser 0.1.1",
        "parse_mode": "parse(source, loc=True)",
        "summary": summary_rows,
        "limitations": (
            "Execution success does not establish AST-field accuracy or "
            "compiler-equivalent semantics."
        ),
    }
    (args.output_dir / "parser_compatibility_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
