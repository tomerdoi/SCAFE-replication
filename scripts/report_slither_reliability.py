#!/usr/bin/env python3
"""Build an auditable parser/Slither reliability report from existing runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "data/derived/slither_results/slither_per_contract.csv"
OUT = REPO / "data/derived/slither_reliability"
CONSTRUCTS = {
    "delegatecall": re.compile(r"\bdelegatecall\s*\(", re.I),
    "inline_assembly": re.compile(r"\bassembly\s*\{", re.I),
    "interfaces": re.compile(r"\binterface\s+[A-Za-z_]\w*", re.I),
    "proxy_indicators": re.compile(
        r"\b(?:implementation|upgradeTo|upgradeToAndCall|ERC1967|TransparentUpgradeableProxy|UUPSUpgradeable)\b",
        re.I,
    ),
}


def classify_source(path: str) -> dict[str, bool]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return {name: bool(pattern.search(text)) for name, pattern in CONSTRUCTS.items()}


def summarize(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    result = (
        frame.groupby(group, dropna=False, sort=True)
        .agg(n=("slither_success", "size"), n_success=("slither_success", "sum"))
        .reset_index()
    )
    result["n_failed"] = result["n"] - result["n_success"]
    result["success_rate"] = result["n_success"] / result["n"].where(result["n"] > 0, 1)
    result["fail_rate"] = 1.0 - result["success_rate"]
    return result


def main() -> None:
    frame = pd.read_csv(INPUT, low_memory=False)
    frame["slither_success"] = frame["slither_success"].astype(bool)
    frame["sol_major_minor"] = (
        frame["solc_version"].fillna("unknown").astype(str).str.extract(r"^(\d+\.\d+)", expand=False).fillna("unknown")
    )
    construct_frame = frame["sol_path"].map(classify_source).apply(pd.Series)
    frame = pd.concat([frame, construct_frame], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    summarize(frame, ["sol_major_minor"]).to_csv(OUT / "by_solidity_version.csv", index=False)
    summarize(frame, list(CONSTRUCTS)).to_csv(OUT / "by_source_construct.csv", index=False)
    summary = {
        "input_rows": int(len(frame)),
        "successes": int(frame["slither_success"].sum()),
        "failures": int((~frame["slither_success"]).sum()),
        "success_rate": float(frame["slither_success"].mean()),
        "coverage_of_input": float(frame["slither_success"].mean()),
        "construct_definitions": list(CONSTRUCTS),
        "correlation_artifact": "data/derived/slither_correlation/correlation_results.csv",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
