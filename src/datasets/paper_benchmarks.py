"""Adapters for exact public benchmark slices used by recent papers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class BenchmarkRecord:
    """A source-bearing, contract-level benchmark record."""

    benchmark: str
    record_id: str
    source_path: str
    label_status: str
    vuln_type: str


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def trace_cve_records(repo_root: Path) -> list[BenchmarkRecord]:
    """Load TRACE's 15 CVE-associated vulnerable Solidity contracts."""

    source_root = Path(repo_root) / "datasets" / "CVEs"
    return [
        BenchmarkRecord(
            benchmark="TRACE-CVE",
            record_id=source.stem,
            source_path=str(source.resolve()),
            label_status="vulnerable",
            vuln_type="access_control",
        )
        for source in sorted(source_root.glob("*.sol"))
    ]


def propertygpt_cve_records(repo_root: Path) -> list[BenchmarkRecord]:
    """Load the 13 contracts in PropertyGPT's published CVE table.

    The repository contains one additional Solidity file that is not part of
    the paper's 13-CVE evaluation table; the workbook is used to select the
    exact published slice rather than silently evaluating all files.
    """

    source_root = Path(repo_root) / "RQ2 cve" / "CVEs"
    workbook = Path(repo_root) / "RQ2 cve" / "CVEs_comparision.xlsx"
    table = pd.read_excel(workbook, sheet_name="Sheet1")
    names = {
        _normal_name(str(value))
        for value in table["contract_name"].dropna().tolist()
    }
    source_aliases = {
        "simplifiederc721consecutive": "simplifiederc721",
        "doftcoin": "simplifieddoftcoin",
    }
    records: list[BenchmarkRecord] = []
    for source in sorted(source_root.glob("*.sol")):
        source_name = _normal_name(source.stem)
        matched_name = next(
            (
                name
                for name in names
                if source_name == name or source_aliases.get(name) == source_name
            ),
            None,
        )
        if matched_name is None:
            continue
        row = table[
            table["contract_name"].map(
                lambda value: _normal_name(str(value)) == matched_name
            )
        ].iloc[0]
        records.append(
            BenchmarkRecord(
                benchmark="PropertyGPT-CVE",
                record_id=str(row["CVE"]),
                source_path=str(source.resolve()),
                label_status="vulnerable",
                vuln_type=str(row.get("vuln", "unknown")),
            )
        )
    return records


def llm_smartaudit_cve_records(repo_root: Path) -> list[BenchmarkRecord]:
    """Load LLM-SmartAudit's published 13-contract CVE slice."""

    source_root = Path(repo_root) / "evaluation" / "benchmark" / "cve"
    return [
        BenchmarkRecord(
            benchmark="LLM-SmartAudit-CVE",
            record_id=source.stem,
            source_path=str(source.resolve()),
            label_status="vulnerable",
            vuln_type="cve",
        )
        for source in sorted(source_root.glob("*.sol"))
    ]


def llm_smartaudit_common_records(repo_root: Path) -> list[BenchmarkRecord]:
    """Load the 110-contract common-vulnerability benchmark.

    Ten vulnerability directories contain positive examples and
    ``proc_safecontract`` is the released negative control set.
    """

    source_root = Path(repo_root) / "evaluation" / "benchmark" / "Labeled"
    records: list[BenchmarkRecord] = []
    for directory in sorted(source_root.glob("proc_*")):
        if directory.name == "proc_code3rena":
            continue
        positive = directory.name != "proc_safecontract"
        vuln_type = directory.name.removeprefix("proc_")
        for source in sorted(directory.glob("*.sol")):
            records.append(
                BenchmarkRecord(
                    benchmark="LLM-SmartAudit-common",
                    record_id=f"{vuln_type}:{source.stem}",
                    source_path=str(source.resolve()),
                    label_status="vulnerable" if positive else "clean",
                    vuln_type=vuln_type,
                )
            )
    return records


def smartbugs_wild_records(
    dataset_root: Path,
    results_root: Path,
    *,
    category: str = "reentrancy",
) -> list[BenchmarkRecord]:
    """Load SmartBugs Wild contracts with tool-derived category labels.

    The labels are explicitly marked as weak/tool-derived in the benchmark
    name.  A contract is positive when any released SmartBugs tool reported
    the requested category; otherwise it is a negative control.
    """

    dataset_root = Path(dataset_root)
    results = json.loads(
        (
            Path(results_root) / "metadata" / "results_wild.json"
        ).read_text(encoding="utf-8")
    )
    records: list[BenchmarkRecord] = []
    for source in sorted((dataset_root / "contracts").glob("*.sol")):
        address = source.stem.lower()
        result = results.get(address, {})
        tools = result.get("tools", {}) if isinstance(result, dict) else {}
        positive = any(
            int((tool.get("categories") or {}).get(category, 0)) > 0
            for tool in tools.values()
            if isinstance(tool, dict)
        )
        records.append(
            BenchmarkRecord(
                benchmark=f"SmartBugs-Wild-{category}",
                record_id=address,
                source_path=str(source.resolve()),
                label_status="vulnerable" if positive else "clean",
                vuln_type=category,
            )
        )
    return records


def load_exact_benchmarks(root: Path) -> list[BenchmarkRecord]:
    """Load all exact source-bearing slices currently available locally."""

    root = Path(root)
    return (
        trace_cve_records(root / "Trace")
        + propertygpt_cve_records(root / "PropertyGPT")
        + llm_smartaudit_cve_records(root / "LLMSmartAuditTool")
        + llm_smartaudit_common_records(root / "LLMSmartAuditTool")
    )
