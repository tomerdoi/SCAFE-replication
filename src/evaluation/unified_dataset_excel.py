"""
Build a single Excel file that unifies contract rows across the 4 local datasets under `data/raw/`:

- AccessControlVulnerabilities-main
- BadRandomness-SWC120-Dataset-main
- Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts-main
- FORGE-Artifacts-main

The output is intended for *risk score / ML model* evaluation from a single table.

Output (default):
- data/derived/unified_contract_eval_dataset.xlsx

Quick run:
  python3 -m src.evaluation.unified_dataset_excel
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Allow running as a file in IDE/debugger (so `import src...` works)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_DIR, DERIVED_DIR  # noqa: E402


def _is_addr(s: object) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _norm_addr(s: object) -> str:
    return str(s or "").strip().lower()


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_empirical_vuln_types(tag: object) -> Tuple[int, str]:
    t = str(tag or "").strip().lower().strip(";").strip()
    if not t or t in {"no", "none"}:
        return 0, ""
    t = re.sub(r"\b\d+\s*:\s*", "", t)
    kinds = []
    for k in [
        "access_control",
        "reentrancy",
        "time_manipulation",
        "unchecked_low_calls",
        "arithmetic",
        "front_running",
    ]:
        if k in t:
            kinds.append(k)
    # fall back: store raw tag if we couldn't categorize
    if not kinds:
        kinds.append("other")
    return 1, ";".join(sorted(set(kinds)))


def _forge_max_severity(findings: list[dict]) -> str:
    # Ordered severities used by FORGE JSONs
    order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    best = None
    best_v = -1
    for f in findings or []:
        s = str((f or {}).get("severity") or "").strip().lower()
        if s in order and order[s] > best_v:
            best = s
            best_v = order[s]
    return best or ""


def _forge_cwe_list(findings: list[dict], limit: int = 25) -> str:
    cwes: List[str] = []
    for f in findings or []:
        cat = (f or {}).get("category") or {}
        if isinstance(cat, dict):
            for _, arr in cat.items():
                if isinstance(arr, list):
                    for x in arr:
                        xs = str(x).strip()
                        if xs and xs.startswith("CWE-"):
                            cwes.append(xs)
    # de-dup while preserving order
    seen = set()
    out = []
    for c in cwes:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= int(limit):
            break
    return ";".join(out)


def _bad_randomness_indicator(sol_text: str) -> int:
    """
    Best-effort heuristic for SWC-120 style bad randomness sources.
    (Not a ground-truth label; just a helpful column.)
    """
    t = (sol_text or "").lower()
    needles = [
        "block.timestamp",
        " now",
        "blockhash(",
        "block.difficulty",
        "block.prevrandao",
        "block.number",
        "block.coinbase",
    ]
    return int(any(n in t for n in needles))


@dataclass(frozen=True)
class UnifiedRow:
    source_dataset: str
    source_subdataset: str
    chain: str
    address: str
    contract_id: str
    label_status: str  # clean / vulnerable / suspicious / unknown
    vuln_present: int
    vuln_types: str
    vuln_severity: str
    local_source_path: str
    local_bytecode_path: str
    local_metadata_path: str


def _rows_from_empirical() -> List[UnifiedRow]:
    ds_root = DATA_DIR / "raw" / "Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts-main"
    inp = ds_root / "csvs" / "sample_of_interest.csv"
    if not inp.exists():
        return []
    df = pd.read_csv(inp)
    rows: List[UnifiedRow] = []

    for _, r in df.iterrows():
        addr = _norm_addr(r.get("contract"))
        if not _is_addr(addr):
            continue
        sub = str(r.get("label") or "").strip()
        tag = r.get("tag")
        vuln_present, vuln_types = _parse_empirical_vuln_types(tag)
        label_status = "vulnerable" if vuln_present else "clean"

        # local paths (best-effort)
        sol_path = ""
        if sub == "smartbugs_results":
            sol = ds_root / "contracts" / "sb_results_contracts" / f"{addr}.sol"
            if sol.exists():
                sol_path = str(sol)
        elif sub == "smartbugs_curated":
            sol = ds_root / "contracts" / "smartbugs_curated" / f"{addr}.sol"
            if sol.exists():
                sol_path = str(sol)
        elif sub == "zeus_safe":
            sol = ds_root / "contracts" / "zeus_safe_contracts" / f"{addr}.sol"
            if sol.exists():
                sol_path = str(sol)
        elif sub == "zeus_vulnerable":
            sol = ds_root / "contracts" / "zeus_vulnerable_contracts" / f"{addr}.sol"
            if sol.exists():
                sol_path = str(sol)

        hex_path = ""
        # the repo mixes *.hex and *.sol.hex
        for cand in [
            ds_root / "bytecode" / "smartbugs_results" / f"{addr}.hex",
            ds_root / "bytecode" / "smartbugs_curated" / f"{addr}.sol.hex",
            ds_root / "bytecode" / "smartbugs_curated" / f"{addr}.hex",
            ds_root / "bytecode" / "zeus_safe" / f"{addr}.hex",
            ds_root / "bytecode" / "zeus_vulnerable" / f"{addr}.hex",
        ]:
            if cand.exists():
                hex_path = str(cand)
                break

        rows.append(
            UnifiedRow(
                source_dataset="Empirical-Analysis",
                source_subdataset=sub,
                chain="eth",
                address=addr,
                contract_id=f"empirical:{sub}:{addr}",
                label_status=label_status,
                vuln_present=vuln_present,
                vuln_types=vuln_types,
                vuln_severity="",
                local_source_path=sol_path,
                local_bytecode_path=hex_path,
                local_metadata_path=str(inp),
            )
        )

    # Deduplicate exact duplicates (same address+sub+tag) by contract_id
    dedup: Dict[str, UnifiedRow] = {}
    for row in rows:
        dedup[row.contract_id] = row
    return list(dedup.values())


def _rows_from_access_control() -> List[UnifiedRow]:
    ds_root = DATA_DIR / "raw" / "AccessControlVulnerabilities-main" / "datasets" / "DeFiHackLabsCVEs"
    rows: List[UnifiedRow] = []

    if not ds_root.exists():
        return rows

    for d in ds_root.iterdir():
        if not d.is_dir() or "@0x" not in d.name:
            continue
        chain, addr_raw = d.name.split("@", 1)
        addr = _norm_addr(addr_raw)
        if not _is_addr(addr):
            continue
        # treat these as positive access-control related cases (this repo is an access-control benchmark)
        config_path = d / "source_code" / "config.json"
        sol_dir = d / "source_code"
        # bytecode artifacts exist but are split (bin/bin-runtime); keep directory path
        bin_dir = d / "binary"
        if not bin_dir.exists():
            # some datasets place bytecode at root
            bin_dir = d

        rows.append(
            UnifiedRow(
                source_dataset="AccessControlVulnerabilities",
                source_subdataset="DeFiHackLabsCVEs",
                chain=str(chain).strip().lower(),
                address=addr,
                contract_id=f"acv:defihack:{chain}:{addr}",
                label_status="vulnerable",
                vuln_present=1,
                vuln_types="access_control",
                vuln_severity="",
                local_source_path=str(sol_dir) if sol_dir.exists() else "",
                local_bytecode_path=str(bin_dir) if bin_dir.exists() else "",
                local_metadata_path=str(config_path) if config_path.exists() else str(d),
            )
        )

    return rows


def _rows_from_bad_randomness() -> List[UnifiedRow]:
    ds_root = DATA_DIR / "raw" / "BadRandomness-SWC120-Dataset-main" / "ground_truth"
    rows: List[UnifiedRow] = []
    if not ds_root.exists():
        return rows

    for p in ds_root.glob("*.sol"):
        name = p.name
        addr = ""
        m = re.match(r"^(0x[a-fA-F0-9]{40})_", name)
        if m:
            addr = m.group(1).lower()

        text = _safe_read_text(p)
        is_bad_rand = _bad_randomness_indicator(text)
        vuln_types = "bad_randomness_swc120" if is_bad_rand else ""

        rows.append(
            UnifiedRow(
                source_dataset="BadRandomness-SWC120",
                source_subdataset="ground_truth",
                chain="eth" if addr else "",
                address=addr,
                contract_id=f"bad_randomness:{addr or p.stem}",
                label_status="unknown",  # ground_truth folder includes both safe and vulnerable examples
                vuln_present=int(is_bad_rand),
                vuln_types=vuln_types,
                vuln_severity="",
                local_source_path=str(p),
                local_bytecode_path="",
                local_metadata_path=str(p),
            )
        )
    return rows


def _rows_from_forge() -> List[UnifiedRow]:
    ds_root = DATA_DIR / "raw" / "FORGE-Artifacts-main"
    results_root = ds_root / "dataset" / "results"
    contracts_root = ds_root / "dataset" / "contracts"
    rows: List[UnifiedRow] = []
    if not results_root.exists():
        return rows

    # Aggregate multiple JSONs per (chain,address)
    agg_findings: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    any_json_path: Dict[Tuple[str, str], str] = {}
    project_path_rel: Dict[Tuple[str, str], str] = {}

    for jp in results_root.glob("*.json"):
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = obj.get("project_info", {}) if isinstance(obj, dict) else {}
        chain = str(info.get("chain") or "").strip().lower()
        addr = _norm_addr(info.get("address"))
        if not chain or not _is_addr(addr):
            continue
        findings = obj.get("findings", []) if isinstance(obj, dict) else []
        if isinstance(findings, list):
            agg_findings[(chain, addr)].extend([f for f in findings if isinstance(f, dict)])
        any_json_path[(chain, addr)] = str(jp)
        pp = info.get("project_path")
        # project_path is dict: {project_name: "contracts/<...>"}
        if isinstance(pp, dict):
            for _, v in pp.items():
                if isinstance(v, str) and v:
                    project_path_rel[(chain, addr)] = v
                    break

    for (chain, addr), findings in agg_findings.items():
        max_sev = _forge_max_severity(findings)
        # Heuristic label: treat medium+ as "vulnerable", else "suspicious" if only low/info
        sev_order = {"": -1, "info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        if sev_order.get(max_sev, -1) >= 2:
            label_status = "vulnerable"
        elif sev_order.get(max_sev, -1) >= 0:
            label_status = "suspicious"
        else:
            label_status = "unknown"

        vuln_present = int(label_status in {"vulnerable", "suspicious"})
        vuln_types = _forge_cwe_list(findings, limit=25)

        # local source path: use project_path if present (points into dataset/contracts/)
        local_source = ""
        rel = project_path_rel.get((chain, addr), "")
        if rel:
            # rel already begins with "contracts/..."
            local_source = str((ds_root / "dataset" / rel).resolve())
        else:
            local_source = str(contracts_root) if contracts_root.exists() else ""

        rows.append(
            UnifiedRow(
                source_dataset="FORGE",
                source_subdataset="audit_reports",
                chain=chain,
                address=addr,
                contract_id=f"forge:{chain}:{addr}",
                label_status=label_status,
                vuln_present=vuln_present,
                vuln_types=vuln_types,
                vuln_severity=max_sev,
                local_source_path=local_source,
                local_bytecode_path="",
                local_metadata_path=any_json_path.get((chain, addr), ""),
            )
        )

    return rows


def build_unified_df() -> pd.DataFrame:
    rows: List[UnifiedRow] = []
    rows.extend(_rows_from_empirical())
    rows.extend(_rows_from_access_control())
    rows.extend(_rows_from_bad_randomness())
    rows.extend(_rows_from_forge())

    df = pd.DataFrame([r.__dict__ for r in rows])

    # A stable row id for referencing rows in papers/analysis
    df.insert(0, "row_id", [f"row_{i:06d}" for i in range(len(df))])

    # Normalize address empty -> NaN-like empty string (Excel-friendly)
    df["address"] = df["address"].fillna("").astype(str)
    df["chain"] = df["chain"].fillna("").astype(str)

    # Helpful summary columns
    df["is_eth"] = (df["chain"].str.lower() == "eth").astype(int)

    return df


def write_excel(df: pd.DataFrame, out_xlsx: Path) -> Path:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # Ensure we don't accidentally write infinities/NaNs that Excel hates.
    df2 = df.copy()
    df2 = df2.replace({pd.NA: "", float("nan"): ""})

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df2.to_excel(w, sheet_name="contracts", index=False)

        # lightweight summary sheet
        summary = (
            df2.groupby(["source_dataset", "chain"], dropna=False)
            .agg(rows=("contract_id", "count"), unique_addresses=("address", lambda s: len(set([x for x in s if x]))))
            .reset_index()
            .sort_values(["source_dataset", "chain"])
        )
        summary.to_excel(w, sheet_name="summary", index=False)

    return out_xlsx


if __name__ == "__main__":
    out_xlsx = DERIVED_DIR / "unified_contract_eval_dataset.xlsx"
    df = build_unified_df()
    p = write_excel(df, out_xlsx)
    print(f"Wrote: {p} (rows={len(df)})")

