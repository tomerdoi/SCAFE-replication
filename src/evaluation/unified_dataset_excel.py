"""
Build a single Excel file that unifies contract rows across the 3 local datasets under `data/raw/`:

- AccessControlVulnerabilities-main
- FORGE-Artifacts-main
- DAppSCAN-main

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# FORGE helpers
# ---------------------------------------------------------------------------

def _forge_max_severity(findings: list[dict]) -> str:
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
    seen: set = set()
    out: list = []
    for c in cwes:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= int(limit):
            break
    return ";".join(out)


# ---------------------------------------------------------------------------
# Unified row dataclass
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

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
        config_path = d / "source_code" / "config.json"
        sol_dir = d / "source_code"
        bin_dir = d / "binary"
        if not bin_dir.exists():
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


def _rows_from_forge() -> List[UnifiedRow]:
    ds_root = DATA_DIR / "raw" / "FORGE-Artifacts-main"
    results_root = ds_root / "dataset" / "results"
    contracts_root = ds_root / "dataset" / "contracts"
    rows: List[UnifiedRow] = []
    if not results_root.exists():
        return rows

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
        if isinstance(pp, dict):
            for _, v in pp.items():
                if isinstance(v, str) and v:
                    project_path_rel[(chain, addr)] = v
                    break

    for (chain, addr), findings in agg_findings.items():
        max_sev = _forge_max_severity(findings)
        sev_order = {"": -1, "info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        if sev_order.get(max_sev, -1) >= 2:
            label_status = "vulnerable"
        elif sev_order.get(max_sev, -1) >= 0:
            label_status = "suspicious"
        else:
            label_status = "unknown"

        vuln_present = int(label_status in {"vulnerable", "suspicious"})
        vuln_types = _forge_cwe_list(findings, limit=25)

        local_source = ""
        rel = project_path_rel.get((chain, addr), "")
        if rel:
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


def _rows_from_dappscan() -> List[UnifiedRow]:
    """
    Load DAppSCAN-source: each .sol file under contracts/ is a row.
    SWC labels come from matching JSON files under SWCsource/.

    - Files WITH at least one SWC finding → label_status='vulnerable', vuln_present=1
    - Files with NO SWC finding → label_status='clean', vuln_present=0
    """
    ds_root = DATA_DIR / "raw" / "DAppSCAN-main" / "DAppSCAN-source"
    contracts_dir = ds_root / "contracts"
    swc_dir = ds_root / "SWCsource"
    bytecode_dir = DATA_DIR / "raw" / "DAppSCAN-main" / "DAppSCAN-bytecode"

    rows: List[UnifiedRow] = []

    if not contracts_dir.exists():
        return rows

    # ---- Build SWC label index: relpath → list of SWC categories ----
    # The JSON filePath field looks like "DAppSCAN-source/contracts/<project>/.../<File>.sol"
    # We normalize to "contracts/<project>/.../<File>.sol" for matching.
    swc_by_relpath: Dict[str, List[str]] = defaultdict(list)
    labeled_relpaths: Set[str] = set()

    if swc_dir.exists():
        for jp in swc_dir.rglob("*.json"):
            try:
                obj = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            fp = str(obj.get("filePath") or "")
            # strip leading "DAppSCAN-source/" prefix to get relative to ds_root
            fp = fp.replace("DAppSCAN-source/", "", 1)
            swcs = obj.get("SWCs", [])
            if isinstance(swcs, list):
                for swc in swcs:
                    cat = str(swc.get("category", "")) if isinstance(swc, dict) else ""
                    if cat:
                        swc_by_relpath[fp].append(cat)
            if fp:
                labeled_relpaths.add(fp)

    # ---- Walk all .sol files under contracts/ ----
    all_sol = sorted(contracts_dir.rglob("*.sol"))

    for sol_path in all_sol:
        relpath = str(sol_path.relative_to(ds_root))  # "contracts/<project>/.../<File>.sol"

        # Extract project name (first dir component after contracts/)
        parts = sol_path.relative_to(contracts_dir).parts
        project = parts[0] if parts else ""

        # Try to extract an address from the filename or path
        addr = ""
        for m in re.finditer(r"(0x[a-fA-F0-9]{40})", sol_path.name):
            addr = m.group(1).lower()
            break
        if not addr:
            for m in re.finditer(r"(0x[a-fA-F0-9]{40})", str(sol_path)):
                addr = m.group(1).lower()
                break

        # SWC labels for this file
        swc_cats = swc_by_relpath.get(relpath, [])
        if swc_cats:
            label_status = "vulnerable"
            vuln_present = 1
            # Deduplicate and join SWC categories
            seen: set = set()
            unique_cats: list = []
            for c in swc_cats:
                if c not in seen:
                    seen.add(c)
                    unique_cats.append(c)
            vuln_types = ";".join(unique_cats)
        else:
            label_status = "clean"
            vuln_present = 0
            vuln_types = ""

        # Bytecode path (best-effort: check SWCbytecode dir for matching JSON)
        bytecode_path = ""

        rows.append(
            UnifiedRow(
                source_dataset="DAppSCAN",
                source_subdataset=project,
                chain="eth",
                address=addr,
                contract_id=f"dappscan:{project}:{sol_path.stem}",
                label_status=label_status,
                vuln_present=vuln_present,
                vuln_types=vuln_types,
                vuln_severity="",
                local_source_path=str(sol_path),
                local_bytecode_path=bytecode_path,
                local_metadata_path=str(sol_path),
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Build + write
# ---------------------------------------------------------------------------

def build_unified_df() -> pd.DataFrame:
    rows: List[UnifiedRow] = []
    rows.extend(_rows_from_access_control())
    rows.extend(_rows_from_forge())
    rows.extend(_rows_from_dappscan())

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

    df2 = df.copy()
    df2 = df2.replace({pd.NA: "", float("nan"): ""})

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df2.to_excel(w, sheet_name="contracts", index=False)

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

    # Quick summary
    print("\n--- Dataset summary ---")
    for ds, g in df.groupby("source_dataset"):
        n_vuln = int((g["vuln_present"] == 1).sum())
        n_clean = int((g["vuln_present"] == 0).sum())
        n_addr = g["address"].replace("", pd.NA).dropna().nunique()
        print(f"  {ds}: {len(g)} rows, {n_addr} unique addresses, {n_vuln} vulnerable, {n_clean} clean")
