"""
Contract registry builder for the contract-risk-score pipeline.

The goal is to define the *universe of contract addresses* to score.

Default behavior (no extra inputs):
- Use AVS registry + Dune "related contracts" to produce a restaking/AVS-focused contract universe.
  (This keeps continuity with the existing repo data while pivoting the scoring unit to contracts.)

Outputs:
- data/contract_registry.csv with columns: address, sources
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


# src/registry/contract_registry.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _is_addr(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_addresses_file(path: Path) -> List[str]:
    """
    Best-effort address list loader.
    Supports:
    - .csv with an `address` column
    - .txt with one address per line
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Addresses file not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "address" not in df.columns:
            raise ValueError(f"CSV must have an `address` column: {p}")
        addrs = [str(x).strip().lower() for x in df["address"].tolist()]
        return [a for a in addrs if _is_addr(a)]
    # txt/newline list
    lines = (p.read_text(encoding="utf-8", errors="ignore") or "").splitlines()
    addrs = [ln.strip().lower() for ln in lines if ln.strip()]
    return [a for a in addrs if _is_addr(a)]


@dataclass(frozen=True, slots=True)
class ContractRegistrySources:
    avs_registry_all: Path = DATA_DIR / "avs_registry_all.json"
    dune_avs_related_contracts: Path = DATA_DIR / "dune_avs_related_contracts_all.json"
    u1_latest_avs_balances: Path = DATA_DIR / "u1_latest_avs_balances.json"


class ContractRegistryUtil:
    def __init__(self, sources: Optional[ContractRegistrySources] = None) -> None:
        logger = logging.getLogger("contract_registry_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.sources = sources or ContractRegistrySources()

    def _add(self, addr: str, source: str, out: Dict[str, Set[str]]) -> None:
        a = str(addr).strip().lower()
        if not _is_addr(a):
            return
        out.setdefault(a, set()).add(source)

    def _extract_avs_registry(self, out: Dict[str, Set[str]]) -> int:
        p = self.sources.avs_registry_all
        if not p.exists():
            return 0
        obj = _read_json(p)
        rows = obj.get("data", []) if isinstance(obj, dict) else []
        n = 0
        for r in rows:
            if isinstance(r, dict) and _is_addr(r.get("address")):
                self._add(r["address"], "avs_registry_all:avs_address", out)
                n += 1
        return n

    def _extract_u1_avs_addresses(self, out: Dict[str, Set[str]]) -> int:
        p = self.sources.u1_latest_avs_balances
        if not p.exists():
            return 0
        obj = _read_json(p)
        if not isinstance(obj, dict):
            return 0
        n = 0
        for k, v in obj.items():
            if _is_addr(k):
                self._add(k, "u1_latest_avs_balances:key", out)
                n += 1
            if isinstance(v, dict) and _is_addr(v.get("avsAddress")):
                self._add(v["avsAddress"], "u1_latest_avs_balances:value", out)
        return n

    def _extract_related_contracts(self, out: Dict[str, Set[str]]) -> int:
        """
        Pull related contracts discovered by Dune for AVSs; includes both avs and related contract addresses.
        """
        p = self.sources.dune_avs_related_contracts
        if not p.exists():
            return 0
        obj = _read_json(p)
        rows = obj.get("result", {}).get("rows", []) if isinstance(obj, dict) else []
        if not isinstance(rows, list):
            return 0
        n = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            avs = r.get("avs_address") or r.get("avsAddress")
            rel = r.get("related_contract_address") or r.get("relatedContractAddress")
            if _is_addr(avs):
                self._add(avs, "dune_related_contracts:avs_address", out)
                n += 1
            if _is_addr(rel):
                self._add(rel, "dune_related_contracts:related_contract", out)
                n += 1
        return n

    def build_registry(
        self,
        *,
        include_avs: bool = True,
        include_related_contracts: bool = True,
        include_u1_avs: bool = True,
        extra_addresses: Optional[Iterable[str]] = None,
        extra_addresses_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        out: Dict[str, Set[str]] = {}
        hits = 0

        if include_avs:
            hits += self._extract_avs_registry(out)
        if include_u1_avs:
            hits += self._extract_u1_avs_addresses(out)
        if include_related_contracts:
            hits += self._extract_related_contracts(out)

        if extra_addresses_file is not None:
            for a in _read_addresses_file(Path(extra_addresses_file)):
                self._add(a, f"extra_file:{Path(extra_addresses_file).name}", out)
        if extra_addresses is not None:
            for a in extra_addresses:
                self._add(str(a), "extra_inline", out)

        rows = [{"address": a, "sources": "|".join(sorted(list(srcs)))} for a, srcs in sorted(out.items())]
        df = pd.DataFrame(rows)
        self.logger.info(f"Built contract registry: {len(df)} unique addresses (source_hits={hits})")
        return df

    def write_registry(self, df: pd.DataFrame, *, out_csv: Path) -> Path:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        self.logger.info(f"Wrote contract registry: {out_csv} (rows={len(df)})")
        return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contract registry (address universe) for contract risk scoring.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DATA_DIR / "contract_registry.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--no-avs", action="store_true", help="Do not include AVS registry addresses.")
    parser.add_argument("--no-related", action="store_true", help="Do not include Dune related contracts.")
    parser.add_argument("--no-u1-avs", action="store_true", help="Do not include u--1 AVS addresses.")
    parser.add_argument(
        "--extra-addresses-file",
        type=str,
        default="",
        help="Optional path to a .csv (with `address`) or .txt (one per line) to add addresses.",
    )
    args = parser.parse_args()

    util = ContractRegistryUtil()
    df = util.build_registry(
        include_avs=not bool(args.no_avs),
        include_related_contracts=not bool(args.no_related),
        include_u1_avs=not bool(args.no_u1_avs),
        extra_addresses_file=Path(args.extra_addresses_file) if str(args.extra_addresses_file).strip() else None,
    )
    util.write_registry(df, out_csv=Path(args.out_csv))


if __name__ == "__main__":
    main()


