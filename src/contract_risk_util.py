"""
Contract-level technical risk scoring from verified Solidity sources.

This is the pivoted project's core: one row per contract address with:
- extracted feature flags (upgrade/admin surface, dangerous patterns, complexity proxies)
- raw risk score and normalized z-score

Input:
- data/contract_registry.csv            (address universe)
- data/etherscan_sources/<addr>.json    (cached Etherscan payloads from etherscan_source_util)

Output:
- data/derived/contract_risk.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
ETHERSCAN_SOURCES_DIR = DATA_DIR / "etherscan_sources"


def _is_addr(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class ContractRiskConfig:
    # weights for raw risk score
    w_upgradeability: float = 2.0
    w_admin_power: float = 1.0
    w_external_call: float = 1.0
    w_delegatecall: float = 2.0
    w_selfdestruct: float = 5.0
    w_complexity: float = 0.5
    w_safety_pausable: float = -0.5
    w_safety_timelock: float = -0.5


class ContractRiskUtil:
    def __init__(self, cfg: Optional[ContractRiskConfig] = None) -> None:
        logger = logging.getLogger("contract_risk_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.cfg = cfg or ContractRiskConfig()

    def ensure_dirs(self) -> None:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    def _etherscan_payload_path(self, address: str) -> Path:
        return ETHERSCAN_SOURCES_DIR / f"{address.lower()}.json"

    def load_etherscan_source_payload(self, address: str) -> Optional[Dict[str, Any]]:
        p = self._etherscan_payload_path(address)
        if not p.exists():
            return None
        obj = _read_json(p)
        return obj if isinstance(obj, dict) else None

    def extract_contract_features(self, address: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Lightweight, rule-based feature extraction from Etherscan metadata + Solidity source text.
        """
        base: Dict[str, Any] = {
            "address": address.lower(),
            "has_payload": 1 if payload else 0,
            "has_verified_source": 0,
            "n_files": 0,
            "loc": 0,
            "compiler_version": "",
            "is_proxy": 0,
            "implementation_address": "",
            "has_upgrade_patterns": 0,
            "uses_ownable": 0,
            "uses_accesscontrol": 0,
            "uses_pausable": 0,
            "uses_timelock": 0,
            "uses_call": 0,
            "uses_delegatecall": 0,
            "uses_selfdestruct": 0,
        }
        if not payload:
            return base

        etherscan = payload.get("etherscan", {}) if isinstance(payload, dict) else {}
        result = etherscan.get("result", []) if isinstance(etherscan, dict) else []
        row = result[0] if isinstance(result, list) and result and isinstance(result[0], dict) else {}

        base["compiler_version"] = str(row.get("CompilerVersion") or "").strip()
        base["is_proxy"] = 1 if str(row.get("Proxy", "")).strip() == "1" else 0
        impl = row.get("Implementation")
        if isinstance(impl, str) and impl.lower().startswith("0x") and len(impl) == 42:
            base["implementation_address"] = impl.lower()

        ps = payload.get("parsed_sources", {}) if isinstance(payload, dict) else {}
        contracts = ps.get("contracts", []) if isinstance(ps, dict) else []
        if isinstance(contracts, list):
            base["n_files"] = len([c for c in contracts if isinstance(c, str) and c.strip()])
            text = "\n".join([c for c in contracts if isinstance(c, str)])
        else:
            text = ""

        base["loc"] = int(sum(1 for _ in text.splitlines())) if text else 0
        base["has_verified_source"] = 1 if base["n_files"] > 0 else 0

        # patterns (cheap but effective)
        t = text
        base["has_upgrade_patterns"] = 1 if any(
            p in t for p in ("upgradeTo(", "upgradeToAndCall(", "UUPSUpgradeable", "TransparentUpgradeableProxy")
        ) else 0
        base["uses_ownable"] = 1 if ("onlyOwner" in t or re.search(r"\bOwnable\b", t)) else 0
        base["uses_accesscontrol"] = 1 if (
            "onlyRole" in t or "DEFAULT_ADMIN_ROLE" in t or re.search(r"\bAccessControl\b", t)
        ) else 0
        base["uses_pausable"] = 1 if ("whenNotPaused" in t or re.search(r"\bPausable\b", t) or "pause(" in t) else 0
        base["uses_timelock"] = 1 if ("TimelockController" in t or re.search(r"\bTimelock\b", t)) else 0
        base["uses_delegatecall"] = 1 if "delegatecall" in t else 0
        base["uses_call"] = 1 if (".call{" in t or ".call(" in t) else 0
        base["uses_selfdestruct"] = 1 if "selfdestruct" in t else 0

        return base

    def score_contract(self, feats: Dict[str, Any]) -> float:
        """
        Convert feature dict into a single raw risk score (higher => riskier).
        """
        c = self.cfg
        is_upgradeable = 1 if (int(feats.get("is_proxy", 0)) == 1 or int(feats.get("has_upgrade_patterns", 0)) == 1) else 0
        admin_power = 1 if (int(feats.get("uses_ownable", 0)) == 1 or int(feats.get("uses_accesscontrol", 0)) == 1) else 0

        # Complexity: log-scaled LOC
        loc = float(feats.get("loc", 0) or 0)
        complexity = math.log1p(max(0.0, loc) / 500.0)

        score = 0.0
        score += c.w_upgradeability * is_upgradeable
        score += c.w_admin_power * admin_power
        score += c.w_external_call * int(feats.get("uses_call", 0))
        score += c.w_delegatecall * int(feats.get("uses_delegatecall", 0))
        score += c.w_selfdestruct * int(feats.get("uses_selfdestruct", 0))
        score += c.w_complexity * complexity
        score += c.w_safety_pausable * int(feats.get("uses_pausable", 0))
        score += c.w_safety_timelock * int(feats.get("uses_timelock", 0))
        return float(score)

    def run(self, *, registry_csv: Path, out_csv: Path, limit: int = 0) -> Path:
        self.ensure_dirs()
        reg = pd.read_csv(Path(registry_csv))
        if "address" not in reg.columns:
            raise ValueError(f"Registry CSV must include `address` column: {registry_csv}")
        addrs = [str(a).strip().lower() for a in reg["address"].tolist()]
        addrs = [a for a in addrs if _is_addr(a)]
        if limit and limit > 0:
            addrs = addrs[: int(limit)]

        rows = []
        missing = 0
        for i, addr in enumerate(addrs, start=1):
            payload = self.load_etherscan_source_payload(addr)
            if payload is None:
                missing += 1
            feats = self.extract_contract_features(addr, payload)
            raw = self.score_contract(feats) if payload else 0.0
            rows.append({**feats, "risk_raw": float(raw)})
            if i == 1 or i % 200 == 0 or i == len(addrs):
                self.logger.info(f"[contract risk] {i}/{len(addrs)}")

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No rows produced for contract risk scoring.")

        x = pd.to_numeric(df["risk_raw"], errors="coerce").fillna(0.0)
        mu = float(x.mean())
        sigma = float(x.std())
        sigma = sigma if sigma and sigma > 0 else 1.0
        df["risk_z"] = (x - mu) / sigma

        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        self.logger.info(f"Wrote contract risk CSV: {out_csv} (rows={len(df)})")
        self.logger.info(f"Missing etherscan payloads: {missing}/{len(addrs)} (risk_raw=0 for missing)")
        return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute contract-level technical risk scores.")
    parser.add_argument(
        "--registry-csv",
        type=str,
        default=str(DATA_DIR / "contract_registry.csv"),
        help="Contract registry CSV (must have `address`).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DERIVED_DIR / "contract_risk.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0 = no limit).")
    args = parser.parse_args()

    ContractRiskUtil().run(registry_csv=Path(args.registry_csv), out_csv=Path(args.out_csv), limit=int(args.limit))


if __name__ == "__main__":
    main()


