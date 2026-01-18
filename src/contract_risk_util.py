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
from typing import Any, Dict, Optional, Tuple

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

def _hex_to_bytes(hex_str: str) -> bytes:
    s = (hex_str or "").strip()
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        return b""
    try:
        return bytes.fromhex(s)
    except Exception:
        return b""


def _disassemble_opcode_counts(bytecode_hex: str) -> Dict[str, int]:
    """
    Very small EVM disassembler: iterates opcodes and skips PUSH immediate data.
    Returns counts of selected opcodes (by mnemonic).
    """
    b = _hex_to_bytes(bytecode_hex)
    i = 0
    counts: Dict[str, int] = {}

    def inc(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    while i < len(b):
        op = b[i]
        i += 1

        # PUSH1..PUSH32 => skip immediate bytes
        if 0x60 <= op <= 0x7F:
            i += (op - 0x5F)
            continue

        # Selected opcodes
        if op == 0xF1:
            inc("CALL")
        elif op == 0xF2:
            inc("CALLCODE")
        elif op == 0xF4:
            inc("DELEGATECALL")
        elif op == 0xFA:
            inc("STATICCALL")
        elif op == 0xFF:
            inc("SELFDESTRUCT")
        elif op == 0xF0:
            inc("CREATE")
        elif op == 0xF5:
            inc("CREATE2")
        elif op == 0x55:
            inc("SSTORE")
        elif op == 0x54:
            inc("SLOAD")
    return counts


@dataclass(frozen=True, slots=True)
class ContractRiskConfig:
    """
    Scoring weights for a component-based, explainable contract technical risk score.

    Notes:
    - This is intentionally transparent and simple (rule-based). For Q1-style rigor, you’ll
      typically validate/ablate these weights or learn them later; for now they provide a
      reasonable, interpretable baseline.
    """

    # --- component weights (how much each component contributes to risk_raw) ---
    w_upgrade: float = 1.5
    w_privilege: float = 1.0
    w_interaction: float = 1.0
    w_dangerous: float = 1.5
    w_complexity: float = 1.0
    w_opacity: float = 1.0
    w_safety: float = 1.0  # subtracted (safety credit)

    # --- feature-level weights inside components ---
    w_selfdestruct: float = 8.0
    w_delegatecall: float = 3.0
    w_tx_origin: float = 4.0
    w_assembly: float = 1.5
    w_unchecked: float = 1.0

    w_external_call: float = 1.0
    w_transfer_send: float = 1.0

    w_upgrade_patterns: float = 2.0
    w_proxy_flag: float = 1.0

    w_only_owner: float = 1.0
    w_access_control: float = 1.0
    w_role_admin_ops: float = 0.5
    w_initializer: float = 0.5

    # Safety credits (subtract from risk)
    w_reentrancy_guard: float = 1.5
    w_pausable: float = 0.5
    w_timelock: float = 1.0
    w_safe_math: float = 0.25
    w_two_step_ownable: float = 0.5


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

    def _extract_bytecode_hex(self, payload: Dict[str, Any]) -> str:
        """
        Extract bytecode hex from the cached etherscan payload.
        etherscan_source_util stores this under payload["bytecode"]["eth_getCode"]["result"].
        """
        bc = payload.get("bytecode") if isinstance(payload, dict) else None
        if not isinstance(bc, dict):
            return ""
        egc = bc.get("eth_getCode")
        if isinstance(egc, dict) and isinstance(egc.get("result"), str):
            return str(egc.get("result") or "")
        # tolerate alternate storage
        if isinstance(egc, str):
            return egc
        return ""

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
            "uses_two_step_ownable": 0,
            "uses_accesscontrol": 0,
            "uses_pausable": 0,
            "uses_timelock": 0,
            "uses_reentrancy_guard": 0,
            "uses_call": 0,
            "uses_staticcall": 0,
            "uses_delegatecall": 0,
            "uses_selfdestruct": 0,
            "uses_transfer_send": 0,
            "uses_tx_origin": 0,
            "uses_assembly": 0,
            "uses_unchecked": 0,
            "n_external_calls": 0,
            "n_delegatecall": 0,
            "n_functions": 0,
            "n_public_external_functions": 0,
            "n_payable_functions": 0,
            "n_modifiers": 0,
            "uses_initializer": 0,
            "n_role_admin_ops": 0,
            "uses_safe_math": 0,
            "uses_chainlink_oracle": 0,
            "uses_uniswap_oracle": 0,
            "has_fallback_or_receive": 0,
            # Bytecode-backed features (for robustness on unverified contracts)
            "has_bytecode": 0,
            "bytecode_len": 0,
            "bc_has_call": 0,
            "bc_has_delegatecall": 0,
            "bc_has_staticcall": 0,
            "bc_has_selfdestruct": 0,
            "bc_n_call": 0,
            "bc_n_delegatecall": 0,
            "bc_n_selfdestruct": 0,
            "bc_proxy_like": 0,
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
        base["uses_two_step_ownable"] = 1 if ("Ownable2Step" in t) else 0
        base["uses_accesscontrol"] = 1 if (
            "onlyRole" in t or "DEFAULT_ADMIN_ROLE" in t or re.search(r"\bAccessControl\b", t)
        ) else 0
        base["uses_pausable"] = 1 if ("whenNotPaused" in t or re.search(r"\bPausable\b", t) or "pause(" in t) else 0
        base["uses_timelock"] = 1 if ("TimelockController" in t or re.search(r"\bTimelock\b", t)) else 0
        base["uses_reentrancy_guard"] = 1 if ("nonReentrant" in t or re.search(r"\bReentrancyGuard\b", t)) else 0
        base["uses_delegatecall"] = 1 if "delegatecall" in t else 0
        base["uses_call"] = 1 if (".call{" in t or ".call(" in t) else 0
        base["uses_staticcall"] = 1 if ("staticcall" in t) else 0
        base["uses_selfdestruct"] = 1 if "selfdestruct" in t else 0
        base["uses_transfer_send"] = 1 if (".transfer(" in t or ".send(" in t) else 0
        base["uses_tx_origin"] = 1 if ("tx.origin" in t) else 0
        base["uses_assembly"] = 1 if re.search(r"\bassembly\s*\{", t) else 0
        base["uses_unchecked"] = 1 if re.search(r"\bunchecked\s*\{", t) else 0

        # Count low-level calls: `.call(` or `.call{`
        base["n_external_calls"] = int(len(re.findall(r"\.call\s*(\{|\()", t)))
        base["n_delegatecall"] = int(len(re.findall(r"\bdelegatecall\b", t)))

        base["n_functions"] = int(len(re.findall(r"\bfunction\b", t)))
        base["n_public_external_functions"] = int(len(re.findall(r"\bfunction\b[^{;]*\b(public|external)\b", t)))
        base["n_payable_functions"] = int(len(re.findall(r"\bfunction\b[^{;]*\bpayable\b", t)))
        base["n_modifiers"] = int(len(re.findall(r"\bmodifier\b", t)))

        base["uses_initializer"] = 1 if ("initializer" in t or "reinitializer" in t) else 0
        base["n_role_admin_ops"] = int(
            len(re.findall(r"\b(grantRole|revokeRole|renounceRole|_grantRole|_revokeRole)\b", t))
        )

        # “Safety-ish” library usage (weak signal, but cheap)
        base["uses_safe_math"] = 1 if ("SafeMath" in t) else 0

        # Oracle surface (also weak; just indicates dependency)
        base["uses_chainlink_oracle"] = 1 if ("AggregatorV3Interface" in t or "latestRoundData" in t or "chainlink" in t.lower()) else 0
        base["uses_uniswap_oracle"] = 1 if ("IUniswapV3Pool" in t or "observe(" in t or "getReserves" in t) else 0

        base["has_fallback_or_receive"] = 1 if (re.search(r"\bfallback\s*\(", t) or re.search(r"\breceive\s*\(", t)) else 0

        # --- bytecode features (present even when source is unverified) ---
        bytecode_hex = self._extract_bytecode_hex(payload)
        bb = _hex_to_bytes(bytecode_hex)
        base["has_bytecode"] = 1 if len(bb) > 0 else 0
        base["bytecode_len"] = int(len(bb))
        if len(bb) > 0:
            op = _disassemble_opcode_counts(bytecode_hex)
            base["bc_n_call"] = int(op.get("CALL", 0))
            base["bc_n_delegatecall"] = int(op.get("DELEGATECALL", 0))
            base["bc_n_selfdestruct"] = int(op.get("SELFDESTRUCT", 0))
            base["bc_has_call"] = 1 if base["bc_n_call"] > 0 else 0
            base["bc_has_delegatecall"] = 1 if base["bc_n_delegatecall"] > 0 else 0
            base["bc_has_staticcall"] = 1 if int(op.get("STATICCALL", 0)) > 0 else 0
            base["bc_has_selfdestruct"] = 1 if base["bc_n_selfdestruct"] > 0 else 0
            # Proxy-like heuristic: short runtime with delegatecall
            base["bc_proxy_like"] = 1 if (base["bc_has_delegatecall"] == 1 and base["bytecode_len"] <= 600) else 0

        return base

    def _complexity_term(self, feats: Dict[str, Any]) -> float:
        loc = float(feats.get("loc", 0) or 0)
        n_files = float(feats.get("n_files", 0) or 0)
        n_funcs = float(feats.get("n_functions", 0) or 0)
        # Log-scaled, capped-ish complexity proxy
        term = 0.0
        term += math.log1p(max(0.0, loc) / 500.0)
        term += 0.3 * math.log1p(max(0.0, n_files))
        term += 0.2 * math.log1p(max(0.0, n_funcs) / 50.0)
        return float(term)

    def score_components(self, feats: Dict[str, Any]) -> Dict[str, float]:
        """
        Return explainable component scores. Total raw score is a weighted sum of these components.
        """
        c = self.cfg

        # Upgradeability: prefer explicit source/metadata; fall back to bytecode proxy-like signal.
        is_upgradeable = 1 if (
            int(feats.get("is_proxy", 0)) == 1
            or int(feats.get("has_upgrade_patterns", 0)) == 1
            or int(feats.get("bc_proxy_like", 0)) == 1
        ) else 0
        admin_power = 1 if (int(feats.get("uses_ownable", 0)) == 1 or int(feats.get("uses_accesscontrol", 0)) == 1) else 0

        upgrade = (
            (c.w_proxy_flag * int(feats.get("is_proxy", 0)))
            + (c.w_upgrade_patterns * int(feats.get("has_upgrade_patterns", 0)))
            + (0.5 * int(feats.get("bc_proxy_like", 0)))
        )
        # privilege includes “can someone change things?” + “admin ops exist?” + upgradeable init patterns
        privilege = 0.0
        privilege += c.w_only_owner * int(feats.get("uses_ownable", 0))
        privilege += c.w_access_control * int(feats.get("uses_accesscontrol", 0))
        privilege += c.w_role_admin_ops * math.log1p(max(0.0, float(feats.get("n_role_admin_ops", 0) or 0)))
        privilege += c.w_initializer * int(feats.get("uses_initializer", 0)) * float(is_upgradeable)

        # interaction: more external calls / payable surface tends to increase attack surface
        interaction = 0.0
        interaction += c.w_external_call * math.log1p(max(0.0, float(feats.get("n_external_calls", 0) or 0)))
        interaction += 0.5 * int(feats.get("uses_call", 0))
        interaction += 0.3 * int(feats.get("uses_staticcall", 0))
        # Bytecode fallback signals (helpful for unverified sources)
        interaction += 0.4 * math.log1p(max(0.0, float(feats.get("bc_n_call", 0) or 0)))
        interaction += 0.2 * int(feats.get("bc_has_staticcall", 0))
        interaction += c.w_transfer_send * int(feats.get("uses_transfer_send", 0))
        interaction += 0.2 * math.log1p(max(0.0, float(feats.get("n_payable_functions", 0) or 0)))
        # oracle dependencies add complexity/integration risk
        interaction += 0.5 * int(feats.get("uses_chainlink_oracle", 0))
        interaction += 0.5 * int(feats.get("uses_uniswap_oracle", 0))

        # dangerous / footguns
        dangerous = 0.0
        dangerous += c.w_selfdestruct * int(feats.get("uses_selfdestruct", 0))
        dangerous += c.w_delegatecall * (int(feats.get("uses_delegatecall", 0)) + 0.3 * math.log1p(max(0.0, float(feats.get("n_delegatecall", 0) or 0))))
        # Bytecode fallback (unverified)
        dangerous += 0.75 * c.w_selfdestruct * int(feats.get("bc_has_selfdestruct", 0))
        dangerous += 0.75 * c.w_delegatecall * (int(feats.get("bc_has_delegatecall", 0)) + 0.3 * math.log1p(max(0.0, float(feats.get("bc_n_delegatecall", 0) or 0))))
        dangerous += c.w_tx_origin * int(feats.get("uses_tx_origin", 0))
        dangerous += c.w_assembly * int(feats.get("uses_assembly", 0))
        dangerous += c.w_unchecked * int(feats.get("uses_unchecked", 0))
        dangerous += 0.3 * int(feats.get("has_fallback_or_receive", 0))

        complexity = self._complexity_term(feats)

        # opacity: if source isn't verified, risk is harder to assess.
        # If we have bytecode, opacity is reduced (we can still measure opcode surface).
        has_src = float(int(feats.get("has_verified_source", 0)))
        has_bc = float(int(feats.get("has_bytecode", 0)))
        opacity = (1.0 - has_src) * (1.0 - 0.7 * has_bc)  # 1.0 if neither; 0.3 if only bytecode; 0 if src

        # safety credits (subtract)
        safety = 0.0
        safety += c.w_reentrancy_guard * int(feats.get("uses_reentrancy_guard", 0))
        safety += c.w_pausable * int(feats.get("uses_pausable", 0))
        safety += c.w_timelock * int(feats.get("uses_timelock", 0))
        safety += c.w_safe_math * int(feats.get("uses_safe_math", 0))
        safety += c.w_two_step_ownable * int(feats.get("uses_two_step_ownable", 0))

        return {
            "comp_upgrade": float(upgrade),
            "comp_privilege": float(privilege),
            "comp_interaction": float(interaction),
            "comp_dangerous": float(dangerous),
            "comp_complexity": float(complexity),
            "comp_opacity": float(opacity),
            "comp_safety": float(safety),
            "aux_is_upgradeable": float(is_upgradeable),
            "aux_admin_power": float(admin_power),
        }

    def score_contract(self, feats: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Convert feature dict into a single raw risk score (higher => riskier).
        """
        c = self.cfg
        comps = self.score_components(feats)
        score = 0.0
        score += c.w_upgrade * comps["comp_upgrade"]
        score += c.w_privilege * comps["comp_privilege"]
        score += c.w_interaction * comps["comp_interaction"]
        score += c.w_dangerous * comps["comp_dangerous"]
        score += c.w_complexity * comps["comp_complexity"]
        score += c.w_opacity * comps["comp_opacity"]
        score -= c.w_safety * comps["comp_safety"]
        return float(score), comps

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
            if payload:
                raw, comps = self.score_contract(feats)
            else:
                raw, comps = 0.0, self.score_components(feats)
            # Uncertainty score: not a label, just a measurement-quality signal.
            has_src = float(int(feats.get("has_verified_source", 0)))
            has_bc = float(int(feats.get("has_bytecode", 0)))
            risk_uncertainty = (1.0 - has_src) * (0.7 - 0.4 * has_bc) + (1.0 - has_bc) * 0.3
            rows.append({**feats, **comps, "risk_raw": float(raw), "risk_uncertainty": float(max(0.0, risk_uncertainty))})
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

        # Convenience rank columns
        df["risk_rank"] = pd.to_numeric(df["risk_z"], errors="coerce").rank(ascending=False, method="min").astype(int)

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


