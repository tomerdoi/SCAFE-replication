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


# src/risk/contract_risk.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
ETHERSCAN_SOURCES_DIR = DATA_DIR / "etherscan_sources"
ETHERSCAN_ACTIVITY_DIR = DATA_DIR / "etherscan_activity"


def _is_addr(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now_ts() -> int:
    # avoid importing datetime everywhere in hot paths
    import time

    return int(time.time())


def _clamp01(x: float) -> float:
    try:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)
    except Exception:
        return 0.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return int(default)


def _eth_from_wei(wei: Any) -> float:
    try:
        return float(_safe_int(wei, 0)) / 1e18
    except Exception:
        return 0.0

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
    - This is intentionally transparent and simple (rule-based). For Q1-style rigor, you'll
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
    w_activity: float = 1.0  # live on-chain activity / exposure

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

    # Live activity weights (non-code signals)
    w_tx_activity: float = 0.35
    w_unique_senders: float = 0.50
    w_failed_rate: float = 1.25
    w_token_activity: float = 0.25
    w_value_flow_eth: float = 0.20
    w_admin_event_churn: float = 0.65


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

    def _etherscan_activity_path(self, address: str) -> Path:
        return ETHERSCAN_ACTIVITY_DIR / f"{address.lower()}.json"

    def load_etherscan_source_payload(self, address: str) -> Optional[Dict[str, Any]]:
        p = self._etherscan_payload_path(address)
        if not p.exists():
            return None
        obj = _read_json(p)
        return obj if isinstance(obj, dict) else None

    def load_etherscan_activity_payload(self, address: str) -> Optional[Dict[str, Any]]:
        p = self._etherscan_activity_path(address)
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

    # ------------------------------------------------------------------
    # Shared helpers for feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_feature_base(identifier: str) -> Dict[str, Any]:
        """Return a zeroed-out feature dict.  Shared by all extraction paths."""
        return {
            "address": str(identifier).lower(),
            "has_payload": 0,
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
            # Bytecode-backed features
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
            # --- live (non-code) activity features ---
            "has_activity_payload": 0,
            "age_days": 0.0,
            "last_tx_age_days": 0.0,
            "tx_count_7d": 0,
            "tx_count_30d": 0,
            "tx_count_90d": 0,
            "tx_in_30d": 0,
            "tx_out_30d": 0,
            "unique_senders_30d": 0,
            "failed_tx_rate_90d": 0.0,
            "internal_tx_count_30d": 0,
            "erc20_tx_count_30d": 0,
            "erc20_unique_tokens_30d": 0,
            "erc20_unique_senders_30d": 0,
            "erc20_unique_receivers_30d": 0,
            "value_in_eth_30d": 0.0,
            "value_out_eth_30d": 0.0,
            # admin/governance event churn (last 365d)
            "n_evt_upgraded_365d": 0,
            "n_evt_admin_changed_365d": 0,
            "n_evt_ownership_transferred_365d": 0,
            "n_evt_role_granted_365d": 0,
            "n_evt_role_revoked_365d": 0,
            "n_evt_paused_365d": 0,
            "n_evt_unpaused_365d": 0,
            "admin_event_churn_365d": 0.0,
        }

    @staticmethod
    def _extract_source_features(base: Dict[str, Any], text: str, n_files: int) -> Dict[str, Any]:
        """
        Populate source-code-derived features into *base* from concatenated Solidity *text*.
        This is the shared regex/pattern logic used by both Etherscan-payload and
        local-file extraction paths.  Modifies *base* in place and returns it.
        """
        t = text
        base["loc"] = int(sum(1 for _ in t.splitlines())) if t else 0
        base["n_files"] = int(n_files)
        base["has_verified_source"] = 1 if n_files > 0 else 0

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

        base["uses_safe_math"] = 1 if ("SafeMath" in t) else 0
        base["uses_chainlink_oracle"] = 1 if ("AggregatorV3Interface" in t or "latestRoundData" in t or "chainlink" in t.lower()) else 0
        base["uses_uniswap_oracle"] = 1 if ("IUniswapV3Pool" in t or "observe(" in t or "getReserves" in t) else 0
        base["has_fallback_or_receive"] = 1 if (re.search(r"\bfallback\s*\(", t) or re.search(r"\breceive\s*\(", t)) else 0

        return base

    def _enrich_activity_features(self, base: Dict[str, Any], act: Dict[str, Any]) -> None:
        """Populate on-chain activity features into *base* from an activity payload dict."""
        base["has_activity_payload"] = 1
        now_ts = _utc_now_ts()

        def _parse_rows(key: str) -> list:
            obj = act.get(key)
            if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                return [r for r in obj["rows"] if isinstance(r, dict)]
            return []

        tx_rows = _parse_rows("txlist")
        internal_rows = _parse_rows("txlistinternal")
        token_rows = _parse_rows("tokentx")

        def within_days(ts: int, days: int) -> bool:
            return ts >= now_ts - int(days) * 86400

        seen_senders_30: set = set()
        tx_in_30 = tx_out_30 = 0
        v_in_30 = v_out_30 = 0.0
        n_fail_90 = n_total_90 = 0
        ts_min: Optional[int] = None
        ts_max: Optional[int] = None

        for r in tx_rows:
            ts = _safe_int(r.get("timeStamp"), 0)
            if ts <= 0:
                continue
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            n_total_90 += 1
            if str(r.get("isError") or "").strip() == "1":
                n_fail_90 += 1
            if within_days(ts, 7):
                base["tx_count_7d"] += 1
            if within_days(ts, 30):
                base["tx_count_30d"] += 1
                frm = str(r.get("from") or "").lower()
                if _is_addr(frm):
                    seen_senders_30.add(frm)
                to = str(r.get("to") or "").lower()
                if to == base["address"]:
                    tx_in_30 += 1
                    v_in_30 += _eth_from_wei(r.get("value"))
                if frm == base["address"]:
                    tx_out_30 += 1
                    v_out_30 += _eth_from_wei(r.get("value"))
            if within_days(ts, 90):
                base["tx_count_90d"] += 1

        base["unique_senders_30d"] = int(len(seen_senders_30))
        base["tx_in_30d"] = int(tx_in_30)
        base["tx_out_30d"] = int(tx_out_30)
        base["value_in_eth_30d"] = float(v_in_30)
        base["value_out_eth_30d"] = float(v_out_30)
        base["failed_tx_rate_90d"] = float(n_fail_90 / max(1, n_total_90))
        if ts_min is not None:
            base["age_days"] = float(max(0.0, (now_ts - int(ts_min)) / 86400.0))
        if ts_max is not None:
            base["last_tx_age_days"] = float(max(0.0, (now_ts - int(ts_max)) / 86400.0))

        n_int_30 = sum(1 for r in internal_rows if _safe_int(r.get("timeStamp"), 0) > 0 and within_days(_safe_int(r.get("timeStamp"), 0), 30))
        base["internal_tx_count_30d"] = int(n_int_30)

        n_tok_30 = 0
        tok_set: set = set()
        tok_senders: set = set()
        tok_receivers: set = set()
        for r in token_rows:
            ts = _safe_int(r.get("timeStamp"), 0)
            if ts <= 0 or not within_days(ts, 30):
                continue
            n_tok_30 += 1
            token_addr = str(r.get("contractAddress") or "").lower()
            if _is_addr(token_addr):
                tok_set.add(token_addr)
            frm = str(r.get("from") or "").lower()
            to = str(r.get("to") or "").lower()
            if _is_addr(frm):
                tok_senders.add(frm)
            if _is_addr(to):
                tok_receivers.add(to)
        base["erc20_tx_count_30d"] = int(n_tok_30)
        base["erc20_unique_tokens_30d"] = int(len(tok_set))
        base["erc20_unique_senders_30d"] = int(len(tok_senders))
        base["erc20_unique_receivers_30d"] = int(len(tok_receivers))

        logs = act.get("logs", {}) if isinstance(act.get("logs"), dict) else {}

        def _count_logs(key: str) -> int:
            entry = logs.get(key) if isinstance(logs, dict) else None
            resp = entry.get("response") if isinstance(entry, dict) else None
            res = resp.get("result") if isinstance(resp, dict) else None
            return int(len([x for x in res if isinstance(x, dict)])) if isinstance(res, list) else 0

        base["n_evt_upgraded_365d"] = _count_logs("upgraded")
        base["n_evt_admin_changed_365d"] = _count_logs("admin_changed")
        base["n_evt_ownership_transferred_365d"] = _count_logs("ownership_transferred")
        base["n_evt_role_granted_365d"] = _count_logs("role_granted")
        base["n_evt_role_revoked_365d"] = _count_logs("role_revoked")
        base["n_evt_paused_365d"] = _count_logs("paused")
        base["n_evt_unpaused_365d"] = _count_logs("unpaused")
        base["admin_event_churn_365d"] = float(sum(
            base[k] for k in (
                "n_evt_upgraded_365d", "n_evt_admin_changed_365d", "n_evt_ownership_transferred_365d",
                "n_evt_role_granted_365d", "n_evt_role_revoked_365d", "n_evt_paused_365d", "n_evt_unpaused_365d",
            )
        ))

    # ------------------------------------------------------------------
    # Public extraction API: from Etherscan payload (existing path)
    # ------------------------------------------------------------------

    def extract_contract_features(self, address: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Lightweight, rule-based feature extraction from Etherscan metadata + Solidity source text.
        """
        base = self._empty_feature_base(address)
        base["has_payload"] = 1 if payload else 0

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
            n_files = len([c for c in contracts if isinstance(c, str) and c.strip()])
            text = "\n".join([c for c in contracts if isinstance(c, str)])
        else:
            n_files = 0
            text = ""

        self._extract_source_features(base, text, n_files)

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
            base["bc_proxy_like"] = 1 if (base["bc_has_delegatecall"] == 1 and base["bytecode_len"] <= 600) else 0

        # --- live activity payload (optional) ---
        act = self.load_etherscan_activity_payload(address)
        if isinstance(act, dict):
            self._enrich_activity_features(base, act)

        return base

    # ------------------------------------------------------------------
    # Public extraction API: from raw Solidity source text (local files)
    # ------------------------------------------------------------------

    def extract_features_from_source_text(
        self,
        contract_id: str,
        source_text: str,
        *,
        n_files: int = 1,
        address: str = "",
    ) -> Dict[str, Any]:
        """
        Extract features from raw Solidity source text (e.g. a local .sol file).

        Only static source-code features are populated; bytecode and on-chain
        activity features remain at their defaults (0).  If an on-chain *address*
        is also available, cached Etherscan data will be used to enrich bytecode
        and activity features.
        """
        identifier = address if _is_addr(address) else contract_id
        base = self._empty_feature_base(identifier)
        base["has_payload"] = 1
        self._extract_source_features(base, source_text, n_files)

        if _is_addr(address):
            src_payload = self.load_etherscan_source_payload(address)
            if isinstance(src_payload, dict):
                bytecode_hex = self._extract_bytecode_hex(src_payload)
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
                    base["bc_proxy_like"] = 1 if (base["bc_has_delegatecall"] == 1 and base["bytecode_len"] <= 600) else 0
            act = self.load_etherscan_activity_payload(address)
            if isinstance(act, dict):
                self._enrich_activity_features(base, act)

        return base

    def extract_features_from_local_file(
        self,
        contract_id: str,
        sol_path: str,
        *,
        address: str = "",
    ) -> Dict[str, Any]:
        """
        Read a local ``.sol`` file (or directory of .sol files) and extract features.

        If *sol_path* is a directory, all ``*.sol`` files inside are concatenated.
        """
        p = Path(sol_path)
        if p.is_dir():
            sols = sorted(p.rglob("*.sol"))
            texts = []
            for s in sols:
                try:
                    texts.append(s.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
            text = "\n".join(texts)
            n_files = len(texts)
        elif p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            n_files = 1 if text else 0
        else:
            text = ""
            n_files = 0

        return self.extract_features_from_source_text(
            contract_id, text, n_files=n_files, address=address,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

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
        # privilege includes "can someone change things?" + "admin ops exist?" + upgradeable init patterns
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

        # Live activity / exposure: higher usage + higher failure rate + admin churn => higher operational risk.
        # This is intentionally log-scaled (more robust).
        tx_30 = float(_safe_int(feats.get("tx_count_30d"), 0))
        uniq_30 = float(_safe_int(feats.get("unique_senders_30d"), 0))
        tok_30 = float(_safe_int(feats.get("erc20_tx_count_30d"), 0))
        v_flow = float(max(0.0, _safe_float(feats.get("value_in_eth_30d"), 0.0) + _safe_float(feats.get("value_out_eth_30d"), 0.0)))
        fail_rate = _clamp01(_safe_float(feats.get("failed_tx_rate_90d"), 0.0))
        churn_365 = float(max(0.0, _safe_float(feats.get("admin_event_churn_365d"), 0.0)))

        activity = 0.0
        activity += c.w_tx_activity * math.log1p(tx_30)
        activity += c.w_unique_senders * math.log1p(uniq_30)
        activity += c.w_token_activity * math.log1p(tok_30)
        activity += c.w_value_flow_eth * math.log1p(v_flow)
        activity += c.w_failed_rate * fail_rate
        activity += c.w_admin_event_churn * math.log1p(churn_365)

        return {
            "comp_upgrade": float(upgrade),
            "comp_privilege": float(privilege),
            "comp_interaction": float(interaction),
            "comp_dangerous": float(dangerous),
            "comp_complexity": float(complexity),
            "comp_opacity": float(opacity),
            "comp_safety": float(safety),
            "comp_activity": float(activity),
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
        score += c.w_activity * comps["comp_activity"]
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
