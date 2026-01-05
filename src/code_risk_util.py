"""
Code-level AVS risk features from verified Solidity sources.

Inputs:
- data/eigenexplorer_avs_all.json
- data/dune_avs_related_contracts_all.json  (optional but recommended)
- data/etherscan_sources/<address>.json     (from etherscan_source_util)

Output:
- data/derived/avs_code_risk.csv

Notes:
- This is intentionally lightweight: rule/regex-based heuristics over Solidity source + Etherscan proxy metadata.
- It produces a time-invariant AVS code risk score that can be merged into the weekly behavioral risk.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
ETHERSCAN_SOURCES_DIR = DATA_DIR / "etherscan_sources"


class CodeRiskError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodeRiskConfig:
    related_top_k: int = 5
    # blending/weights are used inside the code-risk score (not the final pipeline blend)
    w_upgradeability: float = 2.0
    w_admin_power: float = 1.0
    w_external_call: float = 1.0
    w_delegatecall: float = 2.0
    w_selfdestruct: float = 5.0
    w_complexity: float = 0.5
    w_safety_pausable: float = -0.5
    w_safety_timelock: float = -0.5


class AVSCodeRiskUtil:
    def __init__(self, cfg: Optional[CodeRiskConfig] = None) -> None:
        logger = logging.getLogger("code_risk_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.cfg = cfg or CodeRiskConfig()

    def ensure_dirs(self) -> None:
        try:
            DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during ensure_dirs", exc_info=True)
            raise

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _read_json", exc_info=True)
            raise

    def load_avs_addresses(self) -> List[str]:
        try:
            path = DATA_DIR / "eigenexplorer_avs_all.json"
            obj = self._read_json(path)
            rows = obj.get("data", []) if isinstance(obj, dict) else []
            out = []
            for r in rows:
                if isinstance(r, dict):
                    a = r.get("address")
                    if isinstance(a, str) and a.startswith("0x") and len(a) == 42:
                        out.append(a.lower())
            return sorted(set(out))
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_avs_addresses", exc_info=True)
            raise

    def load_related_contracts(self) -> Dict[str, List[str]]:
        """
        Returns mapping avs_address -> list(related_contract_address) limited to top-k by n_gov_events.
        """
        try:
            path = DATA_DIR / "dune_avs_related_contracts_all.json"
            if not path.exists():
                return {}
            obj = self._read_json(path)
            rows = obj.get("result", {}).get("rows", []) if isinstance(obj, dict) else []
            if not isinstance(rows, list):
                return {}
            df = pd.DataFrame(rows)
            if df.empty:
                return {}

            for c in ("avs_address", "related_contract_address"):
                if c in df.columns:
                    df[c] = df[c].astype(str).str.lower()

            if "n_gov_events" in df.columns:
                df["n_gov_events"] = pd.to_numeric(df["n_gov_events"], errors="coerce").fillna(0.0)
            else:
                df["n_gov_events"] = 0.0

            out: Dict[str, List[str]] = {}
            for avs, g in df.groupby("avs_address"):
                g2 = g.sort_values("n_gov_events", ascending=False)
                rel = [x for x in g2["related_contract_address"].tolist() if isinstance(x, str) and x.startswith("0x")]
                # keep unique order
                seen: Set[str] = set()
                rel_u: List[str] = []
                for x in rel:
                    if x not in seen:
                        seen.add(x)
                        rel_u.append(x)
                out[str(avs)] = rel_u[: max(0, int(self.cfg.related_top_k))]
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_related_contracts", exc_info=True)
            raise

    def build_avs_contract_set(self, avs_addresses: Sequence[str], related: Dict[str, List[str]]) -> Dict[str, List[str]]:
        try:
            out: Dict[str, List[str]] = {}
            for a in avs_addresses:
                s = [a]
                s += related.get(a, [])
                # unique preserve order
                seen: Set[str] = set()
                uu: List[str] = []
                for x in s:
                    x = str(x).lower()
                    if x.startswith("0x") and len(x) == 42 and x not in seen:
                        seen.add(x)
                        uu.append(x)
                out[a] = uu
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during build_avs_contract_set", exc_info=True)
            raise

    def _etherscan_payload_path(self, address: str) -> Path:
        return ETHERSCAN_SOURCES_DIR / f"{address.lower()}.json"

    def load_etherscan_source_payload(self, address: str) -> Optional[Dict[str, Any]]:
        try:
            path = self._etherscan_payload_path(address)
            if not path.exists():
                return None
            obj = self._read_json(path)
            return obj if isinstance(obj, dict) else None
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_etherscan_source_payload", exc_info=True)
            raise

    def _join_sources(self, payload: Dict[str, Any]) -> str:
        try:
            ps = payload.get("parsed_sources", {}) if isinstance(payload, dict) else {}
            contracts = ps.get("contracts", []) if isinstance(ps, dict) else []
            if not isinstance(contracts, list):
                return ""
            parts = [c for c in contracts if isinstance(c, str)]
            return "\n\n".join(parts)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _join_sources", exc_info=True)
            raise

    def extract_contract_features(self, address: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract heuristic features from Etherscan metadata + Solidity source.
        """
        try:
            base: Dict[str, Any] = {
                "address": address.lower(),
                "has_payload": 1 if payload else 0,
                "has_verified_source": 0,
                "n_files": 0,
                "loc": 0,
                "is_proxy": 0,
                "implementation_address": None,
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

            t = text
            # patterns
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
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during extract_contract_features", exc_info=True)
            raise

    def score_contract(self, feats: Dict[str, Any]) -> float:
        """
        Convert contract features into a single raw risk score.
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during score_contract", exc_info=True)
            raise

    def run(self, *, progress_logger: Optional[Any] = None, progress_every: int = 5) -> Path:
        try:
            self.ensure_dirs()

            avs_addrs = self.load_avs_addresses()
            related = self.load_related_contracts()
            avs_contracts = self.build_avs_contract_set(avs_addrs, related)

            rows_avs: List[Dict[str, Any]] = []
            missing_payloads: int = 0
            total_contracts: int = 0

            total_avs = len(avs_contracts)
            for i, (avs, contracts) in enumerate(avs_contracts.items(), start=1):
                total_contracts += len(contracts)
                contract_scores: List[float] = []
                contract_with_src: int = 0
                for caddr in contracts:
                    payload = self.load_etherscan_source_payload(caddr)
                    if payload is None:
                        missing_payloads += 1
                    feats = self.extract_contract_features(caddr, payload)
                    if int(feats.get("has_verified_source", 0)) == 1:
                        contract_with_src += 1
                    contract_scores.append(self.score_contract(feats) if payload else 0.0)

                avs_code_raw_max = float(np.max(contract_scores)) if contract_scores else 0.0
                avs_code_raw_mean = float(np.mean(contract_scores)) if contract_scores else 0.0

                rows_avs.append(
                    {
                        "avs_address": avs,
                        "n_contracts_total": len(contracts),
                        "n_contracts_with_verified_source": contract_with_src,
                        "code_risk_raw_max": avs_code_raw_max,
                        "code_risk_raw_mean": avs_code_raw_mean,
                    }
                )
                if progress_logger and (i == 1 or i % max(1, int(progress_every)) == 0 or i == total_avs):
                    pct = int(round(100.0 * i / max(1, total_avs)))
                    progress_logger.info(f"[code risk] {i}/{total_avs} ({pct}%)")

            df = pd.DataFrame(rows_avs)
            if df.empty:
                raise CodeRiskError("No AVS rows produced")

            # Normalize across AVSs (global z-score)
            x = pd.to_numeric(df["code_risk_raw_max"], errors="coerce").fillna(0.0)
            mu = float(x.mean())
            sigma = float(x.std()) if float(x.std()) > 0 else 1.0
            df["code_risk_z"] = (x - mu) / sigma

            out_path = DERIVED_DIR / "avs_code_risk.csv"
            df.to_csv(out_path, index=False)

            self.logger.info(f"Wrote: {out_path}")
            self.logger.info(
                f"Note: missing etherscan payloads for {missing_payloads}/{total_contracts} contract addresses "
                f"(run etherscan_source_util with FETCH_ALL=True to improve coverage)."
            )
            return out_path
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    AVSCodeRiskUtil().run()

