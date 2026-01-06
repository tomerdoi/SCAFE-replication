"""
Build the broadest AVS address registry we can from available data sources.

Why:
- EigenExplorer AVS inventory can be smaller than other sources (e.g. u--1).
- Downstream steps (Etherscan source fetch, code-risk, Dune panels) should be able to scale with a
  larger AVS set without hard-coding EigenExplorer as the only registry.

Inputs (optional; unioned when present):
- data/eigenexplorer_avs_all.json            (addresses)
- data/u1_latest_avs_balances.json           (dict keyed by avs address)
- data/dune_*_all.json                       (if they contain 'avs_address' fields)

Output:
- data/avs_registry_all.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class AVSRegistryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AVSRegistrySources:
    eigenexplorer_avs_all: Path = DATA_DIR / "eigenexplorer_avs_all.json"
    u1_latest_avs_balances: Path = DATA_DIR / "u1_latest_avs_balances.json"
    dune_candidates: Tuple[Path, ...] = (
        DATA_DIR / "dune_avs_directory_avs_registry_all.json",
        DATA_DIR / "dune_avs_activity_panel_all.json",
        DATA_DIR / "dune_avs_governance_events_panel_all.json",
        DATA_DIR / "dune_avs_governance_events_panel_expanded_all.json",
        DATA_DIR / "dune_avs_ethflow_panel_expanded_all.json",
        DATA_DIR / "dune_avs_tokenflow_panel_expanded_all.json",
        DATA_DIR / "dune_avs_related_contracts_all.json",
    )


class AVSRegistryUtil:
    def __init__(self, sources: Optional[AVSRegistrySources] = None) -> None:
        logger = logging.getLogger("avs_registry_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.sources = sources or AVSRegistrySources()

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _read_json", exc_info=True)
            raise

    def _is_addr(self, s: Any) -> bool:
        return isinstance(s, str) and s.startswith("0x") and len(s) == 42

    def _add(self, addr: str, source: str, out: Dict[str, Set[str]]) -> None:
        a = addr.lower()
        if not self._is_addr(a):
            return
        out.setdefault(a, set()).add(source)

    def _extract_from_eigenexplorer(self, out: Dict[str, Set[str]]) -> int:
        try:
            p = self.sources.eigenexplorer_avs_all
            if not p.exists():
                return 0
            obj = self._read_json(p)
            rows = obj.get("data", []) if isinstance(obj, dict) else []
            n = 0
            for r in rows:
                if isinstance(r, dict) and self._is_addr(r.get("address")):
                    self._add(r["address"], "eigenexplorer", out)
                    n += 1
            return n
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _extract_from_eigenexplorer", exc_info=True)
            raise

    def _extract_from_u1(self, out: Dict[str, Set[str]]) -> int:
        try:
            p = self.sources.u1_latest_avs_balances
            if not p.exists():
                return 0
            obj = self._read_json(p)
            if not isinstance(obj, dict):
                return 0
            n = 0
            for k, v in obj.items():
                if self._is_addr(k):
                    self._add(k, "u1_latest_avs_balances:key", out)
                    n += 1
                if isinstance(v, dict) and self._is_addr(v.get("avsAddress")):
                    self._add(v["avsAddress"], "u1_latest_avs_balances:value", out)
            return n
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _extract_from_u1", exc_info=True)
            raise

    def _extract_from_dune_json(self, path: Path, out: Dict[str, Set[str]]) -> int:
        try:
            if not path.exists():
                return 0
            obj = self._read_json(path)

            # Common structures:
            # - {"result":{"rows":[{"avs_address":"0x.."}, ...]}}
            # - {"rows":[...]} or list[dict]
            rows = None
            if isinstance(obj, dict):
                rows = obj.get("result", {}).get("rows")
                if rows is None:
                    rows = obj.get("rows")
            elif isinstance(obj, list):
                rows = obj

            if not isinstance(rows, list):
                return 0
            n = 0
            for r in rows:
                if not isinstance(r, dict):
                    continue
                # try common keys
                for key in ("avs_address", "avsAddress", "address"):
                    v = r.get(key)
                    if self._is_addr(v):
                        self._add(v, f"dune:{path.name}:{key}", out)
                        n += 1
                        break
            return n
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _extract_from_dune_json", exc_info=True)
            raise

    def build_registry(self) -> Dict[str, Set[str]]:
        try:
            out: Dict[str, Set[str]] = {}
            n_e = self._extract_from_eigenexplorer(out)
            n_u = self._extract_from_u1(out)
            n_d = 0
            for p in self.sources.dune_candidates:
                n_d += self._extract_from_dune_json(p, out)
            self.logger.info(
                f"Discovered {len(out)} unique AVS addresses "
                f"(eigenexplorer_hits={n_e}, u1_hits={n_u}, dune_hits={n_d})."
            )
            if not out:
                raise AVSRegistryError("No AVS addresses discovered from any source")
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during build_registry", exc_info=True)
            raise

    def write_registry(self, out_path: Optional[Path] = None) -> Path:
        try:
            if out_path is None:
                out_path = DATA_DIR / "avs_registry_all.json"

            reg = self.build_registry()
            rows = [
                {"address": a, "sources": sorted(list(srcs))}
                for a, srcs in sorted(reg.items(), key=lambda kv: kv[0])
            ]
            payload = {
                "meta": {"total": len(rows)},
                "data": rows,
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
            self.logger.info(f"Wrote: {out_path}")
            return out_path
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during write_registry", exc_info=True)
            raise


if __name__ == "__main__":
    AVSRegistryUtil().write_registry()


