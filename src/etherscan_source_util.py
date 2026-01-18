"""
Etherscan utility: fetch verified contract source code (v1 endpoint).

Etherscan v2: the source-code endpoint is `module=contract&action=getsourcecode`
served via `/v2/api?chainid=<id>`.

Input:
- data/eigenexplorer_avs_all.json  (AVS addresses) OR an arbitrary address list CSV (pivot)

Output:
- data/etherscan_sources/<address>.json  (raw Etherscan payload per address)
"""

from __future__ import annotations

import json
import logging
import re
import time
import os
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.global_conf import CONF


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "etherscan_sources"


class EtherscanSourceError(RuntimeError):
    pass


@dataclass(slots=True)
class EtherscanSourceClient:
    api_key: str
    base_url: str = CONF.ETHERSCAN_BASE_URL
    chain_id: int = CONF.ETHERSCAN_CHAIN_ID
    timeout_secs: int = CONF.HTTP_TIMEOUT_SECS
    max_retries: int = CONF.HTTP_MAX_RETRIES
    backoff_secs: float = CONF.HTTP_BACKOFF_SECS
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        retry = Retry(
            total=max(0, int(self.max_retries)),
            backoff_factor=float(self.backoff_secs),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        self._session = s

    def get_sourcecode(self, address: str) -> Dict[str, Any]:
        """
        Returns the raw Etherscan response dict.

        Shape (success):
          {"status":"1","message":"OK","result":[{...}]}  (result is list length 1)

        Shape (unverified / other):
          {"status":"0","message":"NOTOK","result":"Contract source code not verified"}
        """
        addr = address.strip().lower()
        if not (addr.startswith("0x") and len(addr) == 42):
            raise ValueError(f"Invalid address: {address!r}")

        params = {
            "chainid": int(self.chain_id),
            "module": "contract",
            "action": "getsourcecode",
            "address": addr,
            "apikey": self.api_key,
        }

        # Etherscan sometimes returns 200 + NOTOK for throttling. Retry with short sleep.
        for attempt in range(1, 6):
            r = self._session.get(f"{self.base_url}/v2/api", params=params, timeout=self.timeout_secs)
            data = r.json()

            if not isinstance(data, dict):
                raise EtherscanSourceError(f"Unexpected response type: {type(data)}")

            # Rate limit often comes back in status=0 form.
            if data.get("status") == "0" and data.get("message") == "NOTOK":
                res = data.get("result")
                if isinstance(res, str) and "rate limit" in res.lower():
                    time.sleep(min(5.0, 0.5 * attempt))
                    continue

            return data

        raise EtherscanSourceError("Etherscan rate limit: exhausted retries")

    def get_runtime_bytecode(self, address: str) -> Dict[str, Any]:
        """
        Fetch runtime bytecode via Etherscan "proxy" module (eth_getCode).

        Response shape (typical):
          {"jsonrpc":"2.0","id":1,"result":"0x..."}
        """
        addr = address.strip().lower()
        if not (addr.startswith("0x") and len(addr) == 42):
            raise ValueError(f"Invalid address: {address!r}")

        params = {
            "chainid": int(self.chain_id),
            "module": "proxy",
            "action": "eth_getCode",
            "address": addr,
            "tag": "latest",
            "apikey": self.api_key,
        }

        for attempt in range(1, 6):
            r = self._session.get(f"{self.base_url}/v2/api", params=params, timeout=self.timeout_secs)
            data = r.json()
            # Sometimes throttling comes back as a dict with message/result text; handle similarly.
            if isinstance(data, dict) and data.get("status") == "0" and data.get("message") == "NOTOK":
                res = data.get("result")
                if isinstance(res, str) and "rate limit" in res.lower():
                    time.sleep(min(5.0, 0.5 * attempt))
                    continue
            return data

        raise EtherscanSourceError("Etherscan rate limit (eth_getCode): exhausted retries")


class EtherscanAVSSourceUtil:
    def __init__(self) -> None:
        logger = logging.getLogger("etherscan_source_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        key = (CONF.ETHERSCAN_API_KEY or "").strip()
        if not key:
            raise EtherscanSourceError("Missing ETHERSCAN_API_KEY (set it in secrets.json or env)")
        self.client = EtherscanSourceClient(api_key=key)

    def ensure_dirs(self) -> None:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during ensure_dirs", exc_info=True)
            raise

    def load_avs_addresses(self, path: Optional[Path] = None) -> List[str]:
        try:
            if path is None:
                # Prefer expanded registry if present.
                p_reg = DATA_DIR / "avs_registry_all.json"
                path = p_reg if p_reg.exists() else (DATA_DIR / "eigenexplorer_avs_all.json")
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = obj.get("data", []) if isinstance(obj, dict) else []
            out: List[str] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                a = r.get("address")
                if isinstance(a, str) and a.startswith("0x") and len(a) == 42:
                    out.append(a.lower())
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_avs_addresses", exc_info=True)
            raise

    def load_addresses_from_csv(self, csv_path: Path) -> List[str]:
        """
        Generic loader for pivoted project: CSV with an `address` column.
        Example: data/contract_registry.csv
        """
        try:
            import pandas as pd

            p = Path(csv_path)
            if not p.exists():
                raise FileNotFoundError(f"Addresses CSV not found: {p}")
            df = pd.read_csv(p)
            if "address" not in df.columns:
                raise ValueError(f"Addresses CSV must contain `address` column: {p}")
            out = [str(x).strip().lower() for x in df["address"].tolist()]
            out = [a for a in out if isinstance(a, str) and a.startswith("0x") and len(a) == 42]
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_addresses_from_csv", exc_info=True)
            raise

    def parse_contract_sources(self, etherscan_resp: Dict[str, Any], address: str) -> List[str]:
        """
        Parse Etherscan `getsourcecode` response into a list of Solidity sources (strings).

        Handles:
        - unverified contracts => []
        - single-file source => [source]
        - multi-file JSON blobs sometimes wrapped in quotes and/or double-braces
        """
        try:
            if not isinstance(etherscan_resp, dict):
                return []
            result = etherscan_resp.get("result", [])
            if not isinstance(result, list) or not result:
                return []
            row = result[0] if isinstance(result[0], dict) else {}
            src = row.get("SourceCode", "")
            if not isinstance(src, str):
                return []
            src = src.strip()
            if not src:
                return []

            # Unwrap nested quoted JSON
            if src.startswith('"') and src.endswith('"'):
                try:
                    src = json.loads(src)
                except json.JSONDecodeError:
                    src = src[1:-1]
            if not isinstance(src, str):
                return []
            src = src.strip()

            # Remove extra outer braces Etherscan sometimes adds ('{{...}}')
            if src.startswith("{{") and src.endswith("}}"):
                src = src[1:-1].strip()

            # Multi-file JSON (standard solc-json-input-ish)
            if src.startswith("{") or src.startswith("["):
                try:
                    sc = json.loads(src)
                except json.JSONDecodeError:
                    return [src]

                sources = sc.get("sources", sc) if isinstance(sc, dict) else sc
                if not isinstance(sources, dict):
                    return [src]

                contract_name = row.get("ContractName") if isinstance(row, dict) else None
                prioritized_key = f"{contract_name}.sol" if isinstance(contract_name, str) and contract_name else None

                contracts: List[str] = []

                def _get_content(v: Any) -> Optional[str]:
                    if isinstance(v, dict):
                        c = v.get("content")
                        return c if isinstance(c, str) and c.strip() else None
                    return None

                # Add prioritized contract first (if we can locate it)
                if prioritized_key:
                    prioritized_contract = sources.get(prioritized_key)
                    if not prioritized_contract:
                        for key in sources.keys():
                            if isinstance(key, str) and key.endswith(f"/{prioritized_key}"):
                                prioritized_contract = sources[key]
                                break
                    pc = _get_content(prioritized_contract)
                    if pc:
                        contracts.append(pc)

                # Add remaining sources
                for k, v in sources.items():
                    if not isinstance(k, str):
                        continue
                    if prioritized_key and (k == prioritized_key or k.endswith(f"/{prioritized_key}")):
                        continue
                    c = _get_content(v)
                    if c:
                        contracts.append(c)

                return contracts if contracts else [src]

            return [src]
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during parse_contract_sources", exc_info=True)
            raise

    def fetch_one(self, address: str, *, sleep_secs: float = 0.25) -> Path:
        try:
            data = self.client.get_sourcecode(address)
            contracts = self.parse_contract_sources(data, address.lower())
            code_resp = self.client.get_runtime_bytecode(address)
            payload = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "address": address.lower(),
                "etherscan": data,
                "bytecode": {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "eth_getCode": code_resp,
                },
                "parsed_sources": {
                    "n_contracts": len(contracts),
                    "contracts": contracts,
                },
            }
            out = OUT_DIR / f"{address.lower()}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            time.sleep(max(0.0, float(sleep_secs)))
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during fetch_one", exc_info=True)
            raise

    def _extract_implementation_address(self, etherscan_resp: Dict[str, Any]) -> Optional[str]:
        """
        If the fetched address is a proxy, Etherscan may provide the implementation address.
        """
        try:
            if not isinstance(etherscan_resp, dict):
                return None
            result = etherscan_resp.get("result", [])
            if not isinstance(result, list) or not result:
                return None
            row = result[0] if isinstance(result[0], dict) else {}
            impl = row.get("Implementation")
            proxy_flag = row.get("Proxy")
            if isinstance(proxy_flag, str) and proxy_flag.strip() == "1" and isinstance(impl, str):
                impl = impl.strip().lower()
                if impl.startswith("0x") and len(impl) == 42 and impl != "0x0000000000000000000000000000000000000000":
                    return impl
            return None
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _extract_implementation_address", exc_info=True)
            raise

    def fetch_one_with_implementation(self, address: str, visited: Set[str]) -> None:
        """
        Fetch a contract address and, if it's a proxy, also fetch its implementation.
        """
        try:
            addr = address.lower()
            if addr in visited:
                return
            visited.add(addr)

            out = OUT_DIR / f"{addr}.json"
            from_cache = False
            if getattr(self, "_skip_existing", False) and out.exists() and out.stat().st_size > 10:
                # Use existing payload to decide whether we also need to fetch implementation.
                try:
                    existing = json.loads(out.read_text(encoding="utf-8"))
                    data = existing.get("etherscan", {}) if isinstance(existing, dict) else {}
                    existing_bytecode = existing.get("bytecode") if isinstance(existing, dict) else None
                    from_cache = True
                except Exception:
                    data = {}
                    existing_bytecode = None
            else:
                data = self.client.get_sourcecode(addr)
                existing_bytecode = None
            contracts = self.parse_contract_sources(data, addr)

            # Always try to have bytecode. Even if we're skipping existing payloads, backfill bytecode if missing.
            need_bytecode = True
            if isinstance(existing_bytecode, dict):
                # consider present if it has eth_getCode result
                res = (existing_bytecode.get("eth_getCode") or {}) if isinstance(existing_bytecode.get("eth_getCode"), dict) else existing_bytecode.get("eth_getCode")
                if isinstance(res, dict) and isinstance(res.get("result"), str):
                    need_bytecode = False
            if need_bytecode:
                code_resp = self.client.get_runtime_bytecode(addr)
                bytecode_payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "eth_getCode": code_resp}
            else:
                bytecode_payload = existing_bytecode

            payload = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "address": addr,
                "etherscan": data,
                "bytecode": bytecode_payload,
                "parsed_sources": {
                    "n_contracts": len(contracts),
                    "contracts": contracts,
                },
            }
            # If we're skipping existing, don't rewrite/sleep (keeps runs fast and avoids extra IO).
            if (not from_cache) or need_bytecode:
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                # If we're only backfilling bytecode, keep this fast but polite.
                time.sleep(0.10 if from_cache else 0.25)

            impl = self._extract_implementation_address(data)
            if impl and impl not in visited:
                self.logger.info(f"Fetching implementation for {addr}: {impl}")
                self.fetch_one_with_implementation(impl, visited)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during fetch_one_with_implementation", exc_info=True)
            raise

    def run(
        self,
        *,
        fetch_all: bool,
        limit: int = 5,
        addresses_csv: Optional[Path] = None,
        progress_logger: Optional[Any] = None,
        progress_every: int = 5,
        skip_existing: bool = False,
    ) -> None:
        try:
            # used by fetch_one_with_implementation
            self._skip_existing = bool(skip_existing)
            self.ensure_dirs()
            if addresses_csv is not None:
                addrs = self.load_addresses_from_csv(Path(addresses_csv))
            else:
                addrs = self.load_avs_addresses()
            if not fetch_all:
                addrs = addrs[: max(0, int(limit))]
            total_base = len(addrs)
            self.logger.info(f"Fetching Etherscan sources for {total_base} addresses")
            visited: Set[str] = set()
            for i, a in enumerate(addrs, start=1):
                self.logger.info(f"Fetching {a}")
                self.fetch_one_with_implementation(a, visited)
                if progress_logger and (i == 1 or i % max(1, int(progress_every)) == 0 or i == total_base):
                    pct = int(round(100.0 * i / max(1, total_base)))
                    progress_logger.info(f"[etherscan sources] base {i}/{total_base} ({pct}%) | total_fetched={len(visited)}")
            self.logger.info(f"Done. Wrote per-address payloads under: {OUT_DIR}")
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch verified contract sources from Etherscan (writes data/etherscan_sources/*.json).")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch all addresses (ignore --limit).")
    parser.add_argument("--limit", type=int, default=int(os.getenv("ETHERSCAN_LIMIT", "5") or "5"), help="Limit addresses when not --fetch-all.")
    parser.add_argument(
        "--addresses-csv",
        type=str,
        default=str(os.getenv("ETHERSCAN_ADDRESSES_CSV", "")).strip(),
        help="Optional CSV with an `address` column (e.g., data/contract_registry.csv). If omitted, uses AVS registry.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip rewriting existing payloads (faster, fewer calls).")
    args = parser.parse_args()

    # Backwards-compatible env behavior:
    #   ETHERSCAN_FETCH_ALL=1 python -m src.etherscan_source_util
    fetch_all = bool(args.fetch_all) or (str(os.getenv("ETHERSCAN_FETCH_ALL", "0")).strip() in ("1", "true", "True", "yes", "YES"))
    addresses_csv = Path(args.addresses_csv) if str(args.addresses_csv).strip() else None

    EtherscanAVSSourceUtil().run(
        fetch_all=bool(fetch_all),
        limit=int(args.limit),
        addresses_csv=addresses_csv,
        skip_existing=bool(args.skip_existing),
    )


