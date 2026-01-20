"""
Etherscan on-chain "activity" enrichment for contract risk scoring.

This complements `src.clients.etherscan_source` (source+bytecode) by fetching non-code signals:
- normal transactions (account txlist)
- internal transactions
- ERC20 transfers (tokentx)
- governance/admin event logs (Upgraded/AdminChanged/OwnershipTransferred/etc.)

All payloads are cached under `data/etherscan_activity/<address>.json` for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.global_conf import CONF
from src.utils.keccak import topic0


# src/clients/etherscan_activity.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "etherscan_activity"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_addr(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return int(default)


@dataclass(slots=True)
class EtherscanActivityClient:
    api_key: str
    base_url: str = CONF.ETHERSCAN_BASE_URL
    chain_id: int = CONF.ETHERSCAN_CHAIN_ID
    timeout_secs: int = CONF.HTTP_TIMEOUT_SECS
    max_retries: int = CONF.HTTP_MAX_RETRIES
    backoff_secs: float = CONF.HTTP_BACKOFF_SECS
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or "https://api.etherscan.io").rstrip("/")
        retry = Retry(
            total=max(0, int(self.max_retries)),
            backoff_factor=float(self.backoff_secs),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        self._session = s

    def _get(self, *, module: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort: try Etherscan v2 first (`/v2/api?chainid=`), then fall back to v1 (`/api`).
        """
        p = dict(params or {})
        p.update({"module": module, "action": action, "apikey": self.api_key})

        # v2
        p2 = dict(p)
        p2["chainid"] = int(self.chain_id)
        for attempt in range(1, 6):
            try:
                r = self._session.get(f"{self.base_url}/v2/api", params=p2, timeout=self.timeout_secs)
                data = r.json()
                if isinstance(data, dict):
                    # Throttling comes back as status=0 and "rate limit" in result
                    res = data.get("result")
                    if data.get("status") == "0" and isinstance(res, str) and "rate limit" in res.lower():
                        time.sleep(min(5.0, 0.5 * attempt))
                        continue
                    return data
            except Exception:
                pass
            time.sleep(min(2.0, 0.2 * attempt))

        # v1 fallback
        for attempt in range(1, 6):
            try:
                r = self._session.get(f"{self.base_url}/api", params=p, timeout=self.timeout_secs)
                data = r.json()
                if isinstance(data, dict):
                    res = data.get("result")
                    if data.get("status") == "0" and isinstance(res, str) and "rate limit" in res.lower():
                        time.sleep(min(5.0, 0.5 * attempt))
                        continue
                    return data
            except Exception:
                pass
            time.sleep(min(2.0, 0.2 * attempt))

        return {"status": "0", "message": "NOTOK", "result": "failed_to_fetch"}

    def get_block_by_time(self, *, ts_unix: int, closest: str = "before") -> Optional[int]:
        data = self._get(module="block", action="getblocknobytime", params={"timestamp": int(ts_unix), "closest": str(closest)})
        res = data.get("result")
        if isinstance(res, str) and res.isdigit():
            return int(res)
        if isinstance(res, dict) and isinstance(res.get("blockNumber"), str) and str(res.get("blockNumber")).isdigit():
            return int(res["blockNumber"])
        return None

    def _paged_account_action(
        self,
        *,
        action: str,
        address: str,
        startblock: int,
        endblock: int = 99999999,
        sort: str = "desc",
        offset: int = 10000,
        max_pages: int = 3,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"truncated": False, "pages": 0, "offset": int(offset)}
        for page in range(1, max(1, int(max_pages)) + 1):
            params: Dict[str, Any] = {
                "address": address,
                "startblock": int(startblock),
                "endblock": int(endblock),
                "page": int(page),
                "offset": int(offset),
                "sort": str(sort),
            }
            if extra_params:
                params.update(extra_params)
            data = self._get(module="account", action=action, params=params)
            res = data.get("result")
            meta["pages"] = int(page)
            if isinstance(res, str):
                # "No transactions found" / errors
                break
            if not isinstance(res, list):
                break
            batch = [r for r in res if isinstance(r, dict)]
            rows.extend(batch)
            if len(batch) < int(offset):
                break
        if meta["pages"] >= int(max_pages) and len(rows) >= int(max_pages) * int(offset):
            meta["truncated"] = True
        return rows, meta

    def txlist(self, *, address: str, startblock: int, max_pages: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return self._paged_account_action(action="txlist", address=address, startblock=startblock, max_pages=max_pages)

    def txlistinternal(self, *, address: str, startblock: int, max_pages: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return self._paged_account_action(action="txlistinternal", address=address, startblock=startblock, max_pages=max_pages)

    def tokentx(self, *, address: str, startblock: int, max_pages: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return self._paged_account_action(action="tokentx", address=address, startblock=startblock, max_pages=max_pages)

    def get_logs_topic0(self, *, address: str, from_block: int, to_block: int, topic0_hash: str) -> Dict[str, Any]:
        # Etherscan expects 0x-prefixed topic0
        return self._get(
            module="logs",
            action="getLogs",
            params={
                "fromBlock": int(from_block),
                "toBlock": int(to_block),
                "address": address,
                "topic0": str(topic0_hash),
            },
        )


# Admin/governance event topics we care about (topic0 keccak of signature)
_EVENTS: List[Tuple[str, str]] = [
    ("Upgraded(address)", "upgraded"),
    ("AdminChanged(address,address)", "admin_changed"),
    ("OwnershipTransferred(address,address)", "ownership_transferred"),
    ("RoleGranted(bytes32,address,address)", "role_granted"),
    ("RoleRevoked(bytes32,address,address)", "role_revoked"),
    ("Paused(address)", "paused"),
    ("Unpaused(address)", "unpaused"),
]


class EtherscanActivityUtil:
    def __init__(self) -> None:
        logger = logging.getLogger("etherscan_activity_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        if not CONF.ETHERSCAN_API_KEY:
            raise RuntimeError("Missing ETHERSCAN_API_KEY (set secrets.json or env ETHERSCAN_API_KEY)")
        self.client = EtherscanActivityClient(api_key=str(CONF.ETHERSCAN_API_KEY))

    def ensure_dirs(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _out_path(self, address: str) -> Path:
        return OUT_DIR / f"{address.lower()}.json"

    def _load_addresses_from_csv(self, csv_path: Path) -> List[str]:
        df = pd.read_csv(Path(csv_path))
        if "address" not in df.columns:
            raise ValueError(f"CSV must have an `address` column: {csv_path}")
        addrs = [str(x).strip().lower() for x in df["address"].tolist()]
        addrs = [a for a in addrs if _is_addr(a)]
        return list(dict.fromkeys(addrs))

    def _compute_window_blocks(self, *, now: datetime) -> Dict[str, int]:
        """
        Compute global start blocks for common windows (chain-wide, not per-address).
        """
        now_ts = int(now.timestamp())
        windows_days = {"7d": 7, "30d": 30, "90d": 90, "365d": 365}
        blocks: Dict[str, int] = {}
        for k, days in windows_days.items():
            ts = now_ts - int(days) * 86400
            b = self.client.get_block_by_time(ts_unix=ts, closest="before")
            blocks[k] = int(b or 0)
        return blocks

    def fetch_one(
        self,
        *,
        address: str,
        blocks: Dict[str, int],
        max_pages: int,
        fetch_tx: bool,
        fetch_internal: bool,
        fetch_tokentx: bool,
        fetch_logs: bool,
        skip_existing: bool,
    ) -> Path:
        addr = address.lower()
        out = self._out_path(addr)
        if skip_existing and out.exists() and out.stat().st_size > 10:
            return out

        now = _utc_now()
        payload: Dict[str, Any] = {
            "fetched_at": now.isoformat(),
            "address": addr,
            "windows": {"blocks": dict(blocks)},
            "txlist": None,
            "txlistinternal": None,
            "tokentx": None,
            "logs": {},
        }

        startblock_90d = int(blocks.get("90d", 0) or 0)
        startblock_365d = int(blocks.get("365d", 0) or 0)

        if fetch_tx:
            rows, meta = self.client.txlist(address=addr, startblock=startblock_90d, max_pages=max_pages)
            payload["txlist"] = {"meta": meta, "rows": rows}
        if fetch_internal:
            rows, meta = self.client.txlistinternal(address=addr, startblock=startblock_90d, max_pages=max_pages)
            payload["txlistinternal"] = {"meta": meta, "rows": rows}
        if fetch_tokentx:
            rows, meta = self.client.tokentx(address=addr, startblock=startblock_90d, max_pages=max_pages)
            payload["tokentx"] = {"meta": meta, "rows": rows}
        if fetch_logs:
            for sig, key in _EVENTS:
                t0 = topic0(sig)
                data = self.client.get_logs_topic0(address=addr, from_block=startblock_365d, to_block=99999999, topic0_hash=t0)
                # Keep the raw response; feature extraction will interpret it.
                payload["logs"][key] = {"sig": sig, "topic0": t0, "response": data}

        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Be polite w.r.t rate limits
        time.sleep(0.15)
        return out

    def run(
        self,
        *,
        addresses_csv: Path,
        fetch_all: bool,
        limit: int = 50,
        skip_existing: bool = False,
        max_pages: int = 2,
        fetch_tx: bool = True,
        fetch_internal: bool = True,
        fetch_tokentx: bool = True,
        fetch_logs: bool = True,
        progress_logger: Optional[Any] = None,
        progress_every: int = 10,
    ) -> None:
        self.ensure_dirs()
        addrs = self._load_addresses_from_csv(Path(addresses_csv))
        if not fetch_all:
            addrs = addrs[: max(0, int(limit))]
        total = len(addrs)
        self.logger.info(f"Fetching Etherscan activity for {total} addresses")

        blocks = self._compute_window_blocks(now=_utc_now())
        for i, a in enumerate(addrs, start=1):
            self.fetch_one(
                address=a,
                blocks=blocks,
                max_pages=max_pages,
                fetch_tx=fetch_tx,
                fetch_internal=fetch_internal,
                fetch_tokentx=fetch_tokentx,
                fetch_logs=fetch_logs,
                skip_existing=skip_existing,
            )
            if progress_logger and (i == 1 or i % max(1, int(progress_every)) == 0 or i == total):
                pct = int(round(100.0 * i / max(1, total)))
                progress_logger.info(f"[etherscan activity] {i}/{total} ({pct}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch on-chain activity for contracts via Etherscan (writes data/etherscan_activity/*.json).")
    parser.add_argument("--addresses-csv", type=str, required=True, help="CSV with an `address` column (e.g., data/contract_registry.csv).")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch all addresses (ignore --limit).")
    parser.add_argument("--limit", type=int, default=int(os.getenv("ETHERSCAN_ACTIVITY_LIMIT", "50") or "50"))
    parser.add_argument("--skip-existing", action="store_true", help="Skip if cached payload exists.")
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("ETHERSCAN_ACTIVITY_MAX_PAGES", "2") or "2"))
    parser.add_argument("--no-tx", action="store_true")
    parser.add_argument("--no-internal", action="store_true")
    parser.add_argument("--no-tokentx", action="store_true")
    parser.add_argument("--no-logs", action="store_true")
    args = parser.parse_args()

    EtherscanActivityUtil().run(
        addresses_csv=Path(args.addresses_csv),
        fetch_all=bool(args.fetch_all),
        limit=int(args.limit),
        skip_existing=bool(args.skip_existing),
        max_pages=int(args.max_pages),
        fetch_tx=not bool(args.no_tx),
        fetch_internal=not bool(args.no_internal),
        fetch_tokentx=not bool(args.no_tokentx),
        fetch_logs=not bool(args.no_logs),
    )

