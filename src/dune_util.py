"""
Minimal Dune API utility (SQL execution) for adding event/time-series signals.

This util is designed to complement the EigenExplorer snapshot datasets you already have:
- AVS registry: data/eigenexplorer_avs_all.json
- Operator↔AVS edges: data/eigenexplorer_operator_avs_edges.json

It runs Dune SQL queries (via API) to extract on-chain "governance risk" signals for AVS contracts:
- Proxy upgrades: Upgraded(address)
- Ownership transfers: OwnershipTransferred(address,address)

Outputs are written into data/ as JSON for reproducibility.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .global_conf import CONF
except ImportError:  # pragma: no cover
    from global_conf import CONF  # type: ignore


class DuneError(RuntimeError):
    pass


@dataclass(slots=True)
class DuneClient:
    api_key: str
    base_url: str = "https://api.dune.com"
    timeout_secs: int = CONF.HTTP_TIMEOUT_SECS
    max_retries: int = CONF.HTTP_MAX_RETRIES
    backoff_secs: float = CONF.HTTP_BACKOFF_SECS
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        retry = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_secs,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _headers(self) -> Dict[str, str]:
        # Dune uses this header name for API v1 SQL execution.
        return {"X-Dune-API-Key": self.api_key, "Content-Type": "application/json"}

    def execute_sql(self, sql: str, performance: str = "medium") -> str:
        """
        Execute raw SQL on Dune.
        Returns execution_id as string/int serialized.
        """
        url = f"{self.base_url}/api/v1/sql/execute"
        payload = {"sql": sql, "performance": performance}
        r = self._session.post(url, headers=self._headers(), json=payload, timeout=self.timeout_secs)
        if r.status_code >= 400:
            raise DuneError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        execution_id = data.get("execution_id") or data.get("executionId") or data.get("id")
        if execution_id is None:
            raise DuneError(f"Unexpected execute response: {data}")
        return str(execution_id)

    def get_results(self, execution_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/execution/{execution_id}/results"
        r = self._session.get(url, headers=self._headers(), timeout=self.timeout_secs)
        if r.status_code >= 400:
            raise DuneError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        if not isinstance(data, dict):
            raise DuneError(f"Unexpected results response type: {type(data)}")
        return data

    def wait_for_results(
        self,
        execution_id: str,
        poll_secs: float = 2.0,
        timeout_secs: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Poll until execution completes (or fails) and return the full results payload.
        """
        start = time.time()
        while True:
            data = self.get_results(execution_id)
            state = data.get("state") or data.get("status")
            if state in ("QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                if state != "QUERY_STATE_COMPLETED":
                    raise DuneError(f"Query did not complete: {state} | {data.get('error') or data}")
                return data
            if time.time() - start > timeout_secs:
                raise DuneError(f"Timed out waiting for execution {execution_id}")
            time.sleep(poll_secs)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_data_dir() -> Path:
    d = _repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_avs_addresses() -> List[str]:
    """
    Load AVS addresses from data/eigenexplorer_avs_all.json.
    """
    p = _repo_root() / "data" / "eigenexplorer_avs_all.json"
    obj = _read_json(p)
    rows = obj.get("data", []) if isinstance(obj, dict) else []
    out: List[str] = []
    for r in rows:
        if isinstance(r, dict):
            addr = r.get("address")
            if isinstance(addr, str) and addr.startswith("0x") and len(addr) == 42:
                out.append(addr.lower())
    return out


def _as_dune_addr_literals(addresses: Sequence[str]) -> str:
    """
    Convert ["0xabc...", ...] into "0xabc...,0xdef..." (DuneSQL address literals).
    """
    cleaned: List[str] = []
    for a in addresses:
        a = a.lower().strip()
        if a.startswith("0x") and len(a) == 42:
            cleaned.append(a)
    if not cleaned:
        raise DuneError("No valid addresses provided")
    return ",\n    ".join(cleaned)


def build_sql_avs_governance_events(
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    DuneSQL query:
    For each AVS contract and week, count governance-ish events.
    """
    addr_list = _as_dune_addr_literals(avs_addresses)

    # Event signature hashes (topic0)
    # Upgraded(address)
    topic_upgraded = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"
    # OwnershipTransferred(address,address)
    topic_ownership = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"

    return f"""
WITH avs(address) AS (
  VALUES
    {addr_list}
),
logs AS (
  SELECT
    l.block_time,
    l.contract_address,
    l.topic0,
    l.tx_hash
  FROM ethereum.logs l
  JOIN avs a ON l.contract_address = a.address
  WHERE l.block_time >= TIMESTAMP '{start_date}'
    AND l.block_time <  TIMESTAMP '{end_date}'
    AND l.topic0 IN ({topic_upgraded}, {topic_ownership})
)
SELECT
  date_trunc('week', block_time) AS week,
  contract_address AS avs_address,
  SUM(CASE WHEN topic0 = {topic_upgraded} THEN 1 ELSE 0 END) AS n_upgraded_events,
  SUM(CASE WHEN topic0 = {topic_ownership} THEN 1 ELSE 0 END) AS n_ownership_transfers,
  COUNT(*) AS n_total_events,
  COUNT(DISTINCT tx_hash) AS n_distinct_txs
FROM logs
GROUP BY 1, 2
ORDER BY 1, 2
""".strip()


def run_avs_governance_events(
    dune: DuneClient,
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
    performance: str = "medium",
    out_name: str = "dune_avs_governance_events.json",
) -> Path:
    sql = build_sql_avs_governance_events(avs_addresses, start_date=start_date, end_date=end_date)
    execution_id = dune.execute_sql(sql, performance=performance)
    results = dune.wait_for_results(execution_id)

    out_dir = _ensure_data_dir()
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Small console summary
    result = results.get("result", {})
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_rows = meta.get("total_row_count") or meta.get("total_rows")
    print(f"wrote results to {out_path} (execution_id={execution_id}, rows={total_rows})")
    return out_path


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    # - FETCH_ALL=True will include all AVS addresses (currently 49).
    # - FETCH_ALL=False uses a smaller subset for cheaper/faster iteration.
    FETCH_ALL = False
    SMALL_N = 5

    # Time range for event extraction
    START_DATE = "2024-01-01"
    END_DATE = "2025-12-31"

    key = (CONF.DUNE_API_KEY or "").strip()
    if not key:
        raise DuneError("Missing DUNE_API_KEY (expected in secrets.json)")

    avs_all = _load_avs_addresses()
    avs = avs_all if FETCH_ALL else avs_all[: max(1, int(SMALL_N))]

    dune = DuneClient(api_key=key)
    run_avs_governance_events(
        dune,
        avs_addresses=avs,
        start_date=START_DATE,
        end_date=END_DATE,
        performance="medium",
        out_name="dune_avs_governance_events_all.json" if FETCH_ALL else "dune_avs_governance_events_small.json",
    )

