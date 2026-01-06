"""
Minimal Dune API utility (SQL execution) for adding event/time-series signals.

This util is designed to complement the EigenExplorer snapshot datasets you already have:
- AVS registry: data/eigenexplorer_avs_all.json
- Operator↔AVS edges: data/eigenexplorer_operator_avs_edges.json

It runs Dune SQL queries (via API) to extract on-chain "governance/incident risk" signals for AVS contracts:
- Proxy upgrades: Upgraded(address)
- Beacon upgrades: BeaconUpgraded(address)
- Proxy admin changes: AdminChanged(address,address)
- Ownership transfers: OwnershipTransferred(address,address)
- Pause/unpause: Paused(address), Unpaused(address)
- AccessControl churn: RoleGranted/RoleRevoked/RoleAdminChanged

Outputs are written into data/ as JSON for reproducibility.

This file now also supports a "juicy" mode:
- Discover related contracts per AVS (proxy/admin/impl emitters) by looking for governance events
  that occur in transactions involving the AVS address.
- Compute AVS-week panels over the expanded contract set.
- Compute AVS activity panels (tx counts / distinct senders) to separate "popular" from "quiet" AVSs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.global_conf import CONF


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


def _load_operator_addresses_from_edges() -> List[str]:
    """
    Load operator addresses from data/eigenexplorer_operator_avs_edges.json.
    """
    p = _repo_root() / "data" / "eigenexplorer_operator_avs_edges.json"
    obj = _read_json(p)
    if not isinstance(obj, list):
        return []
    ops: set[str] = set()
    for e in obj:
        if not isinstance(e, dict):
            continue
        a = e.get("operator_address")
        if isinstance(a, str) and a.startswith("0x") and len(a) == 42:
            ops.add(a.lower())
    return sorted(ops)


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


def _as_dune_pair_literals(pairs: Sequence[Tuple[str, str]]) -> str:
    """
    Convert [(avs, contract), ...] into "(0xavs, 0xcontract), ..." for DuneSQL VALUES.
    """
    cleaned: List[Tuple[str, str]] = []
    for a, c in pairs:
        a = a.lower().strip()
        c = c.lower().strip()
        if a.startswith("0x") and len(a) == 42 and c.startswith("0x") and len(c) == 42:
            cleaned.append((a, c))
    if not cleaned:
        raise DuneError("No valid (avs, contract) pairs provided")
    return ",\n    ".join(f"({a}, {c})" for a, c in cleaned)


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
    # BeaconUpgraded(address)
    topic_beacon_upgraded = "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e"
    # AdminChanged(address,address)
    topic_admin_changed = "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f"
    # OwnershipTransferred(address,address)
    topic_ownership = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"
    # Paused(address)
    topic_paused = "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258"
    # Unpaused(address)
    topic_unpaused = "0x5db9ee0a495bf2e6ff9c91a7834c1ba4fdd244a5e8aa4e537bd38aeae4b073aa"
    # RoleGranted(bytes32,address,address)
    topic_role_granted = "0x2f8788117e7eff1d82e926ec794901d17c78024a50270940304540a733656f0d"
    # RoleRevoked(bytes32,address,address)
    topic_role_revoked = "0xf6391f5c32d9c69d2a47ea670b442974b53935d1edc7fd64eb21e047a839171b"
    # RoleAdminChanged(bytes32,bytes32,bytes32)
    topic_role_admin_changed = "0xbd79b86ffe0ab8e8776151514217cd7cacd52c909f66475c3af44e129f0b00ff"

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
    AND l.topic0 IN (
      {topic_upgraded},
      {topic_beacon_upgraded},
      {topic_admin_changed},
      {topic_ownership},
      {topic_paused},
      {topic_unpaused},
      {topic_role_granted},
      {topic_role_revoked},
      {topic_role_admin_changed}
    )
)
SELECT
  date_trunc('week', block_time) AS week,
  contract_address AS avs_address,
  SUM(CASE WHEN topic0 = {topic_upgraded} THEN 1 ELSE 0 END) AS n_upgraded_events,
  SUM(CASE WHEN topic0 = {topic_beacon_upgraded} THEN 1 ELSE 0 END) AS n_beacon_upgraded_events,
  SUM(CASE WHEN topic0 = {topic_admin_changed} THEN 1 ELSE 0 END) AS n_admin_changed_events,
  SUM(CASE WHEN topic0 = {topic_ownership} THEN 1 ELSE 0 END) AS n_ownership_transfers,
  SUM(CASE WHEN topic0 = {topic_paused} THEN 1 ELSE 0 END) AS n_paused_events,
  SUM(CASE WHEN topic0 = {topic_unpaused} THEN 1 ELSE 0 END) AS n_unpaused_events,
  SUM(CASE WHEN topic0 = {topic_role_granted} THEN 1 ELSE 0 END) AS n_role_granted_events,
  SUM(CASE WHEN topic0 = {topic_role_revoked} THEN 1 ELSE 0 END) AS n_role_revoked_events,
  SUM(CASE WHEN topic0 = {topic_role_admin_changed} THEN 1 ELSE 0 END) AS n_role_admin_changed_events,
  COUNT(*) AS n_total_events,
  COUNT(DISTINCT tx_hash) AS n_distinct_txs
FROM logs
GROUP BY 1, 2
ORDER BY 1, 2
""".strip()


def build_sql_avs_governance_events_panel(
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    Same as build_sql_avs_governance_events(), but returns a FULL panel:
    every (avs_address × week) row in the time range, with missing weeks filled as zeros.
    """
    addr_list = _as_dune_addr_literals(avs_addresses)

    # Reuse the exact same topic hashes as above by calling the other builder and extracting them is messy;
    # keep them duplicated here for clarity and stability.
    topic_upgraded = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"
    topic_beacon_upgraded = "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e"
    topic_admin_changed = "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f"
    topic_ownership = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"
    topic_paused = "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258"
    topic_unpaused = "0x5db9ee0a495bf2e6ff9c91a7834c1ba4fdd244a5e8aa4e537bd38aeae4b073aa"
    topic_role_granted = "0x2f8788117e7eff1d82e926ec794901d17c78024a50270940304540a733656f0d"
    topic_role_revoked = "0xf6391f5c32d9c69d2a47ea670b442974b53935d1edc7fd64eb21e047a839171b"
    topic_role_admin_changed = "0xbd79b86ffe0ab8e8776151514217cd7cacd52c909f66475c3af44e129f0b00ff"

    return f"""
WITH avs(address) AS (
  VALUES
    {addr_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
agg AS (
  SELECT
    date_trunc('week', l.block_time) AS week,
    l.contract_address AS avs_address,
    SUM(CASE WHEN l.topic0 = {topic_upgraded} THEN 1 ELSE 0 END) AS n_upgraded_events,
    SUM(CASE WHEN l.topic0 = {topic_beacon_upgraded} THEN 1 ELSE 0 END) AS n_beacon_upgraded_events,
    SUM(CASE WHEN l.topic0 = {topic_admin_changed} THEN 1 ELSE 0 END) AS n_admin_changed_events,
    SUM(CASE WHEN l.topic0 = {topic_ownership} THEN 1 ELSE 0 END) AS n_ownership_transfers,
    SUM(CASE WHEN l.topic0 = {topic_paused} THEN 1 ELSE 0 END) AS n_paused_events,
    SUM(CASE WHEN l.topic0 = {topic_unpaused} THEN 1 ELSE 0 END) AS n_unpaused_events,
    SUM(CASE WHEN l.topic0 = {topic_role_granted} THEN 1 ELSE 0 END) AS n_role_granted_events,
    SUM(CASE WHEN l.topic0 = {topic_role_revoked} THEN 1 ELSE 0 END) AS n_role_revoked_events,
    SUM(CASE WHEN l.topic0 = {topic_role_admin_changed} THEN 1 ELSE 0 END) AS n_role_admin_changed_events,
    COUNT(*) AS n_total_events,
    COUNT(DISTINCT l.tx_hash) AS n_distinct_txs
  FROM ethereum.logs l
  JOIN avs a ON l.contract_address = a.address
  WHERE l.block_time >= TIMESTAMP '{start_date}'
    AND l.block_time <  TIMESTAMP '{end_date}'
    AND l.topic0 IN (
      {topic_upgraded},
      {topic_beacon_upgraded},
      {topic_admin_changed},
      {topic_ownership},
      {topic_paused},
      {topic_unpaused},
      {topic_role_granted},
      {topic_role_revoked},
      {topic_role_admin_changed}
    )
  GROUP BY 1, 2
)
SELECT
  w.week,
  a.address AS avs_address,
  COALESCE(agg.n_upgraded_events, 0) AS n_upgraded_events,
  COALESCE(agg.n_beacon_upgraded_events, 0) AS n_beacon_upgraded_events,
  COALESCE(agg.n_admin_changed_events, 0) AS n_admin_changed_events,
  COALESCE(agg.n_ownership_transfers, 0) AS n_ownership_transfers,
  COALESCE(agg.n_paused_events, 0) AS n_paused_events,
  COALESCE(agg.n_unpaused_events, 0) AS n_unpaused_events,
  COALESCE(agg.n_role_granted_events, 0) AS n_role_granted_events,
  COALESCE(agg.n_role_revoked_events, 0) AS n_role_revoked_events,
  COALESCE(agg.n_role_admin_changed_events, 0) AS n_role_admin_changed_events,
  COALESCE(agg.n_total_events, 0) AS n_total_events,
  COALESCE(agg.n_distinct_txs, 0) AS n_distinct_txs
FROM weeks w
CROSS JOIN avs a
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.avs_address = a.address
ORDER BY 1, 2
""".strip()


def build_sql_avs_related_contracts(
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    Discover governance-event-emitting contracts that appear in transactions involving an AVS address.

    Logic:
    - First collect tx_hashes where (tx.to == avs) OR (a log emitted by avs exists)
    - Then within those tx_hashes, find logs with governance topic0 and record the emitting contract_address
    - Output a mapping avs_address -> related_contract_address with event counts
    """
    addr_list = _as_dune_addr_literals(avs_addresses)

    topic_upgraded = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"
    topic_beacon_upgraded = "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e"
    topic_admin_changed = "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f"
    topic_ownership = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"
    topic_paused = "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258"
    topic_unpaused = "0x5db9ee0a495bf2e6ff9c91a7834c1ba4fdd244a5e8aa4e537bd38aeae4b073aa"
    topic_role_granted = "0x2f8788117e7eff1d82e926ec794901d17c78024a50270940304540a733656f0d"
    topic_role_revoked = "0xf6391f5c32d9c69d2a47ea670b442974b53935d1edc7fd64eb21e047a839171b"
    topic_role_admin_changed = "0xbd79b86ffe0ab8e8776151514217cd7cacd52c909f66475c3af44e129f0b00ff"

    topics = f"""
      {topic_upgraded},
      {topic_beacon_upgraded},
      {topic_admin_changed},
      {topic_ownership},
      {topic_paused},
      {topic_unpaused},
      {topic_role_granted},
      {topic_role_revoked},
      {topic_role_admin_changed}
    """.strip()

    return f"""
WITH avs(address) AS (
  VALUES
    {addr_list}
),
avs_txs AS (
  -- txs directly sent to the AVS address
  SELECT
    a.address AS avs_address,
    t.hash AS tx_hash
  FROM ethereum.transactions t
  JOIN avs a ON t.to = a.address
  WHERE t.block_time >= TIMESTAMP '{start_date}'
    AND t.block_time <  TIMESTAMP '{end_date}'
),
avs_logs AS (
  -- txs where the AVS address emitted any log (captures internal calls/proxy patterns)
  SELECT DISTINCT
    a.address AS avs_address,
    l.tx_hash
  FROM ethereum.logs l
  JOIN avs a ON l.contract_address = a.address
  WHERE l.block_time >= TIMESTAMP '{start_date}'
    AND l.block_time <  TIMESTAMP '{end_date}'
),
avs_involved AS (
  SELECT * FROM avs_txs
  UNION
  SELECT * FROM avs_logs
),
gov_logs AS (
  SELECT
    i.avs_address,
    l.contract_address AS related_contract_address,
    l.topic0,
    l.tx_hash
  FROM avs_involved i
  JOIN ethereum.logs l
    ON l.tx_hash = i.tx_hash
  WHERE l.block_time >= TIMESTAMP '{start_date}'
    AND l.block_time <  TIMESTAMP '{end_date}'
    AND l.topic0 IN ({topics})
)
SELECT
  avs_address,
  related_contract_address,
  COUNT(*) AS n_gov_events,
  COUNT(DISTINCT tx_hash) AS n_gov_txs
FROM gov_logs
GROUP BY 1, 2
ORDER BY 1, 3 DESC
""".strip()


def build_sql_avs_activity_panel(
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    AVS-week activity panel using transactions directly to AVS address.
    """
    addr_list = _as_dune_addr_literals(avs_addresses)
    return f"""
WITH avs(address) AS (
  VALUES
    {addr_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
agg AS (
  SELECT
    date_trunc('week', t.block_time) AS week,
    a.address AS avs_address,
    COUNT(*) AS n_txs_to_avs,
    COUNT(DISTINCT t."from") AS n_distinct_senders
  FROM ethereum.transactions t
  JOIN avs a ON t.to = a.address
  WHERE t.block_time >= TIMESTAMP '{start_date}'
    AND t.block_time <  TIMESTAMP '{end_date}'
  GROUP BY 1, 2
)
SELECT
  w.week,
  a.address AS avs_address,
  COALESCE(agg.n_txs_to_avs, 0) AS n_txs_to_avs,
  COALESCE(agg.n_distinct_senders, 0) AS n_distinct_senders
FROM weeks w
CROSS JOIN avs a
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.avs_address = a.address
ORDER BY 1, 2
""".strip()


def build_sql_avs_governance_events_panel_expanded(
    avs_addresses: Sequence[str],
    avs_contract_pairs: Sequence[Tuple[str, str]],
    start_date: str,
    end_date: str,
) -> str:
    """
    Build a full AVS×week governance panel, but attribute events using an expanded contract set:
      mapping(avs_address -> contract_address)

    This lets us catch upgrades/admin/role events emitted on proxy/admin/implementation contracts,
    not only on the AVS service-manager address.
    """
    avs_list = _as_dune_addr_literals(avs_addresses)
    pair_list = _as_dune_pair_literals(avs_contract_pairs)

    topic_upgraded = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"
    topic_beacon_upgraded = "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e"
    topic_admin_changed = "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f"
    topic_ownership = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"
    topic_paused = "0x62e78cea01bee320cd4e420270b5ea74000d11b0c9f74754ebdbfc544b05a258"
    topic_unpaused = "0x5db9ee0a495bf2e6ff9c91a7834c1ba4fdd244a5e8aa4e537bd38aeae4b073aa"
    topic_role_granted = "0x2f8788117e7eff1d82e926ec794901d17c78024a50270940304540a733656f0d"
    topic_role_revoked = "0xf6391f5c32d9c69d2a47ea670b442974b53935d1edc7fd64eb21e047a839171b"
    topic_role_admin_changed = "0xbd79b86ffe0ab8e8776151514217cd7cacd52c909f66475c3af44e129f0b00ff"

    return f"""
WITH avs(address) AS (
  VALUES
    {avs_list}
),
mapping(avs_address, contract_address) AS (
  VALUES
    {pair_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
agg AS (
  SELECT
    date_trunc('week', l.block_time) AS week,
    m.avs_address,
    SUM(CASE WHEN l.topic0 = {topic_upgraded} THEN 1 ELSE 0 END) AS n_upgraded_events,
    SUM(CASE WHEN l.topic0 = {topic_beacon_upgraded} THEN 1 ELSE 0 END) AS n_beacon_upgraded_events,
    SUM(CASE WHEN l.topic0 = {topic_admin_changed} THEN 1 ELSE 0 END) AS n_admin_changed_events,
    SUM(CASE WHEN l.topic0 = {topic_ownership} THEN 1 ELSE 0 END) AS n_ownership_transfers,
    SUM(CASE WHEN l.topic0 = {topic_paused} THEN 1 ELSE 0 END) AS n_paused_events,
    SUM(CASE WHEN l.topic0 = {topic_unpaused} THEN 1 ELSE 0 END) AS n_unpaused_events,
    SUM(CASE WHEN l.topic0 = {topic_role_granted} THEN 1 ELSE 0 END) AS n_role_granted_events,
    SUM(CASE WHEN l.topic0 = {topic_role_revoked} THEN 1 ELSE 0 END) AS n_role_revoked_events,
    SUM(CASE WHEN l.topic0 = {topic_role_admin_changed} THEN 1 ELSE 0 END) AS n_role_admin_changed_events,
    COUNT(*) AS n_total_events,
    COUNT(DISTINCT l.tx_hash) AS n_distinct_txs
  FROM ethereum.logs l
  JOIN mapping m
    ON l.contract_address = m.contract_address
  WHERE l.block_time >= TIMESTAMP '{start_date}'
    AND l.block_time <  TIMESTAMP '{end_date}'
    AND l.topic0 IN (
      {topic_upgraded},
      {topic_beacon_upgraded},
      {topic_admin_changed},
      {topic_ownership},
      {topic_paused},
      {topic_unpaused},
      {topic_role_granted},
      {topic_role_revoked},
      {topic_role_admin_changed}
    )
  GROUP BY 1, 2
)
SELECT
  w.week,
  a.address AS avs_address,
  COALESCE(agg.n_upgraded_events, 0) AS n_upgraded_events,
  COALESCE(agg.n_beacon_upgraded_events, 0) AS n_beacon_upgraded_events,
  COALESCE(agg.n_admin_changed_events, 0) AS n_admin_changed_events,
  COALESCE(agg.n_ownership_transfers, 0) AS n_ownership_transfers,
  COALESCE(agg.n_paused_events, 0) AS n_paused_events,
  COALESCE(agg.n_unpaused_events, 0) AS n_unpaused_events,
  COALESCE(agg.n_role_granted_events, 0) AS n_role_granted_events,
  COALESCE(agg.n_role_revoked_events, 0) AS n_role_revoked_events,
  COALESCE(agg.n_role_admin_changed_events, 0) AS n_role_admin_changed_events,
  COALESCE(agg.n_total_events, 0) AS n_total_events,
  COALESCE(agg.n_distinct_txs, 0) AS n_distinct_txs
FROM weeks w
CROSS JOIN avs a
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.avs_address = a.address
ORDER BY 1, 2
""".strip()


def build_sql_avs_tokenflow_panel_expanded(
    avs_addresses: Sequence[str],
    avs_contract_pairs: Sequence[Tuple[str, str]],
    start_date: str,
    end_date: str,
) -> str:
    """
    Build a full AVS×week ERC20 tokenflow panel (expanded contract set attribution).

    This captures a much "juicier" economic signal:
    - number of ERC20 transfers in/out involving AVS-related contracts
    - number of distinct token contracts in/out
    - total raw value in/out (token units, not normalized by decimals)
    """
    avs_list = _as_dune_addr_literals(avs_addresses)
    pair_list = _as_dune_pair_literals(avs_contract_pairs)

    return f"""
WITH avs(address) AS (
  VALUES
    {avs_list}
),
mapping(avs_address, contract_address) AS (
  VALUES
    {pair_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
flows AS (
  -- OUTflows: AVS-related address is the sender
  SELECT
    date_trunc('week', e.evt_block_time) AS week,
    m.avs_address,
    'out' AS direction,
    e.contract_address AS token_address,
    COUNT(*) AS n_transfers,
    SUM(e.value) AS total_value_raw,
    COUNT(DISTINCT e.evt_tx_hash) AS n_distinct_txs
  FROM erc20_ethereum.evt_Transfer e
  JOIN mapping m
    ON e."from" = m.contract_address
  WHERE e.evt_block_time >= TIMESTAMP '{start_date}'
    AND e.evt_block_time <  TIMESTAMP '{end_date}'
  GROUP BY 1, 2, 3, 4

  UNION ALL

  -- INflows: AVS-related address is the receiver
  SELECT
    date_trunc('week', e.evt_block_time) AS week,
    m.avs_address,
    'in' AS direction,
    e.contract_address AS token_address,
    COUNT(*) AS n_transfers,
    SUM(e.value) AS total_value_raw,
    COUNT(DISTINCT e.evt_tx_hash) AS n_distinct_txs
  FROM erc20_ethereum.evt_Transfer e
  JOIN mapping m
    ON e."to" = m.contract_address
  WHERE e.evt_block_time >= TIMESTAMP '{start_date}'
    AND e.evt_block_time <  TIMESTAMP '{end_date}'
  GROUP BY 1, 2, 3, 4
),
agg AS (
  SELECT
    week,
    avs_address,
    SUM(CASE WHEN direction='out' THEN n_transfers ELSE 0 END) AS n_erc20_transfers_out,
    SUM(CASE WHEN direction='in'  THEN n_transfers ELSE 0 END) AS n_erc20_transfers_in,
    COUNT(DISTINCT CASE WHEN direction='out' THEN token_address END) AS n_unique_tokens_out,
    COUNT(DISTINCT CASE WHEN direction='in'  THEN token_address END) AS n_unique_tokens_in,
    SUM(CASE WHEN direction='out' THEN total_value_raw ELSE 0 END) AS total_erc20_value_raw_out,
    SUM(CASE WHEN direction='in'  THEN total_value_raw ELSE 0 END) AS total_erc20_value_raw_in,
    SUM(CASE WHEN direction='out' THEN n_distinct_txs ELSE 0 END) AS n_erc20_distinct_txs_out,
    SUM(CASE WHEN direction='in'  THEN n_distinct_txs ELSE 0 END) AS n_erc20_distinct_txs_in
  FROM flows
  GROUP BY 1, 2
)
SELECT
  w.week,
  a.address AS avs_address,
  COALESCE(agg.n_erc20_transfers_out, 0) AS n_erc20_transfers_out,
  COALESCE(agg.n_erc20_transfers_in, 0) AS n_erc20_transfers_in,
  COALESCE(agg.n_unique_tokens_out, 0) AS n_unique_tokens_out,
  COALESCE(agg.n_unique_tokens_in, 0) AS n_unique_tokens_in,
  COALESCE(agg.total_erc20_value_raw_out, 0) AS total_erc20_value_raw_out,
  COALESCE(agg.total_erc20_value_raw_in, 0) AS total_erc20_value_raw_in,
  COALESCE(agg.n_erc20_distinct_txs_out, 0) AS n_erc20_distinct_txs_out,
  COALESCE(agg.n_erc20_distinct_txs_in, 0) AS n_erc20_distinct_txs_in
FROM weeks w
CROSS JOIN avs a
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.avs_address = a.address
ORDER BY 1, 2
""".strip()


def build_sql_avs_ethflow_panel_expanded(
    avs_addresses: Sequence[str],
    avs_contract_pairs: Sequence[Tuple[str, str]],
    start_date: str,
    end_date: str,
) -> str:
    """
    AVS×week native ETH flow panel using ethereum.traces (captures internal transfers too).

    Measures:
    - count/value of ETH inflows to AVS-related contracts
    - count/value of ETH outflows from AVS-related contracts
    - distinct tx hashes in/out
    """
    avs_list = _as_dune_addr_literals(avs_addresses)
    pair_list = _as_dune_pair_literals(avs_contract_pairs)

    return f"""
WITH avs(address) AS (
  VALUES
    {avs_list}
),
mapping(avs_address, contract_address) AS (
  VALUES
    {pair_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
flows AS (
  -- OUTflows: AVS-related address is the sender
  SELECT
    date_trunc('week', tr.block_time) AS week,
    m.avs_address,
    'out' AS direction,
    COUNT(*) AS n_transfers,
    SUM(tr.value) AS total_value_wei,
    COUNT(DISTINCT tr.tx_hash) AS n_distinct_txs
  FROM ethereum.traces tr
  JOIN mapping m
    ON tr.\"from\" = m.contract_address
  WHERE tr.block_time >= TIMESTAMP '{start_date}'
    AND tr.block_time <  TIMESTAMP '{end_date}'
    AND tr.value > 0
    AND tr.success = TRUE
  GROUP BY 1, 2, 3

  UNION ALL

  -- INflows: AVS-related address is the receiver
  SELECT
    date_trunc('week', tr.block_time) AS week,
    m.avs_address,
    'in' AS direction,
    COUNT(*) AS n_transfers,
    SUM(tr.value) AS total_value_wei,
    COUNT(DISTINCT tr.tx_hash) AS n_distinct_txs
  FROM ethereum.traces tr
  JOIN mapping m
    ON tr.\"to\" = m.contract_address
  WHERE tr.block_time >= TIMESTAMP '{start_date}'
    AND tr.block_time <  TIMESTAMP '{end_date}'
    AND tr.value > 0
    AND tr.success = TRUE
  GROUP BY 1, 2, 3
),
agg AS (
  SELECT
    week,
    avs_address,
    SUM(CASE WHEN direction='out' THEN n_transfers ELSE 0 END) AS n_eth_transfers_out,
    SUM(CASE WHEN direction='in'  THEN n_transfers ELSE 0 END) AS n_eth_transfers_in,
    SUM(CASE WHEN direction='out' THEN total_value_wei ELSE 0 END) AS total_eth_value_wei_out,
    SUM(CASE WHEN direction='in'  THEN total_value_wei ELSE 0 END) AS total_eth_value_wei_in,
    SUM(CASE WHEN direction='out' THEN n_distinct_txs ELSE 0 END) AS n_eth_distinct_txs_out,
    SUM(CASE WHEN direction='in'  THEN n_distinct_txs ELSE 0 END) AS n_eth_distinct_txs_in
  FROM flows
  GROUP BY 1, 2
)
SELECT
  w.week,
  a.address AS avs_address,
  COALESCE(agg.n_eth_transfers_out, 0) AS n_eth_transfers_out,
  COALESCE(agg.n_eth_transfers_in, 0) AS n_eth_transfers_in,
  COALESCE(agg.total_eth_value_wei_out, 0) AS total_eth_value_wei_out,
  COALESCE(agg.total_eth_value_wei_in, 0) AS total_eth_value_wei_in,
  COALESCE(agg.n_eth_distinct_txs_out, 0) AS n_eth_distinct_txs_out,
  COALESCE(agg.n_eth_distinct_txs_in, 0) AS n_eth_distinct_txs_in
FROM weeks w
CROSS JOIN avs a
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.avs_address = a.address
ORDER BY 1, 2
""".strip()


def build_sql_operator_activity_panel(
    operator_addresses: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    Operator×week activity panel using ethereum.transactions.

    Focuses on operators that actually appear in the operator↔AVS exposure graph.
    """
    op_list = _as_dune_addr_literals(operator_addresses)
    return f"""
WITH operators(address) AS (
  VALUES
    {op_list}
),
weeks(week) AS (
  SELECT *
  FROM UNNEST(
    sequence(
      date_trunc('week', TIMESTAMP '{start_date}'),
      date_trunc('week', TIMESTAMP '{end_date}'),
      INTERVAL '7' day
    )
  ) AS t(week)
),
agg AS (
  SELECT
    date_trunc('week', t.block_time) AS week,
    t.\"from\" AS operator_address,
    COUNT(*) AS n_txs_sent,
    SUM(CASE WHEN t.success THEN 0 ELSE 1 END) AS n_failed_txs_sent,
    SUM(t.gas_used) AS total_gas_used,
    AVG(t.gas_used) AS avg_gas_used
  FROM ethereum.transactions t
  JOIN operators o ON t.\"from\" = o.address
  WHERE t.block_time >= TIMESTAMP '{start_date}'
    AND t.block_time <  TIMESTAMP '{end_date}'
  GROUP BY 1, 2
)
SELECT
  w.week,
  o.address AS operator_address,
  COALESCE(agg.n_txs_sent, 0) AS n_txs_sent,
  COALESCE(agg.n_failed_txs_sent, 0) AS n_failed_txs_sent,
  COALESCE(agg.total_gas_used, 0) AS total_gas_used,
  COALESCE(agg.avg_gas_used, 0) AS avg_gas_used
FROM weeks w
CROSS JOIN operators o
LEFT JOIN agg
  ON agg.week = w.week
 AND agg.operator_address = o.address
ORDER BY 1, 2
""".strip()


def run_avs_governance_events(
    dune: DuneClient,
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
    performance: str = "medium",
    out_name: str = "dune_avs_governance_events.json",
    skip_if_exists: bool = False,
) -> Path:
    out_dir = _ensure_data_dir()
    out_path = out_dir / out_name
    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 10:
        print(f"exists, skipping: {out_path}")
        return out_path
    sql = build_sql_avs_governance_events(avs_addresses, start_date=start_date, end_date=end_date)
    execution_id = dune.execute_sql(sql, performance=performance)
    results = dune.wait_for_results(execution_id)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Small console summary
    result = results.get("result", {})
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_rows = meta.get("total_row_count") or meta.get("total_rows")
    print(f"wrote results to {out_path} (execution_id={execution_id}, rows={total_rows})")
    return out_path


def run_avs_governance_events_panel(
    dune: DuneClient,
    avs_addresses: Sequence[str],
    start_date: str,
    end_date: str,
    performance: str = "medium",
    out_name: str = "dune_avs_governance_events_panel.json",
    skip_if_exists: bool = False,
) -> Path:
    out_dir = _ensure_data_dir()
    out_path = out_dir / out_name
    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 10:
        print(f"exists, skipping: {out_path}")
        return out_path
    sql = build_sql_avs_governance_events_panel(avs_addresses, start_date=start_date, end_date=end_date)
    execution_id = dune.execute_sql(sql, performance=performance)
    results = dune.wait_for_results(execution_id, timeout_secs=600.0)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    result = results.get("result", {})
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_rows = meta.get("total_row_count") or meta.get("total_rows")
    print(f"wrote results to {out_path} (execution_id={execution_id}, rows={total_rows})")
    return out_path


def run_sql_to_data(
    dune: DuneClient,
    sql: str,
    out_name: str,
    performance: str = "medium",
    timeout_secs: float = 600.0,
    skip_if_exists: bool = False,
) -> Path:
    out_dir = _ensure_data_dir()
    out_path = out_dir / out_name
    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 10:
        print(f"exists, skipping: {out_path}")
        return out_path
    execution_id = dune.execute_sql(sql, performance=performance)
    results = dune.wait_for_results(execution_id, timeout_secs=timeout_secs)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    result = results.get("result", {})
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_rows = meta.get("total_row_count") or meta.get("total_rows")
    print(f"wrote results to {out_path} (execution_id={execution_id}, rows={total_rows})")
    return out_path


def _load_dune_rows(result_path: Path) -> List[Dict[str, Any]]:
    obj = _read_json(result_path)
    res = obj.get("result", {}) if isinstance(obj, dict) else {}
    rows = res.get("rows", []) if isinstance(res, dict) else []
    return rows if isinstance(rows, list) else []


def build_avs_contract_mapping_pairs(
    avs_addresses: Sequence[str],
    related_rows: List[Dict[str, Any]],
    top_k_per_avs: int = 10,
) -> List[Tuple[str, str]]:
    """
    Build (avs_address, contract_address) pairs for expanded attribution.

    We always include (avs, avs). Then we add up to top_k_per_avs related contracts per AVS,
    ranked by n_gov_events (desc). This avoids exploding the mapping when there is noise.
    """
    avs_set = {a.lower() for a in avs_addresses}
    per: Dict[str, List[Tuple[str, int]]] = {}

    for r in related_rows:
        avs = (r.get("avs_address") or "").lower()
        rel = (r.get("related_contract_address") or "").lower()
        if avs not in avs_set:
            continue
        if not (avs.startswith("0x") and len(avs) == 42 and rel.startswith("0x") and len(rel) == 42):
            continue
        n = r.get("n_gov_events", 0)
        try:
            n_int = int(n)
        except Exception:
            n_int = 0
        per.setdefault(avs, []).append((rel, n_int))

    pairs: List[Tuple[str, str]] = []
    for avs in avs_addresses:
        avs = avs.lower()
        pairs.append((avs, avs))  # always include self
        rels = per.get(avs, [])
        rels_sorted = sorted(rels, key=lambda t: t[1], reverse=True)
        added = 0
        seen: set[str] = {avs}
        for rel, _n in rels_sorted:
            if rel in seen:
                continue
            pairs.append((avs, rel))
            seen.add(rel)
            added += 1
            if added >= top_k_per_avs:
                break

    # de-dupe pairs while preserving order
    out: List[Tuple[str, str]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    for p in pairs:
        if p in seen_pairs:
            continue
        out.append(p)
        seen_pairs.add(p)
    return out


def run_full_year_panels(
    *,
    fetch_all: bool = True,
    small_n: int = 5,
    lookback_days: int = 365,
    performance: str = "medium",
    top_k_related_contracts: int = 10,
    timeout_secs: float = 900.0,
    progress_logger: Optional[Any] = None,
    skip_if_exists: bool = False,
) -> None:
    """
    In-process "main" runner for Dune extraction (same logic as the __main__ block).

    This is intended for orchestration from src.main_pipeline without spawning subprocesses.
    """
    try:
        # Time range: rolling window ending today (UTC)
        _end = datetime.now(timezone.utc).date()
        _start = _end - timedelta(days=int(lookback_days))
        start_date = _start.isoformat()
        end_date = _end.isoformat()

        key = (CONF.DUNE_API_KEY or "").strip()
        if not key:
            raise DuneError("Missing DUNE_API_KEY (expected in secrets.json)")

        avs_all = _load_avs_addresses()
        avs = avs_all if fetch_all else avs_all[: max(1, int(small_n))]

        dune = DuneClient(api_key=key)

        step = 0
        total_steps = 7

        def _log(msg: str) -> None:
            if progress_logger:
                pct = int(round(100.0 * step / max(1, total_steps)))
                progress_logger.info(f"[dune {pct:3d}%] {msg}")

        step = 1
        _log("Governance events (sparse)")
        run_avs_governance_events(
            dune,
            avs_addresses=avs,
            start_date=start_date,
            end_date=end_date,
            performance=performance,
            out_name="dune_avs_governance_events_all.json" if fetch_all else "dune_avs_governance_events_small.json",
            skip_if_exists=skip_if_exists,
        )

        step = 2
        _log("Governance events panel (AVS×week)")
        run_avs_governance_events_panel(
            dune,
            avs_addresses=avs,
            start_date=start_date,
            end_date=end_date,
            performance=performance,
            out_name="dune_avs_governance_events_panel_all.json"
            if fetch_all
            else "dune_avs_governance_events_panel_small.json",
            skip_if_exists=skip_if_exists,
        )

        # Discover related contracts and compute expanded panels
        step = 3
        _log("Related contracts discovery")
        related_sql = build_sql_avs_related_contracts(avs, start_date=start_date, end_date=end_date)
        related_path = run_sql_to_data(
            dune,
            sql=related_sql,
            out_name="dune_avs_related_contracts_all.json" if fetch_all else "dune_avs_related_contracts_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )

        step = 4
        _log("AVS activity panel (txs/senders)")
        activity_sql = build_sql_avs_activity_panel(avs, start_date=start_date, end_date=end_date)
        run_sql_to_data(
            dune,
            sql=activity_sql,
            out_name="dune_avs_activity_panel_all.json" if fetch_all else "dune_avs_activity_panel_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )

        related_rows = _load_dune_rows(related_path)
        pairs = build_avs_contract_mapping_pairs(avs, related_rows, top_k_per_avs=int(top_k_related_contracts))

        step = 5
        _log("Expanded governance panel (AVS×week, expanded attribution)")
        expanded_sql = build_sql_avs_governance_events_panel_expanded(
            avs_addresses=avs,
            avs_contract_pairs=pairs,
            start_date=start_date,
            end_date=end_date,
        )
        run_sql_to_data(
            dune,
            sql=expanded_sql,
            out_name="dune_avs_governance_events_panel_expanded_all.json"
            if fetch_all
            else "dune_avs_governance_events_panel_expanded_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )

        step = 6
        _log("Token flow panel (ERC20 in/out)")
        tokenflow_sql = build_sql_avs_tokenflow_panel_expanded(
            avs_addresses=avs,
            avs_contract_pairs=pairs,
            start_date=start_date,
            end_date=end_date,
        )
        run_sql_to_data(
            dune,
            sql=tokenflow_sql,
            out_name="dune_avs_tokenflow_panel_expanded_all.json"
            if fetch_all
            else "dune_avs_tokenflow_panel_expanded_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )

        step = 7
        _log("ETH flow panel + operator activity panel")
        ethflow_sql = build_sql_avs_ethflow_panel_expanded(
            avs_addresses=avs,
            avs_contract_pairs=pairs,
            start_date=start_date,
            end_date=end_date,
        )
        run_sql_to_data(
            dune,
            sql=ethflow_sql,
            out_name="dune_avs_ethflow_panel_expanded_all.json" if fetch_all else "dune_avs_ethflow_panel_expanded_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )

        # Operator activity panel (operators in exposure graph)
        operator_addresses = _load_operator_addresses_from_edges()
        operator_sql = build_sql_operator_activity_panel(operator_addresses, start_date=start_date, end_date=end_date)
        run_sql_to_data(
            dune,
            sql=operator_sql,
            out_name="dune_operator_activity_panel_all.json" if fetch_all else "dune_operator_activity_panel_small.json",
            performance=performance,
            timeout_secs=timeout_secs,
            skip_if_exists=skip_if_exists,
        )
    except Exception as e:
        raise DuneError(f"run_full_year_panels failed: {e}") from e


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    # - FETCH_ALL=True will include all AVS addresses (currently 49).
    # - FETCH_ALL=False uses a smaller subset for cheaper/faster iteration.
    FETCH_ALL = True
    SMALL_N = 5

    # Time range for event extraction
    # Use a rolling 1-year window ending "today" (UTC) for free-tier feasibility.
    _end = datetime.now(timezone.utc).date()
    _start = _end - timedelta(days=365)
    START_DATE = _start.isoformat()
    END_DATE = _end.isoformat()

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

    # Also write a full AVS×week panel (zeros included), which is typically what you want for modeling.
    run_avs_governance_events_panel(
        dune,
        avs_addresses=avs,
        start_date=START_DATE,
        end_date=END_DATE,
        performance="medium",
        out_name="dune_avs_governance_events_panel_all.json" if FETCH_ALL else "dune_avs_governance_events_panel_small.json",
    )

    # Juicy additions:
    # 1) Discover related contracts per AVS that emit governance events in AVS-involved transactions.
    related_sql = build_sql_avs_related_contracts(avs, start_date=START_DATE, end_date=END_DATE)
    related_path = run_sql_to_data(
        dune,
        sql=related_sql,
        out_name="dune_avs_related_contracts_all.json" if FETCH_ALL else "dune_avs_related_contracts_small.json",
        performance="medium",
        timeout_secs=900.0,
    )

    # 2) AVS-week activity (txs to AVS) panel.
    activity_sql = build_sql_avs_activity_panel(avs, start_date=START_DATE, end_date=END_DATE)
    run_sql_to_data(
        dune,
        sql=activity_sql,
        out_name="dune_avs_activity_panel_all.json" if FETCH_ALL else "dune_avs_activity_panel_small.json",
        performance="medium",
        timeout_secs=900.0,
    )

    # 3) Expanded attribution: build AVS×week governance panel over (avs -> top-K related contracts).
    related_rows = _load_dune_rows(related_path)
    pairs = build_avs_contract_mapping_pairs(avs, related_rows, top_k_per_avs=10)
    expanded_sql = build_sql_avs_governance_events_panel_expanded(
        avs_addresses=avs,
        avs_contract_pairs=pairs,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    run_sql_to_data(
        dune,
        sql=expanded_sql,
        out_name="dune_avs_governance_events_panel_expanded_all.json"
        if FETCH_ALL
        else "dune_avs_governance_events_panel_expanded_small.json",
        performance="medium",
        timeout_secs=1200.0,
    )

    # 4) Expanded attribution: AVS×week ERC20 tokenflow panel (economic signal).
    tokenflow_sql = build_sql_avs_tokenflow_panel_expanded(
        avs_addresses=avs,
        avs_contract_pairs=pairs,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    run_sql_to_data(
        dune,
        sql=tokenflow_sql,
        out_name="dune_avs_tokenflow_panel_expanded_all.json"
        if FETCH_ALL
        else "dune_avs_tokenflow_panel_expanded_small.json",
        performance="medium",
        timeout_secs=1800.0,
    )

    # 5) Expanded attribution: AVS×week native ETH flow panel (economic signal, includes internal transfers).
    ethflow_sql = build_sql_avs_ethflow_panel_expanded(
        avs_addresses=avs,
        avs_contract_pairs=pairs,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    run_sql_to_data(
        dune,
        sql=ethflow_sql,
        out_name="dune_avs_ethflow_panel_expanded_all.json" if FETCH_ALL else "dune_avs_ethflow_panel_expanded_small.json",
        performance="medium",
        timeout_secs=1800.0,
    )

    # 6) Operator×week activity panel for operators that are actually in the exposure graph.
    # We load operators from the exposure edge list (local file) to keep this focused and free-tier friendly.
    # (Dune SQL needs literals, so this list must be reasonably sized.)
    try:
        edges = _read_json(_repo_root() / "data" / "eigenexplorer_operator_avs_edges.json")
        ops = sorted({e["operator_address"] for e in edges if isinstance(e, dict) and isinstance(e.get("operator_address"), str)})
    except Exception:
        ops = []
    if ops:
        op_sql = build_sql_operator_activity_panel(ops, start_date=START_DATE, end_date=END_DATE)
        run_sql_to_data(
            dune,
            sql=op_sql,
            out_name="dune_operator_activity_panel_all.json" if FETCH_ALL else "dune_operator_activity_panel_small.json",
            performance="medium",
            timeout_secs=1800.0,
        )

