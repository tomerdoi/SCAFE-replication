"""
EigenExplorer AVS → operator exposure utility.

Goal:
- Fetch AVS registry (GET /avs)
- For each AVS, fetch its operators (GET /avs/{avsAddress}/operators)
- Extract operator addresses and write an operator↔AVS edge list.

This is the backbone for "operator risk = exposure-weighted AVS risk" papers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .global_conf import CONF
except ImportError:  # pragma: no cover
    from global_conf import CONF  # type: ignore


class EigenExplorerError(RuntimeError):
    pass


@dataclass(slots=True)
class EigenExplorerClient:
    api_key: str
    base_url: str = "https://api.eigenexplorer.com"
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
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Token": self.api_key}

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self._session.get(url, headers=self._headers(), params=params or {}, timeout=self.timeout_secs)
        if r.status_code >= 400:
            raise EigenExplorerError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if not isinstance(data, dict):
            raise EigenExplorerError(f"Unexpected response type: {type(data)}")
        return data

    def list_avs_page(self, skip: int = 0, take: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        payload = self.get("/avs", params={"skip": skip, "take": take})
        rows = payload.get("data", [])
        meta = payload.get("meta", {})
        if not isinstance(rows, list):
            raise EigenExplorerError("Unexpected 'data' field in /avs response")
        if not isinstance(meta, dict):
            meta = {}
        return rows, meta

    def list_operator_sets_page(self, avs_address: str, skip: int = 0, take: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # NOTE: some docs mention operator-sets, but the current public API exposes /avs/{address}/operators.
        # Kept as a placeholder in case operator-sets endpoints are enabled later.
        payload = self.get(f"/avs/{avs_address}/operator-sets", params={"skip": skip, "take": take})
        rows = payload.get("data", [])
        meta = payload.get("meta", {})
        if not isinstance(rows, list):
            raise EigenExplorerError("Unexpected 'data' field in operator-sets response")
        if not isinstance(meta, dict):
            meta = {}
        return rows, meta

    def get_operator_set_details(self, avs_address: str, operator_set_id: int) -> Dict[str, Any]:
        return self.get(f"/avs/{avs_address}/operator-sets/{operator_set_id}")

    def list_avs_operators_page(self, avs_address: str, skip: int = 0, take: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        GET /avs/{address}/operators (paged).

        Response shape:
          {"data":[{"address":...,"metadataName":...,...}], "meta": {...}}
        """
        payload = self.get(f"/avs/{avs_address}/operators", params={"skip": skip, "take": take})
        rows = payload.get("data", [])
        meta = payload.get("meta", {})
        if not isinstance(rows, list):
            raise EigenExplorerError("Unexpected 'data' field in /avs/{address}/operators response")
        if not isinstance(meta, dict):
            meta = {}
        return rows, meta


def _ensure_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_first_address(obj: Dict[str, Any]) -> Optional[str]:
    """
    Try common keys for an AVS "address" (service manager contract).
    """
    for k in ("address", "avsAddress", "serviceManagerAddress", "serviceManager", "contractAddress"):
        v = obj.get(k)
        if isinstance(v, str) and v.startswith("0x") and len(v) == 42:
            return v.lower()
    return None


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _extract_operator_addresses(payload: Dict[str, Any]) -> Set[str]:
    """
    Best-effort extraction of operator addresses from an operator-set details payload.
    Different versions may nest operators under allocations/operators/etc.
    """
    out: Set[str] = set()
    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        for key in ("operator", "operatorAddress", "address"):
            v = node.get(key)
            if isinstance(v, str) and v.startswith("0x") and len(v) == 42:
                out.add(v.lower())
    return out


def main_generate_operator_avs_edges(page_size: int = 100, limit_avs: Optional[int] = None) -> None:
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    client = EigenExplorerClient(api_key=key)

    # 1) Fetch all AVSs
    avs_rows: List[Dict[str, Any]] = []
    skip = 0
    total: Optional[int] = None
    while True:
        rows, meta = client.list_avs_page(skip=skip, take=page_size)
        if not rows:
            break
        avs_rows.extend(rows)
        if total is None:
            t = meta.get("total")
            if isinstance(t, int):
                total = t
        skip += len(rows)
        if total is not None and skip >= total:
            break

    # Optionally limit for debugging
    if limit_avs is not None:
        avs_rows = avs_rows[: max(0, int(limit_avs))]

    edges: List[Dict[str, Any]] = []
    avs_operators_payloads: List[Dict[str, Any]] = []

    for avs in avs_rows:
        avs_addr = _extract_first_address(avs)
        if not avs_addr:
            continue

        # 2) Fetch operators for this AVS
        operators: List[Dict[str, Any]] = []
        op_skip = 0
        op_total: Optional[int] = None
        while True:
            batch, meta = client.list_avs_operators_page(avs_addr, skip=op_skip, take=page_size)
            if not batch:
                break
            operators.extend(batch)
            if op_total is None:
                t = meta.get("total")
                if isinstance(t, int):
                    op_total = t
            op_skip += len(batch)
            if op_total is not None and op_skip >= op_total:
                break

        avs_operators_payloads.append({"avs_address": avs_addr, "operators": operators})

        for op in operators:
            op_addr = op.get("address")
            if isinstance(op_addr, str) and op_addr.startswith("0x") and len(op_addr) == 42:
                edges.append(
                    {
                        "operator_address": op_addr.lower(),
                        "avs_address": avs_addr,
                    }
                )

    out_dir = _ensure_data_dir()
    edges_path = out_dir / "eigenexplorer_operator_avs_edges.json"
    raw_path = out_dir / "eigenexplorer_avs_operators_payloads.json"

    with edges_path.open("w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2)

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(avs_operators_payloads, f, indent=2)

    print(f"wrote {len(edges)} edges to {edges_path}")
    print(f"wrote {len(avs_operators_payloads)} AVS operator payloads to {raw_path}")


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    PAGE_SIZE = 100
    LIMIT_AVS = None  # set to None to process all AVSs (can take a while)

    main_generate_operator_avs_edges(page_size=max(1, int(PAGE_SIZE)), limit_avs=LIMIT_AVS)

