"""
Minimal EigenExplorer API utility.

Includes a __main__ that fetches 100 real rows from EigenExplorer (operators/addresses).

Docs / base URLs:
- Mainnet:  https://api.eigenexplorer.com
- Holesky:  https://api-holesky.eigenexplorer.com

Auth:
- Header: X-API-Token: <YOUR_API_KEY>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.global_conf import CONF


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

    def list_operator_addresses(self, skip: int = 0, take: int = 100) -> List[Dict[str, Any]]:
        """
        GET /operators/addresses
        Response shape (per docs):
          {"data":[{"address":...,"name":...,"logo":...}], "meta": {...}}
        """
        rows, _meta = self.list_operator_addresses_page(skip=skip, take=take)
        return rows

    def list_operator_addresses_page(self, skip: int = 0, take: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fetch a single page of operator address rows + meta.
        """
        payload = self.get("/operators/addresses", params={"skip": skip, "take": take})
        rows = payload.get("data", [])
        meta = payload.get("meta", {})
        if not isinstance(rows, list):
            raise EigenExplorerError("Unexpected 'data' field in response")
        if not isinstance(meta, dict):
            meta = {}
        return rows, meta


def _ensure_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main_generate_100_rows() -> None:
    """
    Fetch 100 rows from EigenExplorer and write them to data/.
    """
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    client = EigenExplorerClient(api_key=key)
    rows = client.list_operator_addresses(skip=0, take=100)

    out_dir = _ensure_data_dir()
    out_json = out_dir / "eigenexplorer_operator_addresses_100.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} rows to {out_json}")
    # Also print a small preview
    for r in rows[:5]:
        addr = r.get("address")
        name = r.get("name")
        print(f"- {addr}  {name}")


def main_generate_all_rows(page_size: int = 100) -> None:
    """
    Fetch ALL operator rows from EigenExplorer by paging, and write them to data/.
    """
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    client = EigenExplorerClient(api_key=key)

    all_rows: List[Dict[str, Any]] = []
    skip = 0
    total: Optional[int] = None

    while True:
        rows, meta = client.list_operator_addresses_page(skip=skip, take=page_size)
        if not rows:
            break
        all_rows.extend(rows)

        # meta expected: {total, skip, take}
        if total is None:
            t = meta.get("total")
            if isinstance(t, int):
                total = t

        skip += len(rows)
        if total is not None:
            pct = int(round(100.0 * skip / max(1, total)))
            print(f"[eigenexplorer operators] {skip}/{total} ({pct}%)")
        if total is not None and skip >= total:
            break

    out_dir = _ensure_data_dir()
    out_json = out_dir / "eigenexplorer_operator_addresses_all.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print(f"wrote {len(all_rows)} rows to {out_json}")
    if total is not None:
        print(f"meta.total reported: {total}")


if __name__ == "__main__":
    # Run:
    #   python3 -m src.eigen_explorer_util
    # or:
    #   python3 src/eigen_explorer_util.py
    #
    # Toggle behavior here (no argparse):
    FETCH_ALL = True
    PAGE_SIZE = 100  # used only when FETCH_ALL=True

    if FETCH_ALL:
        main_generate_all_rows(page_size=max(1, int(PAGE_SIZE)))
    else:
        main_generate_100_rows()

