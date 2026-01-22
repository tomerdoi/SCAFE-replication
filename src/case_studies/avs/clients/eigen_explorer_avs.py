"""
EigenExplorer AVS registry utility.

Fetches AVS deployments from:
  GET /avs

Writes JSON into data/ for reproducible pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.global_conf import CONF
from src.paths import AVS_DATA_DIR


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
        """
        GET /avs (paged).
        """
        payload = self.get("/avs", params={"skip": skip, "take": take})
        rows = payload.get("data", [])
        meta = payload.get("meta", {})
        if not isinstance(rows, list):
            raise EigenExplorerError("Unexpected 'data' field in response")
        if not isinstance(meta, dict):
            meta = {}
        return rows, meta


def _ensure_data_dir() -> Path:
    AVS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return AVS_DATA_DIR


def main_generate_100_rows() -> None:
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    client = EigenExplorerClient(api_key=key)
    rows, meta = client.list_avs_page(skip=0, take=100)

    out_dir = _ensure_data_dir()
    out_json = out_dir / "eigenexplorer_avs_100.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": rows}, f, indent=2)

    print(f"wrote {len(rows)} rows to {out_json}")


def main_generate_all_rows(page_size: int = 100) -> None:
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    client = EigenExplorerClient(api_key=key)

    all_rows: List[Dict[str, Any]] = []
    skip = 0
    total: Optional[int] = None

    while True:
        rows, meta = client.list_avs_page(skip=skip, take=page_size)
        if not rows:
            break
        all_rows.extend(rows)

        if total is None:
            t = meta.get("total")
            if isinstance(t, int):
                total = t

        skip += len(rows)
        if total is not None:
            pct = int(round(100.0 * skip / max(1, total)))
            print(f"[eigenexplorer avs] {skip}/{total} ({pct}%)")
        if total is not None and skip >= total:
            break

    out_dir = _ensure_data_dir()
    out_json = out_dir / "eigenexplorer_avs_all.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": {"total": total, "fetched": len(all_rows)}, "data": all_rows}, f, indent=2)

    print(f"wrote {len(all_rows)} rows to {out_json}")
    if total is not None:
        print(f"meta.total reported: {total}")


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    FETCH_ALL = True
    PAGE_SIZE = 100  # used only when FETCH_ALL=True

    if FETCH_ALL:
        main_generate_all_rows(page_size=max(1, int(PAGE_SIZE)))
    else:
        main_generate_100_rows()

