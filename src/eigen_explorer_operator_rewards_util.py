"""
EigenExplorer operator rewards utility.

Fetches reward strategies and reward token addresses for operators:
  GET /operators/{address}/rewards

By default this runs on the subset of operators that actually appear in the operator↔AVS exposure graph
(data/eigenexplorer_operator_avs_edges.json) to keep the dataset dense and relevant for risk scoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

    def get_operator_rewards(self, operator_address: str) -> Dict[str, Any]:
        op = operator_address.lower()
        return self.get(f"/operators/{op}/rewards")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_data_dir() -> Path:
    d = _repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_edges_operators() -> List[str]:
    p = _repo_root() / "data" / "eigenexplorer_operator_avs_edges.json"
    edges = json.loads(p.read_text(encoding="utf-8"))
    ops: Set[str] = set()
    for e in edges:
        if isinstance(e, dict):
            a = e.get("operator_address")
            if isinstance(a, str) and a.startswith("0x") and len(a) == 42:
                ops.add(a.lower())
    return sorted(ops)


def main_fetch_operator_rewards(
    limit: Optional[int] = None,
    *,
    progress_logger: Optional[Any] = None,
    progress_every: int = 25,
) -> None:
    """
    Fetch operator rewards and write data/eigenexplorer_operator_rewards.json.

    progress_logger: optional logger with .info(...) for percent progress
    """
    key = (CONF.EIGEN_EXPLORER_API_KEY or "").strip()
    if not key:
        raise EigenExplorerError("Missing EIGEN_EXPLORER_API_KEY (expected in secrets.json)")

    ops = _load_edges_operators()
    if limit is not None:
        ops = ops[: max(0, int(limit))]

    client = EigenExplorerClient(api_key=key)
    out: List[Dict[str, Any]] = []

    total = len(ops)
    for i, op in enumerate(ops, start=1):
        try:
            payload = client.get_operator_rewards(op)
            payload["operator_address"] = op
            out.append(payload)
        except Exception as e:
            out.append({"operator_address": op, "error": str(e)})

        if progress_logger and (i == 1 or i % max(1, int(progress_every)) == 0 or i == total):
            pct = int(round(100.0 * i / max(1, total)))
            progress_logger.info(f"[operator rewards] {i}/{total} ({pct}%)")

        # gentle pacing
        if i % 20 == 0:
            time.sleep(0.25)

    out_dir = _ensure_data_dir()
    out_path = out_dir / "eigenexplorer_operator_rewards.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    FETCH_ALL = True
    SMALL_N = 50

    main_fetch_operator_rewards(limit=None if FETCH_ALL else SMALL_N)

