"""
u--1 (u--1.com) EigenLayer live API utility.

Fetches free-tier live snapshots:
- Operators: /v2/latest-operator-balances
- AVSs:      /v2/latest-avs-balances
- Strategies:/v2/strategies
- Pricing:   /v2/latest-pricing
- AVS risks: /v2/avs-risks-summary

Writes JSON snapshots under data/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.global_conf import CONF
from src.paths import AVS_DATA_DIR
from src.paths import REPO_ROOT


class U1Error(RuntimeError):
    pass


@dataclass(slots=True)
class U1Client:
    api_key: str
    base_url: str = CONF.U1_BASE_URL
    timeout_secs: int = CONF.HTTP_TIMEOUT_SECS
    max_retries: int = CONF.HTTP_MAX_RETRIES
    backoff_secs: float = CONF.HTTP_BACKOFF_SECS
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or "https://api.u--1.com").rstrip("/")

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

        # Avoid brotli dependency surprises by not advertising br support explicitly.
        self._session.headers.update({"Accept-Encoding": "gzip, deflate"})

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        q = dict(params or {})
        q["api_key"] = self.api_key
        r = self._session.get(url, params=q, timeout=self.timeout_secs)
        if r.status_code >= 400:
            raise U1Error(f"HTTP {r.status_code}: {r.text[:300]}")
        # API returns JSON arrays/objects
        return r.json()

    def latest_operator_balances(self) -> Any:
        return self.get("/v2/latest-operator-balances")

    def latest_avs_balances(self) -> Any:
        return self.get("/v2/latest-avs-balances")

    def strategies(self) -> Any:
        return self.get("/v2/strategies")

    def latest_pricing(self) -> Any:
        return self.get("/v2/latest-pricing")

    def avs_risks_summary(self) -> Any:
        return self.get("/v2/avs-risks-summary")


def _repo_root() -> Path:
    return REPO_ROOT


def _ensure_data_dir() -> Path:
    AVS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return AVS_DATA_DIR


def _write_json(name: str, payload: Any) -> Path:
    out_dir = _ensure_data_dir()
    path = out_dir / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main_fetch_all_live(*, skip_if_exists: bool = True) -> None:
    key = (CONF.U1_API_KEY or "").strip()
    if not key:
        raise U1Error("Missing U1_API_KEY (add to secrets.json as U1_API_KEY)")

    client = U1Client(api_key=key)

    out_dir = _ensure_data_dir()
    p1 = out_dir / "u1_latest_operator_balances.json"
    if not (skip_if_exists and p1.exists()):
        operators = client.latest_operator_balances()
        p1 = _write_json("u1_latest_operator_balances.json", operators)
        print(f"wrote {p1}")
    else:
        print(f"exists, skipping: {p1}")

    time.sleep(0.2)
    p2 = out_dir / "u1_latest_avs_balances.json"
    if not (skip_if_exists and p2.exists()):
        avs = client.latest_avs_balances()
        p2 = _write_json("u1_latest_avs_balances.json", avs)
        print(f"wrote {p2}")
    else:
        print(f"exists, skipping: {p2}")

    time.sleep(0.2)
    p3 = out_dir / "u1_strategies.json"
    if not (skip_if_exists and p3.exists()):
        strategies = client.strategies()
        p3 = _write_json("u1_strategies.json", strategies)
        print(f"wrote {p3}")
    else:
        print(f"exists, skipping: {p3}")

    time.sleep(0.2)
    p4 = out_dir / "u1_latest_pricing.json"
    if not (skip_if_exists and p4.exists()):
        pricing = client.latest_pricing()
        p4 = _write_json("u1_latest_pricing.json", pricing)
        print(f"wrote {p4}")
    else:
        print(f"exists, skipping: {p4}")

    time.sleep(0.2)
    p5 = out_dir / "u1_avs_risks_summary.json"
    if not (skip_if_exists and p5.exists()):
        risks = client.avs_risks_summary()
        p5 = _write_json("u1_avs_risks_summary.json", risks)
        print(f"wrote {p5}")
    else:
        print(f"exists, skipping: {p5}")


if __name__ == "__main__":
    # Toggle behavior here (no argparse):
    FETCH_ALL = True
    if FETCH_ALL:
        main_fetch_all_live()

