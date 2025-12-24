"""
Minimal Etherscan API v2 utility.

Goal: a small, reliable "smoke test" to confirm the Etherscan v2 API works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    # Preferred: package import (works with `python -m src.etherscan_util`)
    from .global_conf import CONF
except ImportError:  # pragma: no cover
    # Fallback: direct script execution from within `src/` (IDE "Run file")
    from global_conf import CONF  # type: ignore


class EtherscanError(RuntimeError):
    pass


@dataclass(slots=True)
class EtherscanV2Client:
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

    def call(self, module: str, action: str, **params: Any) -> Any:
        """
        Call Etherscan API v2 endpoint.

        Etherscan v2 format:
          GET https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_blockNumber&apikey=...
        """
        q: Dict[str, Any] = {"chainid": self.chain_id, "module": module, "action": action, "apikey": self.api_key}
        q.update(params)

        r = self._session.get(f"{self.base_url}/v2/api", params=q, timeout=self.timeout_secs)
        data = r.json()

        # Success example: {"jsonrpc":"2.0","id":1,"result":"0x..."}
        if isinstance(data, dict) and "error" in data:
            raise EtherscanError(f"Etherscan error: {data['error']}")

        # Some failures come back in v1-ish shape even on v2 URLs
        if isinstance(data, dict) and data.get("status") == "0" and data.get("message") == "NOTOK":
            raise EtherscanError(f"Etherscan NOTOK: {data.get('result')}")

        return data.get("result") if isinstance(data, dict) else data

    def eth_block_number(self) -> int:
        """Return latest block number as int."""
        result = self.call("proxy", "eth_blockNumber")
        if not isinstance(result, str) or not result.startswith("0x"):
            raise EtherscanError(f"Unexpected eth_blockNumber result: {result!r}")
        return int(result, 16)

    def eth_get_transaction_count(self, address: str, tag: str = "latest") -> int:
        """Return tx count for an address at a given tag."""
        result = self.call("proxy", "eth_getTransactionCount", address=address, tag=tag)
        if not isinstance(result, str) or not result.startswith("0x"):
            raise EtherscanError(f"Unexpected eth_getTransactionCount result: {result!r}")
        return int(result, 16)


def smoke_test(api_key: Optional[str] = None) -> int:
    """
    Minimal API smoke test. Returns latest block number as int.
    """
    key = (api_key or CONF.ETHERSCAN_API_KEY or "").strip()
    if not key:
        raise EtherscanError("Missing ETHERSCAN_API_KEY (set it in env or pass api_key=...)")
    return EtherscanV2Client(api_key=key).eth_block_number()


if __name__ == "__main__":
    # Run as a module for package imports:
    #   python3 -m src.etherscan_util
    bn = smoke_test()
    print(f"etherscan v2 ok. latest block: {bn}")

