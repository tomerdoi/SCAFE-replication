"""
Global configuration (keys, secrets, runtime settings).

This repo intentionally keeps configuration in ONE place.
Secrets should be provided via environment variables and never committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    """Read environment variable (empty string treated as missing)."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


@dataclass(frozen=True, slots=True)
class GlobalConf:
    # --- Etherscan API v2 ---
    # NOTE: This is a secret. Prefer using env vars for real projects.
    # Kept here per repo workflow request.
    ETHERSCAN_API_KEY: str = "Y2X5Z21WJP6M9SP5UM6PWD7AHJZJDY6AEA"
    ETHERSCAN_BASE_URL: str = _env("ETHERSCAN_BASE_URL", "https://api.etherscan.io") or "https://api.etherscan.io"
    ETHERSCAN_CHAIN_ID: int = int(_env("ETHERSCAN_CHAIN_ID", "1") or "1")  # 1=Ethereum mainnet

    # --- HTTP / retries ---
    HTTP_TIMEOUT_SECS: int = int(_env("HTTP_TIMEOUT_SECS", "30") or "30")
    HTTP_MAX_RETRIES: int = int(_env("HTTP_MAX_RETRIES", "3") or "3")
    HTTP_BACKOFF_SECS: float = float(_env("HTTP_BACKOFF_SECS", "1.0") or "1.0")


CONF = GlobalConf()

