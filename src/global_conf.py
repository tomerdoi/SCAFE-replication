"""
Global configuration (keys, secrets, runtime settings).

This repo intentionally keeps configuration in ONE place.
Secrets are loaded from repo-root `secrets.json` (not committed).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    # src/global_conf.py -> repo root is one level up from src/
    return Path(__file__).resolve().parents[1]


def _load_secrets() -> dict:
    """
    Load secrets from repo-root secrets file.
    Expected keys: ETHERSCAN_API_KEY, EIGEN_EXPLORER_API_KEY, DUNE_API_KEY
    """
    root = _repo_root()
    for filename in ("secrets.json", "secrets.local.json"):
        path = root / filename
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            continue
        except Exception:
            # Fail closed: if secrets file exists but is malformed, treat as missing.
            return {}
    return {}


_SECRETS = _load_secrets()


def _secret(name: str, default: str | None = None) -> str | None:
    value = _SECRETS.get(name, default)
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value or default


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
    ETHERSCAN_API_KEY: str | None = _secret("ETHERSCAN_API_KEY") or _env("ETHERSCAN_API_KEY")
    ETHERSCAN_BASE_URL: str = _env("ETHERSCAN_BASE_URL", "https://api.etherscan.io") or "https://api.etherscan.io"
    ETHERSCAN_CHAIN_ID: int = int(_env("ETHERSCAN_CHAIN_ID", "1") or "1")  # 1=Ethereum mainnet

    # --- Eigen Explorer / EigenLayer ecosystem APIs (if used later) ---
    EIGEN_EXPLORER_API_KEY: str | None = _secret("EIGEN_EXPLORER_API_KEY") or _env("EIGEN_EXPLORER_API_KEY")

    # --- Dune Analytics ---
    DUNE_API_KEY: str | None = _secret("DUNE_API_KEY") or _env("DUNE_API_KEY")

    # --- u--1 (EigenLayer live API) ---
    U1_API_KEY: str | None = _secret("U1_API_KEY") or _secret("U--1_API_KEY") or _env("U1_API_KEY")
    U1_BASE_URL: str = _env("U1_BASE_URL", "https://api.u--1.com") or "https://api.u--1.com"

    # --- HTTP / retries ---
    HTTP_TIMEOUT_SECS: int = int(_env("HTTP_TIMEOUT_SECS", "30") or "30")
    HTTP_MAX_RETRIES: int = int(_env("HTTP_MAX_RETRIES", "3") or "3")
    HTTP_BACKOFF_SECS: float = float(_env("HTTP_BACKOFF_SECS", "1.0") or "1.0")


CONF = GlobalConf()

