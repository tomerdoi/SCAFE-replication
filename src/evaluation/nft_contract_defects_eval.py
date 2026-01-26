"""
Score contracts from the NFTDefects dataset (NFTContractDefects.csv) using this repo's
contract technical risk score.

Input:
- A CSV with a contract address column (commonly: "Contract Address")

Outputs (under data/derived/ by default):
- nft_contract_defects_registry.csv        (the addresses we scored)
- nft_contract_defects_contract_risk.csv   (raw ContractRiskUtil output)
- nft_contract_defects_scored.csv          (original dataset joined with risk columns)

Quick run (default limit=10):
  python3 -m src.evaluation.nft_contract_defects_eval

Notes:
- By default, this module will try to fetch missing Etherscan *sources+bytecode* caches,
  because NFTDefects addresses are unlikely to already exist in data/etherscan_sources/.
  Set ETHERSCAN_API_KEY (secrets.json or env) for fetching to work.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# Allow running as a file in IDE/debugger (so `import src...` works)
# src/evaluation/nft_contract_defects_eval.py -> repo root is two levels up
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_DIR, DERIVED_DIR  # noqa: E402
from src.risk.contract_risk import ContractRiskUtil  # noqa: E402


def _is_addr(s: object) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _find_address_col(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    normalized = {str(c).strip().lower(): str(c) for c in cols}
    for key in (
        "contract address",
        "contract_address",
        "address",
        "contract",
        "contractaddress",
    ):
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find contract address column. Columns: {cols!r}")


class NFTContractDefectsEvaluation:
    def __init__(self) -> None:
        logger = logging.getLogger("nft_contract_defects_eval")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        self.data_dir = DATA_DIR
        self.derived_dir = DERIVED_DIR

    def run(
        self,
        *,
        input_csv: Path,
        limit: int = 10,
        fetch_sources: bool = True,
        fetch_activity: bool = False,
        skip_existing: bool = True,
        activity_max_pages: int = 2,
        out_scored_csv: Optional[Path] = None,
    ) -> Path:
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        self.derived_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv)
        addr_col = _find_address_col(df)
        self.logger.info(f"Using address column: {addr_col!r}")

        addrs = [str(x).strip().lower() for x in df[addr_col].tolist()]
        addrs = [a for a in addrs if _is_addr(a)]
        if limit and int(limit) > 0:
            addrs = addrs[: int(limit)]
        if not addrs:
            raise ValueError("No valid 0x addresses found in input after filtering.")

        registry_csv = self.derived_dir / "nft_contract_defects_registry.csv"
        risk_csv = self.derived_dir / "nft_contract_defects_contract_risk.csv"
        out_scored_csv = Path(out_scored_csv) if out_scored_csv else (self.derived_dir / "nft_contract_defects_scored.csv")

        reg = pd.DataFrame([{"address": a, "sources": "nftdefects"} for a in addrs])
        reg.to_csv(registry_csv, index=False)
        self.logger.info(f"Wrote registry: {registry_csv} (rows={len(reg)})")

        # Optionally fetch caches for these addresses (helps when dataset addresses aren't already cached).
        if fetch_sources:
            try:
                from src.clients.etherscan_source import EtherscanAVSSourceUtil

                self.logger.info("Fetching Etherscan sources+bytecode (best-effort)…")
                EtherscanAVSSourceUtil().run(
                    fetch_all=True,
                    limit=0,
                    addresses_csv=registry_csv,
                    progress_logger=self.logger,
                    progress_every=5,
                    skip_existing=bool(skip_existing),
                )
            except Exception as e:
                self.logger.warning(f"Skipping source fetch (continuing with cache-only scoring): {type(e).__name__}: {e}")

        if fetch_activity:
            try:
                from src.clients.etherscan_activity import EtherscanActivityUtil

                self.logger.info("Fetching Etherscan activity (best-effort)…")
                EtherscanActivityUtil().run(
                    addresses_csv=registry_csv,
                    fetch_all=True,
                    limit=0,
                    skip_existing=bool(skip_existing),
                    max_pages=int(activity_max_pages),
                    progress_logger=self.logger,
                    progress_every=5,
                )
            except Exception as e:
                self.logger.warning(f"Skipping activity fetch (continuing): {type(e).__name__}: {e}")

        # Score using cached payloads under data/etherscan_sources (+ optional activity payloads).
        ContractRiskUtil().run(registry_csv=registry_csv, out_csv=risk_csv, limit=0)

        risk_df = pd.read_csv(risk_csv)
        risk_df["address"] = risk_df["address"].astype(str).str.strip().str.lower()

        out = df.copy()
        out["_address_norm"] = out[addr_col].astype(str).str.strip().str.lower()
        out = out.merge(risk_df, how="left", left_on="_address_norm", right_on="address", suffixes=("", "_risk"))
        out.drop(columns=[c for c in ["_address_norm"] if c in out.columns], inplace=True)

        out_scored_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_scored_csv, index=False)
        self.logger.info(f"Wrote scored dataset: {out_scored_csv} (rows={len(out)})")
        return out_scored_csv


if __name__ == "__main__":
    # ---- Editable run parameters (no CLI args) ----
    input_csv = DATA_DIR / "raw" / "NFTContractDefects.csv"
    limit = 10  # default run limit

    # Fetch caches via Etherscan (best-effort). Requires ETHERSCAN_API_KEY.
    fetch_sources = True
    fetch_activity = False
    skip_existing = True
    activity_max_pages = 2

    out_scored_csv = DERIVED_DIR / "nft_contract_defects_scored.csv"

    NFTContractDefectsEvaluation().run(
        input_csv=Path(input_csv),
        limit=int(limit),
        fetch_sources=bool(fetch_sources),
        fetch_activity=bool(fetch_activity),
        skip_existing=bool(skip_existing),
        activity_max_pages=int(activity_max_pages),
        out_scored_csv=Path(out_scored_csv),
    )

