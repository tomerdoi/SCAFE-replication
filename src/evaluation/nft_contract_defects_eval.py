"""
Score contracts from the NFTDefects dataset (NFTContractDefects.csv) using this repo's
contract technical risk score.

Input:
- A CSV with a contract address column (commonly: "Contract Address")

Outputs (under data/derived/ by default):
- nft_contract_defects_registry.csv              (the addresses we scored; subset when limit>0)
- nft_contract_defects_contract_risk.csv         (raw ContractRiskUtil output; subset when limit>0)
- nft_contract_defects_scored.csv                (original dataset joined with risk columns; subset filled when limit>0)
- nft_contract_defects_contract_risk_all.csv     (optional: risk output for ALL dataset addresses; best for full runs)
- nft_contract_defects_scored_all.csv            (optional: dataset joined with *_all risk output)

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
        batch_size: int = 0,
        fetch_sources: bool = True,
        fetch_activity: bool = False,
        skip_existing: bool = True,
        activity_max_pages: int = 2,
        out_scored_csv: Optional[Path] = None,
        out_risk_csv: Optional[Path] = None,
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
        risk_csv = Path(out_risk_csv) if out_risk_csv else (self.derived_dir / "nft_contract_defects_contract_risk.csv")
        out_scored_csv = Path(out_scored_csv) if out_scored_csv else (self.derived_dir / "nft_contract_defects_scored.csv")

        reg = pd.DataFrame([{"address": a, "sources": "nftdefects"} for a in addrs])
        reg.to_csv(registry_csv, index=False)
        self.logger.info(f"Wrote registry: {registry_csv} (rows={len(reg)})")

        def maybe_fetch(*, addresses_csv: Path) -> None:
            # Optionally fetch caches for these addresses (helps when dataset addresses aren't already cached).
            if fetch_sources:
                try:
                    from src.clients.etherscan_source import EtherscanAVSSourceUtil

                    self.logger.info("Fetching Etherscan sources+bytecode (best-effort)…")
                    EtherscanAVSSourceUtil().run(
                        fetch_all=True,
                        limit=0,
                        addresses_csv=Path(addresses_csv),
                        progress_logger=self.logger,
                        progress_every=5,
                        skip_existing=bool(skip_existing),
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Skipping source fetch (continuing with cache-only scoring): {type(e).__name__}: {e}"
                    )

            if fetch_activity:
                try:
                    from src.clients.etherscan_activity import EtherscanActivityUtil

                    self.logger.info("Fetching Etherscan activity (best-effort)…")
                    EtherscanActivityUtil().run(
                        addresses_csv=Path(addresses_csv),
                        fetch_all=True,
                        limit=0,
                        skip_existing=bool(skip_existing),
                        max_pages=int(activity_max_pages),
                        progress_logger=self.logger,
                        progress_every=5,
                    )
                except Exception as e:
                    self.logger.warning(f"Skipping activity fetch (continuing): {type(e).__name__}: {e}")

        def score_one_registry(*, addresses_csv: Path, out_csv: Path) -> None:
            # Score using cached payloads under data/etherscan_sources (+ optional activity payloads).
            ContractRiskUtil().run(registry_csv=Path(addresses_csv), out_csv=Path(out_csv), limit=0)

        # If batch_size is set, run in chunks and append to a single risk CSV.
        bs = int(batch_size or 0)
        if bs > 0 and len(addrs) > bs:
            risk_csv.parent.mkdir(parents=True, exist_ok=True)
            if risk_csv.exists():
                risk_csv.unlink()
            self.logger.info(f"Batch mode enabled: batch_size={bs} over {len(addrs)} addresses")

            for start in range(0, len(addrs), bs):
                end = min(len(addrs), start + bs)
                batch_addrs = addrs[start:end]
                batch_registry = self.derived_dir / f"nft_contract_defects_registry_batch_{start}_{end}.csv"
                pd.DataFrame([{"address": a, "sources": "nftdefects"} for a in batch_addrs]).to_csv(batch_registry, index=False)
                self.logger.info(f"Batch {start}:{end} registry written: {batch_registry}")

                maybe_fetch(addresses_csv=batch_registry)

                batch_risk = self.derived_dir / f"nft_contract_defects_contract_risk_batch_{start}_{end}.csv"
                score_one_registry(addresses_csv=batch_registry, out_csv=batch_risk)

                bdf = pd.read_csv(batch_risk)
                write_header = not risk_csv.exists()
                bdf.to_csv(risk_csv, index=False, mode="a", header=write_header)
                self.logger.info(f"Appended batch risk to: {risk_csv} (batch_rows={len(bdf)})")
        else:
            maybe_fetch(addresses_csv=registry_csv)
            score_one_registry(addresses_csv=registry_csv, out_csv=risk_csv)

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
    limit = 0  # 0 = score ALL dataset contracts (16,527)
    batch_size = 250  # set >0 for long runs to save progress incrementally (recommended)

    # Fetch caches via Etherscan (best-effort). Requires ETHERSCAN_API_KEY.
    fetch_sources = True
    fetch_activity = False
    skip_existing = True
    activity_max_pages = 2

    out_risk_csv = DERIVED_DIR / "nft_contract_defects_contract_risk_all.csv"
    out_scored_csv = DERIVED_DIR / "nft_contract_defects_scored_all.csv"

    NFTContractDefectsEvaluation().run(
        input_csv=Path(input_csv),
        limit=int(limit),
        batch_size=int(batch_size),
        fetch_sources=bool(fetch_sources),
        fetch_activity=bool(fetch_activity),
        skip_existing=bool(skip_existing),
        activity_max_pages=int(activity_max_pages),
        out_scored_csv=Path(out_scored_csv),
        out_risk_csv=Path(out_risk_csv),
    )

