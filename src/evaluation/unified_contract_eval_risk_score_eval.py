"""
Score ALL contracts from the unified evaluation Excel (built by `unified_dataset_excel.py`)
using this repo's contract technical risk score.

Input:
- data/derived/unified_contract_eval_dataset.xlsx (sheet: "contracts")

Outputs (under data/derived/ by default):
- unified_contract_eval_registry.csv              (unique addresses scored)
- unified_contract_eval_contract_risk.csv         (one row per unique address; ContractRiskUtil output + recomputed z/rank)
- unified_contract_eval_scored.xlsx               (all original dataset rows joined with risk columns)

Quick run:
  /path/to/venv/bin/python -m src.evaluation.unified_contract_eval_risk_score_eval

Notes:
- We score UNIQUE addresses (deduped) for correctness/perf; the scored Excel preserves ALL original rows.
- Contracts without an on-chain address (address=="") remain in the output but will have blank risk columns.
- Missing cached Etherscan payloads are allowed: those addresses get `has_payload=0` and `risk_raw=0`.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running as a file in IDE/debugger (so `import src...` works)
# src/evaluation/unified_contract_eval_risk_score_eval.py -> repo root is two levels up
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_DIR, DERIVED_DIR  # noqa: E402
from src.risk.contract_risk import ContractRiskUtil  # noqa: E402


def _is_addr(s: object) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _norm_addr(s: object) -> str:
    return str(s or "").strip().lower()


def _stable_dedupe(seq: list[str]) -> list[str]:
    # Preserve order of first appearance
    return list(dict.fromkeys(seq))


def _recompute_z_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute risk_z and risk_rank across the FULL address set.
    Useful when resuming by appending new addresses (batch scoring would otherwise z-score per chunk).
    """
    out = df.copy()
    if "risk_raw" not in out.columns:
        return out
    x = pd.to_numeric(out["risk_raw"], errors="coerce").fillna(0.0)
    mu = float(x.mean())
    sigma = float(x.std())
    sigma = sigma if sigma and sigma > 0 else 1.0
    out["risk_z"] = (x - mu) / sigma
    out["risk_rank"] = pd.to_numeric(out["risk_z"], errors="coerce").rank(ascending=False, method="min").astype(int)
    return out


@dataclass(frozen=True)
class Paths:
    in_xlsx: Path = DERIVED_DIR / "unified_contract_eval_dataset.xlsx"
    out_registry_csv: Path = DERIVED_DIR / "unified_contract_eval_registry.csv"
    out_risk_csv: Path = DERIVED_DIR / "unified_contract_eval_contract_risk.csv"
    out_scored_xlsx: Path = DERIVED_DIR / "unified_contract_eval_scored.xlsx"


class UnifiedContractEvalRiskScoreEvaluation:
    def __init__(self) -> None:
        logger = logging.getLogger("unified_contract_eval_risk_score_eval")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.derived_dir = DERIVED_DIR

    def run(
        self,
        *,
        in_xlsx: Path,
        out_registry_csv: Optional[Path] = None,
        out_risk_csv: Optional[Path] = None,
        out_scored_xlsx: Optional[Path] = None,
        resume: bool = True,
    ) -> Path:
        in_xlsx = Path(in_xlsx)
        if not in_xlsx.exists():
            raise FileNotFoundError(f"Input XLSX not found: {in_xlsx}")

        out_registry_csv = Path(out_registry_csv) if out_registry_csv else (self.derived_dir / "unified_contract_eval_registry.csv")
        out_risk_csv = Path(out_risk_csv) if out_risk_csv else (self.derived_dir / "unified_contract_eval_contract_risk.csv")
        out_scored_xlsx = Path(out_scored_xlsx) if out_scored_xlsx else (self.derived_dir / "unified_contract_eval_scored.xlsx")

        self.derived_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Reading unified dataset: {in_xlsx} (sheet='contracts')…")
        df = pd.read_excel(in_xlsx, sheet_name="contracts")
        if "address" not in df.columns:
            raise ValueError(f"Unified dataset missing required column 'address'. Columns: {list(df.columns)!r}")

        df["_address_norm"] = df["address"].map(_norm_addr)
        all_addrs = [a for a in df["_address_norm"].tolist() if _is_addr(a)]
        uniq_addrs = _stable_dedupe(all_addrs)
        if not uniq_addrs:
            raise ValueError("No valid 0x addresses found in unified dataset.")

        reg = pd.DataFrame([{"address": a, "sources": "unified_contract_eval"} for a in uniq_addrs])
        reg.to_csv(out_registry_csv, index=False)
        self.logger.info(f"Wrote registry: {out_registry_csv} (rows={len(reg)})")

        existing_df: Optional[pd.DataFrame] = None
        existing_addrs: set[str] = set()
        if resume and out_risk_csv.exists():
            try:
                existing_df = pd.read_csv(out_risk_csv)
                if "address" in existing_df.columns:
                    existing_df["address"] = existing_df["address"].map(_norm_addr)
                    existing_addrs = set([a for a in existing_df["address"].tolist() if _is_addr(a)])
                    self.logger.info(f"Resume enabled: found existing risk file with {len(existing_addrs)} addresses.")
            except Exception as e:
                self.logger.warning(f"Failed to read existing risk CSV for resume; will re-score all: {type(e).__name__}: {e}")
                existing_df = None
                existing_addrs = set()

        remaining = [a for a in uniq_addrs if a not in existing_addrs]
        self.logger.info(f"Scoring unique addresses: total={len(uniq_addrs)}, remaining={len(remaining)}")

        new_df: Optional[pd.DataFrame] = None
        if remaining:
            tmp_registry = self.derived_dir / "unified_contract_eval_registry_remaining.csv"
            tmp_risk = self.derived_dir / "unified_contract_eval_contract_risk_remaining.csv"
            pd.DataFrame([{"address": a, "sources": "unified_contract_eval"} for a in remaining]).to_csv(tmp_registry, index=False)
            self.logger.info("Running ContractRiskUtil (cache-only scoring)…")
            ContractRiskUtil().run(registry_csv=tmp_registry, out_csv=tmp_risk, limit=0)
            new_df = pd.read_csv(tmp_risk)
            if "address" in new_df.columns:
                new_df["address"] = new_df["address"].map(_norm_addr)

        if existing_df is None:
            risk_df = new_df if new_df is not None else pd.DataFrame()
        else:
            parts = [existing_df]
            if new_df is not None and not new_df.empty:
                parts.append(new_df)
            risk_df = pd.concat(parts, ignore_index=True)

        if risk_df.empty:
            raise ValueError("Risk scoring produced no rows.")

        # One row per address, deterministically.
        risk_df = risk_df.sort_values(["address"]).drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
        risk_df = _recompute_z_and_rank(risk_df)

        out_risk_csv.parent.mkdir(parents=True, exist_ok=True)
        risk_df.to_csv(out_risk_csv, index=False)
        self.logger.info(f"Wrote risk-by-address CSV: {out_risk_csv} (rows={len(risk_df)})")

        # Merge back onto ALL original rows (preserve dataset columns for vuln/dataset analysis).
        join_df = risk_df.copy()
        join_df["_address_norm"] = join_df["address"].map(_norm_addr)
        scored = df.merge(join_df, how="left", on="_address_norm", suffixes=("", "_risk"))
        # Keep original address column; drop the risk-side duplicate.
        if "address_risk" in scored.columns:
            scored.drop(columns=["address_risk"], inplace=True)
        scored.drop(columns=["_address_norm"], inplace=True)

        # Helpful flags
        scored["has_risk_row"] = scored["risk_raw"].notna().astype(int)

        # Write a scored Excel with separate sheets for easier manual review.
        out_scored_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_scored_xlsx, engine="openpyxl") as w:
            scored.to_excel(w, sheet_name="contracts_scored", index=False)
            risk_df.to_excel(w, sheet_name="risk_by_address", index=False)
            summary = (
                scored.groupby(["source_dataset", "chain"], dropna=False)
                .agg(
                    rows=("contract_id", "count"),
                    rows_with_address=("address", lambda s: int(sum([_is_addr(_norm_addr(x)) for x in s]))),
                    unique_addresses=("address", lambda s: len(set([_norm_addr(x) for x in s if _is_addr(_norm_addr(x))]))),
                    rows_scored=("has_risk_row", "sum"),
                    missing_payloads=("has_payload", lambda s: int(sum(pd.to_numeric(s, errors="coerce").fillna(0).astype(int) == 0))),
                )
                .reset_index()
                .sort_values(["source_dataset", "chain"])
            )
            summary.to_excel(w, sheet_name="summary", index=False)

        self.logger.info(f"Wrote scored Excel: {out_scored_xlsx} (rows={len(scored)})")
        return out_scored_xlsx


if __name__ == "__main__":
    # ---- Editable run parameters (no CLI args) ----
    paths = Paths()
    resume = True

    UnifiedContractEvalRiskScoreEvaluation().run(
        in_xlsx=paths.in_xlsx,
        out_registry_csv=paths.out_registry_csv,
        out_risk_csv=paths.out_risk_csv,
        out_scored_xlsx=paths.out_scored_xlsx,
        resume=bool(resume),
    )

