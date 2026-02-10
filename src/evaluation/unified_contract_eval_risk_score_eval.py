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
import time
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
from src.clients.etherscan_activity import EtherscanActivityUtil  # noqa: E402
from src.clients.etherscan_source import EtherscanAVSSourceUtil  # noqa: E402
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
        out_scored_csv: Optional[Path] = None,
        resume: bool = True,
        fetch_sources: bool = True,
        fetch_activity: bool = False,
        fetch_missing_only: bool = True,
        skip_existing_fetch: bool = True,
        fetch_chunk_size: int = 0,
        sleep_between_chunks_secs: float = 2.0,
        activity_max_pages: int = 2,
    ) -> Path:
        in_xlsx = Path(in_xlsx)
        if not in_xlsx.exists():
            raise FileNotFoundError(f"Input XLSX not found: {in_xlsx}")

        out_registry_csv = Path(out_registry_csv) if out_registry_csv else (self.derived_dir / "unified_contract_eval_registry.csv")
        out_risk_csv = Path(out_risk_csv) if out_risk_csv else (self.derived_dir / "unified_contract_eval_contract_risk.csv")
        out_scored_xlsx = Path(out_scored_xlsx) if out_scored_xlsx else (self.derived_dir / "unified_contract_eval_scored.xlsx")
        out_scored_csv = Path(out_scored_csv) if out_scored_csv else (self.derived_dir / "unified_contract_eval_scored.csv")

        self.derived_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Reading unified dataset: {in_xlsx} (sheet='contracts')…")
        # NOTE: pandas requires openpyxl to read .xlsx. We surface a clear error message if missing.
        try:
            df = pd.read_excel(in_xlsx, sheet_name="contracts")
        except ImportError as e:
            raise ImportError(
                "Reading XLSX requires `openpyxl`. Install it in your active environment, e.g. "
                "`pip install openpyxl`, then rerun."
            ) from e
        if "address" not in df.columns:
            raise ValueError(f"Unified dataset missing required column 'address'. Columns: {list(df.columns)!r}")

        df["_address_norm"] = df["address"].map(_norm_addr)
        all_addrs = [a for a in df["_address_norm"].tolist() if _is_addr(a)]
        uniq_addrs = _stable_dedupe(all_addrs)

        # Rows without on-chain address but with local .sol source (e.g. DAppSCAN)
        has_local_source = (
            df["local_source_path"].fillna("").astype(str).str.strip().str.len() > 0
            if "local_source_path" in df.columns
            else pd.Series([False] * len(df), index=df.index)
        )
        no_addr_mask = ~df["_address_norm"].map(_is_addr)
        local_only_mask = no_addr_mask & has_local_source
        n_local_only = int(local_only_mask.sum())
        self.logger.info(
            f"Dataset: {len(df)} rows, {len(uniq_addrs)} unique addresses, "
            f"{n_local_only} local-source-only rows (no on-chain address)"
        )

        reg = pd.DataFrame([{"address": a, "sources": "unified_contract_eval"} for a in uniq_addrs])
        reg.to_csv(out_registry_csv, index=False)
        self.logger.info(f"Wrote registry: {out_registry_csv} (rows={len(reg)})")

        # --- Optional: fetch missing payloads from Etherscan ---
        # The risk scoring reads cached payloads under data/etherscan_sources/<addr>.json (+ optional activity).
        # If those files aren't present, ContractRiskUtil will score with has_payload=0 and risk_raw=0.
        if fetch_sources or fetch_activity:
            sources_dir = DATA_DIR / "etherscan_sources"
            activity_dir = DATA_DIR / "etherscan_activity"
            sources_dir.mkdir(parents=True, exist_ok=True)
            activity_dir.mkdir(parents=True, exist_ok=True)

            def compute_missing() -> list[str]:
                addrs_to_fetch = uniq_addrs
                if fetch_missing_only:
                    missing_src = [a for a in uniq_addrs if not (sources_dir / f"{a}.json").exists()]
                    if fetch_activity:
                        missing_act = [a for a in uniq_addrs if not (activity_dir / f"{a}.json").exists()]
                        addrs_to_fetch = _stable_dedupe(missing_src + missing_act)
                    else:
                        addrs_to_fetch = missing_src
                return addrs_to_fetch

            missing_all = compute_missing()
            if not missing_all:
                self.logger.info("No missing Etherscan payloads detected (nothing to fetch).")
            else:
                total_missing = len(missing_all)
                # Chunk internally to reduce rate-limit risk, but run end-to-end automatically.
                chunk_size = int(fetch_chunk_size) if int(fetch_chunk_size) > 0 else total_missing
                sleep_s = float(max(0.0, sleep_between_chunks_secs))

                done = 0
                chunk_idx = 0
                while True:
                    missing_all = compute_missing()
                    if not missing_all:
                        break
                    chunk_idx += 1
                    chunk = missing_all[:chunk_size]
                    done = total_missing - len(missing_all)
                    self.logger.info(
                        f"Missing payloads remaining: {len(missing_all)}. "
                        f"Fetching chunk {chunk_idx} (size={len(chunk)})…"
                    )

                    tmp_fetch_registry = self.derived_dir / "unified_contract_eval_registry_fetch_missing.csv"
                    pd.DataFrame([{"address": a, "sources": "unified_contract_eval"} for a in chunk]).to_csv(
                        tmp_fetch_registry, index=False
                    )

                    if fetch_sources:
                        self.logger.info(
                            f"Fetching Etherscan source+bytecode payloads for {len(chunk)} address(es) "
                            f"(missing_only={bool(fetch_missing_only)}, skip_existing={bool(skip_existing_fetch)})…"
                        )
                        EtherscanAVSSourceUtil().run(
                            fetch_all=True,
                            limit=0,
                            addresses_csv=tmp_fetch_registry,
                            progress_logger=self.logger,
                            progress_every=25,
                            skip_existing=bool(skip_existing_fetch),
                        )

                    if fetch_activity:
                        self.logger.info(
                            f"Fetching Etherscan activity payloads for {len(chunk)} address(es) "
                            f"(missing_only={bool(fetch_missing_only)}, skip_existing={bool(skip_existing_fetch)})…"
                        )
                        EtherscanActivityUtil().run(
                            addresses_csv=tmp_fetch_registry,
                            fetch_all=True,
                            limit=0,
                            skip_existing=bool(skip_existing_fetch),
                            max_pages=int(activity_max_pages),
                            progress_logger=self.logger,
                            progress_every=25,
                        )

                    # polite delay between chunks (helps avoid getting throttled / banned)
                    if sleep_s > 0 and len(missing_all) > len(chunk):
                        self.logger.info(f"Sleeping {sleep_s:.1f}s before next fetch chunk…")
                        time.sleep(sleep_s)

        existing_df: Optional[pd.DataFrame] = None
        existing_addrs: set[str] = set()
        existing_missing_payload: set[str] = set()
        existing_missing_activity: set[str] = set()
        if resume and out_risk_csv.exists():
            try:
                existing_df = pd.read_csv(out_risk_csv)
                if "address" in existing_df.columns:
                    existing_df["address"] = existing_df["address"].map(_norm_addr)
                    existing_addrs = set([a for a in existing_df["address"].tolist() if _is_addr(a)])
                    self.logger.info(f"Resume enabled: found existing risk file with {len(existing_addrs)} addresses.")

                    # If we fetched new caches, we may want to re-score previously-missing payload rows.
                    if "has_payload" in existing_df.columns:
                        hp = pd.to_numeric(existing_df["has_payload"], errors="coerce").fillna(0).astype(int)
                        existing_missing_payload = set(existing_df.loc[hp == 0, "address"].tolist())
                    if "has_activity_payload" in existing_df.columns:
                        ha = pd.to_numeric(existing_df["has_activity_payload"], errors="coerce").fillna(0).astype(int)
                        existing_missing_activity = set(existing_df.loc[ha == 0, "address"].tolist())
            except Exception as e:
                self.logger.warning(f"Failed to read existing risk CSV for resume; will re-score all: {type(e).__name__}: {e}")
                existing_df = None
                existing_addrs = set()
                existing_missing_payload = set()
                existing_missing_activity = set()

        # Determine addresses to (re)score:
        # - any address not present in existing risk file
        # - any address previously scored with missing payload/activity, BUT for which we now have cached files
        #   (after the fetch step above).
        remaining: list[str]
        if existing_df is None:
            remaining = uniq_addrs
        else:
            sources_dir = DATA_DIR / "etherscan_sources"
            activity_dir = DATA_DIR / "etherscan_activity"
            need: list[str] = []
            for a in uniq_addrs:
                if a not in existing_addrs:
                    need.append(a)
                    continue
                if fetch_sources and (a in existing_missing_payload) and (sources_dir / f"{a}.json").exists():
                    need.append(a)
                    continue
                if fetch_activity and (a in existing_missing_activity) and (activity_dir / f"{a}.json").exists():
                    need.append(a)
                    continue
            remaining = need
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
            # Put new rows FIRST so they win on dedupe.
            parts = []
            if new_df is not None and not new_df.empty:
                parts.append(new_df)
            parts.append(existing_df)
            risk_df = pd.concat(parts, ignore_index=True)

        # ---- Phase 2: Score local-source-only rows (no on-chain address) ----
        # These rows have a local .sol file but no Etherscan address.
        # We use ContractRiskUtil.extract_features_from_local_file for each row.
        local_risk_rows: list = []
        if n_local_only > 0:
            self.logger.info(f"Scoring {n_local_only} local-source-only rows via extract_features_from_local_file…")
            risk_util = ContractRiskUtil()
            local_df = df[local_only_mask].copy()
            for idx, (_, row) in enumerate(local_df.iterrows(), start=1):
                sol_path = str(row.get("local_source_path", "") or "").strip()
                contract_id = str(row.get("contract_id", "") or "").strip()
                addr = _norm_addr(row.get("address"))
                if not sol_path:
                    continue
                feats = risk_util.extract_features_from_local_file(
                    contract_id=contract_id,
                    sol_path=sol_path,
                    address=addr if _is_addr(addr) else "",
                )
                # Use contract_id as the identifier in the address column for join
                feats["address"] = contract_id
                raw, comps = risk_util.score_contract(feats)
                has_src = float(int(feats.get("has_verified_source", 0)))
                has_bc = float(int(feats.get("has_bytecode", 0)))
                risk_uncertainty = (1.0 - has_src) * (0.7 - 0.4 * has_bc) + (1.0 - has_bc) * 0.3
                local_risk_rows.append({
                    **feats, **comps,
                    "risk_raw": float(raw),
                    "risk_uncertainty": float(max(0.0, risk_uncertainty)),
                })
                if idx == 1 or idx % 2000 == 0 or idx == n_local_only:
                    self.logger.info(f"[local scoring] {idx}/{n_local_only}")

        local_risk_df = pd.DataFrame(local_risk_rows) if local_risk_rows else pd.DataFrame()
        if not local_risk_df.empty:
            self.logger.info(f"Scored {len(local_risk_df)} local-source-only contracts")

        # ---- Combine address-based and local-source risk DataFrames ----
        all_parts = []
        if risk_df is not None and not risk_df.empty:
            all_parts.append(risk_df)
        if not local_risk_df.empty:
            all_parts.append(local_risk_df)

        if not all_parts:
            raise ValueError("Risk scoring produced no rows.")

        combined_risk_df = pd.concat(all_parts, ignore_index=True) if len(all_parts) > 1 else all_parts[0]
        combined_risk_df = combined_risk_df.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
        combined_risk_df = combined_risk_df.sort_values(["address"]).reset_index(drop=True)
        combined_risk_df = _recompute_z_and_rank(combined_risk_df)

        out_risk_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_risk_df.to_csv(out_risk_csv, index=False)
        self.logger.info(f"Wrote risk CSV: {out_risk_csv} (rows={len(combined_risk_df)})")

        # ---- Merge risk scores back onto ALL original rows ----
        # For address-based rows: join on _address_norm
        # For local-source-only rows: join on contract_id
        join_df = combined_risk_df.copy()

        # Address-based merge
        addr_join = join_df[join_df["address"].map(_is_addr)].copy()
        addr_join["_address_norm"] = addr_join["address"].map(_norm_addr)
        scored_addr = df[~local_only_mask].merge(addr_join, how="left", on="_address_norm", suffixes=("", "_risk"))
        if "address_risk" in scored_addr.columns:
            scored_addr.drop(columns=["address_risk"], inplace=True)

        # Local-source merge (join on contract_id)
        local_join = join_df[~join_df["address"].map(_is_addr)].copy()
        local_join = local_join.rename(columns={"address": "_local_join_key"})
        scored_local = df[local_only_mask].copy()
        scored_local["_local_join_key"] = scored_local["contract_id"].fillna("").astype(str)
        scored_local = scored_local.merge(local_join, how="left", on="_local_join_key", suffixes=("", "_risk"))
        scored_local.drop(columns=["_local_join_key"], inplace=True, errors="ignore")

        scored = pd.concat([scored_addr, scored_local], ignore_index=True)
        scored.drop(columns=["_address_norm"], inplace=True, errors="ignore")

        # Helpful flags
        scored["has_risk_row"] = scored["risk_raw"].notna().astype(int)

        # Always write a scored CSV (no optional Excel deps).
        out_scored_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_scored_csv, index=False)
        self.logger.info(f"Wrote scored CSV: {out_scored_csv} (rows={len(scored)})")

        # Also attempt to write a scored Excel with separate sheets for easier manual review (requires openpyxl).
        try:
            out_scored_xlsx.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(out_scored_xlsx, engine="openpyxl") as w:
                scored.to_excel(w, sheet_name="contracts_scored", index=False)
                combined_risk_df.to_excel(w, sheet_name="risk_by_address", index=False)
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
        except ImportError:
            self.logger.warning("openpyxl not installed; skipping Excel output. (CSV was written successfully.)")
            return out_scored_csv


if __name__ == "__main__":
    # ---- Editable run parameters (no CLI args) ----
    paths = Paths()
    resume = True
    fetch_sources = False
    fetch_activity = False
    fetch_missing_only = True
    skip_existing_fetch = True
    # End-to-end run: fetch ALL missing payloads, but do it in internal chunks to reduce rate-limit risk.
    fetch_chunk_size = 500  # set to 0 to do it in a single giant call (not recommended)
    sleep_between_chunks_secs = 2.0
    activity_max_pages = 2

    UnifiedContractEvalRiskScoreEvaluation().run(
        in_xlsx=paths.in_xlsx,
        out_registry_csv=paths.out_registry_csv,
        out_risk_csv=paths.out_risk_csv,
        out_scored_xlsx=paths.out_scored_xlsx,
        resume=bool(resume),
        fetch_sources=bool(fetch_sources),
        fetch_activity=bool(fetch_activity),
        fetch_missing_only=bool(fetch_missing_only),
        skip_existing_fetch=bool(skip_existing_fetch),
        fetch_chunk_size=int(fetch_chunk_size),
        sleep_between_chunks_secs=float(sleep_between_chunks_secs),
        activity_max_pages=int(activity_max_pages),
    )

