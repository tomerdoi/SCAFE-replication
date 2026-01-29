"""
Score contracts from the "Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts"
replication package using this repo's contract technical risk score.

Input (default):
- data/raw/Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts-main/csvs/sample_of_interest.csv
  Columns: label, tag, contract
  - contract: Ethereum address
  - tag: vulnerability tags (or "no")

Outputs (under data/derived/ by default):
- empirical_analysis_registry.csv          (addresses scored; subset when limit>0)
- empirical_analysis_contract_risk.csv     (raw ContractRiskUtil output)
- empirical_analysis_scored.csv            (input joined with risk + parsed label columns)
- empirical_analysis_metrics.csv           (AUC-style summaries for risk vs labels)

Quick run:
  python3 -m src.evaluation.empirical_analysis_eval
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Allow running as a file in IDE/debugger (so `import src...` works)
# src/evaluation/empirical_analysis_eval.py -> repo root is two levels up
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_DIR, DERIVED_DIR  # noqa: E402
from src.risk.contract_risk import ContractRiskUtil  # noqa: E402


def _is_addr(s: object) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _auc_u(scores: pd.Series, labels: pd.Series) -> float:
    """
    AUC via Mann–Whitney U (no sklearn dependency).
    Returns NaN if labels are degenerate.
    """
    s = pd.to_numeric(scores, errors="coerce")
    y = pd.to_numeric(labels, errors="coerce")
    m = s.notna() & y.notna()
    s = s[m]
    y = y[m].astype(int)
    pos = s[y == 1].to_numpy()
    neg = s[y == 0].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = pd.Series(list(pos) + list(neg))
    ranks = all_s.rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _parse_tag_flags(tag: Any) -> Dict[str, int]:
    """
    Parse the Empirical-Analysis `tag` field into coarse vulnerability indicators.

    Examples:
      "no"
      "26: unchecked_low_calls; 26: reentrancy;"
      "none: reentrancy;"
    """
    t = str(tag or "").strip().lower()
    t = t.strip(";").strip()
    if not t or t == "no" or t == "none":
        return {
            "label_any_vuln": 0,
            "label_access_control": 0,
            "label_reentrancy": 0,
            "label_time_manipulation": 0,
            "label_unchecked_low_calls": 0,
            "label_arithmetic": 0,
            "label_front_running": 0,
        }

    # Remove line numbers like "26:" to make substring checks less brittle.
    t_norm = re.sub(r"\b\d+\s*:\s*", "", t)

    def has(substr: str) -> int:
        return int(substr in t_norm)

    # Note: these strings match the repo's common tags.
    flags = {
        "label_any_vuln": 1,
        "label_access_control": has("access_control"),
        "label_reentrancy": has("reentrancy"),
        "label_time_manipulation": has("time_manipulation"),
        "label_unchecked_low_calls": has("unchecked_low_calls") or has("unchecked call"),
        "label_arithmetic": has("arithmetic") or has("integer overflow") or has("overflow") or has("underflow"),
        "label_front_running": has("front_running") or has("front running"),
    }
    return flags


class EmpiricalAnalysisEvaluation:
    def __init__(self) -> None:
        logger = logging.getLogger("empirical_analysis_eval")
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
        limit: int = 100,
        batch_size: int = 0,
        build_local_sources: bool = True,
        overwrite_local_sources: bool = False,
        fetch_sources: bool = False,
        fetch_activity: bool = False,
        skip_existing: bool = True,
        activity_max_pages: int = 2,
        out_scored_csv: Optional[Path] = None,
        out_risk_csv: Optional[Path] = None,
        out_metrics_csv: Optional[Path] = None,
    ) -> Path:
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        self.derived_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv)
        if "contract" not in df.columns:
            raise ValueError(f"Expected 'contract' column. Columns: {list(df.columns)!r}")
        if "tag" not in df.columns:
            raise ValueError(f"Expected 'tag' column. Columns: {list(df.columns)!r}")
        if "label" not in df.columns:
            raise ValueError(f"Expected 'label' column. Columns: {list(df.columns)!r}")

        # Normalize addresses
        df["_address_norm"] = df["contract"].astype(str).str.strip().str.lower()
        addrs = [a for a in df["_address_norm"].tolist() if _is_addr(a)]
        if limit and int(limit) > 0:
            addrs = addrs[: int(limit)]
        if not addrs:
            raise ValueError("No valid 0x addresses found in input after filtering.")

        registry_csv = self.derived_dir / "empirical_analysis_registry.csv"
        risk_csv = Path(out_risk_csv) if out_risk_csv else (self.derived_dir / "empirical_analysis_contract_risk.csv")
        out_scored_csv = (
            Path(out_scored_csv) if out_scored_csv else (self.derived_dir / "empirical_analysis_scored.csv")
        )
        out_metrics_csv = (
            Path(out_metrics_csv) if out_metrics_csv else (self.derived_dir / "empirical_analysis_metrics.csv")
        )

        reg = pd.DataFrame([{"address": a, "sources": "empirical-analysis"} for a in addrs])
        reg.to_csv(registry_csv, index=False)
        self.logger.info(f"Wrote registry: {registry_csv} (rows={len(reg)})")

        dataset_root = input_csv.resolve().parents[1]  # .../Empirical-Analysis-...-main
        contracts_root = dataset_root / "contracts"
        bytecode_root = dataset_root / "bytecode"

        # Pre-index local source/bytecode files for fast lookup (small enough: ~2k files).
        sol_by_addr: Dict[str, Path] = {}
        if contracts_root.exists():
            for p in contracts_root.rglob("*.sol"):
                sol_by_addr[p.stem.strip().lower()] = p

        hex_by_addr: Dict[str, Path] = {}
        if bytecode_root.exists():
            for p in bytecode_root.rglob("*.hex"):
                name = p.name.strip().lower()
                if name.endswith(".sol.hex"):
                    key = name[: -len(".sol.hex")]
                elif name.endswith(".hex"):
                    key = name[: -len(".hex")]
                else:
                    continue
                if key.startswith("0x") and len(key) == 42:
                    hex_by_addr[key] = p

        def build_local_cache_for_addresses(addresses: list[str]) -> None:
            """
            Build `data/etherscan_sources/<addr>.json` from the local `.sol` + `.hex` files
            shipped with the replication package, so scoring can run offline (no Etherscan calls).
            """
            from src.risk import contract_risk as cr

            cr.ETHERSCAN_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

            rows = df[df["_address_norm"].isin(addresses)][["label", "_address_norm"]].drop_duplicates()
            for _, r in rows.iterrows():
                addr = str(r["_address_norm"]).strip().lower()
                label = str(r["label"] or "").strip().lower()
                out_path = cr.ETHERSCAN_SOURCES_DIR / f"{addr}.json"
                if out_path.exists() and not overwrite_local_sources:
                    continue

                # Best-effort map label -> folder conventions in this replication package.
                candidates_sol = []
                if label == "smartbugs_results":
                    candidates_sol.append(contracts_root / "sb_results_contracts" / f"{addr}.sol")
                elif label == "smartbugs_curated":
                    candidates_sol.append(contracts_root / "smartbugs_curated" / f"{addr}.sol")
                elif label == "zeus_safe":
                    candidates_sol.append(contracts_root / "zeus_safe_contracts" / f"{addr}.sol")
                elif label == "zeus_vulnerable":
                    candidates_sol.append(contracts_root / "zeus_vulnerable_contracts" / f"{addr}.sol")

                sol_path = next((p for p in candidates_sol if p.exists()), None) or sol_by_addr.get(addr)

                candidates_hex = []
                if label == "smartbugs_results":
                    candidates_hex.append(bytecode_root / "smartbugs_results" / f"{addr}.hex")
                elif label == "smartbugs_curated":
                    candidates_hex.append(bytecode_root / "smartbugs_curated" / f"{addr}.sol.hex")
                    candidates_hex.append(bytecode_root / "smartbugs_curated" / f"{addr}.hex")
                elif label == "zeus_safe":
                    candidates_hex.append(bytecode_root / "zeus_safe" / f"{addr}.hex")
                elif label == "zeus_vulnerable":
                    candidates_hex.append(bytecode_root / "zeus_vulnerable" / f"{addr}.hex")

                hex_path = next((p for p in candidates_hex if p.exists()), None) or hex_by_addr.get(addr)

                source_text = ""
                if sol_path and sol_path.exists():
                    source_text = sol_path.read_text(encoding="utf-8", errors="ignore")

                bytecode_hex = ""
                if hex_path and hex_path.exists():
                    bytecode_hex = hex_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if bytecode_hex and not bytecode_hex.startswith("0x"):
                        bytecode_hex = "0x" + bytecode_hex

                payload = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "address": addr,
                    # Minimal Etherscan-like shape expected by ContractRiskUtil.
                    "etherscan": {
                        "status": "1",
                        "message": "OK",
                        "result": [{"CompilerVersion": "", "Proxy": "0", "Implementation": ""}],
                    },
                    "bytecode": {"eth_getCode": {"jsonrpc": "2.0", "id": 1, "result": bytecode_hex}},
                    "parsed_sources": {
                        "n_contracts": 1 if source_text.strip() else 0,
                        "contracts": [source_text] if source_text.strip() else [],
                    },
                }

                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def maybe_fetch(*, addresses_csv: Path) -> None:
            # Offline-first: build local cache files from the replication package.
            if build_local_sources:
                try:
                    addf = pd.read_csv(addresses_csv)
                    addrs_local = (
                        addf["address"].astype(str).str.strip().str.lower().tolist()
                        if "address" in addf.columns
                        else []
                    )
                    self.logger.info(f"Building local etherscan_sources cache for {len(addrs_local)} addresses…")
                    build_local_cache_for_addresses(addrs_local)
                except Exception as e:
                    self.logger.warning(f"Skipping local cache build: {type(e).__name__}: {e}")

            if fetch_sources:
                try:
                    from src.clients.etherscan_source import EtherscanAVSSourceUtil

                    self.logger.info("Fetching Etherscan sources+bytecode (best-effort)…")
                    EtherscanAVSSourceUtil().run(
                        fetch_all=True,
                        limit=0,
                        addresses_csv=Path(addresses_csv),
                        progress_logger=self.logger,
                        progress_every=25,
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
                        progress_every=25,
                    )
                except Exception as e:
                    self.logger.warning(f"Skipping activity fetch (continuing): {type(e).__name__}: {e}")

        def score_one_registry(*, addresses_csv: Path, out_csv: Path) -> None:
            ContractRiskUtil().run(registry_csv=Path(addresses_csv), out_csv=Path(out_csv), limit=0)

        bs = int(batch_size or 0)
        if bs > 0 and len(addrs) > bs:
            risk_csv.parent.mkdir(parents=True, exist_ok=True)
            if risk_csv.exists():
                risk_csv.unlink()
            self.logger.info(f"Batch mode enabled: batch_size={bs} over {len(addrs)} addresses")

            for start in range(0, len(addrs), bs):
                end = min(len(addrs), start + bs)
                batch_addrs = addrs[start:end]
                batch_registry = self.derived_dir / f"empirical_analysis_registry_batch_{start}_{end}.csv"
                pd.DataFrame([{"address": a, "sources": "empirical-analysis"} for a in batch_addrs]).to_csv(
                    batch_registry, index=False
                )
                self.logger.info(f"Batch {start}:{end} registry written: {batch_registry}")

                maybe_fetch(addresses_csv=batch_registry)

                batch_risk = self.derived_dir / f"empirical_analysis_contract_risk_batch_{start}_{end}.csv"
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

        # Parse tag flags for evaluation
        flags_df = pd.DataFrame([_parse_tag_flags(t) for t in df["tag"].tolist()])
        out = pd.concat([df.drop(columns=["_address_norm"]), flags_df], axis=1)
        out["_address_norm"] = df["_address_norm"]

        out = out.merge(risk_df, how="left", left_on="_address_norm", right_on="address", suffixes=("", "_risk"))
        out.drop(columns=[c for c in ["_address_norm"] if c in out.columns], inplace=True)

        out_scored_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_scored_csv, index=False)
        self.logger.info(f"Wrote scored dataset: {out_scored_csv} (rows={len(out)})")

        # Metrics: evaluate risk_z (preferred) else risk_raw
        score_col = "risk_z" if "risk_z" in out.columns else "risk_raw"
        metrics_rows = []
        label_cols = [
            "label_any_vuln",
            "label_access_control",
            "label_reentrancy",
            "label_time_manipulation",
            "label_unchecked_low_calls",
            "label_arithmetic",
            "label_front_running",
        ]
        for lab in label_cols:
            if lab not in out.columns:
                continue
            sub = out[out[lab].isin([0, 1]) & out[score_col].notna()].copy()
            if len(sub) == 0:
                continue
            pos_n = int((sub[lab] == 1).sum())
            neg_n = int((sub[lab] == 0).sum())
            auc = _auc_u(sub[score_col], sub[lab])
            metrics_rows.append(
                {
                    "label": lab,
                    "score_col": score_col,
                    "pos_n": pos_n,
                    "neg_n": neg_n,
                    "auc": auc,
                    "pos_median": float(pd.to_numeric(sub.loc[sub[lab] == 1, score_col], errors="coerce").median()),
                    "neg_median": float(pd.to_numeric(sub.loc[sub[lab] == 0, score_col], errors="coerce").median()),
                }
            )

        pd.DataFrame(metrics_rows).to_csv(out_metrics_csv, index=False)
        self.logger.info(f"Wrote metrics: {out_metrics_csv} (rows={len(metrics_rows)})")

        return out_scored_csv


if __name__ == "__main__":
    # ---- Editable run parameters (no CLI args) ----
    dataset_root = DATA_DIR / "raw" / "Empirical-Analysis-of-Vulnerability-Detection-Tools-for-Solidity-Smart-Contracts-main"
    input_csv = dataset_root / "csvs" / "sample_of_interest.csv"

    # For a first run, keep this small (set to 0 for the full 2,182).
    limit = 200
    batch_size = 50  # set >0 to checkpoint progress for longer runs

    # Prefer fully-offline evaluation via the dataset's local `.sol` + `.hex` artifacts.
    build_local_sources = True
    overwrite_local_sources = False

    # Optional: fetch caches via Etherscan (best-effort). Requires ETHERSCAN_API_KEY and working TLS.
    fetch_sources = False
    fetch_activity = False
    skip_existing = True
    activity_max_pages = 2

    out_risk_csv = DERIVED_DIR / "empirical_analysis_contract_risk.csv"
    out_scored_csv = DERIVED_DIR / "empirical_analysis_scored.csv"
    out_metrics_csv = DERIVED_DIR / "empirical_analysis_metrics.csv"

    EmpiricalAnalysisEvaluation().run(
        input_csv=Path(input_csv),
        limit=int(limit),
        batch_size=int(batch_size),
        build_local_sources=bool(build_local_sources),
        overwrite_local_sources=bool(overwrite_local_sources),
        fetch_sources=bool(fetch_sources),
        fetch_activity=bool(fetch_activity),
        skip_existing=bool(skip_existing),
        activity_max_pages=int(activity_max_pages),
        out_scored_csv=Path(out_scored_csv),
        out_risk_csv=Path(out_risk_csv),
        out_metrics_csv=Path(out_metrics_csv),
    )

