"""
Risk score pipeline for paper figures.

Builds:
- AVS-week risk scores from Dune panels (governance + flows) with expanded contract attribution
- Operator-week risk scores by lifting AVS risk via operator↔AVS exposure graph
- Evaluation vs Tokensight/u--1 AVS risk summary (totalRiskScore)

Writes:
- data/derived/*.csv (scores)
- figures/*.png (key plots)
"""

from __future__ import annotations

import logging
import os
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# src/pipelines/risk_score_pipeline.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure headless plotting + writable caches (some environments have read-only $HOME caches)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

# Create cache dirs early to avoid fontconfig/matplotlib crashes
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


class RiskScorePipeline:
    def __init__(self) -> None:
        logger = logging.getLogger("risk_score_pipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        self.repo_root = REPO_ROOT
        self.data_dir = self.repo_root / "data"
        self.derived_dir = self.data_dir / "derived"
        self.figures_dir = self.repo_root / "figures"

        # Inputs
        self.path_dune_gov_panel_expanded = self.data_dir / "dune_avs_governance_events_panel_expanded_all.json"
        self.path_dune_gov_panel_baseline = self.data_dir / "dune_avs_governance_events_panel_all.json"
        self.path_dune_activity_panel = self.data_dir / "dune_avs_activity_panel_all.json"
        self.path_dune_tokenflow_panel = self.data_dir / "dune_avs_tokenflow_panel_expanded_all.json"
        self.path_dune_ethflow_panel = self.data_dir / "dune_avs_ethflow_panel_expanded_all.json"
        self.path_operator_avs_edges = self.data_dir / "eigenexplorer_operator_avs_edges.json"
        self.path_u1_avs_risks_summary = self.data_dir / "u1_avs_risks_summary.json"
        self.path_u1_latest_avs_balances = self.data_dir / "u1_latest_avs_balances.json"
        self.path_avs_code_risk = self.derived_dir / "avs_code_risk.csv"

    # ---------------- helpers ----------------
    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _read_json", exc_info=True)
            raise

    def _dune_rows(self, path: Path) -> List[Dict[str, Any]]:
        try:
            obj = self._read_json(path)
            res = obj.get("result", {}) if isinstance(obj, dict) else {}
            rows = res.get("rows", []) if isinstance(res, dict) else []
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected Dune rows in {path}")
            return rows
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _dune_rows", exc_info=True)
            raise

    def _normalize_name(self, s: str) -> str:
        try:
            s = s.lower().strip()
            s = re.sub(r"\bavs\b", "", s)
            s = re.sub(r"[^a-z0-9]+", "", s)
            return s
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _normalize_name", exc_info=True)
            raise

    def _safe_log1p(self, series: pd.Series) -> pd.Series:
        try:
            series = pd.to_numeric(series, errors="coerce").fillna(0.0)
            series = series.clip(lower=0)
            return np.log1p(series)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _safe_log1p", exc_info=True)
            raise

    def _zscore_by_week(self, df: pd.DataFrame, col: str, week_col: str = "week") -> pd.Series:
        try:
            x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            mu = x.groupby(df[week_col]).transform("mean")
            sigma = x.groupby(df[week_col]).transform("std").replace(0, np.nan)
            z = (x - mu) / sigma
            return z.fillna(0.0)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _zscore_by_week", exc_info=True)
            raise

    def _zscore_global(self, series: pd.Series) -> pd.Series:
        try:
            x = pd.to_numeric(series, errors="coerce").fillna(0.0)
            mu = float(x.mean())
            sigma = float(x.std())
            if sigma == 0.0 or np.isnan(sigma):
                return pd.Series(np.zeros(len(x)), index=x.index)
            return (x - mu) / sigma
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _zscore_global", exc_info=True)
            raise

    # ---------------- loaders ----------------
    def load_tokensight_labels(self) -> pd.DataFrame:
        try:
            raw = self._read_json(self.path_u1_avs_risks_summary)
            rows = []
            for item in raw:
                avs_name = item.get("avsName")
                total = item.get("totalRiskScore")
                if not isinstance(avs_name, str):
                    continue
                row = {"avsName": avs_name, "totalRiskScore": float(total) if total is not None else np.nan}
                for r in item.get("risks", []) or []:
                    name = r.get("riskName")
                    scores = r.get("scores", [])
                    if isinstance(name, str) and isinstance(scores, list) and scores:
                        row[f"risk_{name}"] = str(scores[0])
                rows.append(row)
            df = pd.DataFrame(rows)
            df["avsName_norm"] = df["avsName"].map(self._normalize_name)
            return df
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_tokensight_labels", exc_info=True)
            raise

    def load_u1_avs_name_map(self) -> pd.DataFrame:
        try:
            raw = self._read_json(self.path_u1_latest_avs_balances)
            rows = []
            if isinstance(raw, dict):
                for addr, payload in raw.items():
                    if not (isinstance(addr, str) and addr.startswith("0x") and len(addr) == 42):
                        continue
                    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                    name = meta.get("name") if isinstance(meta, dict) else None
                    tvl = payload.get("totalUsdValue")
                    if isinstance(name, str):
                        rows.append(
                            {
                                "avs_address": addr.lower(),
                                "avsName_u1": name,
                                "avsName_u1_norm": self._normalize_name(name),
                                "tvl_usd": float(tvl) if tvl is not None else np.nan,
                            }
                        )
            return pd.DataFrame(rows)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_u1_avs_name_map", exc_info=True)
            raise

    def load_dune_panels(self) -> Dict[str, pd.DataFrame]:
        try:
            def to_df(path: Path) -> pd.DataFrame:
                df = pd.DataFrame(self._dune_rows(path))
                if "avs_address" in df.columns:
                    df["avs_address"] = df["avs_address"].astype(str).str.lower()
                df["week"] = pd.to_datetime(df["week"], utc=True, errors="coerce")
                return df

            return {
                "gov_expanded": to_df(self.path_dune_gov_panel_expanded),
                "gov_baseline": to_df(self.path_dune_gov_panel_baseline),
                "activity": to_df(self.path_dune_activity_panel),
                "tokenflow": to_df(self.path_dune_tokenflow_panel),
                "ethflow": to_df(self.path_dune_ethflow_panel),
            }
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_dune_panels", exc_info=True)
            raise

    def load_operator_edges(self) -> pd.DataFrame:
        try:
            edges = self._read_json(self.path_operator_avs_edges)
            df = pd.DataFrame(edges)
            df["operator_address"] = df["operator_address"].astype(str).str.lower()
            df["avs_address"] = df["avs_address"].astype(str).str.lower()
            return df
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_operator_edges", exc_info=True)
            raise

    def load_code_risk(self) -> pd.DataFrame:
        try:
            if not self.path_avs_code_risk.exists():
                self.logger.warning(f"Code risk file missing: {self.path_avs_code_risk} (continuing with zeros)")
                return pd.DataFrame(columns=["avs_address", "code_risk_z"])
            df = pd.read_csv(self.path_avs_code_risk)
            if "avs_address" in df.columns:
                df["avs_address"] = df["avs_address"].astype(str).str.lower()
            return df
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during load_code_risk", exc_info=True)
            raise

    # ---------------- scoring ----------------
    def build_avs_week_risk(self, panels: Dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
        try:
            gov = panels["gov_baseline"] if mode == "baseline" else panels["gov_expanded"]
            activity = panels["activity"][["week", "avs_address", "n_txs_to_avs", "n_distinct_senders"]].copy()

            df = gov.merge(activity, on=["week", "avs_address"], how="left")
            df[["n_txs_to_avs", "n_distinct_senders"]] = df[["n_txs_to_avs", "n_distinct_senders"]].fillna(0)

            if mode == "expanded+flows":
                tf = panels["tokenflow"].copy()
                ef = panels["ethflow"].copy()
                df = df.merge(tf, on=["week", "avs_address"], how="left").merge(
                    ef, on=["week", "avs_address"], how="left"
                )
                for c in df.columns:
                    if c.startswith("n_") or c.startswith("total_"):
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            gov_weighted = (
                3 * df.get("n_upgraded_events", 0)
                + 2 * df.get("n_beacon_upgraded_events", 0)
                + 2 * df.get("n_admin_changed_events", 0)
                + 1 * df.get("n_ownership_transfers", 0)
                + 1 * df.get("n_role_granted_events", 0)
                + 1 * df.get("n_role_revoked_events", 0)
            )
            df["gov_weighted"] = pd.to_numeric(gov_weighted, errors="coerce").fillna(0)

            df["gov_per_tx"] = df["gov_weighted"] / (
                1.0 + pd.to_numeric(df["n_txs_to_avs"], errors="coerce").fillna(0.0)
            )
            df["gov_per_tx_log"] = self._safe_log1p(df["gov_per_tx"])
            df["gov_per_tx_z"] = self._zscore_by_week(df, "gov_per_tx_log")

            if mode == "expanded+flows":
                df["erc20_out_log"] = self._safe_log1p(df.get("n_erc20_transfers_out", 0))
                df["eth_out_log"] = self._safe_log1p(df.get("n_eth_transfers_out", 0))
                df["eth_out_value_log"] = self._safe_log1p(df.get("total_eth_value_wei_out", 0))
                df["erc20_out_value_log"] = self._safe_log1p(df.get("total_erc20_value_raw_out", 0))

                df["erc20_out_z"] = self._zscore_by_week(df, "erc20_out_log")
                df["eth_out_z"] = self._zscore_by_week(df, "eth_out_log")
                df["eth_out_value_z"] = self._zscore_by_week(df, "eth_out_value_log")
                df["erc20_out_value_z"] = self._zscore_by_week(df, "erc20_out_value_log")

                df["flow_z"] = (
                    0.25 * df["erc20_out_z"]
                    + 0.25 * df["eth_out_z"]
                    + 0.25 * df["eth_out_value_z"]
                    + 0.25 * df["erc20_out_value_z"]
                )
                df["avs_risk_score"] = 0.6 * df["gov_per_tx_z"] + 0.4 * df["flow_z"]
            else:
                df["avs_risk_score"] = df["gov_per_tx_z"]

            return df[
                ["week", "avs_address", "avs_risk_score", "gov_weighted", "n_txs_to_avs", "n_distinct_senders"]
            ].copy()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during build_avs_week_risk", exc_info=True)
            raise

    def apply_code_risk(self, avs_week: pd.DataFrame, code_risk: pd.DataFrame, alpha: float = 0.8) -> pd.DataFrame:
        """
        Merge time-invariant code risk into the weekly behavioral risk.

        final_score = alpha * behavioral + (1-alpha) * code_risk_z
        """
        try:
            df = avs_week.copy()
            if "avs_address" in df.columns:
                df["avs_address"] = df["avs_address"].astype(str).str.lower()
            cr = code_risk[["avs_address", "code_risk_z"]].copy() if not code_risk.empty else pd.DataFrame(
                columns=["avs_address", "code_risk_z"]
            )
            df = df.merge(cr, on="avs_address", how="left")
            df["code_risk_z"] = pd.to_numeric(df.get("code_risk_z", 0), errors="coerce").fillna(0.0)
            df["avs_risk_behavior"] = pd.to_numeric(df["avs_risk_score"], errors="coerce").fillna(0.0)
            a = float(alpha)
            df["avs_risk_score"] = a * df["avs_risk_behavior"] + (1.0 - a) * df["code_risk_z"]
            return df
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during apply_code_risk", exc_info=True)
            raise

    def score_feature_csv(
        self,
        in_csv: Path,
        *,
        out_csv: Optional[Path] = None,
        mode: str = "expanded+flows",
        alpha: float = 0.8,
        label_col: str = "llm_totalRiskScore",
    ) -> pd.DataFrame:
        """
        Score an LLM-generated feature CSV (one row per AVS-time snapshot) and produce `modelRisk` per row.

        Expected minimum columns (extra columns are allowed and preserved):
        - week: ISO date/time (parsed to datetime; used for per-week z-scoring)
        - avs_id: any string identifier (name/address); used only for readability
        - n_txs_to_avs: non-negative int
        - Governance event counts (non-negative ints):
          - n_upgraded_events
          - n_beacon_upgraded_events
          - n_admin_changed_events
          - n_ownership_transfers
          - n_role_granted_events
          - n_role_revoked_events
        - If mode == "expanded+flows", also provide flow columns (non-negative ints):
          - n_erc20_transfers_out
          - n_eth_transfers_out
          - total_eth_value_wei_out
          - total_erc20_value_raw_out
        - Optional code risk:
          - code_risk_z (preferred; typically ~[-3, +3])
          - OR code_risk_raw_max (will be z-scored globally inside this CSV to create code_risk_z)

        If `label_col` exists and is numeric, we compute Spearman rho between `modelRisk` and that label
        and log it (for quick iteration).
        """
        try:
            in_csv = Path(in_csv)
            if not in_csv.exists():
                raise FileNotFoundError(f"Input CSV not found: {in_csv}")

            df = pd.read_csv(in_csv)
            if df.empty:
                raise ValueError(f"Input CSV is empty: {in_csv}")

            # --- required columns ---
            required = {
                "week",
                "n_txs_to_avs",
                "n_upgraded_events",
                "n_beacon_upgraded_events",
                "n_admin_changed_events",
                "n_ownership_transfers",
                "n_role_granted_events",
                "n_role_revoked_events",
            }
            missing = sorted([c for c in required if c not in df.columns])
            if missing:
                raise ValueError(f"Missing required columns in {in_csv.name}: {missing}")

            df = df.copy()
            # Be permissive: accept ISO (YYYY-MM-DD) and common DD/MM/YYYY exports.
            # `dayfirst=True` still parses ISO correctly while fixing e.g. 13/01/2025.
            df["week"] = pd.to_datetime(df["week"], utc=True, errors="coerce", dayfirst=True)
            df = df.dropna(subset=["week"]).copy()
            if df.empty:
                raise ValueError("All rows have invalid/missing `week` after parsing.")

            # Coerce core numeric columns
            for c in list(required - {"week"}):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)

            # Governance weighted signal (same weights as main pipeline)
            gov_weighted = (
                3 * df["n_upgraded_events"]
                + 2 * df["n_beacon_upgraded_events"]
                + 2 * df["n_admin_changed_events"]
                + 1 * df["n_ownership_transfers"]
                + 1 * df["n_role_granted_events"]
                + 1 * df["n_role_revoked_events"]
            )
            df["gov_weighted"] = pd.to_numeric(gov_weighted, errors="coerce").fillna(0.0)

            df["gov_per_tx"] = df["gov_weighted"] / (1.0 + df["n_txs_to_avs"])
            df["gov_per_tx_log"] = self._safe_log1p(df["gov_per_tx"])
            df["gov_per_tx_z"] = self._zscore_by_week(df, "gov_per_tx_log")

            # Behavioral risk (same structure as build_avs_week_risk)
            if mode == "expanded+flows":
                flow_required = {
                    "n_erc20_transfers_out",
                    "n_eth_transfers_out",
                    "total_eth_value_wei_out",
                    "total_erc20_value_raw_out",
                }
                flow_missing = sorted([c for c in flow_required if c not in df.columns])
                if flow_missing:
                    raise ValueError(
                        f"mode=expanded+flows requires flow columns; missing: {flow_missing}. "
                        f"Either add them or run with mode=baseline/expanded."
                    )

                for c in flow_required:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)

                df["erc20_out_log"] = self._safe_log1p(df["n_erc20_transfers_out"])
                df["eth_out_log"] = self._safe_log1p(df["n_eth_transfers_out"])
                df["eth_out_value_log"] = self._safe_log1p(df["total_eth_value_wei_out"])
                df["erc20_out_value_log"] = self._safe_log1p(df["total_erc20_value_raw_out"])

                df["erc20_out_z"] = self._zscore_by_week(df, "erc20_out_log")
                df["eth_out_z"] = self._zscore_by_week(df, "eth_out_log")
                df["eth_out_value_z"] = self._zscore_by_week(df, "eth_out_value_log")
                df["erc20_out_value_z"] = self._zscore_by_week(df, "erc20_out_value_log")

                df["flow_z"] = (
                    0.25 * df["erc20_out_z"]
                    + 0.25 * df["eth_out_z"]
                    + 0.25 * df["eth_out_value_z"]
                    + 0.25 * df["erc20_out_value_z"]
                )
                df["avs_risk_behavior"] = 0.6 * df["gov_per_tx_z"] + 0.4 * df["flow_z"]
            else:
                if mode not in ("baseline", "expanded"):
                    raise ValueError(f"Unsupported mode: {mode}. Use baseline|expanded|expanded+flows.")
                df["avs_risk_behavior"] = df["gov_per_tx_z"]

            # Code risk (optional)
            if "code_risk_z" not in df.columns:
                if "code_risk_raw_max" in df.columns:
                    df["code_risk_z"] = self._zscore_global(df["code_risk_raw_max"])
                else:
                    df["code_risk_z"] = 0.0
            else:
                df["code_risk_z"] = pd.to_numeric(df["code_risk_z"], errors="coerce").fillna(0.0)

            a = float(alpha)
            df["modelRisk"] = a * df["avs_risk_behavior"] + (1.0 - a) * df["code_risk_z"]

            # Optional: quick eval vs label column (LLM "ground truth")
            if label_col in df.columns:
                y = pd.to_numeric(df[label_col], errors="coerce")
                m = df[["modelRisk"]].copy()
                m[label_col] = y
                m = m.dropna(subset=["modelRisk", label_col])
                if len(m) >= 3:
                    rho = self.spearman(m["modelRisk"], m[label_col])
                    self.logger.info(f"Evaluation vs {label_col} (Spearman ρ): {rho:.3f} (n={len(m)})")
                else:
                    self.logger.info(f"Evaluation vs {label_col}: not enough labeled rows (n={len(m)})")

            if out_csv is not None:
                out_csv = Path(out_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv, index=False)
                self.logger.info(f"Wrote scored CSV: {out_csv}")

            return df
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during score_feature_csv", exc_info=True)
            raise

    def lift_to_operator_week(self, avs_week: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
        try:
            merged = edges.merge(avs_week, on="avs_address", how="left")
            merged["avs_risk_score"] = merged["avs_risk_score"].fillna(0.0)
            g = merged.groupby(["week", "operator_address"])
            out = g.agg(
                operator_risk_mean=("avs_risk_score", "mean"),
                operator_risk_max=("avs_risk_score", "max"),
                n_avs=("avs_address", "nunique"),
            ).reset_index()
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during lift_to_operator_week", exc_info=True)
            raise

    def evaluate_against_tokensight(
        self, avs_week: pd.DataFrame, tokensight: pd.DataFrame, u1_map: pd.DataFrame, score_agg: str = "mean_last4w"
    ) -> pd.DataFrame:
        try:
            m = tokensight.merge(u1_map, left_on="avsName_norm", right_on="avsName_u1_norm", how="left")

            missing = m["avs_address"].isna()
            if missing.any():
                u1_names = u1_map[["avs_address", "avsName_u1_norm"]].dropna().to_dict("records")

                def best_match(norm_name: str) -> Optional[str]:
                    if not isinstance(norm_name, str) or not norm_name:
                        return None
                    candidates = []
                    for r in u1_names:
                        u = r["avsName_u1_norm"]
                        if norm_name in u or u in norm_name:
                            candidates.append((len(u), r["avs_address"]))
                    if not candidates:
                        return None
                    candidates.sort(reverse=True)
                    return candidates[0][1]

                m.loc[missing, "avs_address"] = m.loc[missing, "avsName_norm"].map(best_match)

            labeled = m.dropna(subset=["avs_address", "totalRiskScore"]).copy()

            avs_week = avs_week.dropna(subset=["week"]).copy()
            last_week = avs_week["week"].max()
            if score_agg == "last_week":
                s = avs_week[avs_week["week"] == last_week].groupby("avs_address")["avs_risk_score"].mean()
            else:
                cutoff = last_week - pd.Timedelta(days=28)
                s = avs_week[avs_week["week"] >= cutoff].groupby("avs_address")["avs_risk_score"].mean()
            s = s.rename("modelRisk").reset_index()

            joined = labeled.merge(s, on="avs_address", how="left").dropna(subset=["modelRisk"])
            return joined
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during evaluate_against_tokensight", exc_info=True)
            raise

    def spearman(self, x: Sequence[float], y: Sequence[float]) -> float:
        try:
            xr = pd.Series(x).rank(method="average")
            yr = pd.Series(y).rank(method="average")
            x0 = (xr - xr.mean()).to_numpy()
            y0 = (yr - yr.mean()).to_numpy()
            denom = float(np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum()))
            if denom == 0.0:
                return 0.0
            return float((x0 * y0).sum() / denom)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during spearman", exc_info=True)
            raise

    # ---------------- plotting ----------------
    def ensure_dirs(self) -> None:
        try:
            self.derived_dir.mkdir(parents=True, exist_ok=True)
            self.figures_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during ensure_dirs", exc_info=True)
            raise

    def plot_scatter(self, joined: pd.DataFrame, title: str, out: Path) -> None:
        try:
            plt.figure(figsize=(6, 4))
            sns.regplot(data=joined, x="modelRisk", y="totalRiskScore", scatter_kws={"s": 30, "alpha": 0.8})
            rho = self.spearman(joined["modelRisk"], joined["totalRiskScore"])
            plt.title(f"{title}\nSpearman ρ={rho:.2f} (n={len(joined)})")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during plot_scatter", exc_info=True)
            raise

    def plot_ablation(self, baseline: float, expanded: float, expanded_flows: float, out: Path) -> None:
        try:
            df = pd.DataFrame(
                [
                    {"model": "baseline (service mgr only)", "spearman_rho": baseline},
                    {"model": "expanded contracts", "spearman_rho": expanded},
                    {"model": "expanded + flows", "spearman_rho": expanded_flows},
                ]
            )
            plt.figure(figsize=(7, 3))
            sns.barplot(data=df, x="model", y="spearman_rho")
            plt.ylim(-1, 1)
            plt.xticks(rotation=15, ha="right")
            plt.title("AVS risk score vs Tokensight (Spearman ρ)")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during plot_ablation", exc_info=True)
            raise

    def plot_operator_risk_distribution(self, op_week: pd.DataFrame, out: Path) -> None:
        try:
            latest = op_week["week"].max()
            df = op_week[op_week["week"] == latest].copy()
            plt.figure(figsize=(6, 4))
            sns.histplot(df["operator_risk_max"], bins=40, kde=True)
            plt.title(f"Operator max-risk exposure (latest week)\n(n={len(df)})")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during plot_operator_risk_distribution", exc_info=True)
            raise

    def plot_top_avs_timeseries(self, avs_week: pd.DataFrame, u1_map: pd.DataFrame, out: Path) -> None:
        try:
            top = (
                avs_week.groupby("avs_address")["avs_risk_score"]
                .mean()
                .sort_values(ascending=False)
                .head(8)
                .index.tolist()
            )
            df = avs_week[avs_week["avs_address"].isin(top)].copy()
            df = df.merge(u1_map[["avs_address", "avsName_u1"]], on="avs_address", how="left")
            df["label"] = df["avsName_u1"].fillna(df["avs_address"].str[:10])
            plt.figure(figsize=(9, 4))
            sns.lineplot(data=df, x="week", y="avs_risk_score", hue="label")
            plt.title("Top AVSs by average model risk (time series)")
            plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            plt.tight_layout()
            # legend is outside the axes; ensure it isn't cropped in the saved PNG
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during plot_top_avs_timeseries", exc_info=True)
            raise

    # ---------------- run ----------------
    def run(self) -> None:
        try:
            self.ensure_dirs()

            tokensight = self.load_tokensight_labels()
            u1_map = self.load_u1_avs_name_map()
            panels = self.load_dune_panels()
            edges = self.load_operator_edges()
            code_risk = self.load_code_risk()

            avs_baseline = self.build_avs_week_risk(panels, mode="baseline")
            avs_expanded = self.build_avs_week_risk(panels, mode="expanded")
            avs_expanded_flows = self.build_avs_week_risk(panels, mode="expanded+flows")

            # Integrate code risk into the main model (keep baseline/expanded as pure behavioral for ablations)
            avs_expanded_flows = self.apply_code_risk(avs_expanded_flows, code_risk, alpha=0.8)

            avs_expanded_flows.to_csv(self.derived_dir / "avs_week_risk_expanded_flows.csv", index=False)
            op_week = self.lift_to_operator_week(avs_expanded_flows, edges)
            op_week.to_csv(self.derived_dir / "operator_week_risk.csv", index=False)

            j_base = self.evaluate_against_tokensight(avs_baseline, tokensight, u1_map)
            j_exp = self.evaluate_against_tokensight(avs_expanded, tokensight, u1_map)
            j_flow = self.evaluate_against_tokensight(avs_expanded_flows, tokensight, u1_map)

            rho_base = self.spearman(j_base["modelRisk"], j_base["totalRiskScore"])
            rho_exp = self.spearman(j_exp["modelRisk"], j_exp["totalRiskScore"])
            rho_flow = self.spearman(j_flow["modelRisk"], j_flow["totalRiskScore"])

            self.plot_ablation(rho_base, rho_exp, rho_flow, self.figures_dir / "avs_tokensight_ablation.png")
            self.plot_scatter(
                j_flow,
                "Model AVS risk (expanded+flows) vs Tokensight totalRiskScore",
                self.figures_dir / "avs_tokensight_scatter.png",
            )
            self.plot_operator_risk_distribution(op_week, self.figures_dir / "operator_risk_distribution.png")
            self.plot_top_avs_timeseries(avs_expanded_flows, u1_map, self.figures_dir / "top_avs_risk_timeseries.png")

            self.logger.info("Evaluation (Spearman ρ vs Tokensight totalRiskScore; last-4-weeks avg)")
            self.logger.info(f"baseline:       {rho_base:.3f} (n={len(j_base)})")
            self.logger.info(f"expanded:       {rho_exp:.3f} (n={len(j_exp)})")
            self.logger.info(f"expanded+flows: {rho_flow:.3f} (n={len(j_flow)})")
            self.logger.info(f"Wrote figures to: {self.figures_dir}")
            self.logger.info(f"Wrote derived data to: {self.derived_dir}")
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk score pipeline (full run) or score an LLM feature CSV.")
    parser.add_argument("--feature-csv", type=str, default="", help="Path to LLM-generated feature CSV to score.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional output CSV path. If omitted, nothing is written (but evaluation logs still print).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="expanded+flows",
        help="baseline|expanded|expanded+flows (must match the columns present in your feature CSV).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Blend factor for final score: modelRisk = alpha*behavior + (1-alpha)*code_risk_z",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="llm_totalRiskScore",
        help="Optional label column name to evaluate against (Spearman).",
    )
    args = parser.parse_args()

    pipe = RiskScorePipeline()
    if str(args.feature_csv).strip():
        pipe.score_feature_csv(
            Path(args.feature_csv),
            out_csv=Path(args.out_csv) if str(args.out_csv).strip() else None,
            mode=str(args.mode).strip(),
            alpha=float(args.alpha),
            label_col=str(args.label_col).strip(),
        )
    else:
        pipe.run()

