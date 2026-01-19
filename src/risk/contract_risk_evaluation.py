"""
Evaluation and reporting utilities for the contract risk score.

What this provides (Q1-friendly artifacts):
- Baselines (including legacy v1 heuristic) vs current score
- Component ablations (drop one component at a time)
- Stability diagnostics (rank correlation between variants)
- Optional label-based evaluation if you provide a small labeled set:
  data/labels/contract_labels.csv with columns: address,label (0/1)

Outputs (default):
- data/derived/contract_risk_report.json
- data/derived/contract_risk_ablation.csv
- figures/contract_risk_distribution.png
- figures/contract_risk_components_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# src/risk/contract_risk_evaluation.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
FIGURES_DIR = REPO_ROOT / "figures"
LABELS_DIR = DATA_DIR / "labels"


def _rank_spearman(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    x0 = (xr - xr.mean()).to_numpy()
    y0 = (yr - yr.mean()).to_numpy()
    denom = float(np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((x0 * y0).sum() / denom)


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    ROC AUC without sklearn. Returns None if labels are degenerate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0:
        return None
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    # AUC via rank statistic (Mann–Whitney U)
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Average Precision (area under precision-recall curve) without sklearn.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0:
        return None
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / n_pos

    # AP: sum over recall steps of precision
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        if r > prev_recall:
            ap += float(p) * float(r - prev_recall)
            prev_recall = float(r)
    return float(ap)


@dataclass(frozen=True, slots=True)
class EvalPaths:
    contract_risk_csv: Path = DERIVED_DIR / "contract_risk.csv"
    out_report_json: Path = DERIVED_DIR / "contract_risk_report.json"
    out_ablation_csv: Path = DERIVED_DIR / "contract_risk_ablation.csv"
    labels_csv: Path = LABELS_DIR / "contract_labels.csv"
    fig_distribution: Path = FIGURES_DIR / "contract_risk_distribution.png"
    fig_components_heatmap: Path = FIGURES_DIR / "contract_risk_components_heatmap.png"


class ContractRiskEvaluation:
    def __init__(self, paths: Optional[EvalPaths] = None) -> None:
        logger = logging.getLogger("contract_risk_evaluation")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.paths = paths or EvalPaths()

        # headless plotting + writable caches
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        p = self.paths.contract_risk_csv
        if not p.exists():
            raise FileNotFoundError(f"Missing contract risk CSV: {p}")
        df = pd.read_csv(p)
        if df.empty:
            raise ValueError("contract_risk.csv is empty")
        if "address" not in df.columns:
            raise ValueError("contract_risk.csv missing `address`")
        if "risk_raw" not in df.columns or "risk_z" not in df.columns:
            raise ValueError("contract_risk.csv missing `risk_raw` / `risk_z`")
        df["address"] = df["address"].astype(str).str.lower()
        return df

    def _load_labels(self) -> Optional[pd.DataFrame]:
        p = self.paths.labels_csv
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if df.empty:
            return None
        if "address" not in df.columns or "label" not in df.columns:
            raise ValueError(f"labels csv must include columns address,label: {p}")
        df = df.copy()
        df["address"] = df["address"].astype(str).str.lower()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df = df[df["label"].isin([0, 1])].copy()
        return df

    def _legacy_v1_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Previous v1-ish heuristic baseline (simple weights).
        Uses only columns already present in contract_risk.csv.
        """
        is_upgradeable = ((df.get("is_proxy", 0).astype(int) == 1) | (df.get("has_upgrade_patterns", 0).astype(int) == 1)).astype(int)
        admin_power = ((df.get("uses_ownable", 0).astype(int) == 1) | (df.get("uses_accesscontrol", 0).astype(int) == 1)).astype(int)
        loc = pd.to_numeric(df.get("loc", 0), errors="coerce").fillna(0.0)
        complexity = np.log1p(np.maximum(0.0, loc) / 500.0)
        s = (
            2.0 * is_upgradeable
            + 1.0 * admin_power
            + 1.0 * pd.to_numeric(df.get("uses_call", 0), errors="coerce").fillna(0.0)
            + 2.0 * pd.to_numeric(df.get("uses_delegatecall", 0), errors="coerce").fillna(0.0)
            + 5.0 * pd.to_numeric(df.get("uses_selfdestruct", 0), errors="coerce").fillna(0.0)
            + 0.5 * complexity
            - 0.5 * pd.to_numeric(df.get("uses_pausable", 0), errors="coerce").fillna(0.0)
            - 0.5 * pd.to_numeric(df.get("uses_timelock", 0), errors="coerce").fillna(0.0)
        )
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    def _compute_ablation_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Score variants derived from the current component columns:
        - full: risk_raw
        - drop_<component>: remove one component’s contribution (approx by subtracting that component from risk_raw)
        - baseline_v1: legacy heuristic baseline
        """
        variants: Dict[str, pd.Series] = {"full": pd.to_numeric(df["risk_raw"], errors="coerce").fillna(0.0)}

        comp_cols = [c for c in df.columns if c.startswith("comp_")]
        for ccol in comp_cols:
            variants[f"drop_{ccol}"] = variants["full"] - pd.to_numeric(df.get(ccol, 0.0), errors="coerce").fillna(0.0)

        variants["baseline_v1"] = self._legacy_v1_score(df)
        return variants

    def _plots(self, df: pd.DataFrame) -> None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        import seaborn as sns  # noqa: E402
        from matplotlib import pyplot as plt  # noqa: E402

        # Distribution
        plt.figure(figsize=(7, 4))
        sns.histplot(pd.to_numeric(df["risk_z"], errors="coerce").fillna(0.0), bins=40, kde=True)
        plt.title("Contract risk score distribution (risk_z)")
        plt.tight_layout()
        plt.savefig(self.paths.fig_distribution, dpi=200)
        plt.close()

        # Component heatmap
        comp_cols = [c for c in df.columns if c.startswith("comp_")]
        if comp_cols:
            m = df[comp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            corr = m.corr()
            plt.figure(figsize=(7, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Risk components correlation")
            plt.tight_layout()
            plt.savefig(self.paths.fig_components_heatmap, dpi=200)
            plt.close()

    def run(self) -> Tuple[Path, Path]:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        df = self._load()
        labels = self._load_labels()

        variants = self._compute_ablation_scores(df)
        rows: List[Dict[str, Any]] = []

        # Join labels if available
        y = None
        if labels is not None:
            j = df[["address"]].merge(labels, on="address", how="inner")
            y = j["label"].to_numpy().astype(int)

        for name, s in variants.items():
            # stability vs full
            rho = _rank_spearman(variants["full"], s)
            row: Dict[str, Any] = {"variant": name, "spearman_vs_full": float(rho)}

            if labels is not None:
                # align scores to labels join set
                tmp = df[["address"]].copy()
                tmp["score"] = pd.to_numeric(s, errors="coerce").fillna(0.0)
                j = tmp.merge(labels, on="address", how="inner")
                ys = j["label"].to_numpy().astype(int)
                ss = pd.to_numeric(j["score"], errors="coerce").fillna(0.0).to_numpy()
                row["n_labeled"] = int(len(j))
                row["label_pos"] = int((ys == 1).sum())
                row["auc_roc"] = _auc_roc(ys, ss)
                row["avg_precision"] = _avg_precision(ys, ss)
            rows.append(row)

        ab = pd.DataFrame(rows).sort_values(by=["variant"]).reset_index(drop=True)
        ab.to_csv(self.paths.out_ablation_csv, index=False)

        report: Dict[str, Any] = {
            "n_contracts": int(len(df)),
            "n_verified_source": int((pd.to_numeric(df.get("has_verified_source", 0), errors="coerce").fillna(0).astype(int) == 1).sum()),
            "n_unverified_source": int((pd.to_numeric(df.get("has_verified_source", 0), errors="coerce").fillna(0).astype(int) == 0).sum()),
            "labels_present": bool(labels is not None),
            "labels_path": str(self.paths.labels_csv),
            "ablation_csv": str(self.paths.out_ablation_csv),
            "figures": {
                "distribution": str(self.paths.fig_distribution),
                "components_heatmap": str(self.paths.fig_components_heatmap),
            },
        }
        self.paths.out_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # plots
        self._plots(df)

        self.logger.info(f"Wrote ablation: {self.paths.out_ablation_csv}")
        self.logger.info(f"Wrote report:   {self.paths.out_report_json}")
        if labels is None:
            self.logger.info(f"No labels found (optional). To enable AUC/AP metrics, create: {self.paths.labels_csv}")
        return self.paths.out_report_json, self.paths.out_ablation_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate contract risk score (baselines/ablations/plots).")
    parser.add_argument("--contract-risk-csv", type=str, default=str(EvalPaths().contract_risk_csv))
    parser.add_argument("--labels-csv", type=str, default=str(EvalPaths().labels_csv))
    args = parser.parse_args()

    paths = EvalPaths(contract_risk_csv=Path(args.contract_risk_csv), labels_csv=Path(args.labels_csv))
    ContractRiskEvaluation(paths=paths).run()


if __name__ == "__main__":
    main()


