"""
Code-risk component distribution figures across AVSs.

Reads:
- data/derived/avs_code_risk.csv
- data/eigenexplorer_avs_all.json
- data/dune_avs_related_contracts_all.json (optional)
- data/etherscan_sources/*.json

Writes:
- figures/code_risk_components_distribution.png
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Headless plotting
from src.paths import REPO_ROOT  # repo root resolver
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from src.case_studies.avs.risk.code_risk import AVSCodeRiskUtil
from src.paths import AVS_DERIVED_DIR, AVS_FIGURES_DIR


class CodeRiskFigures:
    def __init__(self) -> None:
        logger = logging.getLogger("code_risk_figures")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.util = AVSCodeRiskUtil()

    def ensure_dirs(self) -> None:
        try:
            AVS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during ensure_dirs", exc_info=True)
            raise

    def build_avs_components_df(self) -> pd.DataFrame:
        """
        Build per-AVS component flags by scanning the AVS contract set and taking ANY(contract has feature).
        """
        try:
            avs_addrs = self.util.load_avs_addresses()
            related = self.util.load_related_contracts()
            avs_contracts = self.util.build_avs_contract_set(avs_addrs, related)

            rows = []
            for avs, contracts in avs_contracts.items():
                flags = {
                    "is_upgradeable_any": 0,
                    "admin_power_any": 0,
                    "uses_call_any": 0,
                    "uses_delegatecall_any": 0,
                    "uses_selfdestruct_any": 0,
                    "uses_pausable_any": 0,
                    "uses_timelock_any": 0,
                }
                n_with_src = 0
                for caddr in contracts:
                    payload = self.util.load_etherscan_source_payload(caddr)
                    feats = self.util.extract_contract_features(caddr, payload)
                    if int(feats.get("has_verified_source", 0)) == 1:
                        n_with_src += 1
                    is_upgradeable = 1 if (int(feats.get("is_proxy", 0)) == 1 or int(feats.get("has_upgrade_patterns", 0)) == 1) else 0
                    admin_power = 1 if (int(feats.get("uses_ownable", 0)) == 1 or int(feats.get("uses_accesscontrol", 0)) == 1) else 0
                    flags["is_upgradeable_any"] = max(flags["is_upgradeable_any"], is_upgradeable)
                    flags["admin_power_any"] = max(flags["admin_power_any"], admin_power)
                    flags["uses_call_any"] = max(flags["uses_call_any"], int(feats.get("uses_call", 0)))
                    flags["uses_delegatecall_any"] = max(flags["uses_delegatecall_any"], int(feats.get("uses_delegatecall", 0)))
                    flags["uses_selfdestruct_any"] = max(flags["uses_selfdestruct_any"], int(feats.get("uses_selfdestruct", 0)))
                    flags["uses_pausable_any"] = max(flags["uses_pausable_any"], int(feats.get("uses_pausable", 0)))
                    flags["uses_timelock_any"] = max(flags["uses_timelock_any"], int(feats.get("uses_timelock", 0)))

                rows.append(
                    {
                        "avs_address": avs,
                        "n_contracts_total": len(contracts),
                        "n_contracts_with_verified_source": n_with_src,
                        **flags,
                    }
                )

            return pd.DataFrame(rows)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during build_avs_components_df", exc_info=True)
            raise

    def plot_components_distribution(self, out_path: Optional[Path] = None) -> Path:
        try:
            self.ensure_dirs()
            if out_path is None:
                out_path = AVS_FIGURES_DIR / "code_risk_components_distribution.png"

            comp = self.build_avs_components_df()
            cr_path = AVS_DERIVED_DIR / "avs_code_risk.csv"
            if cr_path.exists():
                cr = pd.read_csv(cr_path)
                cr["avs_address"] = cr["avs_address"].astype(str).str.lower()
                comp = comp.merge(cr[["avs_address", "code_risk_z", "code_risk_raw_max"]], on="avs_address", how="left")

            # Convert boolean-ish columns to {0,1}
            bool_cols = [
                "is_upgradeable_any",
                "admin_power_any",
                "uses_call_any",
                "uses_delegatecall_any",
                "uses_selfdestruct_any",
                "uses_pausable_any",
                "uses_timelock_any",
            ]
            for c in bool_cols:
                comp[c] = pd.to_numeric(comp[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

            fig, axes = plt.subplots(2, 2, figsize=(11, 7))

            # (A) Share of AVSs with each flag
            shares = pd.DataFrame(
                {
                    "component": bool_cols,
                    "share_of_avss": [float(comp[c].mean()) for c in bool_cols],
                }
            )
            shares["share_of_avss"] = shares["share_of_avss"] * 100.0
            ax = axes[0][0]
            sns.barplot(data=shares, x="share_of_avss", y="component", ax=ax)
            ax.set_title("Code-risk components: % of AVSs with signal (ANY contract)")
            ax.set_xlabel("% of AVSs")
            ax.set_ylabel("")

            # (B) Verified source coverage (# contracts with verified code per AVS)
            ax = axes[0][1]
            sns.histplot(comp["n_contracts_with_verified_source"], bins=15, ax=ax)
            ax.set_title("Verified source coverage per AVS")
            ax.set_xlabel("# contracts with verified source (in AVS contract set)")

            # (C) Raw code risk max distribution
            ax = axes[1][0]
            if "code_risk_raw_max" in comp.columns:
                sns.histplot(comp["code_risk_raw_max"].fillna(0), bins=20, ax=ax)
                ax.set_title("Distribution of code_risk_raw_max across AVSs")
                ax.set_xlabel("code_risk_raw_max")
            else:
                ax.text(0.5, 0.5, "Missing avs_code_risk.csv", ha="center", va="center")
                ax.set_axis_off()

            # (D) code_risk_z distribution
            ax = axes[1][1]
            if "code_risk_z" in comp.columns:
                sns.histplot(comp["code_risk_z"].fillna(0), bins=20, ax=ax)
                ax.set_title("Distribution of code_risk_z across AVSs")
                ax.set_xlabel("code_risk_z")
            else:
                ax.text(0.5, 0.5, "Missing avs_code_risk.csv", ha="center", va="center")
                ax.set_axis_off()

            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Wrote: {out_path}")
            return out_path
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during plot_components_distribution", exc_info=True)
            raise


if __name__ == "__main__":
    CodeRiskFigures().plot_components_distribution()

