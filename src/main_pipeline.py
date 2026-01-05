"""
Main end-to-end pipeline orchestrator.

Runs (in order):
- EigenExplorer: operators, AVSs, operator↔AVS edges, operator rewards
- u--1: live snapshots + Tokensight labels (small)
- Dune: 1-year panels (governance/activity/flows + related contracts)
- GitHub: case-study signals (incident-like issues/PRs + releases)
- Etherscan: verified contract sources (AVSs + proxy implementations)
- Code risk: derive AVS code-risk features
- Risk score pipeline: final AVS/operator risk + figures

Usage:
  python -m src.main_pipeline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as a file in IDE/debugger (so `import src...` works)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.code_risk_util import AVSCodeRiskUtil
from src.dune_util import run_full_year_panels
from src.eigen_explorer_avs_exposure_util import main_generate_operator_avs_edges
from src.eigen_explorer_avs_util import main_generate_all_rows as eigen_avs_all
from src.eigen_explorer_operator_rewards_util import main_fetch_operator_rewards
from src.eigen_explorer_util import main_generate_all_rows as eigen_ops_all
from src.etherscan_source_util import EtherscanAVSSourceUtil
from src.github_incidents_util import GitHubIncidentsUtil
from src.risk_score_pipeline import RiskScorePipeline
from src.u1_util import main_fetch_all_live


class MainPipeline:
    def __init__(self) -> None:
        logger = logging.getLogger("main_pipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        self._total_steps = 7  # keep in sync with run()
        self._step_idx = 0

    def _log_progress(self, msg: str) -> None:
        """
        Log overall pipeline progress as a percentage.
        """
        pct = int(round(100.0 * max(0, self._step_idx) / max(1, self._total_steps)))
        self.logger.info(f"[{pct:3d}%] {msg}")

    def _start_step(self, name: str) -> None:
        self._step_idx += 1
        pct_start = int(round(100.0 * (self._step_idx - 1) / max(1, self._total_steps)))
        self.logger.info(f"[{pct_start:3d}%] START: {name} ({self._step_idx}/{self._total_steps})")

    def _end_step(self, name: str) -> None:
        pct_end = int(round(100.0 * self._step_idx / max(1, self._total_steps)))
        self.logger.info(f"[{pct_end:3d}%] DONE:  {name} ({self._step_idx}/{self._total_steps})")

    def run_eigenexplorer(self) -> None:
        try:
            step = "EigenExplorer"
            self._start_step(step)
            self._log_progress("EigenExplorer: operators inventory (all)")
            eigen_ops_all(page_size=100)
            self._log_progress("EigenExplorer: AVS registry (all)")
            eigen_avs_all(page_size=100)
            self._log_progress("EigenExplorer: operator↔AVS exposure graph")
            main_generate_operator_avs_edges(page_size=100, limit_avs=None)
            self._log_progress("EigenExplorer: operator rewards (all exposed operators)")
            # progress-aware logging inside the util
            main_fetch_operator_rewards(limit=None, progress_logger=self.logger, progress_every=25)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_eigenexplorer", exc_info=True)
            raise

    def run_u1(self) -> None:
        try:
            step = "u--1 snapshots"
            self._start_step(step)
            self._log_progress("u--1: live snapshots + Tokensight labels (small)")
            main_fetch_all_live()
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_u1", exc_info=True)
            raise

    def run_dune(self) -> None:
        try:
            step = "Dune panels"
            self._start_step(step)
            self._log_progress("Dune: 1-year panels (governance/activity/flows + related contracts)")
            run_full_year_panels(fetch_all=True, lookback_days=365, performance="medium", progress_logger=self.logger)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_dune", exc_info=True)
            raise

    def run_github(self) -> None:
        try:
            step = "GitHub case studies"
            self._start_step(step)
            self._log_progress("GitHub: case-study signals (incident-like issues/PRs + releases)")
            util = GitHubIncidentsUtil()
            base_repos = [
                "Layr-Labs/eigenlayer-contracts",
                "Layr-Labs/eigensdk-go",
                "Layr-Labs/eigenda",
                "OpenZeppelin/openzeppelin-contracts",
            ]
            discovered = util.discover_repos_from_u1_metadata()
            repos = sorted(set(base_repos + discovered))
            util.run(repos, since_days=365)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_github", exc_info=True)
            raise

    def run_etherscan_sources(self) -> None:
        try:
            step = "Etherscan sources"
            self._start_step(step)
            self._log_progress("Etherscan: verified contract sources (all AVSs + proxy implementations)")
            EtherscanAVSSourceUtil().run(fetch_all=True, limit=0, progress_logger=self.logger, progress_every=5)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_etherscan_sources", exc_info=True)
            raise

    def run_code_risk(self) -> None:
        try:
            step = "Code risk"
            self._start_step(step)
            self._log_progress("Code risk: derive AVS code-risk features")
            AVSCodeRiskUtil().run(progress_logger=self.logger, progress_every=5)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_code_risk", exc_info=True)
            raise

    def run_risk_score(self) -> None:
        try:
            step = "Risk score + figures"
            self._start_step(step)
            self._log_progress("Risk score: final AVS/operator risk + figures")
            RiskScorePipeline().run()
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_risk_score", exc_info=True)
            raise

    def run(self) -> None:
        try:
            self._step_idx = 0
            self.run_eigenexplorer()
            self.run_u1()
            self.run_dune()
            self.run_github()
            self.run_etherscan_sources()
            self.run_code_risk()
            self.run_risk_score()
            self.logger.info("[100%] End-to-end pipeline completed successfully.")
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    MainPipeline().run()

