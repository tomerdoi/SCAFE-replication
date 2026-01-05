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

from .code_risk_util import AVSCodeRiskUtil
from .dune_util import run_full_year_panels
from .eigen_explorer_avs_exposure_util import main_generate_operator_avs_edges
from .eigen_explorer_avs_util import main_generate_all_rows as eigen_avs_all
from .eigen_explorer_operator_rewards_util import main_fetch_operator_rewards
from .eigen_explorer_util import main_generate_all_rows as eigen_ops_all
from .etherscan_source_util import EtherscanAVSSourceUtil
from .github_incidents_util import GitHubIncidentsUtil
from .risk_score_pipeline import RiskScorePipeline
from .u1_util import main_fetch_all_live


class MainPipeline:
    def __init__(self) -> None:
        logger = logging.getLogger("main_pipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

    def run_eigenexplorer(self) -> None:
        try:
            self.logger.info("EigenExplorer: operators inventory (all)")
            eigen_ops_all(page_size=100)
            self.logger.info("EigenExplorer: AVS registry (all)")
            eigen_avs_all(page_size=100)
            self.logger.info("EigenExplorer: operator↔AVS exposure graph")
            main_generate_operator_avs_edges(page_size=100, limit_avs=None)
            self.logger.info("EigenExplorer: operator rewards (all exposed operators)")
            main_fetch_operator_rewards(limit=None)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_eigenexplorer", exc_info=True)
            raise

    def run_u1(self) -> None:
        try:
            self.logger.info("u--1: live snapshots + Tokensight labels (small)")
            main_fetch_all_live()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_u1", exc_info=True)
            raise

    def run_dune(self) -> None:
        try:
            self.logger.info("Dune: 1-year panels (governance/activity/flows + related contracts)")
            run_full_year_panels(fetch_all=True, lookback_days=365, performance="medium")
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_dune", exc_info=True)
            raise

    def run_github(self) -> None:
        try:
            self.logger.info("GitHub: case-study signals (incident-like issues/PRs + releases)")
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
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_github", exc_info=True)
            raise

    def run_etherscan_sources(self) -> None:
        try:
            self.logger.info("Etherscan: verified contract sources (all AVSs + proxy implementations)")
            EtherscanAVSSourceUtil().run(fetch_all=True, limit=0)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_etherscan_sources", exc_info=True)
            raise

    def run_code_risk(self) -> None:
        try:
            self.logger.info("Code risk: derive AVS code-risk features")
            AVSCodeRiskUtil().run()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_code_risk", exc_info=True)
            raise

    def run_risk_score(self) -> None:
        try:
            self.logger.info("Risk score: final AVS/operator risk + figures")
            RiskScorePipeline().run()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_risk_score", exc_info=True)
            raise

    def run(self) -> None:
        try:
            self.run_eigenexplorer()
            self.run_u1()
            self.run_dune()
            self.run_github()
            self.run_etherscan_sources()
            self.run_code_risk()
            self.run_risk_score()
            self.logger.info("End-to-end pipeline completed successfully.")
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    MainPipeline().run()

