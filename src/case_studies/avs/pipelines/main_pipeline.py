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
  python -m src.case_studies.avs.pipelines.main_pipeline
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running as a file in IDE/debugger (so `import src...` works)
# This file now lives under src/case_studies/avs/... so we use the centralized path resolver.
from src.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.case_studies.avs.risk.code_risk import AVSCodeRiskUtil
from src.case_studies.avs.registry.avs_registry import AVSRegistryUtil
from src.clients.dune import run_avs_directory_avs_registry, run_full_year_panels
from src.clients.eigen_explorer_avs_exposure import main_generate_operator_avs_edges
from src.clients.eigen_explorer_avs import main_generate_all_rows as eigen_avs_all
from src.clients.eigen_explorer_operator_rewards import main_fetch_operator_rewards
from src.clients.eigen_explorer import main_generate_all_rows as eigen_ops_all
from src.clients.etherscan_source import EtherscanAVSSourceUtil
from src.clients.github_incidents import GitHubIncidentsUtil
from src.case_studies.avs.pipelines.risk_score_pipeline import RiskScorePipeline
from src.clients.u1 import main_fetch_all_live

from src.paths import AVS_DATA_DIR, AVS_DERIVED_DIR


class MainPipeline:
    def __init__(self) -> None:
        logger = logging.getLogger("main_pipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        self._total_steps = 9  # keep in sync with run()
        self._step_idx = 0
        self.force = str(os.getenv("PIPELINE_FORCE", "0")).strip() in ("1", "true", "True", "yes", "YES")

        self.repo_root = _REPO_ROOT
        # AVS case study outputs live under data/case_studies/avs/
        self.data_dir = AVS_DATA_DIR
        self.derived_dir = AVS_DERIVED_DIR

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

    def _file_ok(self, path: Path, min_bytes: int = 10) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size >= int(min_bytes)
        except Exception:
            return False

    def _all_ok(self, paths: list[Path]) -> bool:
        return all(self._file_ok(p) for p in paths)

    def run_eigenexplorer(self) -> None:
        try:
            step = "EigenExplorer"
            self._start_step(step)
            out_ops = self.data_dir / "eigenexplorer_operator_addresses_all.json"
            out_avs = self.data_dir / "eigenexplorer_avs_all.json"
            out_edges = self.data_dir / "eigenexplorer_operator_avs_edges.json"
            out_edges_raw = self.data_dir / "eigenexplorer_avs_operators_payloads.json"
            out_rewards = self.data_dir / "eigenexplorer_operator_rewards.json"

            if not self.force and self._all_ok([out_ops, out_avs, out_edges, out_edges_raw, out_rewards]):
                self._log_progress("EigenExplorer outputs exist; skipping API calls")
                self._end_step(step)
                return

            if self.force or not self._file_ok(out_ops):
                self._log_progress("EigenExplorer: operators inventory (all)")
                eigen_ops_all(page_size=100)
            else:
                self._log_progress("EigenExplorer: operators inventory exists; skipping")

            if self.force or not self._file_ok(out_avs):
                self._log_progress("EigenExplorer: AVS registry (all)")
                eigen_avs_all(page_size=100)
            else:
                self._log_progress("EigenExplorer: AVS registry exists; skipping")

            if self.force or not (self._file_ok(out_edges) and self._file_ok(out_edges_raw)):
                self._log_progress("EigenExplorer: operator↔AVS exposure graph")
                main_generate_operator_avs_edges(page_size=100, limit_avs=None)
            else:
                self._log_progress("EigenExplorer: exposure graph exists; skipping")

            if self.force or not self._file_ok(out_rewards):
                self._log_progress("EigenExplorer: operator rewards (all exposed operators)")
                main_fetch_operator_rewards(limit=None, progress_logger=self.logger, progress_every=25)
            else:
                self._log_progress("EigenExplorer: operator rewards exists; skipping")

            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_eigenexplorer", exc_info=True)
            raise

    def run_u1(self) -> None:
        try:
            step = "u--1 snapshots"
            self._start_step(step)
            self._log_progress("u--1: live snapshots + Tokensight labels (small)")
            # u1_util will also skip individual endpoints if files exist
            main_fetch_all_live(skip_if_exists=not self.force)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_u1", exc_info=True)
            raise

    def run_avs_registry(self) -> None:
        """
        Build a broad AVS registry by unioning available sources (EigenExplorer + u--1 + Dune-derived lists if present).
        Downstream steps prefer this file if present.
        """
        try:
            step = "AVS registry (union)"
            self._start_step(step)
            out_path = self.data_dir / "avs_registry_all.json"
            if (not self.force) and self._file_ok(out_path):
                self._log_progress(f"AVS registry exists; skipping: {out_path.name}")
                self._end_step(step)
                return
            self._log_progress("Building unioned AVS registry (max coverage)")
            AVSRegistryUtil().write_registry(out_path=out_path)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_avs_registry", exc_info=True)
            raise

    def run_dune(self) -> None:
        try:
            step = "Dune panels"
            self._start_step(step)
            self._log_progress("Dune: 1-year panels (governance/activity/flows + related contracts)")
            run_full_year_panels(
                fetch_all=True,
                lookback_days=365,
                performance="medium",
                progress_logger=self.logger,
                skip_if_exists=not self.force,
            )
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_dune", exc_info=True)
            raise

    def run_dune_avs_registry(self) -> None:
        """
        Discover AVS addresses directly from on-chain AVSDirectory logs via Dune.
        This is independent of EigenExplorer/u--1 inventories and can expand coverage.
        """
        try:
            step = "Dune AVS registry (on-chain)"
            self._start_step(step)
            out_path = self.data_dir / "dune_avs_directory_avs_registry_all.json"
            if not self.force and self._file_ok(out_path):
                self._log_progress("Dune AVS registry output exists; skipping query")
                self._end_step(step)
                return
            self._log_progress("Dune: discovering AVS addresses from AVSDirectory logs")
            run_avs_directory_avs_registry(performance="medium", skip_if_exists=not self.force)
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_dune_avs_registry", exc_info=True)
            raise

    def run_github(self) -> None:
        try:
            step = "GitHub case studies"
            self._start_step(step)
            self._log_progress("GitHub: case-study signals (incident-like issues/PRs + releases)")
            out_path = self.data_dir / "github_case_studies.json"
            if not self.force and self._file_ok(out_path):
                self._log_progress("GitHub case studies output exists; skipping API calls")
                self._end_step(step)
                return
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
            EtherscanAVSSourceUtil().run(
                fetch_all=True,
                limit=0,
                progress_logger=self.logger,
                progress_every=5,
                skip_existing=not self.force,
            )
            self._end_step(step)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run_etherscan_sources", exc_info=True)
            raise

    def run_code_risk(self) -> None:
        try:
            step = "Code risk"
            self._start_step(step)
            self._log_progress("Code risk: derive AVS code-risk features")
            out_path = self.derived_dir / "avs_code_risk.csv"
            if not self.force and self._file_ok(out_path):
                self._log_progress("AVS code risk CSV exists; skipping computation")
                self._end_step(step)
                return
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
            self.run_dune_avs_registry()
            self.run_avs_registry()
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

