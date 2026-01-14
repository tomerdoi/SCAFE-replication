"""
Main pipeline entrypoint for the pivoted project: contract-level risk scoring.

Runs:
- Build contract registry (address universe)
- Compute contract risk using cached `data/etherscan_sources/*.json`

Usage:
  python -m src.main_contract_risk_pipeline

Optional:
  If you want to refresh sources, run `python -m src.etherscan_source_util --addresses-csv data/contract_registry.csv ...`
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running as a file in IDE/debugger (so `import src...` works)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.contract_registry_util import ContractRegistryUtil  # noqa: E402
from src.contract_risk_util import ContractRiskUtil  # noqa: E402


class MainContractRiskPipeline:
    def __init__(self) -> None:
        logger = logging.getLogger("main_contract_risk_pipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger

        self.repo_root = _REPO_ROOT
        self.data_dir = self.repo_root / "data"
        self.derived_dir = self.data_dir / "derived"

        self.registry_csv = self.data_dir / "contract_registry.csv"
        self.contract_risk_csv = self.derived_dir / "contract_risk.csv"

        self.limit = int(os.getenv("CONTRACT_RISK_LIMIT", "0") or "0")

    def run(self) -> None:
        self.logger.info("Building contract registry…")
        reg = ContractRegistryUtil().build_registry()
        ContractRegistryUtil().write_registry(reg, out_csv=self.registry_csv)

        self.logger.info("Computing contract risk scores from cached Etherscan sources…")
        ContractRiskUtil().run(registry_csv=self.registry_csv, out_csv=self.contract_risk_csv, limit=self.limit)
        self.logger.info("Contract-risk pipeline completed successfully.")


if __name__ == "__main__":
    MainContractRiskPipeline().run()


