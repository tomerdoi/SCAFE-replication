"""DEPRECATED SHIM.

Legacy AVS risk score pipeline moved to: `src.case_studies.avs.pipelines.risk_score_pipeline`.
"""

from __future__ import annotations

from src.case_studies.avs.pipelines.risk_score_pipeline import RiskScorePipeline  # noqa: F401
from src.case_studies.avs.pipelines.risk_score_pipeline import *  # noqa: F401,F403


if __name__ == "__main__":
    RiskScorePipeline().run()
