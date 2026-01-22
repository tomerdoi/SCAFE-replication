"""DEPRECATED SHIM.

Legacy AVS pipeline moved to: `src.case_studies.avs.pipelines.main_pipeline`.
"""

from __future__ import annotations

from src.case_studies.avs.pipelines.main_pipeline import MainPipeline  # noqa: F401
from src.case_studies.avs.pipelines.main_pipeline import *  # noqa: F401,F403


if __name__ == "__main__":
    MainPipeline().run()
