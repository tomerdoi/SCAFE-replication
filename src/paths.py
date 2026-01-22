"""
Centralized repository paths.

Goal:
- Keep the main (general) contract-risk pipeline using `data/` and `data/derived/`.
- Keep AVS-specific case study artifacts under `data/case_studies/avs/` (and corresponding figures).
"""

from __future__ import annotations

from pathlib import Path


# src/paths.py -> repo root is one level up from this file
REPO_ROOT = Path(__file__).resolve().parents[1]

# General (main) locations
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
FIGURES_DIR = REPO_ROOT / "figures"

# Case studies
CASE_STUDIES_DIR = DATA_DIR / "case_studies"
AVS_DATA_DIR = CASE_STUDIES_DIR / "avs"
AVS_DERIVED_DIR = AVS_DATA_DIR / "derived"
AVS_FIGURES_DIR = FIGURES_DIR / "case_studies" / "avs"

