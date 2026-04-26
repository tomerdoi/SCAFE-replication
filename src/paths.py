"""
Centralized repository paths.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
FIGURES_DIR = REPO_ROOT / "figures"
