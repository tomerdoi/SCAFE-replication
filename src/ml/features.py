"""
Feature matrix preparation for ML vulnerability classification.

Responsibilities:
- Load the scored CSV produced by the risk-scoring pipeline.
- Select numeric feature columns (exclude IDs, labels, paths, risk scores).
- Handle NaN / inf values.
- Provide group IDs for leakage-safe GroupKFold splitting.
- Optional feature-set variants (all, static-only, on-chain-only) for ablation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Repo root (src/ml/features.py → 2 parents up)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DERIVED_DIR = _REPO_ROOT / "data" / "derived"
_DEFAULT_SCORED_CSV = _DERIVED_DIR / "unified_contract_eval_scored.csv"

# ── Columns to ALWAYS exclude from feature matrix ──────────────────────────
_ID_COLS = {
    "row_id", "source_dataset", "source_subdataset", "chain",
    "address", "contract_id", "label_status", "vuln_present",
    "vuln_types", "vuln_severity",
    "local_source_path", "local_bytecode_path", "local_metadata_path",
    "compiler_version", "implementation_address",
    "is_eth",
}
_RISK_SCORE_COLS = {
    "risk_raw", "risk_z", "risk_rank", "risk_uncertainty", "has_risk_row",
}
# These fields describe source availability or compiler metadata rather than
# source-derived program content.  They are excluded from every ML matrix to
# prevent provenance/metadata leakage into source-only claims.
_PROVENANCE_METADATA_COLS = {
    "has_verified_source", "is_solc_pre_08",
}
_AUX_COLS = {
    "aux_is_upgradeable", "aux_admin_power",
}
# Label columns that may be created during prepare_dataset
_LABEL_COLS = {
    "y", "_address_norm", "_local_join_key",
}
_EXCLUDE = (
    _ID_COLS
    | _RISK_SCORE_COLS
    | _PROVENANCE_METADATA_COLS
    | _AUX_COLS
    | _LABEL_COLS
)

# ── Feature subsets for ablation studies ────────────────────────────────────
_ONCHAIN_FEATURES = {
    "has_activity_payload", "age_days", "last_tx_age_days",
    "tx_count_7d", "tx_count_30d", "tx_count_90d",
    "tx_in_30d", "tx_out_30d", "unique_senders_30d",
    "failed_tx_rate_90d", "internal_tx_count_30d",
    "erc20_tx_count_30d", "erc20_unique_tokens_30d",
    "erc20_unique_senders_30d", "erc20_unique_receivers_30d",
    "value_in_eth_30d", "value_out_eth_30d",
    "n_evt_upgraded_365d", "n_evt_admin_changed_365d",
    "n_evt_ownership_transferred_365d", "n_evt_role_granted_365d",
    "n_evt_role_revoked_365d", "n_evt_paused_365d", "n_evt_unpaused_365d",
    "admin_event_churn_365d",
    "has_bytecode", "bytecode_len",
    "bc_has_call", "bc_has_delegatecall", "bc_has_staticcall",
    "bc_has_selfdestruct", "bc_n_call", "bc_n_delegatecall",
    "bc_n_selfdestruct", "bc_proxy_like",
}
_COMPONENT_COLS = {
    "comp_upgrade", "comp_privilege", "comp_interaction",
    "comp_dangerous", "comp_complexity", "comp_opacity",
    "comp_code_smell", "comp_safety", "comp_activity",
}


@dataclass
class FeatureSet:
    """Immutable description of which features to include."""
    name: str
    include: Optional[set] = None   # None = all numeric (minus excludes)
    exclude: set = field(default_factory=set)
    description: str = ""


# Pre-defined feature sets for experiments / ablation
FEATURE_SETS = {
    "all": FeatureSet(
        name="all",
        description="All numeric features (static + on-chain + bytecode + components)",
    ),
    "static_only": FeatureSet(
        name="static_only",
        exclude=_ONCHAIN_FEATURES,
        description="Static source-code features only (no on-chain, no bytecode)",
    ),
    "raw_only": FeatureSet(
        name="raw_only",
        exclude=_ONCHAIN_FEATURES | _COMPONENT_COLS,
        description="Raw static features only (no components, no on-chain)",
    ),
    "components_only": FeatureSet(
        name="components_only",
        include=_COMPONENT_COLS,
        description="Only the 9 risk-score component values",
    ),
}


@dataclass
class PreparedDataset:
    """Ready-to-use dataset for model training and evaluation."""
    X: np.ndarray                  # (n_samples, n_features)
    y: np.ndarray                  # (n_samples,) binary labels
    feature_names: List[str]       # column names in X
    groups: np.ndarray             # group IDs for GroupKFold (integer-encoded contract_id)
    df: pd.DataFrame               # original DataFrame (filtered, for post-hoc analysis)
    dataset_filter: str            # description of the filter applied
    feature_set_name: str          # name of the FeatureSet used


def load_scored_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the scored CSV with low_memory=False to avoid dtype warnings."""
    path = Path(path) if path else _DEFAULT_SCORED_CSV
    logger.info(f"Loading scored CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def select_feature_columns(
    df: pd.DataFrame,
    feature_set: FeatureSet,
) -> List[str]:
    """Return the list of numeric feature column names for the given FeatureSet."""
    # Start with all numeric columns
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in _EXCLUDE
    ]

    if feature_set.include is not None:
        # Whitelist mode: only keep columns in the include set
        numeric_cols = [c for c in numeric_cols if c in feature_set.include]

    if feature_set.exclude:
        numeric_cols = [c for c in numeric_cols if c not in feature_set.exclude]

    return sorted(numeric_cols)


def prepare_dataset(
    df: Optional[pd.DataFrame] = None,
    scored_csv: Optional[Path] = None,
    *,
    dataset_filter: str = "DAppSCAN",
    label_filter: str = "clean_vs_vulnerable",
    feature_set_name: str = "all",
    custom_feature_set: Optional[FeatureSet] = None,
) -> PreparedDataset:
    """
    Prepare a ready-to-use dataset for ML training.

    Parameters
    ----------
    df : pre-loaded DataFrame (if None, loads from scored_csv)
    scored_csv : path to scored CSV (used if df is None)
    dataset_filter : which dataset(s) to include.
        "DAppSCAN" | "all" | "Empirical-Analysis" | comma-separated list
    label_filter : how to define binary labels.
        "clean_vs_vulnerable" = label_status ∈ {clean, vulnerable}
    feature_set_name : key into FEATURE_SETS dict
    custom_feature_set : override feature_set_name with a custom FeatureSet
    """
    if df is None:
        df = load_scored_csv(scored_csv)

    # ── Filter by dataset ──
    df = df.copy()
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    if dataset_filter != "all":
        datasets = [d.strip() for d in dataset_filter.split(",")]
        df = df[df["source_dataset"].isin(datasets)].copy()

    # ── Filter by label ──
    if label_filter == "clean_vs_vulnerable":
        df = df[df["label_status"].isin(["clean", "vulnerable"])].copy()
        df["y"] = (df["label_status"] == "vulnerable").astype(int)
    else:
        raise ValueError(f"Unknown label_filter: {label_filter!r}")

    if len(df) == 0:
        raise ValueError(f"No rows after filtering: dataset={dataset_filter}, label={label_filter}")

    n_pos = int(df["y"].sum())
    n_neg = int(len(df) - n_pos)
    logger.info(
        f"Prepared dataset: {len(df)} rows (pos={n_pos}, neg={n_neg}, "
        f"prev={n_pos / len(df):.3f}), filter={dataset_filter}"
    )

    # ── Feature selection (done AFTER y is added, but y is in _EXCLUDE) ──
    fs = custom_feature_set or FEATURE_SETS.get(feature_set_name)
    if fs is None:
        raise ValueError(f"Unknown feature_set_name: {feature_set_name!r}. Available: {list(FEATURE_SETS)}")
    feat_cols = select_feature_columns(df, fs)
    # Safety: ensure no label leakage
    assert "y" not in feat_cols, "FATAL: label column 'y' in feature set!"
    assert "vuln_present" not in feat_cols, "FATAL: label column 'vuln_present' in feature set!"
    if not feat_cols:
        raise ValueError(f"No features selected with feature_set={fs.name}")

    logger.info(f"  Feature set '{fs.name}': {len(feat_cols)} features")

    # ── Build X matrix ──
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float64)
    # Replace any remaining inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df["y"].values.astype(int)

    # ── Group IDs (for GroupKFold — leakage-safe splitting) ──
    # Use contract_id as group; encode to integer
    group_key = df["contract_id"].fillna("").astype(str)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)

    return PreparedDataset(
        X=X,
        y=y,
        feature_names=feat_cols,
        groups=groups,
        df=df.reset_index(drop=True),
        dataset_filter=dataset_filter,
        feature_set_name=fs.name,
    )
