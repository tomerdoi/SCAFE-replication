"""
A-PARSE: Parser reliability analysis for the paper revision.

Measures:
  1. Tree-sitter parse failure rate by Solidity version
  2. Feature NaN rates (indicators of parse failures)
  3. Impact on the ML evaluation (did failed contracts end up in train/test?)

Outputs:
  data/derived/parser_stats/parse_reliability_summary.csv
  data/derived/parser_stats/nan_feature_rates.csv
  figures/fig_parse_reliability.pdf  (if matplotlib available)

Usage:
    python scripts/analyze_parser_reliability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCORED = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
_OUT = _REPO / "data" / "derived" / "parser_stats"


def nan_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature NaN rate across all numeric columns."""
    from src.ml.features import _EXCLUDE

    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in _EXCLUDE]
    nan_rates = {}
    for col in numeric_cols:
        nan_rates[col] = df[col].isna().mean()
    sr = pd.Series(nan_rates).sort_values(ascending=False)
    df_nan = pd.DataFrame({"feature": sr.index, "nan_rate": sr.values})
    return df_nan


def solidity_version_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Break down parse quality by Solidity major.minor version."""
    # Use compiler_version column
    if "compiler_version" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["sol_major_minor"] = (
        df["compiler_version"]
        .fillna("unknown")
        .astype(str)
        .str.extract(r"(\d+\.\d+)", expand=False)
        .fillna("unknown")
    )

    # Define "parse_failed" as contracts with >50% NaN in structural features
    from src.ml.features import _EXCLUDE
    struct_cols = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c])
                   and c not in _EXCLUDE
                   and c not in {"risk_raw", "risk_z", "risk_rank"}]

    if struct_cols:
        nan_frac = df[struct_cols].isna().mean(axis=1)
        df["parse_fail_flag"] = (nan_frac > 0.5).astype(int)
    else:
        df["parse_fail_flag"] = 0

    summary = df.groupby("sol_major_minor").agg(
        n=("contract_id", "count"),
        n_failed=("parse_fail_flag", "sum"),
        fail_rate=("parse_fail_flag", "mean"),
    ).reset_index().sort_values("n", ascending=False)

    return summary


def impact_analysis(df: pd.DataFrame, df_nan: pd.DataFrame) -> None:
    """Report how parse failures affect the ML evaluation set."""
    from src.ml.features import _EXCLUDE
    struct_cols = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c])
                   and c not in _EXCLUDE
                   and c not in {"risk_raw", "risk_z", "risk_rank"}]

    if not struct_cols:
        print("  No structural columns found for impact analysis.")
        return

    nan_frac = df[struct_cols].isna().mean(axis=1)
    failed = df[nan_frac > 0.5]

    print(f"\n=== PARSE FAILURE IMPACT ===")
    print(f"  Total contracts: {len(df):,}")
    print(f"  Contracts with >50% NaN structural features (parse fail): {len(failed):,} ({len(failed)/len(df):.2%})")

    if len(failed) > 0:
        label_dist = failed["label_status"].value_counts()
        print(f"  Label distribution of failed contracts:")
        for lbl, cnt in label_dist.items():
            print(f"    {lbl}: {cnt} ({cnt/len(failed):.1%})")

        print(f"  Note: these contracts will have features filled with 0.0 by prepare_dataset()")
        print(f"  Impact: predictions on these contracts are unreliable — they are flagged in JSONL")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    _OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_SCORED}")
    df = pd.read_csv(_SCORED, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── 1. NaN analysis ──────────────────────────────────────────────────
    print("\n=== NaN ANALYSIS (TOP 20 FEATURES) ===")
    df_nan = nan_analysis(df)
    print(df_nan.head(20).to_string(index=False))
    df_nan.to_csv(_OUT / "nan_feature_rates.csv", index=False)
    print(f"Saved → {_OUT / 'nan_feature_rates.csv'}")

    high_nan = df_nan[df_nan["nan_rate"] > 0.01]
    print(f"\n  Features with NaN rate > 1%: {len(high_nan)} out of {len(df_nan)}")
    if len(high_nan) == 0:
        print("  -> Excellent: no feature has meaningful NaN rate. Parser is reliable.")

    # ── 2. Solidity version analysis ─────────────────────────────────────
    print("\n=== SOLIDITY VERSION BREAKDOWN ===")
    df_versions = solidity_version_analysis(df)
    if len(df_versions) > 0:
        print(df_versions.head(20).to_string(index=False))
        df_versions.to_csv(_OUT / "parse_by_version.csv", index=False)
        print(f"Saved → {_OUT / 'parse_by_version.csv'}")

        total_fail = int(df_versions["n_failed"].sum())
        total_n = int(df_versions["n"].sum())
        print(f"\nOverall parse fail rate: {total_fail}/{total_n} = {total_fail/max(1,total_n):.3%}")
    else:
        print("  compiler_version column not available.")

    # ── 3. Impact analysis ──────────────────────────────────────────────
    impact_analysis(df, df_nan)

    # ── 4. Summary ────────────────────────────────────────────────────────
    summary = {
        "total_contracts": len(df),
        "features_with_any_nan": int((df_nan["nan_rate"] > 0).sum()),
        "features_with_gt1pct_nan": int((df_nan["nan_rate"] > 0.01).sum()),
        "max_nan_rate": float(df_nan["nan_rate"].max()),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(_OUT / "parse_reliability_summary.csv", index=False)
    print(f"\nSummary saved → {_OUT / 'parse_reliability_summary.csv'}")

    print("\n=== PARSER RELIABILITY SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
