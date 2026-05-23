"""
A-FORGE: Derive strict label CSV and ablation variants.

Produces three CSVs:
  (a) strict   — headline: FORGE-vulnerable + DAppSCAN (clean+vulnerable); drop unknown/suspicious
  (b) legacy   — original scheme: FORGE unknown→negative, suspicious dropped
  (c) dappscan — DAppSCAN-only

Also runs A-FORGE-5 fold-balance check.

Usage:
    python scripts/derive_strict_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCORED = _REPO / "data" / "derived" / "unified_contract_eval_scored.csv"
_OUT = _REPO / "data" / "derived"


def _project_group(contract_id: str) -> str:
    """Extract audit-project group from contract_id for GroupKFold."""
    parts = contract_id.split(":")
    if contract_id.startswith("dappscan:") and len(parts) >= 3:
        return parts[1]          # audit project name
    return contract_id           # FORGE / ACV: each contract is its own group


def _fold_balance_check(df: pd.DataFrame, label: str = "strict") -> None:
    """Verify no project dominates a fold (A-FORGE-5)."""
    print(f"\n{'='*60}")
    print(f"FOLD BALANCE CHECK — {label} label scheme")
    print(f"{'='*60}")

    group_key = df["contract_id"].fillna("").astype(str).map(_project_group)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    groups = group_key.map(group_map).values.astype(int)
    y = df["y"].values

    gkf = GroupKFold(n_splits=5)
    max_pct = 0.0
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups)):
        y_test = y[test_idx]
        n_pos = int(y_test.sum())
        n_neg = int((y_test == 0).sum())
        n_total = len(y_test)
        pos_prev = n_pos / max(1, n_total)

        # Find dominant project in this fold
        test_groups = group_key.iloc[test_idx]
        proj_pos = {}
        for g, yi in zip(test_groups, y_test):
            if yi == 1:
                proj_pos[g] = proj_pos.get(g, 0) + 1
        if proj_pos:
            top_proj, top_count = max(proj_pos.items(), key=lambda x: x[1])
            pct = top_count / max(1, n_pos) * 100
            max_pct = max(max_pct, pct)
        else:
            top_proj, top_count, pct = "—", 0, 0.0

        print(f"  Fold {fold_idx}: n={n_total:5d}  pos={n_pos:4d} ({pos_prev:.1%})  "
              f"neg={n_neg:5d}  top_proj={top_proj!r} ({top_count} pos, {pct:.1f}% of fold pos)")

    flag = "⚠️  EXCEEDS 15% — check fold balance" if max_pct > 15 else "✓  OK"
    print(f"  Max single-project share of fold positives: {max_pct:.1f}%  {flag}")


def main() -> None:
    print(f"Loading {_SCORED}")
    df = pd.read_csv(_SCORED, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # ── (a) STRICT label scheme ───────────────────────────────────────────
    # DAppSCAN: keep clean + vulnerable as-is
    # FORGE: keep only 'vulnerable'; drop 'unknown' and 'suspicious'
    # AccessControlVulnerabilities: excluded from training (positive-only sanity set)

    dappscan = df[df["source_dataset"] == "DAppSCAN"].copy()
    forge_vuln = df[(df["source_dataset"] == "FORGE") &
                    (df["label_status"] == "vulnerable")].copy()

    strict = pd.concat([dappscan, forge_vuln], ignore_index=True)
    strict = strict[strict["label_status"].isin(["clean", "vulnerable"])].copy()
    strict["y"] = (strict["label_status"] == "vulnerable").astype(int)

    n_pos = int(strict["y"].sum())
    n_neg = int((strict["y"] == 0).sum())
    print(f"\n(a) STRICT: {len(strict):,} rows — pos={n_pos:,}, neg={n_neg:,}, "
          f"prev={n_pos/len(strict):.3f}")
    print(f"    DAppSCAN={len(dappscan):,}  FORGE-vulnerable={len(forge_vuln):,}")

    out_strict = _OUT / "unified_contract_eval_scored_strict.csv"
    strict.to_csv(out_strict, index=False)
    print(f"    Saved → {out_strict}")

    _fold_balance_check(strict, "strict")

    # ── (b) LEGACY label scheme (original, for ablation) ─────────────────
    # FORGE unknown → negative; suspicious → dropped
    forge_legacy = df[df["source_dataset"] == "FORGE"].copy()
    forge_legacy = forge_legacy[forge_legacy["label_status"] != "suspicious"].copy()
    forge_legacy["label_status"] = forge_legacy["label_status"].replace("unknown", "clean")

    legacy = pd.concat([dappscan, forge_legacy], ignore_index=True)
    legacy = legacy[legacy["label_status"].isin(["clean", "vulnerable"])].copy()
    legacy["y"] = (legacy["label_status"] == "vulnerable").astype(int)

    n_pos_l = int(legacy["y"].sum())
    n_neg_l = int((legacy["y"] == 0).sum())
    print(f"\n(b) LEGACY: {len(legacy):,} rows — pos={n_pos_l:,}, neg={n_neg_l:,}, "
          f"prev={n_pos_l/len(legacy):.3f}")

    out_legacy = _OUT / "unified_contract_eval_scored_legacy.csv"
    legacy.to_csv(out_legacy, index=False)
    print(f"    Saved → {out_legacy}")

    # ── (c) DAPPSCAN-ONLY (robustness check) ─────────────────────────────
    dappscan_only = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])].copy()
    dappscan_only["y"] = (dappscan_only["label_status"] == "vulnerable").astype(int)

    n_pos_d = int(dappscan_only["y"].sum())
    n_neg_d = int((dappscan_only["y"] == 0).sum())
    print(f"\n(c) DAPPSCAN-ONLY: {len(dappscan_only):,} rows — pos={n_pos_d:,}, neg={n_neg_d:,}, "
          f"prev={n_pos_d/len(dappscan_only):.3f}")

    out_dappscan = _OUT / "unified_contract_eval_scored_dappscan_only.csv"
    dappscan_only.to_csv(out_dappscan, index=False)
    print(f"    Saved → {out_dappscan}")

    _fold_balance_check(dappscan_only, "dappscan-only")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LABEL SCHEME SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scheme':<20} {'N':>7} {'Pos':>6} {'Neg':>7} {'Prev':>7}")
    print(f"  {'-'*50}")
    print(f"  {'strict (headline)':<20} {len(strict):>7,} {n_pos:>6,} {n_neg:>7,} {n_pos/len(strict):>7.3f}")
    print(f"  {'legacy (ablation)':<20} {len(legacy):>7,} {n_pos_l:>6,} {n_neg_l:>7,} {n_pos_l/len(legacy):>7.3f}")
    print(f"  {'dappscan-only':<20} {len(dappscan_only):>7,} {n_pos_d:>6,} {n_neg_d:>7,} {n_pos_d/len(dappscan_only):>7.3f}")
    print(f"\nDone. All CSVs written to {_OUT}")


if __name__ == "__main__":
    main()
