#!/usr/bin/env python3 -u
"""
Deep error analysis: SCAFE-Sem vs CodeBERT.

Identifies exactly which contracts each method gets right/wrong,
characterizes the error patterns, and suggests targeted improvements.

Use --protocol strict to load STRICT-label OOFs (excludes FORGE-unknown).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"
SCORED_STRICT = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["legacy", "strict"], default="legacy")
    args = parser.parse_args()
    protocol = args.protocol

    if protocol == "strict":
        sem_oof = OOF_DIR / "sem_Vote_RF_GBC_XGB_strict_struct_plus_both.csv"
        if not sem_oof.exists():
            # Fall back to the strict ml-Vote SCAFE OOF (no semantic) if SCAFE-Sem strict not available
            sem_oof = OOF_DIR / "ml_Vote_RF_GBC_XGB_strict_merged.csv"
        cb_oof = OOF_DIR / "dl_codebert_strict_merged.csv"
        scored_csv = SCORED_STRICT
        forge_labels = ["vulnerable"]  # strict: drop unknown + suspicious
    else:
        sem_oof = OOF_DIR / "sem_Vote_RF_GBC_XGB_merged.csv"
        cb_oof = OOF_DIR / "dl_codebert_Merged.csv"
        scored_csv = SCORED_V2
        forge_labels = ["vulnerable", "unknown"]

    print("=" * 100)
    print(f"ERROR ANALYSIS: SCAFE-Sem vs CodeBERT  [protocol={protocol}]")
    print(f"  SEM OOF: {sem_oof.name}")
    print(f"  CB  OOF: {cb_oof.name}")
    print("=" * 100)

    if not sem_oof.exists() or not cb_oof.exists():
        missing = [p.name for p in (sem_oof, cb_oof) if not p.exists()]
        print(f"  Missing OOF: {missing}.  Aborting.")
        return

    sem = pd.read_csv(sem_oof)
    cb = pd.read_csv(cb_oof)

    # Merge
    df = sem.merge(
        cb[['contract_id', 'fold', 'y_prob']],
        on=['contract_id', 'fold'],
        suffixes=('_sem', '_cb')
    )
    print(f"Merged: {len(df)} samples, {df['y_true'].sum()} positives")

    # Find optimal thresholds per fold
    sem_preds = np.zeros(len(df))
    cb_preds = np.zeros(len(df))
    for fid in sorted(df['fold'].unique()):
        m = df['fold'] == fid
        yt = df.loc[m, 'y_true'].values

        t_sem, _ = optimal_f1_threshold(yt, df.loc[m, 'y_prob_sem'].values)
        t_cb, _ = optimal_f1_threshold(yt, df.loc[m, 'y_prob_cb'].values)

        sem_preds[m.values] = (df.loc[m, 'y_prob_sem'].values >= t_sem).astype(int)
        cb_preds[m.values] = (df.loc[m, 'y_prob_cb'].values >= t_cb).astype(int)

    df['pred_sem'] = sem_preds.astype(int)
    df['pred_cb'] = cb_preds.astype(int)
    y_true = df['y_true'].values

    # ── 1) Agreement analysis ───────────────────────────────────────────────
    print("\n--- Agreement Analysis ---")
    both_correct = ((df['pred_sem'] == y_true) & (df['pred_cb'] == y_true)).sum()
    both_wrong = ((df['pred_sem'] != y_true) & (df['pred_cb'] != y_true)).sum()
    sem_only_correct = ((df['pred_sem'] == y_true) & (df['pred_cb'] != y_true)).sum()
    cb_only_correct = ((df['pred_cb'] == y_true) & (df['pred_sem'] != y_true)).sum()

    print(f"  Both correct:         {both_correct:6d} ({both_correct/len(df)*100:.1f}%)")
    print(f"  Both wrong:           {both_wrong:6d} ({both_wrong/len(df)*100:.1f}%)")
    print(f"  SCAFE-Sem only right: {sem_only_correct:6d} ({sem_only_correct/len(df)*100:.1f}%)")
    print(f"  CodeBERT only right:  {cb_only_correct:6d} ({cb_only_correct/len(df)*100:.1f}%)")

    # ── 2) Confusion breakdown for positives ────────────────────────────────
    pos = df[y_true == 1]
    neg = df[y_true == 0]
    print(f"\n--- Positive class (n={len(pos)}) ---")
    tp_sem = (pos['pred_sem'] == 1).sum()
    fn_sem = (pos['pred_sem'] == 0).sum()
    tp_cb = (pos['pred_cb'] == 1).sum()
    fn_cb = (pos['pred_cb'] == 0).sum()
    print(f"  SCAFE-Sem: TP={tp_sem} ({tp_sem/len(pos)*100:.1f}%), FN={fn_sem} ({fn_sem/len(pos)*100:.1f}%)")
    print(f"  CodeBERT:  TP={tp_cb} ({tp_cb/len(pos)*100:.1f}%), FN={fn_cb} ({fn_cb/len(pos)*100:.1f}%)")

    # Contracts CodeBERT catches but we miss
    cb_catches_we_miss = pos[(pos['pred_cb'] == 1) & (pos['pred_sem'] == 0)]
    we_catch_cb_misses = pos[(pos['pred_sem'] == 1) & (pos['pred_cb'] == 0)]
    print(f"\n  CB catches, we miss: {len(cb_catches_we_miss)}")
    print(f"  We catch, CB misses: {len(we_catch_cb_misses)}")

    print(f"\n--- Negative class (n={len(neg)}) ---")
    tn_sem = (neg['pred_sem'] == 0).sum()
    fp_sem = (neg['pred_sem'] == 1).sum()
    tn_cb = (neg['pred_cb'] == 0).sum()
    fp_cb = (neg['pred_cb'] == 1).sum()
    print(f"  SCAFE-Sem: TN={tn_sem} ({tn_sem/len(neg)*100:.1f}%), FP={fp_sem} ({fp_sem/len(neg)*100:.1f}%)")
    print(f"  CodeBERT:  TN={tn_cb} ({tn_cb/len(neg)*100:.1f}%), FP={fp_cb} ({fp_cb/len(neg)*100:.1f}%)")

    # ── 3) Probability distribution analysis ────────────────────────────────
    print("\n--- Probability Distribution Analysis ---")
    for label_name, subset in [("Positive", pos), ("Negative", neg)]:
        print(f"\n  {label_name} class:")
        for col, name in [('y_prob_sem', 'SCAFE-Sem'), ('y_prob_cb', 'CodeBERT')]:
            vals = subset[col].values
            print(f"    {name:12s}: mean={np.mean(vals):.3f}, median={np.median(vals):.3f}, "
                  f"std={np.std(vals):.3f}, [p10={np.percentile(vals,10):.3f}, "
                  f"p25={np.percentile(vals,25):.3f}, p75={np.percentile(vals,75):.3f}, "
                  f"p90={np.percentile(vals,90):.3f}]")

    # ── 4) Feature analysis of errors ───────────────────────────────────────
    print("\n--- Feature Analysis of Errors ---")
    scored = pd.read_csv(scored_csv, low_memory=False)
    scored['source_dataset'] = scored['source_dataset'].fillna('').astype(str)
    scored['label_status'] = scored['label_status'].fillna('').astype(str)

    dappscan = scored[scored['source_dataset'] == 'DAppSCAN'].copy()
    dappscan = dappscan[dappscan['label_status'].isin(['clean', 'vulnerable'])]
    forge = scored[scored['source_dataset'] == 'FORGE'].copy()
    forge = forge[forge['label_status'].isin(forge_labels)]
    data = pd.concat([dappscan, forge], ignore_index=True)
    data['y'] = (data['label_status'] == 'vulnerable').astype(int)
    data['contract_id'] = data['contract_id'].fillna('').astype(str)

    # Join features to predictions
    df_feat = df.merge(data[['contract_id', 'y', 'local_source_path'] +
                            [c for c in data.columns if c.startswith(('comp_', 'halstead_',
                             'token_', 'trigram_', 'state_', 'call_', 'loc_', 'n_func',
                             'vuln_', 'security_', 'risk_'))
                             and not c.startswith('risk_')]],
                       on='contract_id', how='left')

    key_features = [
        'loc_total', 'n_functions', 'comp_complexity',
        'halstead_volume', 'halstead_effort',
        'token_diversity', 'trigram_diversity', 'trigram_entropy',
        'state_vars_written', 'call_graph_edges',
    ]
    existing_features = [f for f in key_features if f in df_feat.columns]

    # Group 1: CB catches, we miss (our false negatives that CB gets right)
    group_cb_wins = df_feat[(df_feat['pred_cb'] == 1) & (df_feat['pred_sem'] == 0) & (df_feat['y_true'] == 1)]
    # Group 2: We catch, CB misses
    group_we_win = df_feat[(df_feat['pred_sem'] == 1) & (df_feat['pred_cb'] == 0) & (df_feat['y_true'] == 1)]
    # Group 3: Both catch
    group_both_tp = df_feat[(df_feat['pred_sem'] == 1) & (df_feat['pred_cb'] == 1) & (df_feat['y_true'] == 1)]
    # Group 4: Both miss
    group_both_fn = df_feat[(df_feat['pred_sem'] == 0) & (df_feat['pred_cb'] == 0) & (df_feat['y_true'] == 1)]
    # Group 5: Our false positives
    group_our_fp = df_feat[(df_feat['pred_sem'] == 1) & (df_feat['y_true'] == 0)]
    # Group 6: CB's false positives
    group_cb_fp = df_feat[(df_feat['pred_cb'] == 1) & (df_feat['y_true'] == 0)]

    print(f"\n  Vulnerable contracts breakdown:")
    print(f"    Both TP:          {len(group_both_tp):5d}")
    print(f"    CB catches, we miss: {len(group_cb_wins):5d}")
    print(f"    We catch, CB misses: {len(group_we_win):5d}")
    print(f"    Both FN:          {len(group_both_fn):5d}")

    if existing_features:
        print(f"\n  Feature comparison (median values):")
        print(f"    {'Feature':30s} {'Both TP':>10s} {'CB wins':>10s} {'We win':>10s} {'Both FN':>10s} {'Our FP':>10s}")
        for feat in existing_features:
            vals = []
            for grp in [group_both_tp, group_cb_wins, group_we_win, group_both_fn, group_our_fp]:
                if feat in grp.columns and len(grp) > 0:
                    vals.append(f"{grp[feat].median():.1f}")
                else:
                    vals.append("N/A")
            print(f"    {feat:30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s} {vals[4]:>10s}")

    # ── 5) Probability calibration analysis ─────────────────────────────────
    print("\n--- Probability Calibration (where are we losing?) ---")
    # Bin by SCAFE-Sem probability and see where CodeBERT does better
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"  {'Prob range':>12s} {'N':>6s} {'Pos%':>6s} {'SEM ROC':>10s} {'CB ROC':>10s} {'Gap':>8s}")
    for lo, hi in bins:
        mask = (df['y_prob_sem'] >= lo) & (df['y_prob_sem'] < hi)
        n = mask.sum()
        if n < 20:
            continue
        pos_rate = df.loc[mask, 'y_true'].mean()
        if df.loc[mask, 'y_true'].nunique() < 2:
            continue
        roc_sem = roc_auc_score(df.loc[mask, 'y_true'], df.loc[mask, 'y_prob_sem'])
        roc_cb = roc_auc_score(df.loc[mask, 'y_true'], df.loc[mask, 'y_prob_cb'])
        gap = roc_cb - roc_sem
        print(f"  [{lo:.1f}, {hi:.1f})  {n:6d} {pos_rate:6.1%} {roc_sem:10.4f} {roc_cb:10.4f} {gap:+8.4f}")

    # ── 6) Per-fold analysis ────────────────────────────────────────────────
    print("\n--- Per-Fold ROC-AUC Comparison ---")
    for fid in sorted(df['fold'].unique()):
        m = df['fold'] == fid
        yt = df.loc[m, 'y_true'].values
        roc_sem = roc_auc_score(yt, df.loc[m, 'y_prob_sem'].values)
        roc_cb = roc_auc_score(yt, df.loc[m, 'y_prob_cb'].values)
        print(f"  Fold {fid}: SEM={roc_sem:.4f}, CB={roc_cb:.4f}, gap={roc_cb-roc_sem:+.4f}")

    # ── 7) Source dataset analysis ──────────────────────────────────────────
    print("\n--- Source Dataset Breakdown ---")
    df_src = df.merge(data[['contract_id', 'source_dataset']], on='contract_id', how='left')
    if 'source_dataset' in df_src.columns:
        for src in df_src['source_dataset'].unique():
            mask = df_src['source_dataset'] == src
            n = mask.sum()
            if n < 50:
                continue
            yt = df_src.loc[mask, 'y_true'].values
            if len(np.unique(yt)) < 2:
                continue
            roc_sem = roc_auc_score(yt, df_src.loc[mask, 'y_prob_sem'].values)
            roc_cb = roc_auc_score(yt, df_src.loc[mask, 'y_prob_cb'].values)
            pos = yt.sum()
            print(f"  {src:15s}: n={n:6d}, pos={pos:5d}, SEM ROC={roc_sem:.4f}, CB ROC={roc_cb:.4f}, gap={roc_cb-roc_sem:+.4f}")

    # ── 8) Contract size analysis ───────────────────────────────────────────
    print("\n--- Performance by Contract Size ---")
    if 'loc_total' in df_feat.columns:
        for lo, hi, label in [(0, 50, "tiny"), (50, 200, "small"), (200, 500, "medium"),
                               (500, 1000, "large"), (1000, 99999, "very large")]:
            mask = (df_feat['loc_total'] >= lo) & (df_feat['loc_total'] < hi)
            n = mask.sum()
            if n < 30:
                continue
            yt = df_feat.loc[mask, 'y_true'].values
            if len(np.unique(yt)) < 2:
                continue
            roc_sem = roc_auc_score(yt, df_feat.loc[mask, 'y_prob_sem'].values)
            roc_cb = roc_auc_score(yt, df_feat.loc[mask, 'y_prob_cb'].values)
            pos = yt.sum()
            print(f"  {label:12s} (LOC {lo}-{hi}): n={n:6d}, pos={pos:4d}, "
                  f"SEM={roc_sem:.4f}, CB={roc_cb:.4f}, gap={roc_cb-roc_sem:+.4f}")

    # ── 9) Correlation between error probability gap and features ───────────
    print("\n--- Feature Correlations with Performance Gap ---")
    df_feat['prob_gap'] = df_feat['y_prob_cb'] - df_feat['y_prob_sem']
    pos_feat = df_feat[df_feat['y_true'] == 1]
    if existing_features and len(pos_feat) > 50:
        print("  Correlation of (CB_prob - SEM_prob) with features (positive class):")
        correlations = []
        for feat in existing_features:
            if feat in pos_feat.columns:
                valid = pos_feat[[feat, 'prob_gap']].dropna()
                if len(valid) > 30:
                    corr = valid[feat].corr(valid['prob_gap'])
                    correlations.append((feat, corr))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in correlations:
            print(f"    {feat:30s}: r={corr:+.3f}")

    # ── 10) Summary & Recommendations ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY OF FINDINGS")
    print("=" * 100)
    print(f"\n  Total gap to close: ROC-AUC = {0.944 - 0.932:.3f}, F1 = {0.664 - 0.611:.3f}")
    print(f"  Contracts where CB wins (TP we miss): {len(group_cb_wins)}")
    print(f"  Contracts where we win (TP CB misses): {len(group_we_win)}")
    print(f"  Our extra false positives vs CB: {fp_sem - fp_cb}")
    print(f"  Our extra false negatives vs CB: {fn_sem - fn_cb}")


if __name__ == "__main__":
    main()
