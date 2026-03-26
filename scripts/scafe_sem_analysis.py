#!/usr/bin/env python3 -u
"""
SCAFE-Sem post-experiment analysis:
  1. Per-fold comparison against DL baselines
  2. Statistical tests (Welch's t-test)
  3. Hybrid ensemble (SCAFE-Sem + CodeBERT)
  4. Two-stage pipeline (SCAFE-Sem pre-filter → CodeBERT)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)

warnings.filterwarnings("ignore")

OOF_DIR = Path(__file__).resolve().parent.parent / "data" / "derived" / "oof_predictions"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "derived" / "ml_results"


def load_oof(prefix, model, dataset):
    return pd.read_csv(OOF_DIR / f"{prefix}_{model}_{dataset}.csv")


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def per_fold_metrics(df):
    """Compute per-fold ROC-AUC, PR-AUC, F1 from OOF predictions."""
    folds = sorted(df['fold'].unique())
    rocs, prs, f1s, precs, recs = [], [], [], [], []
    for fid in folds:
        m = df['fold'] == fid
        yt = df.loc[m, 'y_true'].values
        yp = df.loc[m, 'y_prob'].values
        rocs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
        t, f = optimal_f1_threshold(yt, yp)
        f1s.append(f)
        pred = (yp >= t).astype(int)
        precs.append(precision_score(yt, pred, zero_division=0))
        recs.append(recall_score(yt, pred, zero_division=0))
    return {
        'roc_folds': rocs, 'roc_mean': np.mean(rocs), 'roc_std': np.std(rocs),
        'pr_folds': prs, 'pr_mean': np.mean(prs), 'pr_std': np.std(prs),
        'f1_folds': f1s, 'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        'prec_mean': np.mean(precs), 'rec_mean': np.mean(recs),
    }


def welch_ttest(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return t, p, d


def analyze_dataset(dataset_label):
    """Full analysis for one dataset (merged or dappscan)."""
    print(f"\n{'='*100}")
    print(f"ANALYSIS: {dataset_label}")
    print(f"{'='*100}")

    # Load SCAFE-Sem models
    sem_models = {}
    for name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'ExtraTrees', 'Vote_RF_GBC_XGB']:
        try:
            sem_models[name] = load_oof('sem', name, dataset_label)
        except FileNotFoundError:
            pass

    # Load original SCAFE models
    orig_models = {}
    for name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'Vote_RF_GBC_XGB']:
        try:
            orig_models[name] = load_oof('ml', name, dataset_label)
        except FileNotFoundError:
            pass

    # Load DL baselines
    ds_cap = "Merged" if dataset_label == "merged" else "DAppSCAN"
    dl_models = {}
    for name in ['textcnn', 'bilstm', 'codebert']:
        try:
            dl_models[name] = load_oof('dl', name, ds_cap)
        except FileNotFoundError:
            pass

    # ── 1) Per-fold metrics comparison ──────────────────────────────────────
    print("\n--- Per-fold ROC-AUC Comparison ---")
    all_metrics = {}

    for name, df in orig_models.items():
        m = per_fold_metrics(df)
        all_metrics[f'SCAFE_{name}'] = m
        print(f"  SCAFE {name:25s} ROC={m['roc_mean']:.4f}±{m['roc_std']:.4f}  "
              f"PR={m['pr_mean']:.4f}  F1={m['f1_mean']:.4f}  "
              f"P={m['prec_mean']:.3f} R={m['rec_mean']:.3f}")

    print()
    for name, df in sem_models.items():
        m = per_fold_metrics(df)
        all_metrics[f'SCAFE-Sem_{name}'] = m
        print(f"  SCAFE-Sem {name:21s} ROC={m['roc_mean']:.4f}±{m['roc_std']:.4f}  "
              f"PR={m['pr_mean']:.4f}  F1={m['f1_mean']:.4f}  "
              f"P={m['prec_mean']:.3f} R={m['rec_mean']:.3f}")

    print()
    for name, df in dl_models.items():
        m = per_fold_metrics(df)
        all_metrics[f'DL_{name}'] = m
        print(f"  DL {name:28s} ROC={m['roc_mean']:.4f}±{m['roc_std']:.4f}  "
              f"PR={m['pr_mean']:.4f}  F1={m['f1_mean']:.4f}  "
              f"P={m['prec_mean']:.3f} R={m['rec_mean']:.3f}")

    # ── 2) Statistical tests ────────────────────────────────────────────────
    print("\n--- Statistical Tests (Welch's t-test on 5-fold ROC-AUC) ---")
    sem_vote = all_metrics.get('SCAFE-Sem_Vote_RF_GBC_XGB')
    if sem_vote:
        for cmp_name in ['DL_textcnn', 'DL_bilstm', 'DL_codebert',
                         'SCAFE_Vote_RF_GBC_XGB']:
            if cmp_name in all_metrics:
                t, p, d = welch_ttest(
                    sem_vote['roc_folds'],
                    all_metrics[cmp_name]['roc_folds']
                )
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  SCAFE-Sem Vote vs {cmp_name:30s}: t={t:+.3f}, p={p:.4f} ({sig}), "
                      f"Cohen's d={d:+.3f}")

    # ── 3) Hybrid ensemble (SCAFE-Sem + CodeBERT) ──────────────────────────
    print("\n--- Hybrid Ensemble (SCAFE-Sem + CodeBERT) ---")
    if 'Vote_RF_GBC_XGB' in sem_models and 'codebert' in dl_models:
        sem_df = sem_models['Vote_RF_GBC_XGB']
        cb_df = dl_models['codebert']
        merged = sem_df.merge(
            cb_df[['contract_id', 'fold', 'y_prob']],
            on=['contract_id', 'fold'],
            suffixes=('_sem', '_cb')
        )

        for w_sem in [0.3, 0.4, 0.5]:
            w_cb = 1 - w_sem
            merged['y_hybrid'] = w_sem * merged['y_prob_sem'] + w_cb * merged['y_prob_cb']

            folds = sorted(merged['fold'].unique())
            rocs, f1s, prs = [], [], []
            for fid in folds:
                m = merged['fold'] == fid
                yt = merged.loc[m, 'y_true'].values
                yh = merged.loc[m, 'y_hybrid'].values
                rocs.append(roc_auc_score(yt, yh))
                prs.append(average_precision_score(yt, yh))
                _, f = optimal_f1_threshold(yt, yh)
                f1s.append(f)

            print(f"  w_sem={w_sem:.1f}, w_cb={w_cb:.1f}: "
                  f"ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
                  f"PR={np.mean(prs):.4f}  F1={np.mean(f1s):.4f}")

        # Also try with individual RF
        for rf_name in ['RandomForest', 'GradientBoosting']:
            if rf_name in sem_models:
                sem_rf = sem_models[rf_name]
                merged_rf = sem_rf.merge(
                    cb_df[['contract_id', 'fold', 'y_prob']],
                    on=['contract_id', 'fold'],
                    suffixes=('_sem', '_cb')
                )
                merged_rf['y_hybrid'] = 0.4 * merged_rf['y_prob_sem'] + 0.6 * merged_rf['y_prob_cb']
                folds = sorted(merged_rf['fold'].unique())
                rocs, f1s = [], []
                for fid in folds:
                    m2 = merged_rf['fold'] == fid
                    yt2 = merged_rf.loc[m2, 'y_true'].values
                    yh2 = merged_rf.loc[m2, 'y_hybrid'].values
                    rocs.append(roc_auc_score(yt2, yh2))
                    _, f = optimal_f1_threshold(yt2, yh2)
                    f1s.append(f)
                print(f"  Hybrid(SCAFE-Sem {rf_name}+CB, 0.4/0.6): "
                      f"ROC={np.mean(rocs):.4f}±{np.std(rocs):.4f}  "
                      f"F1={np.mean(f1s):.4f}")

    # ── 4) Two-stage pipeline ───────────────────────────────────────────────
    print("\n--- Two-Stage Pipeline (SCAFE-Sem pre-filter → CodeBERT) ---")
    if 'Vote_RF_GBC_XGB' in sem_models and 'codebert' in dl_models:
        sem_df = sem_models['Vote_RF_GBC_XGB']
        cb_df = dl_models['codebert']
        merged = sem_df.merge(
            cb_df[['contract_id', 'fold', 'y_prob']],
            on=['contract_id', 'fold'],
            suffixes=('_scafe', '_cb')
        )

        for target_recall in [0.99, 0.97, 0.95, 0.90]:
            fold_metrics = []
            for fid in sorted(merged['fold'].unique()):
                m = merged['fold'] == fid
                sub = merged[m].copy()
                yt = sub['y_true'].values
                sp = sub['y_prob_scafe'].values

                # Find HIGHEST threshold that still achieves target recall
                thresholds = np.linspace(0.5, 0.001, 500)
                best_t = 0.001
                for t in thresholds:
                    rec = recall_score(yt, (sp >= t).astype(int), zero_division=1.0)
                    if rec >= target_recall:
                        best_t = t
                        break

                scafe_pos = sp >= best_t
                pct_passed = scafe_pos.mean()

                final_prob = np.where(scafe_pos, sub['y_prob_cb'].values, 0.0)
                roc = roc_auc_score(yt, final_prob)
                _, f = optimal_f1_threshold(yt, final_prob)
                fold_metrics.append({
                    'roc': roc, 'f1': f, 'pct_passed': pct_passed,
                })

            rocs = [x['roc'] for x in fold_metrics]
            f1s = [x['f1'] for x in fold_metrics]
            pcts = [x['pct_passed'] for x in fold_metrics]
            speedup = 1 / np.mean(pcts)

            print(f"  Recall≥{target_recall:.2f}: "
                  f"ROC={np.mean(rocs):.4f}  F1={np.mean(f1s):.4f}  "
                  f"Pass={np.mean(pcts)*100:.1f}%  Speedup={speedup:.1f}x")

    return all_metrics


def main():
    print("=" * 100)
    print("SCAFE-Sem Comprehensive Analysis")
    print("=" * 100)

    results = {}
    for ds in ["merged", "dappscan"]:
        results[ds] = analyze_dataset(ds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for ds, metrics in results.items():
        serializable[ds] = {}
        for k, v in metrics.items():
            serializable[ds][k] = {
                'roc_mean': round(v['roc_mean'], 4),
                'roc_std': round(v['roc_std'], 4),
                'pr_mean': round(v['pr_mean'], 4),
                'f1_mean': round(v['f1_mean'], 4),
                'f1_std': round(v['f1_std'], 4),
                'prec_mean': round(v['prec_mean'], 3),
                'rec_mean': round(v['rec_mean'], 3),
            }
    out = RESULTS_DIR / "scafe_sem_comparison.json"
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()
