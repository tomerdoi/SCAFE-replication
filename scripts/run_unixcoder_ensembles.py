#!/usr/bin/env python3 -u
"""
Ensemble & two-stage analysis using UniXcoder (and CodeBERT for comparison).

Computes:
  1. Individual baselines (all ML + all DL including UniXcoder)
  2. Nested weighted-average fusion: α·SCAFE + (1-α)·DL
  3. Two-stage pipeline: SCAFE pre-filter → DL on flagged subset
  4. Stacking meta-learner

All results saved to JSON for paper integration.
Runs on CPU only (uses pre-computed OOF predictions).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
RESULTS_DIR = _REPO / "data" / "derived" / "ml_results"


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 199):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def score_fold(y_true, y_prob):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    t, f1 = optimal_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return roc, pr, f1, prec, rec


def per_fold_metrics(base, y_prob):
    folds = base["fold"].values
    y = base["y_true"].values
    rocs, prs, f1s, precs, recs = [], [], [], [], []
    for f in sorted(set(folds)):
        mask = folds == f
        r, p, f1, prec, rec = score_fold(y[mask], y_prob[mask])
        rocs.append(r); prs.append(p); f1s.append(f1)
        precs.append(prec); recs.append(rec)
    return {
        "roc_auc_mean": np.mean(rocs), "roc_auc_std": np.std(rocs),
        "pr_auc_mean": np.mean(prs), "pr_auc_std": np.std(prs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "precision_mean": np.mean(precs), "recall_mean": np.mean(recs),
    }


def load_oof(prefix, model, dataset):
    fname = f"{prefix}_{model}_{dataset}.csv"
    path = OOF_DIR / fname
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("sample_idx").reset_index(drop=True)


def nested_weighted_avg(base, ml_prob, dl_prob, steps=51):
    """Nested leave-one-fold-out α selection."""
    folds = base["fold"].values
    y = base["y_true"].values
    unique_folds = sorted(set(folds))

    fold_metrics = []
    fold_alphas = []

    for test_fold in unique_folds:
        test_mask = folds == test_fold
        train_mask = ~test_mask

        best_alpha, best_roc = 0.5, 0.0
        for alpha in np.linspace(0, 1, steps):
            p_train = alpha * ml_prob[train_mask] + (1 - alpha) * dl_prob[train_mask]
            roc = roc_auc_score(y[train_mask], p_train)
            if roc > best_roc:
                best_alpha, best_roc = alpha, roc

        p_test = best_alpha * ml_prob[test_mask] + (1 - best_alpha) * dl_prob[test_mask]
        r, p, f1, prec, rec = score_fold(y[test_mask], p_test)
        fold_metrics.append((r, p, f1, prec, rec))
        fold_alphas.append(best_alpha)

    m = np.array(fold_metrics)
    return {
        "alpha_mean": np.mean(fold_alphas), "alpha_std": np.std(fold_alphas),
        "roc_auc_mean": m[:, 0].mean(), "roc_auc_std": m[:, 0].std(),
        "pr_auc_mean": m[:, 1].mean(), "pr_auc_std": m[:, 1].std(),
        "f1_mean": m[:, 2].mean(), "f1_std": m[:, 2].std(),
        "precision_mean": m[:, 3].mean(), "recall_mean": m[:, 4].mean(),
    }


def two_stage_analysis(base, scafe_prob, dl_prob, recall_targets):
    """Two-stage pipeline at various SCAFE recall operating points."""
    folds = base["fold"].values
    y = base["y_true"].values
    results = []

    for target in [1.0] + recall_targets:
        fold_data = []
        for f in sorted(set(folds)):
            mask = folds == f
            y_f = y[mask]
            sp = scafe_prob[mask]
            dp = dl_prob[mask]

            if target == 1.0:
                final_prob = dp
                frac = 1.0
            else:
                pos_probs = sp[y_f == 1]
                threshold = np.percentile(pos_probs, (1.0 - target) * 100)
                passed = sp >= threshold
                frac = passed.mean()
                final_prob = np.where(passed, dp, 0.0)

            r, p, f1, prec, rec = score_fold(y_f, final_prob)
            fold_data.append((r, p, f1, prec, rec, frac))

        m = np.array(fold_data)
        results.append({
            "recall_target": target,
            "roc_auc_mean": m[:, 0].mean(), "roc_auc_std": m[:, 0].std(),
            "pr_auc_mean": m[:, 1].mean(), "pr_auc_std": m[:, 1].std(),
            "f1_mean": m[:, 2].mean(), "f1_std": m[:, 2].std(),
            "precision_mean": m[:, 3].mean(), "recall_mean": m[:, 4].mean(),
            "fraction_sent_mean": m[:, 5].mean(), "fraction_sent_std": m[:, 5].std(),
        })
    return results


def stacking(base, prob_cols_dict):
    """Leave-one-fold-out logistic meta-learner."""
    y = base["y_true"].values
    folds = base["fold"].values
    X = np.column_stack(list(prob_cols_dict.values()))
    meta_pred = np.zeros(len(y))

    for f in sorted(set(folds)):
        train_mask = folds != f
        test_mask = folds == f
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X[train_mask], y[train_mask])
        meta_pred[test_mask] = lr.predict_proba(X[test_mask])[:, 1]

    return per_fold_metrics(base, meta_pred)


def fmt(d, key):
    return f"{d[key+'_mean']:.4f}±{d[key+'_std']:.4f}"


def analyse(dataset):
    ds_tag = "Merged" if dataset == "merged" else "DAppSCAN"
    print(f"\n{'='*80}")
    print(f"  ENSEMBLE ANALYSIS — {dataset.upper()}")
    print(f"{'='*80}\n")

    # Load SCAFE OOF
    if dataset == "merged":
        scafe_df = load_oof("ml", "Vote_RF_GBC_XGB", "merged")
    else:
        scafe_df = load_oof("ml", "RandomForest", "dappscan")

    if scafe_df is None:
        print("  SCAFE OOF not found, skipping.")
        return None

    base = scafe_df[["sample_idx", "contract_id", "fold", "y_true"]].copy()
    scafe_prob = scafe_df["y_prob"].values

    # Load DL OOFs
    cb_df = load_oof("dl", "codebert", ds_tag)
    ux_df = load_oof("dl", "unixcoder", ds_tag)

    dl_models = {}
    if cb_df is not None:
        dl_models["codebert"] = cb_df.sort_values("sample_idx")["y_prob"].values
    if ux_df is not None:
        dl_models["unixcoder"] = ux_df.sort_values("sample_idx")["y_prob"].values

    if not dl_models:
        print("  No DL OOF found, skipping.")
        return None

    scafe_label = "Vote" if dataset == "merged" else "RF"
    results = {"dataset": dataset}

    # 1. Individual baselines
    print("--- Individual baselines ---")
    scafe_m = per_fold_metrics(base, scafe_prob)
    print(f"  SCAFE ({scafe_label}):  ROC={fmt(scafe_m,'roc_auc')}  PR={fmt(scafe_m,'pr_auc')}  F1={fmt(scafe_m,'f1')}")
    results[f"scafe_{scafe_label}"] = scafe_m

    for name, prob in dl_models.items():
        m = per_fold_metrics(base, prob)
        print(f"  {name:15s}:  ROC={fmt(m,'roc_auc')}  PR={fmt(m,'pr_auc')}  F1={fmt(m,'f1')}")
        results[name] = m

    # 2. Weighted fusion for each DL model
    print("\n--- Nested weighted-average fusion ---")
    for name, prob in dl_models.items():
        wt = nested_weighted_avg(base, scafe_prob, prob)
        label = f"SCAFE+{name}"
        print(f"  {label:25s}: α={wt['alpha_mean']:.2f}±{wt['alpha_std']:.2f}  "
              f"ROC={fmt(wt,'roc_auc')}  PR={fmt(wt,'pr_auc')}  F1={fmt(wt,'f1')}")
        results[f"fusion_{name}"] = wt

    # 3. Two-stage pipeline for each DL model
    recall_targets = [0.99, 0.97, 0.95, 0.90]
    print("\n--- Two-stage pipeline (SCAFE pre-filter → DL) ---")
    for name, prob in dl_models.items():
        ts = two_stage_analysis(base, scafe_prob, prob, recall_targets)
        print(f"\n  SCAFE → {name}:")
        print(f"  {'Target':>8}  {'%Sent':>7}  {'ROC-AUC':>14}  {'PR-AUC':>14}  {'F1':>14}")
        for r in ts:
            tgt = "DL-only" if r["recall_target"] == 1.0 else f"{r['recall_target']:.0%}"
            print(f"  {tgt:>8}  {r['fraction_sent_mean']*100:5.1f}%  "
                  f"{fmt(r,'roc_auc'):>14}  {fmt(r,'pr_auc'):>14}  {fmt(r,'f1'):>14}")
        results[f"two_stage_{name}"] = ts

    # 4. Stacking for each DL model
    print("\n--- Stacking meta-learner ---")
    for name, prob in dl_models.items():
        st = stacking(base, {"scafe": scafe_prob, name: prob})
        label = f"Stack(SCAFE+{name})"
        print(f"  {label:30s}: ROC={fmt(st,'roc_auc')}  PR={fmt(st,'pr_auc')}  F1={fmt(st,'f1')}")
        results[f"stack_{name}"] = st

    # 5. If both DL models present, also stack all three
    if len(dl_models) == 2:
        all_probs = {"scafe": scafe_prob, **dl_models}
        st_all = stacking(base, all_probs)
        print(f"  {'Stack(SCAFE+CB+UX)':30s}: ROC={fmt(st_all,'roc_auc')}  PR={fmt(st_all,'pr_auc')}  F1={fmt(st_all,'f1')}")
        results["stack_all"] = st_all

    return results


def main():
    print("=" * 80)
    print("ENSEMBLE & TWO-STAGE ANALYSIS (UniXcoder + CodeBERT)")
    print("=" * 80)

    all_results = {}
    for ds in ["merged", "dappscan"]:
        r = analyse(ds)
        if r:
            all_results[ds] = r

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ensemble_unixcoder_results.json"

    def convert(o):
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(f"Not serializable: {type(o)}")

    out.write_text(json.dumps(all_results, indent=2, default=convert))
    print(f"\nResults saved to {out}")
    print("\nDONE")


if __name__ == "__main__":
    main()
