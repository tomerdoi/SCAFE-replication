"""
Ensemble analysis: combine ML and DL (CodeBERT) out-of-fold predictions
to explore whether a hybrid approach beats either family alone.

All metrics are computed PER-FOLD and reported as mean ± std, consistent
with the main experiment tables. The weighted-average α is selected via
nested leave-one-fold-out CV to avoid optimistic bias.

Protocol selection
------------------
By default (legacy), loads OOF files matching ml_{Model}_{merged|dappscan}.csv
and dl_{model}_{Merged|DAppSCAN}.csv.

Pass --protocol strict to load STRICT-label OOFs (excludes FORGE-unknown):
  ml_{Model}_strict_merged.csv,  dl_{model}_strict_merged.csv  (etc.)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

OOF_DIR = Path(__file__).resolve().parent.parent / "data" / "derived" / "oof_predictions"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "derived" / "ml_results"

ML_MODELS = [
    "RandomForest", "GradientBoosting", "XGBoost",
    "ExtraTrees", "HistGradientBoosting", "LogisticRegression",
]
DL_MODELS = ["textcnn", "bilstm", "codebert", "unixcoder"]

# Maps logical dataset name to (ml_suffix, dl_suffix) for OOF file names.
_DATASET_TAGS = {
    ("legacy", "merged"): ("merged", "Merged"),
    ("legacy", "dappscan"): ("dappscan", "DAppSCAN"),
    ("strict", "merged"): ("strict_merged", "strict_merged"),
    ("strict", "dappscan"): ("strict_dappscan", "strict_dappscan"),
}


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
    _, f1 = optimal_f1_threshold(y_true, y_prob)
    return roc, pr, f1


def per_fold_score(base, y_prob):
    """Compute metrics per fold, return mean ± std."""
    folds = base["fold"].values
    y = base["y_true"].values
    rocs, prs, f1s = [], [], []
    for f in sorted(set(folds)):
        mask = folds == f
        r, p, f1 = score_fold(y[mask], y_prob[mask])
        rocs.append(r)
        prs.append(p)
        f1s.append(f1)
    return {
        "ROC-AUC": np.mean(rocs), "ROC-AUC_std": np.std(rocs),
        "PR-AUC": np.mean(prs), "PR-AUC_std": np.std(prs),
        "F1": np.mean(f1s), "F1_std": np.std(f1s),
    }


def fmt(d, key):
    return f"{d[key]:.3f} ± {d[key+'_std']:.3f}"


def load_oof(prefix, model, dataset):
    fname = f"{prefix}_{model}_{dataset}.csv"
    path = OOF_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.sort_values("sample_idx").reset_index(drop=True)


def load_all(dataset, protocol="legacy"):
    """Load all OOF predictions for a dataset under the given protocol."""
    base = None
    probs = {}

    ml_tag, dl_tag = _DATASET_TAGS[(protocol, dataset)]

    for m in ML_MODELS:
        df = load_oof("ml", m, ml_tag)
        if df is None:
            continue
        if base is None:
            base = df[["sample_idx", "contract_id", "fold", "y_true"]].copy()
        probs[f"ml_{m}"] = df["y_prob"].values

    vote_df = load_oof("ml", "Vote_RF_GBC_XGB", ml_tag)
    if vote_df is not None:
        probs["ml_Vote"] = vote_df["y_prob"].values

    for m in DL_MODELS:
        df = load_oof("dl", m, dl_tag)
        if df is None:
            continue
        probs[f"dl_{m}"] = df.sort_values("sample_idx")["y_prob"].values

    if base is None:
        return None, None
    prob_df = pd.DataFrame(probs, index=base.index)
    return base, prob_df


def run_individual_baselines(base, prob_df):
    """Score each model individually (per-fold)."""
    rows = []
    for col in prob_df.columns:
        s = per_fold_score(base, prob_df[col].values)
        rows.append({"Model": col, **s})
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return df


def nested_weight_grid(base, prob_df, ml_col, dl_col="dl_codebert", steps=51):
    """Nested leave-one-fold-out α selection to avoid optimistic bias.

    For each test fold:
      1. Select α on the remaining 4 folds (grid search maximizing ROC-AUC)
      2. Apply that α to the test fold and compute metrics
    Report mean ± std across the 5 test folds.
    """
    folds = base["fold"].values
    y = base["y_true"].values
    ml_p = prob_df[ml_col].values
    dl_p = prob_df[dl_col].values
    unique_folds = sorted(set(folds))

    fold_rocs, fold_prs, fold_f1s, fold_alphas = [], [], [], []

    for test_fold in unique_folds:
        test_mask = folds == test_fold
        train_mask = ~test_mask

        best_alpha, best_inner_roc = 0.5, 0.0
        for alpha in np.linspace(0, 1, steps):
            p_train = alpha * ml_p[train_mask] + (1 - alpha) * dl_p[train_mask]
            inner_roc = roc_auc_score(y[train_mask], p_train)
            if inner_roc > best_inner_roc:
                best_alpha, best_inner_roc = alpha, inner_roc

        p_test = best_alpha * ml_p[test_mask] + (1 - best_alpha) * dl_p[test_mask]
        r, p, f1 = score_fold(y[test_mask], p_test)
        fold_rocs.append(r)
        fold_prs.append(p)
        fold_f1s.append(f1)
        fold_alphas.append(best_alpha)

    return {
        "alpha_mean": np.mean(fold_alphas), "alpha_std": np.std(fold_alphas),
        "ROC-AUC": np.mean(fold_rocs), "ROC-AUC_std": np.std(fold_rocs),
        "PR-AUC": np.mean(fold_prs), "PR-AUC_std": np.std(fold_prs),
        "F1": np.mean(fold_f1s), "F1_std": np.std(fold_f1s),
    }


def run_stacking(base, prob_df, feature_cols):
    """Leave-one-fold-out logistic meta-learner, per-fold metrics."""
    y = base["y_true"].values
    folds = base["fold"].values
    X_meta = prob_df[feature_cols].values
    meta_pred = np.zeros(len(y))

    unique_folds = sorted(set(folds))
    for f in unique_folds:
        train_mask = folds != f
        test_mask = folds == f
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_meta[train_mask], y[train_mask])
        meta_pred[test_mask] = lr.predict_proba(X_meta[test_mask])[:, 1]

    return per_fold_score(base, meta_pred)


def analyse_dataset(dataset, protocol="legacy"):
    print(f"\n{'='*72}")
    print(f"  ENSEMBLE ANALYSIS — {dataset.upper()} dataset  [protocol={protocol}]")
    print(f"  (all metrics: per-fold mean ± std, 5-fold GroupKFold)")
    print(f"{'='*72}\n")

    base, prob_df = load_all(dataset, protocol=protocol)
    if base is None:
        print("  No data found, skipping.")
        return

    n_pos = base["y_true"].sum()
    n_tot = len(base)
    print(f"  Samples: {n_tot}  |  Positive: {n_pos} ({100*n_pos/n_tot:.1f}%)\n")

    # --- 1. Individual baselines ---
    print("─" * 70)
    print("  1) INDIVIDUAL MODEL BASELINES")
    print("─" * 70)
    ind = run_individual_baselines(base, prob_df)
    for _, row in ind.iterrows():
        print(f"  {row['Model']:30s}  ROC={fmt(row,'ROC-AUC')}  PR={fmt(row,'PR-AUC')}  F1={fmt(row,'F1')}")

    # --- 2. Nested weighted average: best ML + CodeBERT ---
    best_ml_row = ind[ind["Model"].str.startswith("ml_")].iloc[0]
    best_ml_name = best_ml_row["Model"]
    print(f"\n{'─'*70}")
    print(f"  2) NESTED WEIGHTED AVG:  α·{best_ml_name} + (1-α)·CodeBERT")
    print(f"     (α selected on training folds, applied to held-out fold)")
    print("─" * 70)
    wt = nested_weight_grid(base, prob_df, best_ml_name)
    print(f"  α = {wt['alpha_mean']:.2f} ± {wt['alpha_std']:.2f}  →  "
          f"ROC={fmt(wt,'ROC-AUC')}  PR={fmt(wt,'PR-AUC')}  F1={fmt(wt,'F1')}")

    wt_vote = None
    if "ml_Vote" in prob_df.columns and best_ml_name != "ml_Vote":
        wt_vote = nested_weight_grid(base, prob_df, "ml_Vote")
        print(f"\n  ml_Vote variant:")
        print(f"  α = {wt_vote['alpha_mean']:.2f} ± {wt_vote['alpha_std']:.2f}  →  "
              f"ROC={fmt(wt_vote,'ROC-AUC')}  PR={fmt(wt_vote,'PR-AUC')}  F1={fmt(wt_vote,'F1')}")

    # --- 3. Stacking meta-learner ---
    print(f"\n{'─'*70}")
    print("  3) STACKING META-LEARNER (leave-one-fold-out logistic regression)")
    print("─" * 70)

    stack_2 = run_stacking(base, prob_df, [best_ml_name, "dl_codebert"])
    print(f"  {best_ml_name} + CodeBERT  →  "
          f"ROC={fmt(stack_2,'ROC-AUC')}  PR={fmt(stack_2,'PR-AUC')}  F1={fmt(stack_2,'F1')}")

    all_ml = [c for c in prob_df.columns if c.startswith("ml_")]
    stack_all_ml_cb = run_stacking(base, prob_df, all_ml + ["dl_codebert"])
    print(f"  All ML + CodeBERT           →  "
          f"ROC={fmt(stack_all_ml_cb,'ROC-AUC')}  PR={fmt(stack_all_ml_cb,'PR-AUC')}  F1={fmt(stack_all_ml_cb,'F1')}")

    all_cols = list(prob_df.columns)
    stack_full = run_stacking(base, prob_df, all_cols)
    print(f"  All ML + All DL             →  "
          f"ROC={fmt(stack_full,'ROC-AUC')}  PR={fmt(stack_full,'PR-AUC')}  F1={fmt(stack_full,'F1')}")

    stack_vote_cb = None
    if "ml_Vote" in prob_df.columns:
        stack_vote_cb = run_stacking(base, prob_df, ["ml_Vote", "dl_codebert"])
        print(f"  ml_Vote + CodeBERT          →  "
              f"ROC={fmt(stack_vote_cb,'ROC-AUC')}  PR={fmt(stack_vote_cb,'PR-AUC')}  F1={fmt(stack_vote_cb,'F1')}")

    # --- Summary ---
    print(f"\n{'─'*70}")
    print("  SUMMARY")
    print("─" * 70)
    cb = ind[ind["Model"] == "dl_codebert"].iloc[0]
    configs = [
        (f"Best ML ({best_ml_name})", best_ml_row),
        ("CodeBERT", cb),
        (f"Weighted avg (α≈{wt['alpha_mean']:.2f})", wt),
        (f"Stack: {best_ml_name}+CB", stack_2),
        ("Stack: All ML+CB", stack_all_ml_cb),
        ("Stack: All ML+All DL", stack_full),
    ]
    for label, d in configs:
        print(f"  {label:35s}  ROC={fmt(d,'ROC-AUC')}  PR={fmt(d,'PR-AUC')}  F1={fmt(d,'F1')}")

    return {
        "individual": ind,
        "best_weighted": wt,
        "best_weighted_vote": wt_vote,
        "stack_best_ml_cb": stack_2,
        "stack_vote_cb": stack_vote_cb,
        "stack_all_ml_cb": stack_all_ml_cb,
        "stack_full": stack_full,
    }


def _serialize(obj):
    """Convert per_fold_score dict / DataFrame to plain-JSON friendly."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["legacy", "strict"], default="legacy")
    parser.add_argument(
        "--datasets", nargs="+", default=["merged", "dappscan"],
        help="Subset of datasets to analyse",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional JSON output path. Defaults to "
             "ml_results/ensemble_results_{protocol}.json",
    )
    args = parser.parse_args()

    results = {}
    for ds in args.datasets:
        results[ds] = analyse_dataset(ds, protocol=args.protocol)
    print("\n\nDone.")

    # Write JSON summary so the orchestrator can pick it up
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out) if args.out
        else RESULTS_DIR / f"ensemble_results_{args.protocol}.json"
    )
    serial = {}
    for ds, r in results.items():
        if r is None:
            continue
        serial[ds] = {k: _serialize(v) for k, v in r.items()}
    out_path.write_text(json.dumps(serial, indent=2, default=str))
    print(f"  Saved ensemble summary → {out_path}")
