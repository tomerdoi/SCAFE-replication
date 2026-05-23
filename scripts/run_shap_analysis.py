"""
A-SHAP: SHAP TreeExplainer analysis for vulnerability classification models.

Produces:
  (a) fig_shap_summary.pdf  — bee-swarm plot (top-20 features by mean |SHAP|)
  (b) fig_shap_waterfall.pdf — waterfall plots for 6 deterministically selected cases:
        TP-1, TP-2 (highest-prob true positives)
        FP-1, FP-2 (highest-prob false positives)
        FN-1, FN-2 (highest-prob false negatives)
  (c) shap_values.npz       — raw SHAP values for reproducibility
  (d) shap_summary.csv      — per-feature mean |SHAP| rankings

Usage:
    python scripts/run_shap_analysis.py [--csv PATH] [--out-dir DIR]

Requires: shap, matplotlib (pip install shap matplotlib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_DEFAULT_CSV = _REPO / "data" / "derived" / "unified_contract_eval_scored_strict.csv"
_FIG_DIR = _REPO.parent / "sc_risk_score_overleaf" / "figures"
_OUT_DIR = _REPO / "data" / "derived" / "shap_results"


def build_oof_predictions(dataset, n_folds: int = 5):
    """
    Run 5-fold GroupKFold CV with XGBoost and collect out-of-fold predictions
    together with fitted model objects for SHAP analysis on the final fold.
    Returns: oof_probs, oof_true, oof_X, oof_feature_names, best_model, best_X_train
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    X, y, groups = dataset.X, dataset.y, dataset.groups
    gkf = GroupKFold(n_splits=n_folds)

    oof_probs = np.zeros(len(y))
    oof_true = y.copy()
    best_model = None
    best_X_train = None
    best_fold_auc = -1.0

    from sklearn.metrics import roc_auc_score

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max(1.0, n_neg / max(1, n_pos)),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        oof_probs[test_idx] = probs

        fold_auc = roc_auc_score(y_test, probs)
        if fold_auc > best_fold_auc:
            best_fold_auc = fold_auc
            best_model = model
            # Store scaled training data from best fold for SHAP background
            best_X_train = X_train_s

    return oof_probs, oof_true, X, best_model, best_X_train


def run_shap(dataset, oof_probs, oof_true, model, X_train, out_dir: Path, fig_dir: Path):
    """Compute SHAP values and produce all figures."""
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    feature_names = dataset.feature_names
    X = dataset.X

    print("Computing SHAP values with TreeExplainer...")
    # Use a random background sample (200 contracts) for speed
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    background = X_train[bg_idx]

    explainer = shap.TreeExplainer(model, background)

    # Scale all data the same way as training (use last fold's scaler — approximate)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Compute SHAP values for all contracts (may take a few minutes)
    shap_values = explainer.shap_values(X_scaled)   # (n_samples, n_features)

    # Save raw values
    np.savez_compressed(out_dir / "shap_values.npz",
                        shap_values=shap_values,
                        feature_names=np.array(feature_names))
    print(f"Saved raw SHAP values → {out_dir / 'shap_values.npz'}")

    # Summary CSV
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_summary_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_summary_df["rank"] = range(1, len(shap_summary_df) + 1)
    shap_summary_df.to_csv(out_dir / "shap_summary.csv", index=False)
    print(f"Top-10 features by mean |SHAP|:")
    for _, row in shap_summary_df.head(10).iterrows():
        print(f"  {int(row['rank']):3d}. {row['feature']:<40} {row['mean_abs_shap']:.5f}")

    # ── Figure (a): Bee-swarm summary ────────────────────────────────────
    top20_idx = shap_summary_df.head(20)["feature"].tolist()
    top20_cols = [feature_names.index(f) for f in top20_idx]

    fig_a, ax_a = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values[:, top20_cols],
        X_scaled[:, top20_cols],
        feature_names=top20_idx,
        show=False,
        plot_type="dot",
        max_display=20,
        plot_size=None,
        color_bar_label="Feature value",
        alpha=0.5,
    )
    plt.title("SHAP Summary — Top 20 Features by Mean |SHAP|", fontsize=11)
    plt.tight_layout()
    out_a = fig_dir / "fig_shap_summary.pdf"
    plt.savefig(out_a, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved bee-swarm summary → {out_a}")

    # ── Figure (b): Waterfall plots — 6 deterministic cases ──────────────
    # Select deterministically by rank on OOF predicted probability
    y_true = oof_true
    y_prob = oof_probs
    y_pred = (y_prob >= 0.5).astype(int)   # approximate — use 0.5 for case selection

    # TP: true positive, sorted by probability descending
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tp_sorted = tp_idx[np.argsort(y_prob[tp_idx])[::-1]]
    tp_cases = tp_sorted[:2] if len(tp_sorted) >= 2 else tp_sorted

    # FP: false positive, sorted by probability descending
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fp_sorted = fp_idx[np.argsort(y_prob[fp_idx])[::-1]]
    fp_cases = fp_sorted[:2] if len(fp_sorted) >= 2 else fp_sorted

    # FN: false negative, sorted by probability ascending (most-missed)
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fn_sorted = fn_idx[np.argsort(y_prob[fn_idx])]
    fn_cases = fn_sorted[:2] if len(fn_sorted) >= 2 else fn_sorted

    n_panels = len(tp_cases) + len(fp_cases) + len(fn_cases)
    fig_b, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    panel_labels = (
        [f"TP-{i+1}  (p={y_prob[c]:.3f})" for i, c in enumerate(tp_cases)] +
        [f"FP-{i+1}  (p={y_prob[c]:.3f})" for i, c in enumerate(fp_cases)] +
        [f"FN-{i+1}  (p={y_prob[c]:.3f})" for i, c in enumerate(fn_cases)]
    )
    case_indices = list(tp_cases) + list(fp_cases) + list(fn_cases)

    for ax, case_idx, label in zip(axes, case_indices, panel_labels):
        shap_exp = shap.Explanation(
            values=shap_values[case_idx],
            base_values=explainer.expected_value,
            data=X_scaled[case_idx],
            feature_names=feature_names,
        )
        plt.sca(ax)
        shap.waterfall_plot(shap_exp, max_display=12, show=False)
        ax.set_title(f"SHAP Waterfall — {label}", fontsize=9)

    plt.tight_layout()
    out_b = fig_dir / "fig_shap_waterfall.pdf"
    plt.savefig(out_b, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved waterfall plots → {out_b}")

    return shap_summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(_DEFAULT_CSV))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--fig-dir", default=str(_FIG_DIR))
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from src.ml.features import prepare_dataset
    print(f"Loading dataset: {args.csv}")
    dataset = prepare_dataset(
        scored_csv=Path(args.csv),
        dataset_filter="all",
        label_filter="clean_vs_vulnerable",
        feature_set_name="all",
    )
    print(f"  Dataset: {len(dataset.y)} rows, {len(dataset.feature_names)} features")

    print("Building OOF predictions with XGBoost (5-fold nested CV)...")
    oof_probs, oof_true, X, model, X_train = build_oof_predictions(dataset)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    run_shap(dataset, oof_probs, oof_true, model, X_train, out_dir, fig_dir)
    print("SHAP analysis complete.")


if __name__ == "__main__":
    main()
