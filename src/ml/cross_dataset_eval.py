"""
Cross-dataset evaluation for vulnerability classification.

Experiments:
  A. Train on DAppSCAN → predict on FORGE & AccessControlVulnerabilities
  B. Train on FORGE (vulnerable vs unknown) → predict on DAppSCAN
  C. Merged training (DAppSCAN + FORGE) with GroupKFold CV
  D. Per-dataset CV (DAppSCAN CV, FORGE CV separately)

Usage:
    /path/to/venv/bin/python -m src.ml.cross_dataset_eval
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.features import load_scored_csv, select_feature_columns, FEATURE_SETS, _EXCLUDE
from src.ml.models import get_all_models

logger = logging.getLogger(__name__)

_OUT_DIR = _REPO_ROOT / "data" / "derived" / "ml_results"

BEST_MODELS = ["rf", "xgb", "gbc", "hgb"]


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 199)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0.0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def _prepare_features(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _get_groups(df: pd.DataFrame) -> np.ndarray:
    group_key = df["contract_id"].fillna("").astype(str)
    unique_groups = sorted(group_key.unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    return group_key.map(group_map).values.astype(int)


def _train_and_predict(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    scale: bool = True,
):
    """Train a model and return predicted probabilities on X_test."""
    all_models = get_all_models()
    est = all_models[model_name].build()

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if hasattr(est, "scale_pos_weight"):
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        est.set_params(scale_pos_weight=max(1.0, n_neg / max(1, n_pos)))

    est.fit(X_train, y_train)

    if hasattr(est, "predict_proba"):
        y_prob = est.predict_proba(X_test)[:, 1]
    elif hasattr(est, "decision_function"):
        y_prob = est.decision_function(X_test)
    else:
        y_prob = est.predict(X_test).astype(float)

    return y_prob, est


def _eval_metrics(y_true, y_prob, dataset_name=""):
    """Compute all metrics and return as dict."""
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    opt_t, opt_f1 = _optimal_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= opt_t).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "prev": float(y_true.mean()),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "f1": round(opt_f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(opt_t, 4),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def experiment_A_cross_predict(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Experiment A: Train on full DAppSCAN → predict on FORGE & AccessControl.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT A: Train on DAppSCAN → Predict on FORGE & AccessControl")
    logger.info("=" * 80)

    # Training set: DAppSCAN (clean vs vulnerable)
    train_df = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    train_df["y"] = (train_df["label_status"] == "vulnerable").astype(int)
    X_train = _prepare_features(train_df, feat_cols)
    y_train = train_df["y"].values

    logger.info(f"  Train: DAppSCAN {len(train_df)} samples (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")

    # Test set 1: FORGE (vulnerable vs unknown)
    forge_df = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"].isin(["vulnerable", "unknown"]))
    ].copy()
    forge_df["y"] = (forge_df["label_status"] == "vulnerable").astype(int)
    X_forge = _prepare_features(forge_df, feat_cols)
    y_forge = forge_df["y"].values

    # Test set 1b: FORGE all (vulnerable + suspicious vs unknown)
    forge_all_df = df[df["source_dataset"] == "FORGE"].copy()
    forge_all_df["y"] = (forge_all_df["label_status"].isin(["vulnerable", "suspicious"])).astype(int)
    X_forge_all = _prepare_features(forge_all_df, feat_cols)
    y_forge_all = forge_all_df["y"].values

    # Test set 2: AccessControlVulnerabilities (all vulnerable, no negatives)
    ac_df = df[df["source_dataset"] == "AccessControlVulnerabilities"].copy()
    X_ac = _prepare_features(ac_df, feat_cols)

    logger.info(f"  Test FORGE (vuln vs unknown): {len(forge_df)} (pos={y_forge.sum()}, neg={len(y_forge)-y_forge.sum()})")
    logger.info(f"  Test FORGE (all): {len(forge_all_df)} (pos={y_forge_all.sum()}, neg={len(y_forge_all)-y_forge_all.sum()})")
    logger.info(f"  Test AccessControl: {len(ac_df)} (all vulnerable)")

    rows = []
    for model_name in BEST_MODELS:
        logger.info(f"\n  Training {model_name} on DAppSCAN...")
        y_prob_forge, est = _train_and_predict(model_name, X_train, y_train, X_forge)
        y_prob_forge_all = est.predict_proba(X_forge_all)[:, 1] if hasattr(est, "predict_proba") else est.predict(X_forge_all).astype(float)
        y_prob_ac = est.predict_proba(X_ac)[:, 1] if hasattr(est, "predict_proba") else est.predict(X_ac).astype(float)

        # Re-apply scaler to forge_all and AC (need to redo with scaler)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        if hasattr(est, "scale_pos_weight"):
            est2 = get_all_models()[model_name].build()
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            est2.set_params(scale_pos_weight=max(1.0, n_neg / max(1, n_pos)))
            est2.fit(X_train_s, y_train)
        else:
            est2 = get_all_models()[model_name].build()
            est2.fit(X_train_s, y_train)

        X_forge_s = scaler.transform(X_forge)
        X_forge_all_s = scaler.transform(X_forge_all)
        X_ac_s = scaler.transform(X_ac)

        if hasattr(est2, "predict_proba"):
            y_prob_forge = est2.predict_proba(X_forge_s)[:, 1]
            y_prob_forge_all = est2.predict_proba(X_forge_all_s)[:, 1]
            y_prob_ac = est2.predict_proba(X_ac_s)[:, 1]
        else:
            y_prob_forge = est2.predict(X_forge_s).astype(float)
            y_prob_forge_all = est2.predict(X_forge_all_s).astype(float)
            y_prob_ac = est2.predict(X_ac_s).astype(float)

        # Metrics on FORGE (vuln vs unknown)
        m = _eval_metrics(y_forge, y_prob_forge, "FORGE (vuln_vs_unknown)")
        m["experiment"] = "A_cross_predict"
        m["model"] = model_name
        m["train_on"] = "DAppSCAN"
        logger.info(f"    → FORGE (vuln vs unknown): ROC={m['roc_auc']}, PR={m['pr_auc']}, F1={m['f1']}, Prec={m['precision']}, Rec={m['recall']}")
        rows.append(m)

        # Metrics on FORGE (all: vuln+suspicious vs unknown)
        m2 = _eval_metrics(y_forge_all, y_prob_forge_all, "FORGE (all)")
        m2["experiment"] = "A_cross_predict"
        m2["model"] = model_name
        m2["train_on"] = "DAppSCAN"
        logger.info(f"    → FORGE (all):             ROC={m2['roc_auc']}, PR={m2['pr_auc']}, F1={m2['f1']}, Prec={m2['precision']}, Rec={m2['recall']}")
        rows.append(m2)

        # AccessControl: recall only (all positive)
        opt_t, _ = _optimal_f1_threshold(y_forge, y_prob_forge)
        ac_preds = (y_prob_ac >= opt_t).astype(int)
        ac_recall = ac_preds.mean()
        ac_mean_prob = y_prob_ac.mean()
        logger.info(f"    → AccessControl (94 vuln): recall={ac_recall:.3f} (mean_prob={ac_mean_prob:.3f}, threshold={opt_t:.3f})")
        rows.append({
            "experiment": "A_cross_predict",
            "model": model_name,
            "train_on": "DAppSCAN",
            "dataset": "AccessControlVuln",
            "n_samples": len(ac_df),
            "n_pos": len(ac_df),
            "n_neg": 0,
            "prev": 1.0,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": round(ac_recall, 4),
            "threshold": round(opt_t, 4),
            "TP": int(ac_preds.sum()),
            "FN": int(len(ac_df) - ac_preds.sum()),
            "FP": 0,
            "TN": 0,
            "ac_mean_prob": round(ac_mean_prob, 4),
        })

    return pd.DataFrame(rows)


def experiment_B_reverse(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Experiment B: Train on FORGE (vulnerable vs unknown) → predict on DAppSCAN.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT B: Train on FORGE → Predict on DAppSCAN")
    logger.info("=" * 80)

    # Training set: FORGE (vulnerable vs unknown, drop suspicious)
    train_df = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"].isin(["vulnerable", "unknown"]))
    ].copy()
    train_df["y"] = (train_df["label_status"] == "vulnerable").astype(int)
    X_train = _prepare_features(train_df, feat_cols)
    y_train = train_df["y"].values

    # Test set: DAppSCAN
    test_df = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    test_df["y"] = (test_df["label_status"] == "vulnerable").astype(int)
    X_test = _prepare_features(test_df, feat_cols)
    y_test = test_df["y"].values

    logger.info(f"  Train: FORGE {len(train_df)} (pos={y_train.sum()}, neg={len(y_train)-y_train.sum()})")
    logger.info(f"  Test: DAppSCAN {len(test_df)} (pos={y_test.sum()}, neg={len(y_test)-y_test.sum()})")

    rows = []
    for model_name in BEST_MODELS:
        logger.info(f"\n  Training {model_name} on FORGE...")
        y_prob, _ = _train_and_predict(model_name, X_train, y_train, X_test)
        m = _eval_metrics(y_test, y_prob, "DAppSCAN")
        m["experiment"] = "B_reverse"
        m["model"] = model_name
        m["train_on"] = "FORGE"
        logger.info(f"    → DAppSCAN: ROC={m['roc_auc']}, PR={m['pr_auc']}, F1={m['f1']}, Prec={m['precision']}, Rec={m['recall']}")
        rows.append(m)

    return pd.DataFrame(rows)


def experiment_C_merged_cv(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Experiment C: Merge DAppSCAN + FORGE, GroupKFold CV.
    DAppSCAN: clean=0, vulnerable=1
    FORGE: unknown=0, vulnerable=1 (drop suspicious)
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT C: Merged (DAppSCAN + FORGE) with GroupKFold CV")
    logger.info("=" * 80)

    dappscan_df = df[
        (df["source_dataset"] == "DAppSCAN") &
        (df["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    dappscan_df["y"] = (dappscan_df["label_status"] == "vulnerable").astype(int)

    forge_df = df[
        (df["source_dataset"] == "FORGE") &
        (df["label_status"].isin(["vulnerable", "unknown"]))
    ].copy()
    forge_df["y"] = (forge_df["label_status"] == "vulnerable").astype(int)

    merged = pd.concat([dappscan_df, forge_df], ignore_index=True)
    X = _prepare_features(merged, feat_cols)
    y = merged["y"].values
    groups = _get_groups(merged)

    logger.info(f"  Merged: {len(merged)} samples (pos={y.sum()}, neg={len(y)-y.sum()}, prev={y.mean():.3f})")
    logger.info(f"  DAppSCAN: {len(dappscan_df)} | FORGE: {len(forge_df)}")

    n_folds = 5
    n_groups = len(np.unique(groups))
    if n_groups < n_folds:
        logger.warning(f"  Only {n_groups} groups, falling back to StratifiedKFold")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(X, y, groups)

    rows = []
    for model_name in BEST_MODELS:
        logger.info(f"\n  {model_name} — {n_folds}-fold GroupKFold on merged data...")
        fold_metrics = []

        if n_groups < n_folds:
            split_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y)
        else:
            split_iter = GroupKFold(n_splits=n_folds).split(X, y, groups)

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_prob, _ = _train_and_predict(model_name, X_train, y_train, X_test)
            fm = _eval_metrics(y_test, y_prob, f"merged_fold{fold_idx}")
            fold_metrics.append(fm)

        roc_aucs = [fm["roc_auc"] for fm in fold_metrics]
        pr_aucs = [fm["pr_auc"] for fm in fold_metrics]
        f1s = [fm["f1"] for fm in fold_metrics]
        precs = [fm["precision"] for fm in fold_metrics]
        recs = [fm["recall"] for fm in fold_metrics]
        tps = sum(fm["TP"] for fm in fold_metrics)
        fns = sum(fm["FN"] for fm in fold_metrics)
        fps = sum(fm["FP"] for fm in fold_metrics)
        tns = sum(fm["TN"] for fm in fold_metrics)

        row = {
            "experiment": "C_merged_cv",
            "model": model_name,
            "train_on": "DAppSCAN+FORGE",
            "dataset": "merged",
            "n_samples": len(merged),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
            "prev": round(float(y.mean()), 4),
            "roc_auc": round(float(np.mean(roc_aucs)), 4),
            "roc_auc_std": round(float(np.std(roc_aucs)), 4),
            "pr_auc": round(float(np.mean(pr_aucs)), 4),
            "pr_auc_std": round(float(np.std(pr_aucs)), 4),
            "f1": round(float(np.mean(f1s)), 4),
            "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4),
            "TP": tps, "FN": fns, "FP": fps, "TN": tns,
        }
        logger.info(
            f"    → merged CV: ROC={row['roc_auc']}±{row.get('roc_auc_std','')}, "
            f"PR={row['pr_auc']}, F1={row['f1']}, Prec={row['precision']}, Rec={row['recall']}"
        )
        rows.append(row)

    return pd.DataFrame(rows)


def experiment_D_per_dataset_cv(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Experiment D: Separate CV on each dataset for direct comparison.
    - DAppSCAN: clean vs vulnerable (5-fold GroupKFold)
    - FORGE: vulnerable vs unknown (5-fold GroupKFold)
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT D: Per-dataset CV (DAppSCAN and FORGE separately)")
    logger.info("=" * 80)

    rows = []
    for ds_name, pos_label, neg_labels in [
        ("DAppSCAN", "vulnerable", ["clean"]),
        ("FORGE", "vulnerable", ["unknown"]),
    ]:
        sub = df[
            (df["source_dataset"] == ds_name) &
            (df["label_status"].isin([pos_label] + neg_labels))
        ].copy()
        sub["y"] = (sub["label_status"] == pos_label).astype(int)
        X = _prepare_features(sub, feat_cols)
        y = sub["y"].values
        groups = _get_groups(sub)

        n_folds = 5
        n_groups = len(np.unique(groups))
        logger.info(f"\n  {ds_name}: {len(sub)} samples (pos={y.sum()}, neg={len(y)-y.sum()}, prev={y.mean():.3f}, groups={n_groups})")

        for model_name in BEST_MODELS:
            fold_metrics = []

            if n_groups < n_folds:
                split_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y)
            else:
                split_iter = GroupKFold(n_splits=n_folds).split(X, y, groups)

            for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                y_prob, _ = _train_and_predict(model_name, X_train, y_train, X_test)
                fm = _eval_metrics(y_test, y_prob, f"{ds_name}_fold{fold_idx}")
                fold_metrics.append(fm)

            roc_aucs = [fm["roc_auc"] for fm in fold_metrics]
            pr_aucs = [fm["pr_auc"] for fm in fold_metrics]
            f1s = [fm["f1"] for fm in fold_metrics]
            precs = [fm["precision"] for fm in fold_metrics]
            recs = [fm["recall"] for fm in fold_metrics]
            tps = sum(fm["TP"] for fm in fold_metrics)
            fns = sum(fm["FN"] for fm in fold_metrics)
            fps = sum(fm["FP"] for fm in fold_metrics)
            tns = sum(fm["TN"] for fm in fold_metrics)

            row = {
                "experiment": "D_per_dataset_cv",
                "model": model_name,
                "train_on": ds_name,
                "dataset": ds_name,
                "n_samples": len(sub),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
                "prev": round(float(y.mean()), 4),
                "roc_auc": round(float(np.mean(roc_aucs)), 4),
                "roc_auc_std": round(float(np.std(roc_aucs)), 4),
                "pr_auc": round(float(np.mean(pr_aucs)), 4),
                "pr_auc_std": round(float(np.std(pr_aucs)), 4),
                "f1": round(float(np.mean(f1s)), 4),
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "TP": tps, "FN": fns, "FP": fps, "TN": tns,
            }
            logger.info(
                f"    {model_name:5s} → {ds_name}: ROC={row['roc_auc']}±{row.get('roc_auc_std','')}, "
                f"PR={row['pr_auc']}, F1={row['f1']}, Prec={row['precision']}, Rec={row['recall']}"
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scored_csv()
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    feat_cols = select_feature_columns(df, FEATURE_SETS["all"])
    logger.info(f"Feature columns: {len(feat_cols)}")

    results = []

    # Experiment A
    df_a = experiment_A_cross_predict(df, feat_cols)
    results.append(df_a)

    # Experiment B
    df_b = experiment_B_reverse(df, feat_cols)
    results.append(df_b)

    # Experiment C
    df_c = experiment_C_merged_cv(df, feat_cols)
    results.append(df_c)

    # Experiment D
    df_d = experiment_D_per_dataset_cv(df, feat_cols)
    results.append(df_d)

    combined = pd.concat(results, ignore_index=True)
    out_path = _OUT_DIR / "cross_dataset_evaluation.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"\nWrote: {out_path}")

    # Print summary
    print("\n" + "=" * 120)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 120)

    for exp_name in ["A_cross_predict", "B_reverse", "C_merged_cv", "D_per_dataset_cv"]:
        sub = combined[combined["experiment"] == exp_name]
        if sub.empty:
            continue
        print(f"\n{'─' * 100}")
        print(f"  {exp_name}")
        print(f"{'─' * 100}")
        print(f"  {'Model':<8} {'Train On':<16} {'Test On':<25} {'N':>6} {'Pos':>5} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>7} {'Prec':>7} {'Rec':>7} {'TP':>5} {'FN':>5} {'FP':>5} {'TN':>6}")
        for _, row in sub.iterrows():
            roc_str = f"{row['roc_auc']:.4f}" if not pd.isna(row["roc_auc"]) else "  N/A"
            pr_str = f"{row['pr_auc']:.4f}" if not pd.isna(row["pr_auc"]) else "  N/A"
            f1_str = f"{row['f1']:.4f}" if not pd.isna(row["f1"]) else "  N/A"
            prec_str = f"{row['precision']:.4f}" if not pd.isna(row["precision"]) else "  N/A"
            rec_str = f"{row['recall']:.4f}" if not pd.isna(row["recall"]) else "  N/A"
            print(
                f"  {row['model']:<8} {row.get('train_on',''):<16} {row['dataset']:<25} "
                f"{row['n_samples']:>6} {row['n_pos']:>5} "
                f"{roc_str:>9} {pr_str:>9} {f1_str:>7} {prec_str:>7} {rec_str:>7} "
                f"{row['TP']:>5} {row['FN']:>5} {row['FP']:>5} {row['TN']:>6}"
            )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    logger.info(f"Total cross-dataset evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
