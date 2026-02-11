"""
End-to-end ML training and evaluation pipeline.

Orchestrates:
1. Load scored data → prepare feature matrix
2. Evaluate rule-based baseline (risk_z)
3. Train & evaluate all ML models with GroupKFold CV
4. Optionally run ablation (static-only, components-only feature sets)
5. Save comparison table, feature importances, and per-fold predictions

Quick run:
    /path/to/venv/bin/python -m src.ml.train_pipeline

All outputs go to data/derived/ml_results/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure repo root is on path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.features import (  # noqa: E402
    FEATURE_SETS,
    PreparedDataset,
    load_scored_csv,
    prepare_dataset,
)
from src.ml.models import get_all_models, ModelConfig  # noqa: E402
from src.ml.evaluation import (  # noqa: E402
    ModelResult,
    build_comparison_table,
    evaluate_baseline,
    evaluate_model,
)

logger = logging.getLogger(__name__)

_OUT_DIR = _REPO_ROOT / "data" / "derived" / "ml_results"


@dataclass
class PipelineConfig:
    """Configuration for a training run."""
    # Data
    scored_csv: Optional[Path] = None
    dataset_filter: str = "DAppSCAN"
    label_filter: str = "clean_vs_vulnerable"
    # Models
    model_names: Optional[List[str]] = None   # None = all available
    # Evaluation
    n_folds: int = 5
    scale_features: bool = True
    # Ablation: run on multiple feature sets?
    feature_sets: Optional[List[str]] = None  # None = ["all"] only
    # Output
    out_dir: Path = _OUT_DIR


def run_pipeline(config: PipelineConfig) -> Dict[str, List[ModelResult]]:
    """
    Execute the full training + evaluation pipeline.

    Returns
    -------
    dict mapping feature_set_name → list of ModelResult (including baseline)
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data once ──
    df = load_scored_csv(config.scored_csv)

    # ── Resolve which feature sets to evaluate ──
    feature_set_names = config.feature_sets or ["all"]
    for fs_name in feature_set_names:
        if fs_name not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {fs_name!r}. Available: {list(FEATURE_SETS)}")

    # ── Resolve which models to evaluate ──
    all_models = get_all_models()
    if config.model_names:
        model_configs = {k: all_models[k] for k in config.model_names if k in all_models}
    else:
        model_configs = all_models

    logger.info(f"Models to evaluate: {list(model_configs.keys())}")
    logger.info(f"Feature sets: {feature_set_names}")

    all_results: Dict[str, List[ModelResult]] = {}
    all_comparison_rows: List[pd.DataFrame] = []

    for fs_name in feature_set_names:
        logger.info(f"\n{'='*70}")
        logger.info(f"Feature set: {fs_name} ({FEATURE_SETS[fs_name].description})")
        logger.info(f"{'='*70}")

        # Prepare dataset with this feature set
        dataset = prepare_dataset(
            df=df,
            dataset_filter=config.dataset_filter,
            label_filter=config.label_filter,
            feature_set_name=fs_name,
        )

        results: List[ModelResult] = []

        # ── Baseline ──
        if "risk_z" in dataset.df.columns:
            logger.info("Evaluating baseline (risk_z)...")
            baseline = evaluate_baseline(dataset, score_column="risk_z")
            results.append(baseline)

        # ── ML models ──
        for model_name, model_cfg in model_configs.items():
            logger.info(f"Evaluating {model_cfg.display_name}...")
            try:
                result = evaluate_model(
                    model_cfg,
                    dataset,
                    n_folds=config.n_folds,
                    scale_features=config.scale_features,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"  FAILED: {model_name}: {type(e).__name__}: {e}")

        all_results[fs_name] = results

        # ── Save per-feature-set results ──
        comparison = build_comparison_table(results)
        comparison["feature_set"] = fs_name
        all_comparison_rows.append(comparison)

        # Save feature importance for each model
        for r in results:
            if r.feature_importance is not None:
                imp_path = config.out_dir / f"feature_importance_{fs_name}_{r.model_name}.csv"
                r.feature_importance.to_csv(imp_path, index=False)

    # ── Save combined comparison table ──
    combined = pd.concat(all_comparison_rows, ignore_index=True)
    combined_path = config.out_dir / "model_comparison.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"\nWrote comparison table: {combined_path}")

    # ── Save run metadata ──
    meta = {
        "dataset_filter": config.dataset_filter,
        "label_filter": config.label_filter,
        "n_folds": config.n_folds,
        "scale_features": config.scale_features,
        "feature_sets": feature_set_names,
        "models": list(model_configs.keys()),
    }
    meta_path = config.out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # ── Print summary ──
    _print_summary(combined, all_results)

    return all_results


def _print_summary(comparison: pd.DataFrame, all_results: Dict[str, List[ModelResult]]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON — sorted by ROC-AUC (descending)")
    print("=" * 90)
    print(f"{'Model':<30} {'Feature Set':<15} {'ROC-AUC':>12} {'PR-AUC':>12} "
          f"{'F1':>8} {'Prec':>8} {'Recall':>8} {'Time':>7}")
    print("-" * 90)

    for _, row in comparison.iterrows():
        roc_str = f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.3f}"
        pr_str = f"{row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.3f}"
        print(
            f"  {row['display_name']:<28} {row['feature_set']:<15} {roc_str:>12} {pr_str:>12} "
            f"{row['f1_mean']:>8.4f} {row['precision_mean']:>8.4f} {row['recall_mean']:>8.4f} "
            f"{row['train_time_s']:>6.1f}s"
        )

    print()

    # Top features from the best model
    for fs_name, results in all_results.items():
        ml_results = [r for r in results if r.family != "baseline" and r.feature_importance is not None]
        if not ml_results:
            continue
        best = max(ml_results, key=lambda r: r.roc_auc_mean)
        if best.feature_importance is not None:
            print(f"Top 15 features ({best.display_name}, feature_set={fs_name}):")
            for _, frow in best.feature_importance.head(15).iterrows():
                print(f"  {frow['rank']:>3}. {frow['feature']:<35} importance={frow['importance_mean']:.4f}")
            print()


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        dataset_filter="DAppSCAN",
        label_filter="clean_vs_vulnerable",
        n_folds=5,
        scale_features=True,
        # Run main + ablation
        feature_sets=["all", "static_only", "components_only"],
        model_names=None,  # all available
    )

    t0 = time.time()
    run_pipeline(config)
    elapsed = time.time() - t0
    logger.info(f"Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
