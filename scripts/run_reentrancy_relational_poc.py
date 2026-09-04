#!/usr/bin/env python3
"""Run the isolated relational-feature PoC for reentrancy detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.baseline_comparison import (
    DatasetSpec,
    add_canonical_vulnerability_types,
    load_scored_frame,
)
from src.ml.reentrancy_relational_eval import (
    canonicalize_reentrancy_source_rows,
    load_or_build_relational_features,
    run_reentrancy_relational_poc,
    summarize_poc_metrics,
    write_poc_metadata,
)


DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "unified_contract_eval_scored.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "ml_results"
    / "reentrancy_relational_poc"
)
DATASET_SPECS = {
    "DAppSCAN": DatasetSpec("DAppSCAN", group_by_project=True),
    "DIVE": DatasetSpec("DIVE"),
    "Empirical-Analysis": DatasetSpec("Empirical-Analysis"),
    # LISA-Bench is positive-only for the broad task, but has both classes for
    # reentrancy one-vs-rest because non-reentrancy findings act as controls.
    "LISA-Bench": DatasetSpec("LISA-Bench"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument("--models", nargs="+", default=["rf", "hgb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help=(
            "Run a directional pilot with at most this many reentrancy and "
            "non-reentrancy rows per dataset."
        ),
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Calibrate the decision threshold on training data only.",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Ignore an existing PoC feature cache and extract sources again.",
    )
    return parser.parse_args()


def _stratified_pilot_sample(
    frame,
    datasets: list[str],
    per_class: int,
    *,
    random_state: int,
):
    normalized = add_canonical_vulnerability_types(frame)
    normalized["_pilot_y"] = normalized["taxonomy_types"].map(
        lambda value: int("reentrancy" in str(value).split(";"))
    )
    parts = []
    for dataset in datasets:
        group = normalized[normalized["source_dataset"] == dataset]
        for _, label_group in group.groupby("_pilot_y"):
            parts.append(
                label_group.sample(
                    n=min(per_class, len(label_group)),
                    random_state=random_state,
                )
            )
    return pd.concat(parts, ignore_index=True).drop(columns="_pilot_y")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    if args.sample_per_class is not None:
        if args.sample_per_class < 10:
            raise ValueError("--sample-per-class must be at least 10")
        args.output_dir = (
            args.output_dir / f"sample_{args.sample_per_class}_per_class"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scored contracts from %s", args.input)
    frame = load_scored_frame(str(args.input))
    frame, source_audit = canonicalize_reentrancy_source_rows(frame)
    logger.info("Pre-LODO source audit: %s", source_audit)
    if args.sample_per_class is not None:
        frame = _stratified_pilot_sample(
            frame,
            args.datasets,
            args.sample_per_class,
            random_state=args.seeds[0],
        )
        logger.info(
            "Directional pilot selected %d rows (%d maximum per class/dataset)",
            len(frame),
            args.sample_per_class,
        )
    cache_path = args.output_dir / "relational_feature_cache.csv"
    logger.info(
        "%s relational feature cache %s",
        "Rebuilding" if args.rebuild_feature_cache else "Loading/building",
        cache_path,
    )
    frame, cache_reused = load_or_build_relational_features(
        frame,
        cache_path,
        rebuild=args.rebuild_feature_cache,
        progress_callback=lambda completed, total: logger.info(
            "Extracted relational features for %d/%d rows (%.1f%%)",
            completed,
            total,
            100.0 * completed / max(total, 1),
        ),
    )
    logger.info(
        "Relational features ready for %d rows (cache_reused=%s)",
        len(frame),
        cache_reused,
    )

    specs = [DATASET_SPECS[name] for name in args.datasets]
    metrics, predictions = run_reentrancy_relational_poc(
        frame,
        specs,
        model_names=args.models,
        seeds=args.seeds,
        calibrate_threshold=args.calibrate_threshold,
    )
    summary = summarize_poc_metrics(metrics)

    metrics_path = args.output_dir / "metrics.csv"
    predictions_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.csv"
    metadata_path = args.output_dir / "protocol.json"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_poc_metadata(
        metadata_path,
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        cache_reused=cache_reused,
        calibrate_threshold=args.calibrate_threshold,
    )
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", predictions_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
