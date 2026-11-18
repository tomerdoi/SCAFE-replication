"""Exact conditional sensitivity of two fixed rankings; never relabels a dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from paired_nuisance import TARGETS, SEEDS, NUISANCE_FEATURES, digest, load_targets, align

FRACTIONS = (.01, .05, .10)


def twice_midrank(scores):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("Nonempty finite score vector required")
    _, inverse, count = np.unique(s, return_inverse=True, return_counts=True)
    return (2 * np.cumsum(count) - count + 1)[inverse].astype(np.int64)


def extrema(y, full, nuisance):
    """Integer numerator extrema for at most j disjoint class-preserving swaps.

    For fixed P and Q, AUC(full)-AUC(nuisance) = sum(y*d)/(2*P*Q),
    where d is the difference between the two twice-midranks. At exactly j swaps,
    minimize by dropping the j largest positive d and adding j smallest negative d.
    Prefix minima/maxima give at-most-j extrema. This is elementary rank algebra,
    not a novel optimizer, inferred noise process, or correction recommendation.
    """
    labels = np.asarray(y)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Observed binary vector required")
    a = twice_midrank(full); b = twice_midrank(nuisance)
    if a.shape != labels.shape or b.shape != labels.shape:
        raise ValueError("Aligned vectors required")
    p = int(labels.sum()); q = len(labels) - p
    if not p or not q:
        raise ValueError("Both classes required")
    d = a - b
    if int(d.sum()) != 0:
        raise AssertionError("Rank conservation failed")
    positive = np.sort(d[labels == 1]); negative = np.sort(d[labels == 0])
    m = min(p, q); base = int(positive.sum())
    lower = np.r_[base, base + np.cumsum(negative[:m] - positive[::-1][:m])]
    upper = np.r_[base, base + np.cumsum(negative[::-1][:m] - positive[:m])]
    return {"n": len(labels), "p": p, "q": q, "denominator": 2*p*q,
            "base": base, "lower": np.minimum.accumulate(lower),
            "upper": np.maximum.accumulate(upper)}


def summarize(e):
    n, p, q = e["n"], e["p"], e["q"]
    denominator = e["denominator"]
    base = e["base"]

    def first(strict):
        if base == 0:
            return 0 if not strict else None
        curve = e["lower"] if base > 0 else e["upper"]
        hit = (curve < 0 if strict else curve <= 0) if base > 0 else (curve > 0 if strict else curve >= 0)
        index = np.flatnonzero(hit)
        return int(index[0]) if len(index) else None

    def cost(j):
        if j is None:
            return None
        return {"swaps": j, "changed_records": 2*j, "fraction_all_records": 2*j/n,
                "fraction_native_positives": j/p, "fraction_native_negatives": j/q}

    rows = []
    for fraction in FRACTIONS:
        j = min(int(np.floor(fraction*n/2)), p, q)
        rows.append({"requested_changed_record_fraction": fraction, **cost(j),
                     "difference_lower": int(e["lower"][j])/denominator,
                     "difference_upper": int(e["upper"][j])/denominator})
    return {"n_records": n, "n_positive": p, "n_negative": q,
            "native_difference": base/denominator, "integer_numerator": base,
            "integer_denominator": denominator, "budget_bounds": rows,
            "minimum_reach_zero_or_reverse": cost(first(False)),
            "minimum_strict_reversal": cost(first(True))}


def run(root, output, protocol_path):
    if output.exists():
        raise FileExistsError("Output must be new")
    protocol = json.loads(protocol_path.read_text()); frozen = digest(protocol_path)
    if protocol["version"] != "fixed-ranking-class-preserving-label-sensitivity-v1" or tuple(protocol["budgets_fraction_of_all_records_changed"]) != FRACTIONS:
        raise ValueError("Frozen protocol differs")
    base = root / "data/derived/sanitized_v1/ml_results"
    paths = [base/"primary_lodo/scafe_repeated_lodo_predictions.csv", base/"nuisance_baseline/predictions.csv"]
    hashes = [digest(p) for p in paths]
    if hashes != [protocol["inputs"]["full"], protocol["inputs"]["nuisance"]]:
        raise ValueError("Cache bytes changed")
    full_meta = json.loads((base/"primary_lodo/scafe_repeated_lodo.json").read_text())
    nuisance_meta = json.loads((base/"nuisance_baseline/protocol.json").read_text())
    if full_meta["input_sha256"] != nuisance_meta["input_sha256"] or full_meta["input_sha256"] != protocol["inputs"]["original_scored_input"] or full_meta["feature_count"] != 244 or full_meta["seeds"] != list(SEEDS) or nuisance_meta["seeds"] != list(SEEDS) or tuple(nuisance_meta["features"]) != NUISANCE_FEATURES:
        raise ValueError("Original feature/input/seed lineage differs")
    start = time.process_time(); paired = align(load_targets(paths[0]), load_targets(paths[1]))
    if {name: len(item["y"]) for name, item in paired.items()} != {"DIVE": 21939, "Empirical-Analysis": 2172}:
        raise ValueError("Pinned cohorts differ")
    results = []
    for name in TARGETS:
        if time.process_time() - start > 120:
            raise TimeoutError("CPU budget exceeded")
        item = paired[name]; y = item["y"]; full = item["full"].mean(axis=1); nuisance = item["nuisance"].mean(axis=1)
        result = summarize(extrema(y, full, nuisance))
        direct = float(roc_auc_score(y, full) - roc_auc_score(y, nuisance))
        if abs(direct - result["native_difference"]) > 1e-12:
            raise AssertionError("Native AUC/rank identity disagrees")
        control = summarize(extrema(y, full, full))
        if control["native_difference"] != 0 or any(r["difference_lower"] != 0 or r["difference_upper"] != 0 for r in control["budget_bounds"]):
            raise AssertionError("Identical-score control failed")
        result.update(dataset=name, ordered_identity_sha256=item["ordered_identity_sha256"], native_sklearn_difference=direct, identical_score_control=True)
        results.append(result)
    if hashes != [digest(p) for p in paths] or digest(protocol_path) != frozen:
        raise ValueError("Inputs changed during calculation")
    report = {"version": protocol["version"], "protocol_sha256": frozen, "cache_sha256": hashes,
              "results": results, "cpu_seconds": time.process_time()-start,
              "role": "Exact hypothetical sensitivity conditional on fixed inspected score/label cohorts; not actual label errors, correction truth, noise estimate or independent confirmation",
              "alternative_labels_or_record_ids_persisted": False,
              "new_fits_or_dataset_changes": 0, "dappscan_outcomes_used": False}
    output.mkdir(); (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True); args = p.parse_args()
    print(json.dumps(run(args.root, args.output, args.protocol), indent=2))
