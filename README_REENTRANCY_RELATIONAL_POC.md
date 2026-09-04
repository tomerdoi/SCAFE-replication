# Relational Reentrancy PoC

This is an isolated, compilation-free proof of concept for improving
reentrancy detection with relational source-code features. It is experimental
and is not part of SCAFE's primary feature inventory, reported manuscript
results, or production scoring pipeline.

## Scope

The PoC represents function-level state effects, external interactions, target
provenance, guards, and internal-call reachability. It produces 28 numeric
`rel_*` features, including:

- untrusted and value-bearing external calls;
- direct and transitive call-to-state-write ordering;
- state-write-to-untrusted-callback paths;
- reentrancy-guard coverage;
- calls in loops and multi-call entry points.

It can recursively read local Solidity imports, but it is deliberately an
approximation. It does not resolve overloads, dynamic dispatch, storage
aliases, or external dependency imports.

## PoC files

| File | Purpose |
| --- | --- |
| `src/ml/reentrancy_relational_features.py` | Standalone source/import reader and relational feature extractor. |
| `src/ml/reentrancy_relational_eval.py` | PoC feature caching, exact-source canonicalization, OVR LODO variants, and metric summaries. |
| `scripts/run_reentrancy_relational_poc.py` | Command-line runner for sampled pilots or a full PoC evaluation. |
| `tests/test_reentrancy_relational_features.py` | Extractor tests for direct/transitive paths, guards, receive functions, storage targets, comments, and imports. |
| `tests/test_reentrancy_relational_eval.py` | Cache identity/versioning and exact-source conflict handling tests. |

The modules only reuse generic utilities from `src/ml/baseline_comparison.py`
for dataset loading, taxonomy normalization, duplicate exclusion, model fitting,
and metrics. They do not modify `src/risk/contract_risk.py`, the 257-feature
primary SCAFE extractor, or manuscript artifacts.

## Run a directional pilot

```bash
python3 scripts/run_reentrancy_relational_poc.py \
  --models rf hgb \
  --seeds 42 43 44 \
  --sample-per-class 1000
```

The sampled run uses at most the specified number of reentrancy-labelled and
non-reentrancy rows from each selected dataset. It is a directional experiment:
the balanced sample changes prevalence, so its absolute PR-AUC, F1, precision,
and recall must not be presented as corpus-level results.

The runner evaluates three explicit variants:

1. `structural_only`: current 257 source-derived structural features;
2. `relational_only`: the 28 PoC relational features;
3. `structural_plus_relational`: direct feature concatenation.

The threshold is fixed at 0.5 unless `--calibrate-threshold` is used. That
option calibrates the threshold using training data only.

## Generated artifacts

By default, outputs are written under:

`data/derived/ml_results/reentrancy_relational_poc/`

For a sampled run, outputs are in:

`data/derived/ml_results/reentrancy_relational_poc/sample_<N>_per_class/`

| Artifact | Contents |
| --- | --- |
| `relational_feature_cache.csv` | Cached per-contract relational features with strict row identity/version validation. |
| `metrics.csv` | Per-seed, per-dataset, per-model, per-variant metrics. |
| `summary.csv` | Mean and standard deviation across seeds, with deltas from structural-only. |
| `predictions.csv` | Per-contract predictions for all variants, models, and seeds. |
| `protocol.json` | Reproducibility metadata and feature/protocol definition. |

An interrupted feature-extraction pass writes an append-only
`relational_feature_cache.csv.partial.jsonl` file. A compatible rerun resumes
from completed contract rows; successful cache creation removes it.

Generated artifacts are excluded from source-control by default unless a
separate decision is made to publish a specific, reviewed artifact set.

## Current evidence and next gate

The broader directional pilot used 5,473 contracts, RF and HGB, and seeds
42--44. Direct concatenation produced a small mean ROC-AUC gain (+0.009) but
was mixed across datasets. An exploratory 80% structural / 20% relational
late-fusion diagnostic improved ROC-AUC in 23 of 24 seed-level comparisons and
PR-AUC in 20 of 24.

That blend weight was selected after inspecting the pilot, so it is not an
unbiased result and must not be reported as primary evidence. The next valid
gate is a train-only selection of any fusion weight, followed by repeated
duplicate-safe LODO on the full corpus and a manual label-quality audit.

## Validation

```bash
python3 -m pytest \
  tests/test_reentrancy_relational_features.py \
  tests/test_reentrancy_relational_eval.py
```
