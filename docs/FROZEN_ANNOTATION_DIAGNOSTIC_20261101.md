# Frozen annotation sensitivity diagnostic — 1 November 2026

## Question and fixed protocol

Motivated by the paired lexical-cue experiments in Susan et al. (2026 extended
preprint, PDF pp.4–7), this diagnostic asks whether a newly hash-bound frozen
function-pattern checkpoint changes outputs when DAppSCAN annotation comments
are removed. It does not test vulnerability detection accuracy. See
`SCAFE_REFERENCE_REVIEW_20261101.md` for critically reviewed context.

The sample was fixed before inference by SHA-256 ordering, one function from
each of 24 nonempty `source_subdataset` project groups, among 417 eligible
primary-development functions from 212 groups. Eligibility required a previously
observed annotation-containing token window. No model scores or labels selected
the sample. Grouping metadata do not establish independent developers or near-clone
separation. All 11 outputs were retained; no external set was read or tuned.

Each function has three evaluations under identical frozen weights: raw text,
canonical sanitized text, and raw-window annotation-token masking. Masking changes
only annotation-overlapping token IDs to the existing mask token; attention masks,
window length, token positions and remaining context are held fixed. It is a
secondary control because mask tokens are not a neutral natural-code comment.
Deletion can also change truncated context. First three raw inputs were repeated.

## Results

| Descriptive endpoint | Annotation removal vs raw | Fixed-window masking vs raw |
|---|---:|---:|
| Mean per-function maximum absolute output change over 11 patterns | 0.075962 | 0.109704 |
| Conditional project-resample 95% interval | [0.038019, 0.125323] | [0.062750, 0.166493] |
| Median maximum change | 0.026987 | 0.045311 |
| Largest single output change | 0.551014 | 0.469841 |
| Mean absolute change over all function/pattern cells | 0.009658 | 0.014118 |
| Functions with any uncalibrated 0.5-output-boundary crossing | 1 / 24 | 2 / 24 |

The repeated-raw control maximum absolute difference was 0.0. The run took 6.590
seconds after imports on one CPU thread; no training or downloads. Intervals use
2,000 bootstrap resamples of the 24 observed project groups, seed 20261101. They
condition on this selected cohort and frozen checkpoint; they do not quantify
corpus, training, label, project-definition, or selection uncertainty. No p-value
or detector-superiority interpretation is warranted.

All prespecified per-pattern effects are reported below. A signed change means
intervention output minus raw output; pattern outputs are not vulnerability
probabilities.

| Pattern | Removal mean signed | Removal mean absolute | Masking mean signed | Masking mean absolute |
|---|---:|---:|---:|---:|
| `EXTERNAL_CALL_BEFORE_STATE_UPDATE` | 0.000357 | 0.000821 | -0.000382 | 0.001734 |
| `MISSING_INPUT_VALIDATION` | 0.003873 | 0.012665 | 0.003970 | 0.027365 |
| `MISSING_ZERO_ADDRESS_CHECK` | -0.008975 | 0.011238 | -0.008255 | 0.010137 |
| `PRECISION_LOSS_IN_DIVISION` | 0.000415 | 0.000711 | 0.004153 | 0.005023 |
| `ROLE_OR_OWNER_REASSIGNMENT` | -0.000027 | 0.000141 | -0.000110 | 0.000152 |
| `SENSITIVE_STATE_READ_WITHOUT_LOCK` | 0.002362 | 0.003433 | -0.005499 | 0.005905 |
| `STATE_UPDATE_ONLY_ON_SUCCESS` | -0.000092 | 0.022008 | -0.024927 | 0.031976 |
| `UNCHECKED_ARITHMETIC_OPERATION` | -0.000664 | 0.020886 | 0.016421 | 0.036299 |
| `UNGUARDED_SELFDESTRUCT` | -0.000304 | 0.000394 | -0.000537 | 0.000671 |
| `UNSAFE_DELEGATECALL` | -0.000613 | 0.001077 | -0.000833 | 0.001091 |
| `WEAK_CALLER_VALIDATION` | 0.024665 | 0.032862 | -0.024897 | 0.034952 |

## Lineage and limits

- Current checkpoint SHA-256: `eb242c96741d8e985e623bdb34fa39e61015fff67fa0ee206956fa170cd078b7`.
- Shared registry SHA-256: `9bd0bfeb51b46df80a602eb334e7ebdcaf14d5777ce69ccce76a223e5283cfcb`.
- Tokenizer SHA-256: `6143ead457e07ba54af91ee518dadb0468b939c892550b0fe04d939b94c248a8`.
- Frozen selection SHA-256: `097b3876d642bf3280bebbd021457347e6079f494c38f106502ee63a3b21054e`.
- Restricted tensor-only checkpoint loading (`weights_only=True`), strict state
  matching, offline config/tokenizer, deterministic CPU evaluation, no gradients.
- The current checkpoint identity is checked against the prior audit, then
  rechecked after inference; the registry is likewise checked before and after.
- Historical score metadata lacks a checkpoint checksum. This newly identified
  diagnostic does not reproduce old scores or repair old fusion results.
- The original auxiliary CSV is missing; full training reproduction remains
  unavailable. Synthetic-label, missing-context and provenance limits remain.
- The finding is nuisance sensitivity. It establishes neither improved accuracy
  nor an exploitable vulnerability, a correct pattern judgment, calibrated risk,
  or generalization to unannotated sources.

## Reproduction and evidence

```sh
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python scripts/run_frozen_annotation_diagnostic.py \
  --output-dir data/derived/sanitized_v1/ml_results/frozen_annotation_diagnostic_NEW \
  --projects 24
```

The output directory must not already exist. The script holds both the
cross-paper `/tmp/paper-research-compute.lock` and SCAFE advisory lock throughout
model work and fails if busy. It enforces a 240-second inference deadline.

The dated ignored output directory contains `summary.json`,
`selected_functions.csv`, and all 264 paired-pattern rows in
`paired_pattern_outputs.csv`. These artifacts are preserved locally, not
force-added to Git. Future supplementary packaging must preserve their hashes.

Next: a provenance-rich context-aware challenge set with real hard negatives,
near-clone controls, explicit unknown labels and a prospectively frozen cohort.
Any fusion or retraining claim needs new sanitized scoring lineage and
development-only selection.
