# DAppSCAN leakage remediation and rerun plan

## Confirmed invalidation

DAppSCAN vulnerability labels are derived from released SWC findings, while the
corresponding Solidity files contain injected `SWC-<number>` comments. These
comments perfectly identify the canonical binary labels and were visible to raw
source models. Results produced from the legacy scored table must not be used as
evidence until regenerated.

Invalidated result families:

- DAppSCAN within-corpus source-model results.
- Feature or model-family selection performed on legacy DAppSCAN features.
- Cross-corpus models whose training data include legacy DAppSCAN features or
  source text.
- Derived ablations, SHAP explanations, and sensitivity analyses based on those
  fitted models.

The raw corpus and legacy artifacts remain unchanged for auditability.

## Corrected protocol

1. Remove only DAppSCAN SWC annotation comments before any model-visible
   feature extraction or tokenization. Remove a standalone annotation's entire
   physical line so that it cannot leak through LOC, while preserving Solidity
   preceding an inline annotation.
2. Preserve raw, comment-insensitive source hashes for duplicate control.
3. Generate collision-free DAppSCAN record IDs from project-relative paths.
4. Isolate corrected artifacts under `data/derived/sanitized_v1/`.
5. Route handcrafted, semantic, strict-DL, and transformer source readers
   through the same sanitizer; legacy callers additionally use a conservative
   DAppSCAN path fallback.
6. Recompute every source-bearing row with one extractor revision. Recomputing
   only annotated positives creates a label-correlated extractor-version batch
   effect and is invalid.
7. Record hashes of the extractor and sanitizer in the rebuild manifest.
8. Include labels, source identities, groups/configuration, and sanitization
   version in checkpoint manifests.
9. Use deterministic epoch shuffling and chunk-balanced training loss.
10. Flag near-constant prediction folds.

Corpus-wide sanitizer audit: 948 of 21,457 DAppSCAN files were changed, all
SWC-number markers were removed, 1,621 standalone annotation lines were
removed, and the 25 inline annotations retained their preceding Solidity.

## Validation gates

Before expensive training:

- Zero SWC-number markers remain in every model-visible canonical DAppSCAN
  source.
- All generated DAppSCAN record IDs are unique.
- Source identity hashes remain unchanged by annotation sanitization.
- Focused and full test suites pass.
- A cheap lexical baseline drops materially after sanitization.
- Corrected SCAFE within-corpus results are plausible and fold-stable.

## Staged commands

Rebuild the versioned unified table:

```bash
venv/bin/python -u scripts/rebuild_sanitized_unified_eval.py --workers 8
```

Run inexpensive SCAFE validation first:

```bash
venv/bin/python -u scripts/run_scafe_within_corpus_cv.py \
  --datasets DAppSCAN --models rf et hgb xgb --seeds 42 43 44 45 46
```

Then rerun the primary repeated LODO protocol and feature ablations:

```bash
venv/bin/python -u scripts/run_scafe_repeated_lodo.py
venv/bin/python -u scripts/run_scafe_feature_ablations.py
```

Run transformer experiments one model and seed at a time. Each invocation writes
to the isolated v3 campaign and resumes only matching corrected checkpoints:

```bash
venv/bin/python -u scripts/run_dl_within_corpus_cv.py \
  --models codebert --seeds 42 --device mps --resume
venv/bin/python -u scripts/run_dl_within_corpus_cv.py \
  --models unixcoder --seeds 42 --device mps --resume
```

Expand to seeds 43--46 only after seed 42 passes collapse and fold-stability
checks. Update manuscript tables and claims only after all corrected validators
pass.
