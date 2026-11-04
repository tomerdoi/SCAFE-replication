# Partial-source-supervision loss — 4 November 2026, 12:00 UTC

Implemented the missing three-category loss/batch component while qualified
source-quality adjudication remains pending. This is a working mathematical
and data-loading component for a future gated pilot, not a trained CodeBERT
classifier, a complete training protocol, a new source label or accuracy evidence.
No model, tokenizer, checkpoint or optimizer was loaded or run.

## Before and after

The source adapter exported 33 observed RE source proxies and masks, but had no
compatible loss. The historical `run_function_semantic_transfer.py` is a separate
legacy eleven-output, closed-set learner using dense BCE. It is unchanged and
must not receive this three-category partial-label dataset. Its existing
unknown-label refusal remains; the counterfactual control below is not a newly
measured legacy bug.

`scripts/function_partial_supervision.py` now provides:

- `load_source_batch`: verifies an explicitly supplied manifest SHA-256 and all
  release artifacts, requires the source-proxy schema, exact ID/hash joins and
  pinned code-only annotation-v2 inputs, and defaults to the adapter's existing
  training-readiness check. Diagnostic loading must explicitly request
  `for_training=False`; it does not promote any example.
- `collate_source_examples`: returns ordered record IDs, code strings, targets
  and inseparable boolean observation masks. Source comments/rationales and
  metadata are not concatenated to model code. Duplicate IDs, unsupported
  views, extra input fields and mismatched hashes fail closed.
- `observed_bce_with_logits`: requires the explicit category order
  `(reentrancy, unchecked_external_call, authorization)` and aligned batch-by-three
  tensors. It selects observed cells **before** BCE, sums their losses, then
  divides by the number of observed cells. The current RE-only source layer
  supplies no supervision for the other two outputs; no three-category detector
  performance is implied. This source-proxy namespace is not legacy eleven labels
  or independent gold.

For fixed positive-class weights, the observed objective is the mean of
`-w[c]*y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z))` over observed cells only,
implemented with numerically stable BCEWithLogits. Default weights are one;
optional weights must be finite positive constants on the same device/dtype.
The function never estimates weights from unknowns. Any eventual choice must be
predeclared/fitted on legitimate training data only, with ablations as appropriate.
An observed-cell mean is a micro objective: differing category/row coverage can
change influence and needs explicit experimental controls, not hidden balancing.

Unknown target placeholders are disabled, not negatives. If no cell is observed,
`MaskedLoss.loss` is `None` and `must_skip` is true: a future trainer must skip
backward **and the optimizer step**, including weight decay. Observed labels must
be finite exact zero/one; masks must be boolean. Wrong order/width, learnable
labels/weights, invalid values or an overflowed objective raise before updating.
Disabled cells may contain arbitrary values in robustness diagnostics because
selection precedes BCE. This is not permission to conceal unexpected nonfinite
model outputs: a future real run must report and diagnose model-health failures
separately. No real model outputs were generated in this iteration.

## Predeclared diagnostic and results

The hypothesis, acceptance rules, input hashes and small CPU-only budget were
frozen in workspace `scafe_iteration_20261104_1200UTC/run.json` before the tests.
Used existing PyTorch 2.10.0, one CPU thread and float64 fixed tensors. No software
was installed, no model/corpus downloaded and no pilot data uploaded.

On the immutable 2,323-row source release:

| Controlled check | Result |
|---|---:|
| Total category cells / observed source proxies / unknown | 6,969 / 33 / 6,936 |
| Observed source RE positive / negative | 14 / 19 |
| Fixed observed logit | 0 |
| Mean observed loss with unit weights | 0.6931471805599453 |
| Unknown cells with nonzero gradient | 0 |
| Maximum absolute error versus analytic observed gradients | 0 |
| Default training-purpose loading | refused: readiness gates unpassed |

Changing every disabled logit/target from NaN to 100,000,000 leaves loss and
observed gradients unchanged. At zero observed logits, each observed gradient
has magnitude `0.5/33`; positive and negative signs differ as expected. An
intentionally incorrect dense-placeholder counterfactual produces nonzero
gradients on all 6,936 unknown cells and dilutes observed gradient magnitudes by
`6969/33 = 211.1818...`. This is an objective-construction illustration on copied
fixed diagnostic tensors, not trained-model results, source-label quality or a
measurement of current legacy behavior. Synthetic unit fixtures also test manual
loss/analytic gradients, non-dilution by unobserved rows, observed negatives,
all-unknown skip, weighted terms, invalid values/schema and manifest tampering.

Reproduce the focused software checks with installed dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 venv/bin/python -m unittest discover \
  -s tests -p 'test_function_partial_supervision.py' -v
```

The fixed-logit experiment/configuration, environment, exact input hash and
results are retained in workspace
`scafe_iteration_20261104_1200UTC/partial_supervision_diagnostic.json` with its
reproduction script. Hash-pinned loading and code-only collation do not certify
full ancestor/source authenticity, licensing, independent labels, group splits
or absence of residual code/style cues; those existing gates remain necessary.
No derived dataset release or original file was changed by this implementation.

## Readiness and next outcome

All native/source-proxy labels, model views, grouping/quarantine evidence and
prior audit artifacts remain unchanged. Independent/adjudicated labels,
training/evaluation eligibility and assigned splits remain zero. The two specific
POOHMO/MonkeyScam rationale concerns in the previous method note remain unresolved;
the loss implementation neither adjudicates them nor flips labels.

This completes the narrow mask-compatible loss and batching deliverable. It does
not clear the broader `compatible_masked_training_protocol_pending` gate: a
frozen three-output model/training configuration, tokenization/context/truncation,
checkpoint/input lineage, skip behavior in the actual optimizer loop and
reproduction still need validation. The legacy eleven-head script is not wired
to this module. Data-quality/evaluation/group/provenance/license and matched
baseline gates remain before any CodeBERT fitting or function-to-contract fusion.
Qualified adjudication of the two rationale mechanisms remains the immediate
semantic dependency. While awaiting it, supported source/context mapping or
independent baseline/training-protocol implementation can proceed without
promoting readiness. Do not require two new reviews for every training function.
The reference-to-gap review's supervision/context/independent-cohort controls
remain in force; this iteration supplies no new novelty or publication claim.

Applicable tests, both integrity validators, fresh full paper build/visual checks,
backed-up verified PDF refresh and final/staged privacy/Git review are mandatory.
Actual outcomes, exact commands/hashes and publication state are in the workspace
report; compilation does not validate labels or detector performance.
