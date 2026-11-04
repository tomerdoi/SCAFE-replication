# Initial source-label adapter — 4 November 2026, 10:00 UTC

A new versioned adapter reuses existing SCRUBD RE annotations instead of requiring
fresh independent labels for every training row. It preserves all 110 native RE
claims and 71 observed native UX claims separately, and maps a first conservative
33-function subset with verified qualified source spans into an explicit
**source-assessed reentrancy-susceptibility proxy** layer. It does not promote those
claims to independently reviewed gold, syntactic pattern presence or exploitability.

| Measure | Original staging / prior views | New source-supervision layer |
|---|---:|---:|
| Candidate records / model-code views | 2,323 | unchanged |
| Preserved native SCRUBD RE claims | 110: 52 positive, 58 negative | unchanged |
| Preserved native UX | 39 positive, 32 negative, 39 unknown | unchanged |
| Observed source-proxy target cells | no such exported layer | 33 RE cells |
| Proxy-positive / proxy-negative rows | — | 14 / 19 |
| Distinct mapped code byte strings / token groups | — | 32 / 31 |
| Native RE rows pending exact local span mapping | — | 77 |
| New unchecked-call / authorization targets | — | 0 / 0 |
| Independently reviewed / training eligible / evaluation eligible / splits | 0 / 0 / 0 / 0 | unchanged |

The original 6,969 harmonized staging cells remain unknown in their immutable
files. The new 33 observed cells belong to a separately defined source-proxy
supervision layer; they are not 33 new independent human annotations. All 33
mapped rows have direct existing-corpus context overlap and retain quarantine.
No exact-input RE contradiction occurs in the current source claims, but the
adapter masks such conflicts and preserves both claims if encountered. Row/token
counts do not establish independent projects, class support or a fresh holdout.

## Predeclared mapping and semantic boundary

The hypothesis, acceptance rules and bounds were recorded before generation in
workspace `scafe_iteration_20261104_1000UTC/run.json`. Only explicit native RE0/1
assessments map. The label means **the upstream source assesses that exact function
as RE-positive/negative under its susceptibility definition**. This is a weak/source
proxy for a reentrancy-related research task, not a silent conversion to the
review kit's `pattern_presence`, `contextual_suspicion` or `supported_vulnerability`
axes. Validate target alignment and source quality before training or comparison;
no automatic mapping into the legacy eleven-output model is provided.

Mapping requires the original delivery/revision, current v2 code-view hash, exact
whole-context hash and raw function span, and the existing unique qualified lexical
owner evidence. That yields 33 of the 110 native-labeled SCRUBD rows. For the other
77, source RE claims remain available but the current local qualified exact-span
evidence is missing. They are not dropped or made negative; mapping can expand as
supported source evidence is recovered. This is an automated evidence gate, not a
requirement for two independent human reviews of every source-labeled function.

Native UX remains in its original unhandled-exception namespace; it is not silently
mapped to unchecked external calls. Authorization is unobserved. HF finding/no-known
claims, CoinFabrik folders/findings, absent labels and patched functions do not
supply new targets in v1. No severity-based confidence or numeric reliability is
invented. Source revision/identity, student-involvement flag, original meanings,
qualified spans and source/project/group limitations remain available for auditing.

## Artifact and loader contract

Local derivative:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/source_supervision_v1_20261104T1000UTC/`

Six files contain separate `source_targets.jsonl` and `model_inputs.jsonl`, an
8-row stratified pilot audit queue, summary, data card and manifest. Code inputs
are byte-identical to the immutable annotation-v2 view: only record ID, model code,
code hash and input-protocol fields are copied. Audit descriptions, source label
metadata, rationales and targets are not concatenated to code. This does not
certify removal of every possible source/format cue; existing leakage controls,
truncation checks and nuisance baselines remain required.

The manifest binds the delivery, context, scope, model-view and helper hashes.
Per-row source supervision preserves current evidence/strict-template components,
code/token signatures and direct/propagated overlap constraints. No split is assigned.
Unknown labels are null with false observation masks. `masked_source_example`
validates IDs, code hashes, source-proxy schema and label/mask consistency, then
returns code and targets together with masks. Its zero placeholders in disabled
cells are not negatives: a future compatible learner must reduce loss only over
observed cells and skip unobserved supervision. `for_training=True` fails while
readiness gates remain unmet; no masked trainer/loss or CodeBERT fitting is claimed.

All rows remain training/evaluation-ineligible pending the targeted source-quality
assessment, independent evaluation cohort, frozen grouped splits, provenance/license
eligibility and compatible learning protocol. Existing corpus matches cannot
become fresh confirmation examples. Source/weak training and independent gold
evaluation remain distinct even after later readiness gates pass.

## Targeted audit and next critical step

The audit frame is the 33 mapped source proxies, stratified by native RE and
recorded student flag. Seed 202611041000 samples up to two rows per stratum: eight
rows in four strata, without replacement. This is a small mapping/meaning pilot,
not a statistically calibrated noise estimate, completed review or independent
evaluation cohort. No reviewer has been contacted; decisions remain null. Audit
mapping/target meaning, record genuine disagreement/context missingness, and size
any subsequent quality sample from a declared precision/risk objective. Do not use
this source-visible audit queue as independent model test truth.

This advances the source-label-adapter milestone with a usable reproducible export
and fail-closed join, not with more source claims or a detector accuracy gain.
Next validate the eight-row source-mapping/meaning pilot, justify a label-quality
protocol and independent grouped evaluation, and implement compatible masked
learning before matched simple baselines and a bounded CodeBERT pilot. Continue
supported automated mapping for pending rows without demanding blanket relabeling.
The reference-to-gap review's context, supervision quality, independent cohorts,
matched-baseline and uncertainty requirements remain unchanged.

## Reproduction and checks

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/adapt_function_source_labels.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --scope-release-dir /absolute/path/scope_mappings_v1_20261103T1300 \
  --view-dir /absolute/path/annotation_view_v2_20261103T1100 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The output must be new. Six-file replay is byte-identical; independent checks bind
all source/native counts, mapped labels, raw spans, unchanged code inputs, group
constraints, unknown masks and eight uncompleted audit rows. Originals/ancestors
are preserved and no pilot data is uploaded. Regression tests cover source/unknown
parsing, negative versus disabled-mask distinction, missing/invalid span/owner,
conflicting identical inputs, audit-text exclusion, parent/review-state tampering,
replay and training-readiness refusal. Focused tests, both integrity validators,
fresh full manuscript/PDF validation and final/staged review are required before
any authorized Git publication. Exact outcomes are in the current workspace report.
