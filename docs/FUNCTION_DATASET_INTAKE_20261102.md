# Function-dataset handoff and intake safety — 2 November 2026

The user supplied the local `scafe_function_dataset_pilot` directory from the
separate dataset-construction work. No restricted chat memory was accessed.
Original files were inspected read-only; no acquisition/build script was run,
no record was relabeled, no split was assigned, and no duplicate training began.

## Current supervision policy update — 4 November 2026

The handoff counts below describe the 2 November 2026 staging snapshot. Importer
behavior and readiness observations describe their dated inspection, not a
permanent current-state assertion; verify relevant current code/artifacts before
reusing an availability or implementation claim.
Upstream labels/findings are already preserved; null harmonized targets do not
mean those source annotations are absent. Future training may reuse justified
source/weak supervision with provenance, exact function/category mapping, partial
observation masks, targeted stratified sample auditing and ambiguous-case
adjudication. There is no requirement for two new independent reviews of every
training example. Keep a separate independently reviewed/adjudicated evaluation
cohort, project/clone-aware splits and rigorous matched simple baselines before
a bounded CodeBERT pilot. No absent-label, patched-function or no-known-vulnerability
blanket negatives, audit-text leakage or circular rule-based evaluation is allowed.
See the standing source-supervision policy and rolling outcomes in
`docs/SCAFE_EXPERIMENT_LEDGER.md` for the current protocol direction.

The 4 November policy update itself did not implement the adapter, change label
fields, make the pilot training-eligible, or start training. Judge subsequent
readiness from implemented code, artifact lineage and declared gates, not this
historical implementation snapshot. All existing validation,
privacy, compute and Git safeguards remain in force. The delivered pilot originals
and prior independently reviewed/unreviewed status remain unchanged.

## Verified handoff

All **18 delivered file hashes** match
`DELIVERY_SHA256.json`. Its SHA-256 is
`d842e2a01bbe5a060454547058673799d93cd11075b6eb27411cadfd8e603437`; the source manifest SHA-256 is
`21f25878f0bab4c0869004c47ec10571ed260341a9999d70ccd9084ff67942d7`. Checks were repeated after reading.
Pinned source revisions are retained in the original manifest.

| Audit quantity | All staging records | Standalone candidates |
|---|---:|---:|
| Records | 4299 | 2323 |
| Unknown unified pattern cells | 12897 | 6969 |
| Independently reviewed records | 0 | 0 |
| Training-eligible records | 0 | 0 |
| Assigned splits | 0 | 0 |
| Records without project ID | 3656 | 1680 |
| Producer-reported missing context | 2910 | 1570 |
| Producer-reported source-file availability | 755 | 753 |

There are 15 named source/project pairs and one producer-reported exact-token
group spanning sources. These metadata do not establish independent projects,
semantic clone exclusion, contradiction of a source claim, or context sufficiency.
The original review queue contains 400 candidates awaiting review, not 400
completed annotations. No new review queue was created.

The pilot is deliberately **unreviewed staging**, not a training/evaluation
benchmark. Its unified categories (`reentrancy`, `unchecked_external_call`,
`authorization`) have no automatic one-to-one mapping to the legacy eleven
suspicious-pattern outputs. Source audit/tool/finding claims are distinct from
unified human-reviewed labels. Unknowns remain unknown; no global CLEAN labels
or audited vulnerability truth were inferred. Distribution rights still need
per-artifact checking; originals/data are not included in this code commit.

## Independent intake flaw fixed

Previously `_multihot` silently encoded UNKNOWN, CONTEXT_REQUIRED, missing values,
and whitespace-padded known labels as all-zero targets. It also accepted
CLEAN mixed with a positive pattern. Controlled software fixtures produced
positive-target counts `[0,0,0,0,1]` for those five respective cases. These are
regression fixtures, not research observations or detector-performance data.

Now unresolved/missing/unsupported annotations and CLEAN-plus-positive mixtures
raise before training. Known labels allow surrounding whitespace. Only explicit
sole CLEAN produces the all-zero legacy target. CLEAN means the annotator's
closed-set no-listed-pattern claim; it is not proof of security.

Required fields, unique IDs, nonempty code, complete declared train/val/test
splits, exact-text crossing splits, and same-text conflicting claims are checked
before output creation, registry access or model imports. Conflicting claims
require context review; the gate does not choose which claim is correct.
Exact UTF-8 source identities preserve literal whitespace and are not a near-clone
or project-leakage detector. Within-split duplicate rows are not automatically
dropped or silently relabeled.

Resume configuration and training/overlap metadata now include a SHA-256 over
ordered IDs, source text, labels and splits after overlap filtering. Same-row-count
label corrections or reordered data cannot silently reuse an incompatible
resumable state. Historical checkpoints and artifacts were not modified.

This legacy encoder remains a closed-set learner. **Partial labels need an
explicit observation mask and a reviewed schema before any adapter/training.**
Passing these software checks does not certify label quality, provenance, project
independence, correctness or benchmark readiness.

## Reproduce the read-only handoff check

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/audit_function_pilot_handoff.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --output-dir data/derived/sanitized_v1/ml_results/function_pilot_handoff_NEW
```

The report directory must be new and outside the original pilot. The audit
rejects changed delivery hashes, path escapes, unexpected schema or promoted
labels/states. It does not execute source code or construction scripts. Its
counts use exported provenance/context fields; tokenization and semantic claims
were not independently rederived. The dated ignored local report is
`data/derived/sanitized_v1/ml_results/function_pilot_handoff_20261102_1000/summary.json`.

## Next research gate

Implement a justified source-label crosswalk and provenance-preserving masked
training adapter, using targeted stratified auditing and ambiguous-case adjudication.
Retain a separate blinded independently reviewed/adjudicated evaluation cohort.
Resolve project/context/localization gaps and cluster versions/pairs/near clones
against SCAFE before freezing cohorts. There is no blanket re-review requirement
for every source-labeled training function. Do not train on this pilot until the
adapter, supervision-quality, independent-evaluation, grouped-split, matched
simple-baseline and compute gates pass. Follow the ledger execution sequence
through gated CodeBERT, credible function-level evaluation, frozen contract-score
integration and evidence-bounded manuscript claims. The prior
reference review supports these gates: MuSe's manual mutation validation (PDF
pp.5–6), Susan et al.'s paired lexical-cue controls (pp.4–7), and BRLF's explicit
preprocessing/split documentation (pp.14–16). Those publications do not validate
our labels or make a journal-acceptance guarantee.

No accuracy gain or Q1 readiness is claimed. The primary manuscript and its
scientific results are unchanged. Current tests, fresh build, PDF refresh and
commit/push evidence are recorded in this run's local report.
