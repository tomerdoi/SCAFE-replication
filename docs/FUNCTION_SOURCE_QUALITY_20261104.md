# Targeted source-quality pilot — 4 November 2026, 11:00 UTC

The prior source adapter is implemented and published in `8f8df8b`. This iteration
completed a **single AI-assisted, source-visible technical review** of the eight
already selected pilot cases. It did not recruit human reviewers or create
independent evaluation labels. The 2,323 candidates, 110 native SCRUBD RE claims,
33 mapped source proxies (14 positive / 19 negative), observation masks and
training/evaluation eligibility are unchanged. No model was fitted or scored.

## Protocol fixed before inspection

The hypothesis, sample, decision boundaries and resource budget were recorded in
workspace `scafe_iteration_20261104_1100UTC/run.json` before the eight source reviews.
The immutable adapter selected two cases per native-RE/student-flag stratum with
seed 202611041000. All eight were inspected without replacement or favorable-case
selection. They represent six distinct contexts and six evidence components,
not eight independent projects. These are previously observed corpus matches;
they cannot become a fresh confirmation set.

Each review separates (a) exact qualified local source/span consistency, (b)
lexical source observations, (c) explicitly AI-proposed interpretation and (d)
unresolved semantic/deployment questions. Source labels and rationales were
visible, so the reviews are not blinded. Raw UTF-8/CRLF character spans and evidence
line hashes are checked against the pinned context; rationale hashes bind the
original source comments. Hash validation proves evidence identity, not that the
AI interpretation or upstream label is correct. The source-proxy target is still
source-assessed susceptibility, not syntactic presence or proven exploitability.

The pinned SCRUBD README describes RE as a vulnerability assessment, with original
rationales and student-involvement metadata. Preserve that meaning and revision;
do not infer a common operational threat model or source-label accuracy from it.
Reference: [SCRUBD README, pinned revision, label definitions and explanations](https://github.com/sujeetc/SCRUBD/blob/9dba1928d4aea9a5d27014e5f31a8ed0b703f711/README.md#example-structure-of-the-labels-files-scrubd-cddatalabelscsv).
The existing reference-to-gap review, P1 context/dependency and matched-cohort
gaps, motivates this concrete target-alignment decision rather than a new
literature inventory or another unchanged intake audit.

## Observed outcome and decisions

| Quantity | Before this pilot | After |
|---|---:|---:|
| Source-visible preliminary AI reviews | 0 | 8 |
| Qualified local span bindings rechecked in this sample | — | 8 |
| Reviewed cases with interaction before later state writes | — | 7 |
| Cases with visible guards outside the function snippet | — | 5 |
| Cases identifying unverified dependency implementations | — | 3 |
| Rationale issues prioritized for qualified adjudication | — | 2 |
| Independently reviewed / adjudicated labels | 0 / 0 | unchanged |
| Training eligible / evaluation eligible / assigned splits | 0 / 0 / 0 | unchanged |

These are case observations, not detector precision, source-label error rates,
prevalence or a calibrated label-noise estimate. Seven observed orderings include
three of the four source-negative cases; that demonstrates why native RE0 cannot
be treated as an absence-of-ordering syntactic label. It does not demonstrate
three incorrect susceptibility labels. The one other sampled negative uses
CREATE and memory operations; no function-local storage write is visible, which
also does not prove contract safety. Runtime/guard/callee analysis is still needed.

Two concrete unresolved mechanisms narrow the next adjudication work:

- One RE0 rationale discusses an admin payment but omits a separate low-level
  value call preceding state updates. Caller guards and pre-call round-state
  changes are visible outside the function; the callee is constructor-supplied.
  The omission is not proof that RE0 is wrong.
- One RE1 rationale names `getPlayerName`, declared `view` in the local interface,
  while an earlier `getPlayerID` call is non-`view`. The actual compiler, deployed
  callee and caller restrictions must distinguish the mechanisms. RE1 is not
  declared false merely because the named mechanism is questionable.

The remaining positive cases also require execution/caller analysis; an ordering
observation is not an exploit proof. No one-pass AI conclusion is substituted for
independent contextual suspicion, and no labels are flipped or synthesized.
The code-only model-input view and all source-target files remain byte-identical.

## Local artifact and enforceable boundary

New versioned derivative (private local research, not staged in Git):

`scafe_function_dataset_pilot/derivatives/source_quality_ai_v1_20261104T1100UTC/`

Four files contain original AI submissions, provenance-enriched preliminary
reviews, summary and manifest. `scripts/review_function_source_quality.py` checks
exact frozen-sample membership, native target/span agreement, original rationale
and evidence hashes, parent/delivery lineage and immutable outputs. It refuses
human/gold/adjudicated status, eligibility promotion, replacement labels or model
inputs in this AI-review schema. These constraints apply only to this specific
preliminary-review export; they do not impose new reviews of every training row.
The existing adapter/source claims remain immutable. Deterministic replay is
required. Tests use synthetic fixtures and establish tooling behavior only.

Reproduce with installed Python, in a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/review_function_source_quality.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --source-release-dir /absolute/path/source_supervision_v1_20261104T1000UTC \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --reviews /absolute/path/source_quality_ai_v1_20261104T1100UTC/review_submissions.jsonl \
  --output-dir /absolute/path/NEW_VERSION
```

## Next outcome and acceptance gates

The immediate deliverable is a qualified semantic decision on the two cited
rationale issues, recording compiler/EVM, required callee/caller evidence and
whether the source-proxy target is fit for the intended research question.
A qualified reviewer should inspect the eight local preliminary records and
prioritize those two; retain unknown/context-required decisions where evidence
is insufficient. These source-visible records must not be used as independently
blinded evaluation truth. No review submission or turnaround is promised.

Then audit a justified source-quality sample under an explicit risk/precision
objective with adjudication, separately obtain independently reviewed grouped
evaluation, and finish provenance/license, frozen splits and mask-compatible
learning gates before matched simple baselines and a bounded CodeBERT pilot.
Do not wait for double manual review of every training example. If qualified
review is unavailable, continue independent supported source mapping/context
recovery or compatible learning implementation without promoting eligibility.
The methodological decision is concrete: preserve the source-proxy namespace;
training/evaluation cannot silently optimize a different pattern-presence task.

Applicable tests, both integrity validators, fresh isolated full manuscript build,
visual checks, verified local PDF refresh and explicit privacy/Git review remain
mandatory. Exact results and publication state are recorded in this run's report;
compilation does not validate the scientific claims above.
