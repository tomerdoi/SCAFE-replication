# Actionable function annotation pilot — 2 November 2026

This iteration prepares a concrete independent-review deliverable, not new training
labels or a model result. It preserves the delivered dataset and all unknowns.

## What changed

The delivered 400-record blinded queue has a nullable single-category annotation
object. The new versioned 24-function feasibility package explicitly asks **both**
reviewers about **all three categories on all three axes**: visible pattern
presence, contextual suspicion, and supported vulnerability. These axes must not
be collapsed into one target. There are 216 decision cells per reviewer, 432 across
the two blank packets. A stable pseudonymous reviewer identity, reason for every
cell, and exact character-offset evidence for observed decisions are required.

The independent first-pass files withhold source claims and source/provenance
metadata. A separate coordinator key binds each review ID to its original record,
pinned source revision, project/path metadata, context hash and license status.
The readable blinded Markdown uses variable-length code fences so embedded source
text cannot escape its code block. Code identifiers can still reveal origin;
perfect reviewer blinding is not established. Do not execute source snippets.

Comparison validates immutable input/context identity, complete coverage, distinct
reviewer identities, exact quoted spans and category/axis schemas. Unknown/unknown
is reported separately from agreement on jointly observed cells; an all-unknown
packet never yields a perfect observed-label agreement. Disagreements, including
known/unknown differences, enter an explicit adjudication queue. No training-label
export or automatic promotion is performed. Two different IDs alone cannot prove
independent qualified review or annotation truth.

## Measured artifact coverage, not classifier performance

| Property | Delivered handoff | New versioned review package |
|---|---:|---:|
| Standalone candidates retained in original dataset | 2,323 | 2,323, unchanged |
| Focused review functions | existing broad 400-record queue | 24 short functions |
| Source balance in focused queue | not the new protocol | 8 per each of 3 sources |
| Explicit category × axis cells per reviewer | nullable single-category object | 216 |
| Distinct exact model-input hashes / reported family hashes | not a global certification | 24 / 24 in selected queue |
| Verified full-source context documents in new package | not supplied | 0 |
| Completed independent annotations / training-eligible records | 0 / 0 | 0 / 0 |
| Frozen train/development/test assignments | 0 | 0 |

Selection is deterministic hash order within round-robin source quotas, restricted
to 100–4,000-character standalone candidates. Exact model-input identities are
independently computed; family hashes are producer metadata. This favors short,
lexically complete candidates and is neither a prevalence sample nor a frozen
classifier evaluation cohort. It does not establish project/near-clone independence.
Sixteen selected records lack project IDs. Sixteen report source-file availability
and eight report missing context, but the recovered snapshot cache is **not in the
local delivered handoff**. A context hash or URL is not source bytes or verified
context. Every new review item therefore truthfully states `snippet_only`.

In this version reviewers may assess supported visible pattern presence, including
unknown when evidence is insufficient. Contextual suspicion and vulnerability
axes must remain unknown. The category rubric explicitly treats authorization-
sensitive operations separately from missing authorization, and suspicious syntax
separately from vulnerability evidence. The three categories are not mapped to the
legacy eleven-label encoder. Completing this exercise does not finalize the data.

## Local deliverable and concrete review action

The versioned output is stored outside Git at:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/review_pilot_v1_20261102T1200/`

Give `REVIEW_INSTRUCTIONS.md`, `blinded_functions.md` and a separate copy of
`reviewer_A.jsonl` or `reviewer_B.jsonl` to each of two qualified reviewers for
independent first-pass work. Keep `coordinator_key.jsonl` and original source claims
with the coordinator, outside first-pass review. Preserve submitted packets as
new files rather than overwriting the immutable templates. Each reviewer supplies
a stable pseudonym, completes all cells with reasons, and sets `completed=true`.
Evidence uses zero-based Unicode character offsets into the exact `model_code`.
This is a concrete pending review action; no reviewers have been contacted and no
review has been performed or attributed to a human by the automation.

Run from `sc_review` using existing Python dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/prepare_function_review.py prepare \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION

PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/prepare_function_review.py compare \
  --kit-dir /absolute/path/review_pilot_v1_20261102T1200 \
  --review-a /absolute/path/completed_A.jsonl \
  --review-b /absolute/path/completed_B.jsonl \
  --output /absolute/path/NEW_comparison.json
```

Generation refuses existing output directories and checks all 18 delivered hashes
before and after input preparation. Comparison refuses incomplete packets and an
existing output file. All final generated files were identical on a second replay;
the delivered original files remained unchanged. Original data and generated review
code snippets are not added to Git or uploaded.

## Remaining gates and next useful work

Recover and verify exact pinned source-context bytes, localize functions including
overloads, and register context documents in a new version before contextual
judgments become eligible. The absence of the snapshot cache is a concrete source
availability limitation; no download or reconstruction was attempted. Independent
annotation and adjudication must preserve pre-adjudication decisions, evidence,
reviewer independence/qualification records and per-category disagreement. Do not
substitute an automated source-claim copy for that review. Missing licenses,
project/provenance identities, template/near-clone/SCAFE overlap and group-wise
frozen splits still gate dataset readiness and CodeBERT fine-tuning. Partial-label
training needs explicit observation masks and compatible loss/evaluation. Reviewed
core versus weak-expansion comparisons and lexical/rule/source-nuisance baselines
remain necessary after those gates pass.

The reference review supports these choices: MuSe's manual validity checks (PDF
pp.5–6), Susan's paired/lexical controls (pp.4–7), and the representation survey's
limitations/reproducibility discussion are methodological guidance, not proof of
our label quality. See `SCAFE_REFERENCE_REVIEW_20261101.md` for exact references,
versions and limitations. No paper results were copied or reclassified as labels.

## Validation evidence

The task workspace `scafe_iteration_20261102_1200/` records predeclared success
criteria, the measured kit manifest/evaluation, final test results and fresh
manuscript build/PDF-refresh evidence. The 14 new standard-library regression tests
cover deterministic replay, original preservation, source-claim withholding,
embedded code fences, duplicate/family limits, evidence offsets, incomplete and
same-reviewer submissions, unknown agreement and blocked context overclaims.
All 50 focused function/diagnostic tests and both project integrity validators
passed. Full pytest remains unavailable and was not installed or run. Compilation validates the unchanged manuscript inputs;
it is not a scientific improvement or proof of submission readiness.
