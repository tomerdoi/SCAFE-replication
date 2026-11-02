# SCAFE experiment ledger

## 2026-11-01 — Function-model annotation boundary audit

Status: active. Exclusive lock: `/tmp/scafe-research-iteration.lock`.
Baseline: replication `4fd7f45`, manuscript `7e4cd05`; both initially clean.
Process inventory found no matching SCAFE research jobs before lock acquisition.
No training, target selection, manuscript edits, commit, push, installation, or upload is authorized by this run.

Hypothesis fixed before evaluation: the function pipeline's raw reader may expose
DAppSCAN SWC annotation comments despite a sanitized registry path. Reusing the
canonical dataset-aware sanitizer removes this direct label channel from model
inputs while preserving intended Solidity code. This is an integrity improvement,
not a hypothesis of increased detector accuracy.

Success criteria:

1. Audit all DAppSCAN registry rows deterministically, examining extracted functions
   only when canonical sanitization changes their source. Do not fit a model or
   read external target sources/labels for selection.
2. Route scoring and overlap filtering through canonical sanitization, with
   dataset-aware caches and recorded sanitization version.
3. Reject legacy/unversioned aggregate evaluation and mismatched aggregate hashes.
4. Pass meaningful regression tests and existing integrity/submission validators.

Primary evidence will be saved to
`data/derived/sanitized_v1/ml_results/function_source_sanitization_audit_20261101/`.
Original source, checkpoints, prior scores, and manuscript remain intact.
The previously inspected DIVE/Empirical sets are not fresh confirmation data.

Validation commands planned: focused function/sanitizer tests, then applicable
broader tests if needed; `validate_experiment_integrity.py` and
`validate_submission_artifacts.py`. No manuscript compilation is needed because
this iteration does not change the manuscript.

### Completed result

The predeclared hypothesis was confirmed at the extracted-text boundary.
All 21,457 DAppSCAN registry rows were audited in 16.738 seconds; 21,455
were readable/nonempty and two were empty. Canonical sanitization changed
948 source records. Among those records, 18,069 function occurrences were
extracted before and after sanitization, with identical function-name sequences.
493 function occurrences in 325 records exposed removable annotation comments
under the legacy reader; zero did after sanitization. All 493 changed function
texts retained the same comment/whitespace-normalized identities. Of the 14,791
unique primary Extra Trees development records, 298 have affected function input.
Counts denote records/function occurrences, not independent projects.

The audit does not establish how many annotations survived the tokenizer's
512-token window or how much they affected predictions. No model was loaded by
the audit, no accuracy experiment was run, and no external target was used for
new selection. The prior function-level CodeBERT fusion remains exploratory and
requires sanitized rescoring plus repeated development-only selection before
supporting new conclusions; prior scores and checkpoints were preserved.
This finding does not invalidate the separately sanitized primary SCAFE tables.

Implemented local changes:

- The function scorer and overlap filter use the canonical dataset-aware source
  reader; caches/deduplication include dataset and path.
- Training overlap audit, resume configuration, training metadata and scoring
  metadata record the sanitization version.
- Scoring metadata records the aggregate SHA-256. Evaluation rejects missing,
  legacy, incompatible or hash-mismatched metadata before reading scores.
  Actual historical aggregates were verified to be rejected as expected.
- Added a CPU-only reproducible exposure audit and focused regression tests.

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -p 'test_function*.py' -v`: **9 passed**, including four rejection subcases,
  directory reading, dataset-specific overlap filtering/caching, and a stub-model
  integration check that observes the text sent to tokenization.
- All **9 existing source-sanitization test functions passed** through
  `unittest.FunctionTestCase`, using temporary directories for their `tmp_path`
  fixtures. No test expectations or source-sanitization implementation changed.
- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/validate_experiment_integrity.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/validate_submission_artifacts.py`: passed.
- AST syntax checks passed for all four changed/new Python files;
  `git diff --check` passed.
- Initial attempted pytest command failed with `No module named pytest`.
  Global and both project Python environments were checked; none had pytest.
  Nothing was installed. The full pytest suite remains **not run**.
- Manuscript was not changed; no fresh manuscript build was attempted.

Reproduce audit into a new, nonexisting directory:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/audit_function_source_sanitization.py \
  --output-dir data/derived/sanitized_v1/ml_results/function_source_sanitization_audit_NEW
```

Evidence: the dated audit directory contains `summary.json` with code/input
hashes and `record_audit.csv`. The sanitized registry SHA-256 is
`9bd0bfeb51b46df80a602eb334e7ebdcaf14d5777ce69ccce76a223e5283cfcb`.

Next bounded step: quantify actual tokenizer-window annotation exposure with the
saved local tokenizer, then plan sanitized development rescoring in a new output
directory. Verify checkpoint and score lineage before attempting incremental
reuse; do not relabel old raw scores as sanitized. Do not promote historical
external improvements as fresh confirmation. Context extraction, hard negatives,
near-clone/template grouping and provenance remain unresolved research issues.

Outcome: useful integrity improvement; **no demonstrated predictive gain**.
No commit/push approval is requested for this iteration. No commit, push, PR,
publication, training, software installation, paid compute, or private upload.

## 2026-11-01 16:00 Israel — Saved-tokenizer annotation exposure audit

Status at registration: active; exclusive lock `/tmp/scafe-research-iteration.lock`.
Baseline: replication `a83f518`, manuscript `7e4cd05`, both clean. Previous five-file
integrity fix is now committed and will be preserved. Historical heading above
said active when first registered; that earlier run completed and released its
lock. No matching research jobs were found for this iteration. A separate
data-centric audit may be active; this run is single-threaded tokenizer work and
will not write any shared baseline/feature CSV, train, or load model weights.

Hypotheses and configuration fixed before evaluation:

- Some of the 493 previously identified annotation-bearing function occurrences
  expose annotation comment text or full SWC-number markers within the saved
  CodeBERT 512-token window.
- Canonical sanitization may change actual token IDs. Count changes without
  inferring prediction effects or improved vulnerability detection.
- Use saved local tokenizer, right truncation and special tokens. Verify token IDs
  and offsets against the historical local AutoTokenizer path for every raw and
  sanitized input; do not download or substitute a tokenizer.
- Audit all previously affected DAppSCAN records. Mark the primary development
  subset using existing IDs; do not use external labels or fit/select any model.
- Before considering rescoring, inspect checkpoint/score lineage and auxiliary
  input availability. A current checkpoint hash cannot retrospectively prove
  which checkpoint produced old scores.

Success criteria: reproducible complete/partial annotation-window counts, no
annotation exposure after sanitization, focused tests and both integrity
validators passing, followed by a fresh isolated pdflatex/BibTeX build at the end
of the iteration, including for code-only changes. Preserve all originals and
previous paper PDF. No commit/push/PR/publication/install/upload is authorized.

Planned evidence directory:
`data/derived/sanitized_v1/ml_results/function_tokenizer_exposure_20261101_1600/`.

### Tokenizer audit result and final validation

The hypothesis was confirmed: with the actual saved local CodeBERT tokenizer,
right truncation, special tokens and a 512-token input window, **450 of 493**
previously affected function occurrences expose complete SWC-number markers.
These functions occur in **304 of 325** audited records. Removing annotations
changes the token IDs for exactly 450 functions; the other 43 annotated functions
have no annotation tokens inside their retained window and unchanged input IDs.
There are 553 marker occurrences in the affected function texts, of which 494
complete markers survive truncation. Zero sanitized function windows contain
removable annotation comments.

Within the existing primary Extra Trees development subset, **417 of 457**
affected function occurrences expose complete markers, spanning **278 of 298**
records; sanitization changes the same 417 input-ID sequences and removes all
annotation exposure. Counts are occurrences, not independent projects. Token IDs
and offsets were checked against local AutoTokenizer for every raw/sanitized
pair (986 encodings). The CPU audit took 2.538 seconds after imports. No model
weights were loaded, and no detector metric or prediction effect was estimated.
No external dataset source or labels were used for new model selection.

Historical scoring metadata records a checkpoint path but no checkpoint checksum.
The current checkpoint SHA-256 was recorded for future identity checking; it
cannot establish which weights produced historical scores. The auxiliary CSV
at its recorded training path is still unavailable. Consequently, sanitized
prediction comparisons/rescoring were deferred in this bounded iteration. These
are evidence/lineage limits, not a claim that new inference is impossible.
Old fusion results remain unvalidated for new scientific claims.

Changes: added `scripts/audit_function_tokenizer_exposure.py` and
`tests/test_function_tokenizer_exposure.py`; appended this ledger entry.
The previously committed pipeline fix, raw data, shared feature CSVs, model
checkpoints, saved scores, TeX and repository PDF are preserved. New ignored
results live in the planned dedicated directory as `summary.json` and
`function_window_audit.csv`, with tokenizer/checkpoint/input hashes.

Validation before final manuscript build:

- `PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 venv/bin/python -m unittest discover -s tests -p 'test_function*.py' -v`: **15 tests passed**.
- Both `validate_experiment_integrity.py` and
  `validate_submission_artifacts.py`: **passed**.
- AST parsing of both new Python files and `git diff --check`: **passed**.
- Full pytest suite: **not run**, pytest still absent; nothing installed.
- Actual tokenizer IDs/offsets match the historical local AutoTokenizer path on
  all raw/sanitized pairs, so this finding is stronger than extracted-text counts.

Mandatory final isolated manuscript build evidence is recorded outside both
repositories at
`/Users/tomer.doitshman/Documents/Codex/2026-11-01/task/scafe_iteration_20261101_1600/manuscript_build/build_report.json`.
That report is the source of truth for compilation outcome; this entry does not
prejudge the pending build. The run must not be reported complete without that
fresh build check. The earlier one-time authorization to replace the repository
PDF does not authorize replacement by this iteration.

Next step: establish explicit checkpoint/registry/tokenizer hashes for a new
sanitized scoring campaign, then bound any raw-versus-sanitized diagnostic under
the same frozen weights. Keep development-only selection and acknowledge that
previously inspected external corpora are exploratory, not fresh confirmation.
Dataset provenance, paired/template grouping, context extraction, hard negatives,
and reliable suspicious-pattern labels remain research priorities.

Outcome: confirmed model-window leakage and reproducible forensic evidence;
**no predictive improvement demonstrated**. No commit/push approval requested.

## 2026-11-01 — Reference-informed frozen-weight sensitivity pilot

Status at registration: reference review completed; diagnostic pending. Standing
user authorization now permits reviewed task-owned commits/pushes after tests,
fresh isolated manuscript compilation and verified local PDF refresh. Older
ask-each-run/no-push clauses above describe historical runs and are superseded
for current work. No force push, private-data upload or ignored-artifact staging.

The prior tokenizer audit/tests/ledger were checked against their saved hashes:
all are task-owned and unchanged since the completed 16:00 iteration. Existing
replication HEAD is `a83f518`; manuscript HEAD `7e4cd05`. No other user edits were
present. Current upstreams are GitHub SCAFE-replication `main` and the existing
Overleaf repository `master`; verify before any push. The paper PDF was refreshed
at 17:06 Israel and will be backed up before this iteration's final refresh.

Reference review: `docs/SCAFE_REFERENCE_REVIEW_20261101.md` inventories all eleven
original PDFs, distinguishes verified publication years, conference/journal/
preprint/pre-proof versions, and provides page/section citations, protocol
cautions and prioritized gaps. Originals and shared data remain read-only.

Predefined diagnostic (before model outputs):

- New protocol `frozen-annotation-diagnostic-v1`; local checkpoint hash must match
  the hash established by the tokenizer audit. Restricted `weights_only=True`
  loading, strict state-dict match, offline cached config/tokenizer only, one CPU
  thread, no training and a 240-second inference deadline.
- Sample exactly 24 eligible nonempty DAppSCAN `source_subdataset` project groups,
  one exposed primary-development function per project, selected by fixed SHA-256
  ordering without model scores or labels. Project names are grouping metadata,
  not a verified independent-population guarantee.
- For each function compare raw text, canonical annotation removal, and a secondary
  fixed-window control masking only annotation-overlapping token IDs. Keep all
  11 pattern outputs; duplicate the first three raw inputs to check numerical
  repeatability. Masking retains context/positions but introduces mask tokens.
- Primary descriptive endpoint: per-project maximum absolute output change across
  the 11 labels, its mean and a fixed-seed 2,000 project-resample interval.
  Report all per-label deltas and uncalibrated 0.5-boundary crossings secondarily.
- This is a conditional nuisance-sensitivity diagnostic, not a vulnerability
  accuracy test, calibrated risk estimate, repaired-code test, historical-score
  reproduction, external validation, or improvement claim. Deletion may also
  alter truncation/context; the mask control is not a neutral natural-code edit.
- Cross-project compute lock `/tmp/paper-research-compute.lock`, plus the existing
  SCAFE advisory lock, must be held throughout model loading/inference. Do not
  run if locks are busy or another heavy paper job is detected. No GPU use.

Planned isolated evidence directory:
`data/derived/sanitized_v1/ml_results/frozen_annotation_diagnostic_20261101/`.
Preserve original scores/checkpoint and never relabel them as newly sanitized.

### Frozen-weight pilot outcome

The preregistered sample included 24 distinct project metadata groups from 212
eligible groups (417 functions). All 24 raw/sanitized/masked triplets completed
in 6.590 seconds after imports, under the cross-paper and SCAFE advisory locks,
on one CPU thread. The first three duplicate raw controls matched exactly.

Annotation removal: mean per-function maximum absolute change across 11 pattern
outputs **0.075962**, conditional project-resample 95% interval
**[0.038019, 0.125323]**, median **0.026987**, largest **0.551014**. One of 24
functions crossed an uncalibrated 0.5 output boundary on at least one pattern.
Mean absolute change across all 264 function/pattern cells was **0.009658**.

Fixed-window annotation masking: corresponding mean maximum change **0.109704**,
interval **[0.062750, 0.166493]**, median **0.045311**, largest **0.469841**;
two functions crossed an uncalibrated boundary. This control holds token/window
positions and context fixed but introduces mask tokens, so it is not a neutral
natural-code edit. Deletion also changes truncation/context. Report both
interventions, all per-label effects and the deterministic control rather than
selecting a favorable pattern. Intervals condition on the selected 24 project
names and frozen weights; no population, training or label uncertainty is covered.

Current checkpoint identity is explicitly bound to inputs/config/code/selection
hashes. Tensor-only checkpoint loading and strict state matching succeeded. Old
historical scores were not reused, repaired or claimed reproduced. No model was
trained, downloaded or selected; no external labels or sources were consulted.
Shared registry and checkpoint hashes matched before and after inference.
This confirms pattern-output nuisance sensitivity, **not an accuracy gain**.

`docs/FROZEN_ANNOTATION_DIAGNOSTIC_20261101.md` contains the full prespecified
result and limits. The new reference review inventories eleven editions, with
page citations, exact PDF hashes, verified/uncertain publication status, protocol
criticisms and prioritized testable gaps. Original papers remain read-only.

The primary manuscript's scientific claims are unchanged: this small conditional
pilot is retained in the research record rather than promoted to primary
vulnerability evidence. A larger provenance-rich hard-negative evaluation and
recovered training data are still needed. The mandatory fresh isolated build and
local PDF refresh are final validation steps even though TeX is unchanged.
The report at the task workspace's
`scafe_iteration_20261101_refs/manuscript_build/build_report.json`, together with
`pdf_refresh_result.json` and `run.json`, records actual validation outcomes.

The prior 16:00 tokenizer audit/tests/ledger are attributed to this task and may
be included with the reviewed new work under the user's current standing
commit/push authorization. Only explicitly reviewed source/tests/docs may be
staged; no ignored data, paper PDFs, raw references, private files or unrelated
edits. Commit/push results and any failures are reported after final validation.

## 2026-11-02 10:00 Israel — Safe function-label intake and pilot handoff

Baseline: clean replication `98e029f`; clean manuscript `7e4cd05`. No relevant
Python jobs were observed and both compute locks were available. Shared registry
SHA-256 still matches the frozen manifest. No previous experiment was rerun.

The user identified the new pilot at
`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot`.
Only authorized files were read; no restricted cross-chat memory was accessed.
README/report/ledger establish bounded public-source construction with pinned
revisions, no training, no independent review, all unified pattern labels unknown,
and no assigned splits. All eighteen delivered hashes match. Originals remain
read-only, including scripts, manifests, queues and data. No builder/acquisition
script will be executed and no duplicate dataset/training run will be started.

Predeclared bounded advance: validate this handoff read-only, preserving staging
states, and fix the independent legacy encoder intake flaw. Controlled pre-change
unit inputs UNKNOWN, CONTEXT_REQUIRED, missing, a whitespace-padded known label,
and CLEAN+positive encoded to counts [0,0,0,0,1]. These are software regression
fixtures, not labeled research examples or performance evidence.

Success criteria:

- Explicit unknown/context-dependent/missing/unsupported annotations must raise
  before becoming negative targets. CLEAN is accepted only alone; whitespace
  around known pattern labels is handled deliberately.
- Validate required CSV fields, unique IDs, all declared splits, exact-text
  cross-split duplicates and same-text conflicting claims before output creation,
  registry access or model imports. Do not infer ground truth from these checks.
- Verify the new pilot's delivered identities, counts, pinned source revisions,
  unknown labels, review/training/split states and provenance/context coverage
  without modifying it, constructing labels, assigning splits, or training.
- Do not automatically map its three categories into the legacy eleven-label
  closed-set encoder. Partial labels need explicit observation masks and a
  reviewed schema; unlisted labels are not automatically audited absences.
- Run focused regressions and both existing integrity validators. Finish with a
  fresh isolated full TeX/BibTeX build, visual checks, backed-up verified local PDF
  refresh, then reviewed task-owned commit/push under standing authorization.

Use cross-paper compute and SCAFE advisory locks for the bounded audit and build.
No installs, downloads, model inference/training, original-data edits or scientific
accuracy/Q1-readiness claims are part of this iteration.

### Verified handoff and intake outcome

All 18 delivery hashes were independently verified before and after the read-only
handoff audit. There are 4,299 staging records and 2,323 standalone candidates.
The latter retain 6,969 unknown unified pattern cells, zero reviewed/training-
eligible records and zero assigned splits. Of the candidates, 1,680 lack project
IDs and 1,570 are marked as missing context; 753 report source-file availability,
which does not establish sufficient caller/modifier/state context. There are 15
named source/project metadata pairs and one reported token group spanning sources.
These are producer metadata/count checks, not semantic or clone revalidation.

The new pilot's path, pinned revisions and intended unreviewed state are now
established from authorized files. Training remains blocked by design until
independent annotation/adjudication, context/localization review, explicit label
semantics/observation masks, project/near-clone overlap controls and frozen splits
are supplied. Its three categories are not mapped into the eleven-output legacy
classifier. No acquisition, extraction, data construction, relabeling or training
was repeated, and no original or queue was modified.

The independent legacy intake bug is fixed: unknown, missing, unsupported and
contradictory CLEAN+positive labels now raise; surrounding whitespace on valid
labels is recognized. Only sole explicit CLEAN yields the all-zero legacy
closed-set target. No claim that CLEAN means verified security is made. Malformed
CSV fields, duplicate IDs, incomplete/invalid splits, exact-text split leakage and
same-text conflicting claims fail before output creation, registry access or
model imports. Exact identities do not collapse whitespace within literals and
do not purport to detect near clones or project leakage.

Code review also found that resume configuration used row count without binding
label/source/split contents. A new ordered input fingerprint is stored in resume,
overlap and training metadata. Same-count label/code/split edits and row-order
changes now alter that identity; incompatible historical resumable states fail
the existing configuration equality check. No historical state was edited.

The 36 focused unittest tests (31 function/intake/handoff tests plus five frozen-
diagnostic tests), both integrity validators, and `git diff --check` passed.
The full pytest suite remains unavailable and was not claimed run. Tests use
controlled software fixtures, not fabricated research annotations. Details and
limits are in `docs/FUNCTION_DATASET_INTAKE_20261102.md`. The independent handoff
report is in the new ignored `function_pilot_handoff_20261102_1000/` result directory.

The primary manuscript claims are unchanged. Final fresh isolated compilation,
visual inspection, safe backed-up PDF refresh, final/staged review and authorized
commit/push are required before reporting this iteration complete. Actual final
outcomes are recorded outside the repos in the task workspace's
`scafe_iteration_20261102_1000/run.json` and `manuscript_build/build_report.json`.
No accuracy improvement or Q1-readiness claim is made.


## 2 November 2026, 12:00 Israel — actionable function review pilot

Hypothesis and success criteria were recorded before evaluation in the task
workspace `scafe_iteration_20261102_1200/run.json`. Prepare 24 source-balanced,
short, distinct-input/family candidates with independent two-reviewer templates,
three categories and three explicit axes, exact evidence spans, unknown-preserving
comparison and an adjudication queue. Preserve all delivered originals and source
claims separately; do not create training labels, assign splits or run a model.
No full-source snapshot cache is present in the delivered pilot, so provided
context is snippet-only and contextual/vulnerability labels cannot be promoted.
This is a feasibility/review-tooling advance, not classifier evaluation.

### Verified review-pilot outcome

Created the new local derivative
`scafe_function_dataset_pilot/derivatives/review_pilot_v1_20261102T1200/`: 24
functions, eight per source, 216 explicit category/axis cells per reviewer (432
across two blank packets), a readable claim-withheld view, coordinator-only
provenance key, rubric and immutable artifact manifest. Distinct exact model-input
hashes and producer family hashes in the selected queue are 24/24, not a global
near-clone or project-independence certification. Sixteen selected candidates lack
project IDs; 16 report source availability and eight missing context, but no
full-source snapshot bytes are present in the delivered handoff. This version
therefore records snippet-only context and blocks non-unknown contextual suspicion
or vulnerability judgments. It neither relabels the originals nor assigns splits.

All seven final kit files, including its manifest, were byte-identical on replay.
The 18 delivered originals passed before/after hash checks. Blank actual packets
were rejected by the comparison CLI with no output artifact. The two reviewer
forms are pending independent completion, not completed annotations. All 2,323
original candidates retain unknown labels, zero reviewed/training-eligible rows
and zero assigned splits. No human review or model training was performed.

Fifty focused stdlib unittest tests passed (45 function/intake/handoff/review tests
plus five frozen-diagnostic tests), including 14 new review regressions; both
integrity validators passed. Full pytest remains unavailable; no installation was
attempted. Scientific value is an executable, evidence-preserving annotation
workflow and concrete small review action, not measured accuracy improvement.
See `docs/FUNCTION_REVIEW_PILOT_20261102.md` for rubric, limitations and commands.

Required end-of-run fresh isolated manuscript compilation/visual checks, safe
backed-up local PDF refresh, final/staged review and authorized commit/push are
recorded in the task workspace `scafe_iteration_20261102_1200/run.json` and its
`manuscript_build/build_report.json`. Manuscript claims are unchanged. The next
independent advance is recovering pinned context bytes/provenance and registering
a new context-enriched release while arranging the bounded first-pass review.
This pilot cannot become a training benchmark merely through reviewer agreement.


## 2 November 2026, 13:00 Israel — pinned local context recovery

Predeclared hypothesis, bounds and success criteria are in the task workspace
`scafe_iteration_20261102_1300/run.json`: recover only exact pinned context texts
from existing authorized local corpora, preserve raw/text hashes and original
labels, quarantine existing-corpus overlap, and report coverage honestly. No
network acquisition, source execution, training or split assignment is authorized
for this bounded advance. The initial directory walk hit its 100,000-directory
budget before output creation; bounded null-delimited rg listing replaced it.

### Verified context recovery outcome

Recovered 21 exact pinned source texts from 22 existing local files, covering
136/2,323 candidates (5.85%): 112 CoinFabrik and 24 SCRUBD. All 136 linked raw
function fragments passed exact hash/character-span checks. This establishes
inspectable bytes and lexical fragment identity, not sufficient semantic context,
AST validity or annotation truth. For CoinFabrik the pinned producer offset is
checked; the single candidate span status is not global semantic uniqueness.

The new local derivative is
`scafe_function_dataset_pilot/derivatives/local_context_v1_20261102T1300/`.
Its 21 source files and five metadata files reproduce byte-for-byte on a second
run. Of 212,823 listed local Solidity paths, only 48 basename-matched files were
read. The initial stdlib walk exceeded its directory budget before output; the
bounded existing rg listing resolved that resource issue. All 18 delivered hashes
and all 48 selected local source hashes were checked unchanged. No sources were
downloaded, executed, modified or uploaded.

Every recovered candidate is flagged `quarantine_existing_corpus_overlap`, since
matches reside in local FORGE or smartbugs-wild corpora. This is presence evidence,
not a determination of historical train/test membership. Audit-folder hints do
not become project IDs. All original labels, unresolved licenses and missing
project IDs remain unchanged; zero independent annotations, training-eligible
records or frozen splits were added. 2,187 candidates still lack verified context
in this derivative. CodeBERT was not trained and no accuracy gain is claimed.

Four records in the old 24-function review queue have recovered source hashes;
that immutable snippet-only queue is not retroactively upgraded. Full source may
still omit imports, callers or relevant callees, and its comments may reveal
annotation claims. A separately versioned context-enriched review protocol and
context-sufficiency checks are next; raw evidence is not a sanitized model input.
See `docs/FUNCTION_CONTEXT_RECOVERY_20261102.md` for bounds, schema and limits.

Sixty-two focused unittest tests passed (57 function-related tests including 12
new context-recovery regressions, plus five frozen-diagnostic tests). Both project
integrity validators passed. Full pytest remains unavailable and was not installed
or claimed run. Final fresh isolated TeX/BibTeX build, PDF visual checks/backed-up
local refresh, final/staged review and commit/push outcomes are recorded in the
task workspace `scafe_iteration_20261102_1300/run.json` and its build report. The
manuscript claims are unchanged; context availability is the measured data gain.


## 2 November 2026, 14:00 Israel — context-enriched review release

The interrupted startup left no staging artifacts, owner file or active run.
After reconnection, clean worktrees, free locks and all 18 original hashes were
verified before resuming the same iteration. Predeclared goals and input hashes
are in task workspace `scafe_iteration_20261102_1400/run.json`. Prepare a separate
context-sufficiency review release for the four exact recoverable records from
the prior review queue, preserving raw sources, claims and all pattern unknowns.
Mask comments while preserving literals, Unicode offsets and source line endings;
expose dependency gaps without declaring closure or promoting labels.

The exact-legacy-view criterion initially failed: three fragments were identical,
one differed only in 15 comment carriage returns formerly masked as spaces. The
amended boundary permits only raw CR/legacy-space at verified comment positions
or endings, records both hashes/counts and rejects every other difference. No
output existed at that failed check. This is a separately versioned review view,
not a silent change to original model inputs or checkpoint lineage.

### Verified context-review outcome

Created the separate local derivative
`scafe_function_dataset_pilot/derivatives/context_review_v1_20261102T1400/` with
four full-source review views, exact function links and 12 category-specific
context-assessment cells per reviewer. This is the exact recoverable intersection
of the earlier 24-function queue: three SCRUBD records and one CoinFabrik record.
All raw contexts and prior packets remain unchanged. 118 comment spans are masked
with string literals, Unicode offsets and source CR/LF preserved; one import
directive is exposed as unresolved. Dependency closure is unverified for all four,
including zero-import cases. Both legacy/new hashes and the one 15-character CR
comment-view difference are recorded. Generator and both input manifest hashes
bind the output. All final artifacts were byte-identical on replay.

Two independent context-assessment templates require per-category reasons, exact
document-bound evidence, assumptions/dependency reasoning for sufficient proposals
and concrete missing evidence for insufficient proposals. Every cell, including
agreements, enters independent context adjudication; the tool never passes a
context gate, promotes pattern labels or assigns eligibility/splits. Actual blank
packets were rejected by the CLI with no output. No reviewer was contacted, no
independent review was completed and no training occurred. Existing-corpus
quarantine and license/project unknowns remain. Overall recovered raw context
coverage stays 136/2,323; first-pass context-view availability improves from zero
to four. This is review-tooling/data-readiness work, not accuracy improvement.

All 74 focused unittest tests passed (69 function-related, including 12 new
context-review tests, plus five frozen-diagnostic tests). Both integrity validators
passed. Full pytest remains unavailable and was not installed/run. The brief
shared-compute-lock contention deferred preparation without overlapping jobs; it
resolved before final generation. The earlier interrupted startup was confirmed
not to have created a stage or active run before safely resuming.

Fresh isolated TeX/BibTeX build/visual checks, backed-up verified local PDF refresh
and reviewed commit/push evidence are in task workspace
`scafe_iteration_20261102_1400/run.json` and its build report. The manuscript claims
are unchanged. The next gate is independent category-specific context adjudication
and resulting evidence requests, followed by separately versioned pattern-label
annotation, project/clone controls and frozen evaluation cohorts. Agreement is not
truth, sufficient context is not exploitability, and unknowns are never negatives.
