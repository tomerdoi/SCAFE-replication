# SCAFE experiment ledger

## Standing operating direction — fast, evidence-backed outcomes (4 November 2026)

Strongly optimize for the shortest credible path to a defensible research
contribution and a useful working detection pipeline. Prefer fast, high-quality,
evidence-backed improvements. This direction supplements the existing task;
it changes no automation schedule, objective, reporting rule,
permission, security/privacy or no-pay constraint, scientific-integrity rule,
validation requirement, known limitation, PDF safeguard or Git safeguard.
Historical entries below describe their own runs; they do not override current
standing instructions. No new experiment is authorized by this maintenance entry.

Maintain at most three rolling outcome milestones. Each must name a concrete
deliverable, measurable completion checks, dependencies, the current critical
path and a realistic conditional effort/time estimate. Prefer milestones that
fit days or a few existing bounded sessions when evidence and capacity support
that estimate. Do not invent deadlines, promise successful research/publication,
or stretch work into months without naming the dependency that requires it.

Once a baseline is established, an audit must resolve a specific decision, risk
or regression and lead to implementation or a decisive experiment. Reuse valid
prior results; repeat a check only when relevant inputs changed or a standing
validation requirement explicitly requires it. Do not count another status check,
repackaged queue, tool or unchanged audit as an outcome improvement. During
existing authorized work sessions, execute the highest-value critical-path step.
Safely parallelize genuinely independent work only within the applicable agent,
lock and resource rules; avoid conflicting writes and duplicate experiments.
When a dependency blocks one step, continue another useful authorized step.

Report concrete evidence of what improved, the next deliverable and actionable
blockers under the existing reporting rules. If a milestone slips, identify the
actual dependency/failure and revise its conditional estimate or deliver a smaller
credible result; do not repeat an open-ended plan. Keep all quality gates: tests,
baselines, ablations, reproducibility, uncertainty, negative results and limitations
must not be weakened to appear faster. Reviewed-label coverage, justified training
eligibility, decisive experiments and reproducible evidence are outcome measures.
Candidate/source/rule proposals never become ground truth by relabeling, and
training must never be implied when it did not occur.

### Source-supervised training and independent evaluation — 4 November 2026

Reuse existing upstream labels and audit findings efficiently as source claims,
with pinned provenance, original meanings, exact source/function references and
explicit supervision tiers. A source-derived training label is not an independently
validated evaluation label. Map a claim into a target category only when its
definition and exact function span are justified. Preserve per-category observation
masks; absent, unsupported or context-dependent categories remain unknown.

Training may use source/weak supervision under a documented protocol with targeted
stratified sample auditing, measured label quality and adjudication of ambiguous
or conflicting cases. There is **no requirement for two new independent reviews
of every training example**. Machine/heuristic/LLM annotations remain explicitly
weak/proposed rather than independent ground truth; retain their provenance and
audit evidence. Do not require manual independent relabeling of the entire training
corpus before making justified source-supervised progress.

Maintain a separate, independently reviewed and adjudicated evaluation cohort.
Use qualified independent review, appropriate blinding and explicit unknowns for
that cohort and for cases requiring adjudication. Keep project/clone-aware grouped
splits and the existing near-clone, pair/version, source-overlap and external-set
controls. Compare rigorous matched majority/nuisance/rule/lexical baselines before
a bounded CodeBERT pilot; source/weak labels are not independent test truth.

Do not fabricate category mappings or synthetic negative labels. Patched functions
and no-known-vulnerability examples are not blanket verified negatives; a verified
repair is finding-specific evidence, not proof of global safety or absence of
other category instances. Keep audit descriptions/explanations and source-label
cues out of model inputs. Do not evaluate against labels created by the same
extraction/labeling rule and call that independent correctness.

At the 4 November 2026 policy review, the inspected importer initialized all three
harmonized targets to null and preserved upstream annotations separately; the
adapter, supervision tiers, observation-mask handling, stratified audit and
independent evaluation protocol were not implemented by that documentation update.
Verify current implementation and artifact status in future sessions rather than
treating this dated observation as a permanent blocker. No staging label is
promoted and no training starts merely because instructions are revised.
Source-supervised eligibility must be explicitly justified; it need not wait for
fresh independent review of every training row. The three-category/legacy-eleven
schema boundary and all compute, privacy, validation, PDF and Git safeguards remain.
The immutable delivered README/reviewer kits describe their original snapshot;
this standing policy governs future supervision work and narrows any blanket
review interpretation in older instructions.

### Rolling outcomes and current critical path

The historical baseline (3 November, 14:00 entry) has zero independent reviews,
zero training-eligible records and no assigned splits. It provides 28 call-shaped
sites in 21 functions, with ten priority functions for possible ignored-result
review. These are source observations, not confirmed suspicious patterns. Refresh
this baseline from current artifacts at the next scheduled work session; this
instruction update performs no new dataset assessment or experiment. These zero
counts refer to the harmonized/independent-review fields, not an absence of upstream
annotations. Reuse justified source labels through a versioned adapter rather than
require new independent labels for every training function.

| Outcome milestone | Concrete deliverable and completion checks | Dependencies / critical path | Conditional estimate |
|---|---|---|---|
| 1. Reusable source-supervision core | Deliver a versioned category crosswalk and source-label adapter with pinned provenance, exact function mappings, original meanings, supervision tiers and partial observation masks. Begin with justified native annotations (such as SCRUBD RE); leave other categories unknown. Report mapped/observed class supports, exclusions and source-training eligibility separately from independent review. | Highest-value next step: compatible target definitions, exact function linkage, existing source metadata and the current fail-closed intake. Resolve ambiguity; do not invent labels. The ten call-evidence priority functions are a targeted audit aid, not a mandatory manual pass over the whole training corpus. | Target one or two existing bounded sessions for a source-supervision export/protocol if definitions and local evidence suffice. If blocked, finish the supported subset and precise mapping/dependency decisions in that window; no training is started by this policy update. |
| 2. Measured label quality and independent evaluation core | Audit a justified stratified sample of source/weak labels and adjudicate ambiguous/conflicting cases; report uncertainty, error rates and resulting exclusions. Separately create an independently reviewed/adjudicated evaluation cohort with project/clone-aware frozen splits. Preserve unknowns and quarantine, and document source-supervised training eligibility without calling it gold truth. | Milestone 1 plus sample design, qualified independent evaluation review, context/provenance/license sufficiency and legitimate grouped cohort gates. Two new reviews of every training row are not required. Missing review blocks the dependent evaluation claim, not all useful source-supervision work. | Target one bounded session for the audit/cohort protocol once the adapter is available; auditing/adjudication effort depends on sample size and actual ambiguity. Integrate complete review submissions in a bounded session when available. Reviewer turnaround is unconfirmed; set no invented calendar deadline and continue independent supported work. |
| 3. Decisive source-supervised pipeline experiment | Run predefined matched-cohort majority/nuisance/rule/lexical baselines and, only when data/checkpoint/compute gates pass, a bounded CodeBERT pilot with a loss compatible with observation masks. Deliver reproducible inputs/checkpoints, grouped development selection, source-label versus reviewed-evaluation results, per-class supports/metrics/uncertainty, error analysis and a proceed/change/stop decision for integration. A negative result is a decision, not an accuracy gain. | Milestones 1–2 with sufficient class/project support and measured supervision quality, a frozen independently reviewed evaluation protocol, existing dependencies/checkpoint and an affordable measured runtime. Source/weak training supervision is permitted; no inspected external set becomes a fresh test and no extraction rule supplies its own correctness oracle. | Aim for one bounded baseline session after gates pass. Estimate the encoder pilot from actual volume/checkpoint/runtime; if blocked, deliver the runnable protocol and measured blocker rather than launch a large campaign or promise strong results. |

### Latest adapter checkpoint — 4 November 2026, 10:00 UTC

Initial source-supervision adapter implemented in a separate versioned layer:
33 exact-span RE proxies (14 positive/19 negative), preserving all 110 native RE
claims; 77 remain mapping-pending. These are source assessments, not independently
reviewed gold or three-axis pattern labels. Eight stratified pilot audit rows are
prepared, not completed. Training/evaluation eligibility and splits remain zero.
Next critical path: targeted source-quality/meaning audit, independent grouped
evaluation readiness and compatible masked learning, then matched baselines and
gated CodeBERT. Current artifacts/entry below supersede historical readiness counts.

### Execution gates and unambiguous run states — 4 November 2026

Use this dependency sequence; the rolling table groups these gates into three
near-term outcomes rather than omitting later integration/publication work:

1. **Adapter/data definition:** justified category definitions and exact function
   mappings; a versioned provenance-preserving source-label export with observation
   masks and a compatible loader/loss protocol. A policy document alone is not an
   implemented adapter or a training-eligible dataset.
2. **Supervision/evaluation readiness:** targeted stratified source-label audits,
   quality measurement and ambiguous-case adjudication; adequate documented class
   and project support; a separate independently reviewed/adjudicated evaluation
   cohort; frozen project/clone-aware memberships and leakage controls. Declare
   acceptance checks before evaluation. Source/weak training labels remain distinct
   from evaluation truth; no blanket double re-review of every training row.
3. **Baseline decision:** matched majority/nuisance/source-only, rule and lexical
   baselines under the same legitimate grouped protocol and label masks, with
   class supports, uncertainty and error analysis. Advance after the declared gate
   passes; a valid negative result may justify changing or stopping that hypothesis.
4. **Bounded CodeBERT pilot:** only after the preceding gates and existing-checkpoint,
   dependency and compute readiness pass. Freeze inputs/configuration/checkpoint
   lineage, use legitimate development selection and retain the required ablations,
   truncation/leakage analysis, uncertainty and seed reporting. Do not imply that
   an unrun pilot trained or improved a model.
5. **Contract-score integration and article evidence:** only after credible
   function-level evaluation, test frozen aggregation/fusion against fair contract
   baselines. Update manuscript claims from reproducible measured evidence, with
   failures and limitations. Compilation is not scientific validity; Q1 publication
   remains a goal with no acceptance or performance guarantee.

At session start, read current code/artifacts and current standing guidance.
Every count, metric, checkpoint claim and availability observation in dated entries
is historical evidence unless reverified for the present decision. Reuse valid
prior evidence; do not redo unchanged audits merely to show activity. A completed
gate should lead to the next authorized critical-path deliverable. Missing review
or resources block dependent work, not unrelated supported adapter/data work.

Record each step as **not attempted**, **running/checkpointed**, **passed**,
**failed**, or **blocked (denied/pending/uncertain)**. A run is complete only when
its authorized deliverable and required checks pass; otherwise checkpoint safely,
identify the exact unmet gate/action/target/reason and give a smaller deliverable
or revised conditional estimate. Distinguish local edits, actual label promotion,
training/inference, local commits and remotely verified pushes. A review-only turn
may correctly end with unstaged local instruction edits and no Git publication.

Use the SCAFE run lock `/tmp/scafe-research-iteration.lock` and shared compute lock
`/tmp/paper-research-compute.lock`, checking ownership/liveness. Coordinate or defer
your conflicting writes; never stop another writer/job or remove its lock. Release
only your own locks on safe completion/checkpoint. Respect denials and standing
stop/report rules; no indirect retry through a different command or another commit.
This instruction review changes no permission, schedule, enabled state, timezone,
notification destination, runtime/model setting or other safeguard.

Update this short table as outcomes and dependencies change. The immediate next
proposed deliverable is milestone 1; this maintenance update does not start it.

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


## 2 November 2026, 15:00 Israel — all-candidate split constraints

Predeclared hypothesis/protocol and input identities are recorded in task workspace
`scafe_iteration_20261102_1500/run.json`. Build exact/provenance constraints for all
2,323 candidates, then a separate consistent-identifier-template sensitivity graph.
Preserve literal values/member selectors and distinguish verified input identities
from source-asserted project/address/context metadata. Propagate existing raw-context
corpus-presence seeds conservatively, without claiming extra confirmed source
matches, new project identities, semantic clones or actual train/test assignments.
No labels, model training or performance-based graph selection are involved.

### Verified grouping outcome

All 2,323 candidates now have two versioned split-constraint memberships, stable
component IDs and typed edge provenance. The exact/provenance graph has 1,326
components (largest 68, one cross-source component, no multi-project metadata
component). Adding identifier-template sensitivity constraints gives 1,228
components (largest 170, four cross-source components, three multi-project metadata
components of sizes 170/150/109 spanning 4/3/2 recorded project anchors). Template
relationships may be common boilerplate or distinct callees/types; they are not
verified semantic clones, project identities or proof of leakage under an unrun
split. The source/metadata versus template distinction is retained explicitly.

The 136 directly verified local-context corpus-presence seeds remain 136. Their
conservative component quarantine propagates to 255 evidence-graph records (+119)
and 307 strict-template records (+171). These are policy-derived exclusion scopes,
not new independently confirmed raw-source matches. Missing original project IDs
remain 1,680; no new IDs, labels, eligibility or split assignments were inferred.
Raw context coverage remains 136/2,323 and independently reviewed labels remain zero.

Lexical signatures were computed for 2,321 records (1,865 token signatures; 1,743
identifier-template signatures). Two HF records fail the new heuristic literal
scanner and are retained with explicit representation-review status and no guessed
token/template edges. This is not a compiler rejection; original snippets remain
unchanged. The new local derivative is
`scafe_function_dataset_pilot/derivatives/constraint_groups_v1_20261102T1500/`.
All five artifacts reproduce byte-for-byte. An independent registry check verified
both complete/disjoint partitions, recomputed component IDs, all 2,771 typed spanning
edges held within their declared graphs, monotonic quarantine and unchanged unknown
labels/false eligibility/unassigned splits. All 18 original hashes and the recovery
release are preserved; no source/data/reviewer payload was uploaded.

All 83 focused unittest tests passed (78 function-related, including nine new
grouping regressions, plus five frozen-diagnostic tests), as did both integrity
validators. Full pytest remains unavailable and was not installed/run. Scientific
value is measured prospective evaluation constraints, template sensitivity and two
representation-review findings, not classifier accuracy or dataset-finalization.
See `docs/FUNCTION_GROUPING_20261102.md` for protocol, reproduction and limitations.

Fresh isolated TeX/BibTeX build/visual/PDF-refresh checks, final/staged review and
commit/push evidence are in task workspace `scafe_iteration_20261102_1500/run.json`
and its build report. Manuscript scientific claims are unchanged. Next inspect
strict-template bridges/overmerge, recover the two flagged representations, resolve
provenance and independent labels, then prespecify frozen grouped evaluation rather
than selecting a favorable grouping using model performance. No training occurred.


## 3 November 2026, 10:00 Israel — inline audit-prose model-view filter

Predeclared hypothesis, narrow acceptance rule and input identities are in task
workspace `scafe_iteration_20261103_1000/run.json`. Two literal-scanner failures
contain audit-issue prose after completed statements, outside comments. Audit the
whole 2,323-record cohort and create a separate view masking only that exact
post-semicolon form outside literals/comments. Preserve CR/LF/Unicode offsets,
original files, unknown labels and all code outside recorded spans. Report every
residual honestly; recompute signatures/groups without stale producer hashes.
No training, compiler-validity, gold-label or accuracy claim is made.

### Verified inline annotation-view outcome

The whole-cohort audit found seven declared inline audit-marker occurrences.
Six post-semicolon tails in six records satisfy the predeclared outside-literal/
comment rule and are masked in the new versioned view. One marker follows a block
opener and remains untouched with explicit representation-review flags; the rule
was not broadened to conceal that residual. Both known apostrophe-induced literal
scanner failures resolve under the frozen heuristic tokenizer: successes rise
2,321→2,323, failures 2→0. This is not compiler/semantic validation or recovery of
an independently audited original contract. Source claims never become labels.

The new local derivative is
`scafe_function_dataset_pilot/derivatives/annotation_view_v1_20261103T1000/`.
All 2,323 views bind parent/new input hashes, exact masking spans and ancestry;
producer token/template fields are omitted instead of copied stale. Three-category
labels remain null with explicit all-false observation masks (unknown, not negative).
All rows remain unreviewed, training-ineligible and unsplit. Original records and
prior derivatives remain intact, with annotation prose available only in parent
evidence rather than new model code for the six handled tails.

Recomputed grouping uses the frozen implementation and actual context-overlap
seeds. Evidence components decrease 1,326→1,324; strict-template components decrease
1,228→1,226, exposing additional token-signature duplicate constraints. Raw-context
overlap stays 136, conservative quarantine stays 255/307, and all provenance/clone/
overmerge limitations remain. Old input signatures, checkpoints and group releases
cannot be silently mixed with the new representation. No model was trained or
rescored and no accuracy gain is claimed.

All nine final artifact files reproduce byte-for-byte. Independent checks verify
all changes lie inside the six hashed masking spans, unchanged Unicode offsets
and CR/LF positions, all unknown/unobserved targets, no eligibility/splits or stale
producer fields, and all 2,775 recomputed graph edges held inside their declared
complete/disjoint partitions. All 18 delivered originals and input release hashes
were preserved. All 94 focused unittest tests passed (89 function-related including
11 new inline-filter regressions, plus five frozen-diagnostic tests), and both
integrity validators passed. Full pytest remains unavailable; no installation/run
was claimed.

Fresh isolated full TeX/BibTeX build/visual/PDF-refresh and reviewed commit/push
proof is recorded in task workspace `scafe_iteration_20261103_1000/run.json` and
its build report. See `docs/FUNCTION_INLINE_ANNOTATIONS_20261103.md` for acceptance
rule, residual stable ID, schema, lineage and limits. Next resolve the remaining
block-opener artifact from original evidence and independent context/category
review. Other cue types, labels, licenses, provenance and frozen-evaluation gates
remain unresolved; tokenizability is not training readiness or Q1 readiness.


## 3 November 2026, 11:00 Israel — guarded if-block view v2

Work began after the scheduled 11:00 Israel start, following read-only preflight.
Predeclared hypothesis and acceptance gates are in task workspace
`scafe_iteration_20261103_1100/run.json`. Extend the filter only via explicit v2
opt-in and an immutable prior v1 release, for a same-line complete balanced
`if (...) {` header outside literals/comments, rejecting any brace or semicolon
in the annotation tail. Reproduce every default v1 code view before extension,
allow changes only to prior residual-flagged records, preserve all prior masks,
source offsets, unknown labels and ancestor hashes. No training, compiler/gold-label
or performance claim is involved. The staged-import precedence issue rejected a
new keyword before processing; explicit loaded-module identity corrected it.

### Verified v2 outcome

All 2,323 default v1 code views reproduced against their immutable parent before
extension. Exactly one record changes relative to v1:
`b407a429c804fffa264515bba167e0c7bcf394622854252df70c3bb0baf50db8`.
The new event follows a complete balanced same-line if header and its annotation
tail contains no brace/semicolon. The condition, block delimiters and statements
remain outside the recorded mask span. All six old semicolon masks are unchanged.

The separate local release is
`scafe_function_dataset_pilot/derivatives/annotation_view_v2_20261103T1100/`.
There are seven total masks versus delivered originals, zero residual occurrences
of the declared audit-issue marker and zero flags for this declared representation
issue. Lexical-signature success remains 2,323/2,323, evidence/strict components stay
1,324/1,226, direct raw-context overlap stays 136 and conservative quarantine remains
255/307. These are source-view/constraint facts, not compiler, semantic, gold-label,
performance or complete cue-removal validation. Independent reviews, observed
labels, training eligibility and actual splits remain zero.

Every v2 model record binds original and direct-parent input hashes/protocols;
prior manifests, ancestor hashes and generator/helpers are retained. Existing
human labels/review states in a parent require explicit migration rather than a
silent reset. All nine generated artifacts reproduce byte-for-byte. Independent
checks verify one prior-view delta, seven hashed spans (six old + one if block),
unchanged Unicode/CR/LF positions and out-of-span code, fresh group input hashes,
complete/disjoint partitions and held-together edges, with unknown/unobserved
labels and no eligibility/splits. Originals and prior releases remain unchanged;
no source/candidate/reviewer data was uploaded.

All 99 focused unittest tests passed (94 function-related, including five new
if-block regressions, plus five frozen-diagnostic tests), as did both integrity
validators. Full pytest remains unavailable and was not installed/run. The old
module selected during first staged validation rejected the new keyword before
processing; corrected path precedence plus loaded-file assertion resolved it.

Fresh isolated full TeX/BibTeX build/visual/PDF-refresh and final/staged/Git proof
are recorded in task workspace `scafe_iteration_20261103_1100/run.json` and its
build report. See `docs/FUNCTION_IF_BLOCK_ANNOTATIONS_20261103.md` for exact scope,
reproduction and limits. Next pursue independent context/category evidence and
provenance/license/frozen-evaluation gates; finishing this known annotation form
is not training readiness or a Q1-journal performance result. No training occurred.

## 3 November 2026, 12:00 Israel — bounded filename-independent context recovery

Predeclared hypothesis: source files missed by the prior basename prefilter may
still exist under different names in authorized local corpora. Before evaluation,
workspace `scafe_iteration_20261103_1200/run.json` fixed a corpus-round-robin sorted
search, at most 2 GiB / 90,000 read files, 16 MiB/file and 500,000 listed paths.
Only exact producer-pinned whole-source text hashes count as recovered context;
original function hashes/spans are checked separately. Similarity cannot establish
source identity. Shared compute locks serialized scan/replay/tests with the active
data-centric task; one validation attempt did not start because its lock was busy.

The search listed 212,823 Solidity files and checked 90,000, reading 1,071,278,311
bytes before the file limit. Five newly matching whole-source texts cover nine
additional SCRUBD functions. Context coverage increases 136/2,323 (5.85%) to
145/2,323 (6.24%); documents increase 21 to 26. Six new functions have unique exact
original-fragment spans; three have multiple identical fragments and stay ambiguous.
Total unique original-fragment localizations are 142, not 145. Dependency closure,
compiler validity, source authenticity beyond producer pins, license clearance and
independent category truth are not established by these lexical source matches.

Two new text matches occur in DIVE and three in local paper-benchmark corpora.
All nine are new direct corpus-presence overlap seeds: 136 to 145. On the unchanged
current annotation-v2 model views, evidence/strict component counts remain
1,324/1,226; conservative quarantine increases 255 to 264 and 307 to 316. Propagated
quarantine remains a sensitivity constraint, not independently confirmed extra
raw-source overlap. No candidate is promoted to a fresh holdout or training set.

Separate local derivative:
`scafe_function_dataset_pilot/derivatives/content_context_v1_20261103T1200/`.
All 52 release files reproduce byte-for-byte in a full second preparation. The
manifest binds the original delivery, prior recovery, current model view, generator
and helpers. Combined links, contexts and group constraints retain unknown labels,
no observation/eligibility promotion and no assigned splits. Original 18 delivered
files and earlier derivatives remain unchanged; no raw data is staged or uploaded.
Independent checks passed for context/fragment hashes, six unique versus three
ambiguous new localizations, unchanged old links, input hashes, partitions/edges,
unknown labels and corpus quarantine. The reference review's P1 context/dependency
and cohort-separation gaps motivate this evidence work, not an architecture claim.

The first staged preparation failed a stored-context checksum because universal
newline reading changed an old CRLF file. It produced only partial workspace
output, retained separately without a release manifest. Exact byte decoding fixed
it and a CRLF regression now passes. Direct scanner tests also exposed unresolved
macOS /var aliases; root resolution was made consistent without relaxing containment.
Failures and resolutions are preserved in the workspace, not hidden as successful
runs. No original/source input changed.

All 107 focused tests passed: 102 function-related including eight new recovery
regressions, plus five frozen-diagnostic tests. Both experiment/submission integrity
validators passed. Full pytest remains unavailable and was not installed or run.
Fresh isolated full TeX/BibTeX build, visual/PDF-refresh and final/staged/Git evidence
are recorded in the current workspace run report and build report. Manuscript
scientific claims and all current model-code views are unchanged. Independent
reviews, training-eligible records and assigned splits remain zero; no model
training, inference or accuracy gain is claimed. Next inspect the newly available
context for category/dependency evidence and resolve ambiguous enclosing-contract
mappings before any adjudication or frozen evaluation design.

## 3 November 2026, 13:00 Israel — qualified lexical function scopes

Predeclared scope-mapping hypothesis and conservative acceptance rules are in
workspace `scafe_iteration_20261103_1300/run.json`. Across all 145 context-covered
records, test whether the explicit producer contract-qualified function identifier
agrees with one exact original fragment at direct member depth in its lexical
contract/library/interface declaration. Comments/literals cannot create scopes;
braces/body boundaries must match. Repeated owners, multiple alternatives,
unsupported qualifiers/forms and malformed scope evidence stay unresolved. No
first-match selection, source execution, corpus rescan, inference or training.

All three previously ambiguous records have one owner-compatible lexical selection:
RC.giveToken at character 11,315 (line 314), RC.refundEther at 12,328 (line 345),
and RCpro.refundEther at 19,566 (line 549). RC starts at source line 168 and RCpro
at 360. The refund functions have identical bytes; qualifiers distinguish their
lexical definitions conditionally. Producer qualifier truth is not independently
verified. Original raw uniqueness remains 142 and all three multiple-span lists
are preserved; do not relabel this as 145 unique raw fragments.

Whole covered-cohort results: 139 unique named-function/qualified-owner mappings,
six unsupported special forms (five constructor qualifiers and one receive), and
2,178 records without supplied context. The six are not demonstrated owner errors
or bad labels. Initial generic unsupported/conflict diagnostics were refined to
name special forms; selection and the predeclared named-function-only rule stayed
unchanged. Context documents/coverage remain 26/145, direct overlap remains 145,
and conservative quarantine remains 264/316. No new source, project/license truth,
label, reviewed annotation, training eligibility or split was created.

New local four-file derivative:
`scafe_function_dataset_pilot/derivatives/scope_mappings_v1_20261103T1300/`.
The manifest binds the original delivery, prior context release and exact helpers.
Context artifact references resolve against that immutable input release; no raw
source is copied or uploaded. All four files reproduce byte-for-byte. Independent
checks confirm chosen raw hashes, declaration/function boundaries and qualifiers,
three explicit selections, six unsupported forms, all original alternatives,
unknown/unobserved labels, unchanged parent contexts/groups and original 18 files.
This is contextual provenance evidence, not a Solidity AST/inheritance resolver,
compiler validity, dependency sufficiency, suspicious-pattern truth or accuracy gain.

All 117 focused tests passed: 112 function-related including ten new scope
regressions, plus five frozen-diagnostic tests. Both integrity validators passed;
full pytest is absent and was not installed/run. The reference review's P1
context/dependency gap motivates this evidence step. Fresh full isolated paper
compilation, visual checks, backed-up PDF refresh and final/staged/Git evidence
are recorded in the current workspace run/build reports. Manuscript scientific
claims and current model code remain unchanged. Next use the localized context
for concrete category/dependency review; independent adjudication, provenance/
license controls, frozen grouped evaluation and simple baselines still gate CodeBERT.

## 3 November 2026, 14:00 Israel — call-result source evidence and review queue

Before evaluation, workspace `scafe_iteration_20261103_1400/run.json` fixed all 139
qualified named functions and the member spellings call/delegatecall/staticcall/
callcode/send. Transfer was excluded because address/token meaning is type-dependent.
Extract exact source spans and immediate result-use shapes; do not infer external
call semantics, safe negatives, suspicious-pattern truth or training labels.

All 139 functions were analyzed across 26 existing contexts. 28 sites occur in 21 functions:
14 call and 14 send spellings; 15 direct, 10 legacy gas/value and 3 modern-option invocations.
Fourteen sites in ten functions match the conservative complete bare-expression
result-discard rule; seven are lexical guard-expression contexts, three assignment
contexts and four ambiguous/other. No returned-expression sites occur here.
Guards/assignments are comparators, not verified handled failures or hard negatives.
Indexed/complex receivers and compound-flow windows can conservatively remain
ambiguous even when their result is actually ignored. These counts are descriptive
source facts, not precision, recall, prevalence or confirmed vulnerability yield.

New local five-file release:
`scafe_function_dataset_pilot/derivatives/call_evidence_v1_20261103T1400/`.
It includes 2,323 function status rows, 28 hash/offset-bound call-site records, a complete
21-function coordinator queue, summary and manifest. Queue priorities retain the
ten bare-result-candidate functions first and all other observed-site functions
as comparators. No reviewer was contacted or decision fabricated. Context/selected
function references bind the immutable prior context and scope releases; codeviews,
originals, groups, 145 direct overlap seeds and 264/316 quarantine counts stay unchanged.

All 28 site excerpts were inspected as source evidence. Compound/guard lexical windows
can span multiple statements, so the field is explicitly statement_window_span,
not an AST statement. This naming clarification did not change sites, selection
or dispositions. Member/invocation/window raw and masked hashes, absolute Unicode
spans, queue membership, immutable parents and unknown/unobserved label fields
passed independent checks. All five release files reproduce byte-for-byte.
Missing-context 2,178 records and six unsupported special functions have null analysis
counts, never negative labels. An analyzed function with no observed spelling is
not clean. Receiver types, CFG/value propagation, dependency sufficiency, qualifier
truth, licenses and independent category decisions remain review requirements.

All 127 focused tests passed: 122 function-related including ten new evidence regressions,
plus five frozen diagnostics. Both integrity validators passed. Full pytest is
absent and was not installed/run. Fresh isolated full paper compilation, visual
checks, backed-up PDF refresh and final/staged/Git evidence are recorded in the
current workspace run/build reports. The reference review's P1 context-aware call
question motivates this concrete evidence, not a source-label shortcut. Scientific
manuscript claims remain unchanged. Independent reviews, training-eligible records
and splits remain zero; no model inference, training, accuracy or Q1-readiness gain.
Next establish receiver/callee and result-propagation evidence for the ten priority
functions and seek independent category adjudication before CodeBERT evaluation.

## 4 November 2026, 10:00 UTC — initial source-label adapter

Recovered the prior instruction-review interruption before writes: old PID 32734
was absent, the OS SCAFE/compute flocks were free, and only old owner metadata
remained. Under the newly acquired SCAFE lock, archived that stale metadata without
removing a lock file or stopping a process. Prior guidance edits were already
committed in 202b0eb; no duplicate edits were made. Repaired the missing prior final
checkpoint report from the saved prompt/settings/build/PDF artifacts and verified
recovery state. This run used its own live owner metadata and shared compute flock
for generation/replay, tests and build; data-centric work retained its own lock.

Predeclared hypothesis and acceptance rules are in workspace
`scafe_iteration_20261104_1000UTC/run.json`. Implement source-label reuse rather
than repeat unchanged audits or require fresh double review of every training row.
The adapter preserves all 110 native SCRUBD RE claims (52 positive/58 negative) and
native UX (39 positive/32 negative/39 unknown) separately. A distinct target definition
maps source-assessed RE susceptibility as an explicit source/weak proxy only when
an original exact local function span and unique qualified owner are justified.
It is not a conversion into syntactic pattern presence, independently reviewed
contextual suspicion, exploitability or the legacy eleven-output model.

New source-proxy layer: 33 observed RE cells, 14 positive/19 negative rows, 32 distinct
model byte strings and 31 token groups. The other 77 SCRUBD RE claims remain available
but unobserved in this target layer pending exact local qualified span evidence.
There are no contradictory identical-code RE claims in this snapshot; the adapter
quarantines such conflicts if present. No UX-to-unchecked-call, authorization,
missing-label, folder, patched or no-known-vulnerability negatives are created.
Original harmonized staging cells remain unchanged/null; new source proxies are
not independent gold annotations. All 33 mapped rows retain direct corpus overlap
and group/quarantine evidence, so none is a fresh independent confirmation example.

Versioned local derivative:
`scafe_function_dataset_pilot/derivatives/source_supervision_v1_20261104T1000UTC/`.
Six files contain separate code-only model inputs and provenance-bound masked
source targets, an 8-row stratified audit pilot, summary, data card and manifest.
Inputs are byte-identical to annotation-v2; audit metadata is not concatenated to
code. Masks distinguish observed RE negatives from disabled unknown cells.
`masked_source_example` validates source-proxy schema, identities, hashes and
label/mask consistency; requesting training fails while readiness blockers remain.
No masked trainer/loss, inference, CodeBERT fitting or accuracy gain is claimed.

Audit seed 202611041000 selects up to 2 rows per RE/student-flag stratum: 8 rows across
four strata. Decisions remain null; no reviewer contacted, no noise estimate or
independent evaluation is claimed. Targeted audit, independent reviewed evaluation,
frozen grouped splits, provenance/license eligibility and compatible masked learning
remain gates. Training/evaluation eligibility and actual splits remain zero.
Independent checks passed for all source/target counts, exact spans, unchanged
inputs/groups/ancestors, masks and 8 uncompleted audit rows; six-file replay is
byte-identical. All 18 delivered originals remain unchanged and no pilot data uploaded.

All 137 focused tests passed: 132 function-related (including ten new adapter
regressions) plus five frozen diagnostics. Both integrity validators passed.
Full pytest remains absent and was not installed/run. Fresh full isolated paper
build, representative visual checks, backed-up verified PDF refresh, final/staged
privacy review and Git outcome are recorded in this run's workspace reports.
Manuscript scientific claims remain unchanged. This advances the adapter/data
milestone with real source-supervision exports, not detector-performance evidence.
Next validate the 8-row mapping/meaning pilot and source-label quality protocol,
complete independent grouped evaluation/compatible learning readiness, and expand
exact source mappings where authorized evidence supports them. Do not wait for
blanket manual relabeling of the whole training corpus or call this pilot gold.

## 4 November 2026, 11:00 UTC — source-quality target-alignment pilot

Continued the published source adapter rather than repeating unchanged intake.
The hypothesis, fixed eight-case sample and AI-versus-independent-review boundary
were declared before source inspection in workspace
`scafe_iteration_20261104_1100UTC/run.json`. Completed a single source-visible
AI technical review of all eight preselected RE/student-stratified cases, with
exact qualified spans, source-rationale hashes and 24 line-bound observations.
The eight cases span six raw contexts and six evidence components, not eight
independent projects. This is preliminary source-quality evidence, not human
review, independent evaluation, adjudicated labels or a calibrated noise estimate.

All eight pinned qualified mappings are consistent in this sample. Seven cases
show interaction before later state writes, including three source-negative
cases; five expose relevant guards outside the function-only snippet, and three
identify unverified dependency implementations. Two concrete rationale issues
need qualified adjudication: an RE0 explanation omits a separate low-level value
call; an RE1 explanation names a locally view-declared method while another
call is non-view. Neither observation proves that the original RE label is wrong.
Compiler/EVM, caller guards, callee implementations and cross-function context
remain unresolved; no vulnerability decision or label flip is fabricated.

New four-file local derivative:
`scafe_function_dataset_pilot/derivatives/source_quality_ai_v1_20261104T1100UTC/`.
It preserves original AI submissions, enriched preliminary reviews, summary and
manifest. The compiler verifies the immutable sample/source/lineage/evidence,
refuses gold/human/eligibility promotion and replacement targets/model inputs,
and supports byte-identical replay. Its integrity checks do not certify semantic
reasoning. The 2,323 candidates, 110 native RE claims, 33 mapped source proxies
(14 positive/19 negative), original 6,969 unknown staging cells, masks, model
inputs, overlap/group constraints and delivered originals remain unchanged.
Independent/adjudicated reviews, training/evaluation eligibility and splits remain
zero; no model training/inference, accuracy gain or publication readiness claim.

Milestone 1 has an implemented conservative adapter (8f8df8b); this run advances
milestone 2 with actual preliminary evidence, not independent gold. The next
small deliverable is qualified adjudication of the two precise rationale issues
with documented compiler/callee/caller sufficiency and target meaning. Review
turnaround is unconfirmed; processing complete submissions fits a bounded session.
Do not require two new reviews of every training function. Independent grouped
evaluation, justified source-quality measurement, provenance/licenses, frozen
splits, matched baselines and mask-compatible learning still gate CodeBERT.
If review is unavailable, continue supported mapping/context or independent
compatible-learning work without promoting eligibility. The reference-to-gap
review's P1 context/dependency concern motivates this decision. Detailed protocol
and evidence boundaries are in `docs/FUNCTION_SOURCE_QUALITY_20261104.md`.

Applicable tests, both integrity validators, fresh full isolated TeX/BibTeX build,
visual checks, outside-tracked PDF backup and exact-input verified refresh,
final/staged privacy review and actual Git outcome are recorded in this run's
workspace report. Scientific manuscript claims are unchanged; compilation is not
scientific validation. Keep pilot/review data local and never force-add it to Git.

## 4 November 2026, 12:00 UTC — observed-cell loss and source batch loader

Continued from 8f8df8b/259282b without repeating the eight-case audit. Qualified
adjudication of the two rationale issues is unavailable in this automated run;
advanced the independent mask-compatible learning dependency. The hypothesis,
success checks, fixed inputs and CPU-only budget were recorded before evaluation
in workspace `scafe_iteration_20261104_1200UTC/run.json`. Existing PyTorch 2.10.0
was used with one CPU thread; no model/tokenizer/checkpoint/optimizer, fitting,
inference, install, download, paid compute or private upload occurred.

Implemented `scripts/function_partial_supervision.py`: explicitly hash-pinned
source-release loading, source-proxy/schema/ID/code-hash validation, code-only
collation with inseparable masks and a three-category observed-cell-only stable
BCEWithLogits objective. Category order and source meaning are explicit; legacy
11-output code is unchanged. Optional positive weights are declared constants,
not inferred from unknowns. Invalid observed values/order/width/masks/weights and
an overflowed objective fail closed. All-unknown batches return no loss and
require skipping the optimizer step, including weight decay.

The immutable 2,323-row diagnostic has 33 observed RE proxies (14 positive/19
negative) and 6,936 unknown cells. At fixed zero observed logits, unit-weight loss
is 0.6931471805599453, unknown gradients are exactly zero and analytic observed
gradient error is zero. Changing disabled cells from NaN to 100,000,000 leaves the
objective/observed gradients unchanged. A deliberately incorrect dense-placeholder
counterfactual gives all 6,936 unknown cells gradients and dilutes observed
gradients by 211.1818; this illustrates objective construction, not current legacy
behavior, label quality or model accuracy. The default training loader correctly
refuses the actual release because readiness gates are unpassed. No eligibility
flag, label, input, group, source/audit artifact, original delivery or split changed.

This finishes a narrow usable loss/batching component, not the full training
protocol or a CodeBERT model. The current source layer observes RE only; other
outputs receive no source supervision and cannot be claimed trained. The broader
learning-protocol gate remains pending actual model/config/tokenization/optimizer
integration and reproducibility, in addition to semantic quality, independent
reviewed evaluation, grouped splits, provenance/licenses and matched baselines.
Source-proxy versus independent truth and the two unresolved rationale concerns
remain unchanged. No independent/adjudicated labels, eligible records or splits
were created. No accuracy or publication-readiness improvement is claimed.

Next semantic deliverable remains qualified adjudication of the two precise
rationale mechanisms. Reviewer turnaround is unconfirmed. Supported mapping or
independent protocol/baseline implementation can continue in bounded sessions
without waiting for double relabeling of every training function. Detailed API,
controls, limitations and reproduction are in
`docs/FUNCTION_PARTIAL_SUPERVISION_20261104.md`. Applicable tests, both integrity
validators, fresh full TeX/BibTeX/visual checks, backed-up input-matched PDF refresh,
final/staged privacy review and actual Git outcome are retained in this run's
workspace report. Scientific manuscript claims are unchanged.

## 4 November 2026, 13:00 UTC — resolve compiler part of a source-rationale concern

Selected a concrete context dependency rather than another unchanged audit or
more training infrastructure. Existing direct Solidity 0.4.24/0.4.25/0.4.26/0.5.3
binaries were found inside the project virtual environment; no installation,
download or solc-select configuration change. Workspace
`scafe_iteration_20261104_1300UTC/run.json` pinned inputs/binaries, the seven-case
matrix and optimizer-disabled Byzantium settings before compilation.

The unchanged full MonkeyScam caller compiles with 0.4.24–0.4.26 without errors.
All three selected PlayerBook calls map to CALL, including locally view-declared
getPlayerName/getPlayerLAff. PCs are 3877/4069/4261 across those three builds.
The identical two-call control uses CALL/CALL with old versions and CALL/STATICCALL
with 0.5.3. The actual source was not patched for0.5.3. Four original warnings per
actual build and zero per control are preserved, not fixed/hidden. Source-map
instruction decoding accounts for PUSH data, UTF-8 byte locations and inherited
fields, requiring one exact uniquely localized instruction per expression.

This narrows/corrects the version-insensitive implication of the prior AI concern:
view syntax alone is not a compiler-level reason to reject the source rationale
under these old builds. Solidity's official0.5.0 notes document this semantic
transition. It does not prove deployed compiler/bytecode, callee callback behavior,
reachability, unsafe state or RE=1 correctness. Qualified adjudication still needs
the actual callee/caller/deployment context. The POOHMO concern is untouched.
CALL/STATICCALL observations are not new RE labels or independent evaluation truth.

New private local 28-file derivative:
`scafe_function_dataset_pilot/derivatives/compiler_call_evidence_v1_20261104T1300UTC/`.
It contains pinned config/version outputs, seven complete compiler input/output/
stderr bundles, report and manifest. All final files reproduce byte-for-byte.
Initial/prototype/final/replay preparations total35 small compiler invocations;
all attempts are retained. No contract bytecode or model was executed, no original
source was modified and no payload is staged/uploaded. All native/source proxy
claims,33observed masks, code views, group/overlap restrictions, prior AI reviews,
18delivered originals and zero independent/adjudicated labels/eligibility/splits
remain unchanged. Scientific value is conditional language-semantics evidence
that removes a misleading review shortcut, not detector-performance improvement.

`scripts/probe_function_compiler_calls.py`, decoder regressions and
`docs/FUNCTION_COMPILER_CALLS_20261104.md` make the experiment reproducible.
Applicable focused tests, both integrity validators, fresh full isolated TeX/
BibTeX/visual checks, backed-up input-matched PDF refresh and final/staged privacy/
Git outcome are recorded in the run report. Next semantic dependency is actual
callee/caller/deployment evidence or qualified adjudication; source completion can
continue independently without blanket manual relabeling. Scientific manuscript
claims, training gates and automation schedule remain unchanged.

## 5 November 2026, 10:00 UTC — localized CoinFabrik finding proxies

Reused workspace `scafe_iteration_20261105_0900UTC/run.json` rather than repeating
the completed V6/source scan: retained totals agree with V6, exact tag/snapshot
identity is unverified, and110000checked paths are recorded without a label change.
Continued
independently of student delivery using existing CoinFabrik finding/function links
and retained raw context. Workspace `scafe_iteration_20261105_1000UTC/run.json`
predeclared643candidates/171findingobjects, source/category definitions, vulnerable-
role and exact source/span/symbol/line-anchor gates before export. No download,
installation, scan, new source revision, model/tokenizer/checkpoint/optimizer,
training or inference occurred. Shared-compute locks serialized generation/tests.

New separate source-specific layer: three positive cells across three functions,
two projects and two contexts. Two PegStability swap functions receive source-
reported RE/order concerns anchored at lines163/189. The internal Neptune
_setPool function receives an authority/privilege-policy concern with two retained
findings anchored at18–68. The two supporting findings are one target cell, not
independent labels. All three functions retain existing-corpus overlap/group
quarantine. No independently reviewed truth, score gain or fresh evaluation claim.

These definitions are not missing-guard/vulnerability labels, canonical pattern
truth, legacy11outputs or automatically harmonized SCRUBD susceptibility targets.
Venus nonReentrant modifiers and Neptune caller/privilege/initialization context
remain material quality questions. Only explicit source class/CWE plus exact
local linkage maps; guard tokens/rules do not create independent labels. All
312patched rows stay unknown, as do266without specific links and62linked but
unmapped rows. No negative cells, missing-label negatives or patched clean labels.
The new observed cells are RE2/authority1/unchecked-call0; training/evaluation
eligibility, independent gold and splits remain zero.

Private seven-file derivative:
`scafe_function_dataset_pilot/derivatives/coinfabrik_source_supervision_v1_20261105T1000UTC/`.
It preserves all171findingobjects byte-for-byte,643code-only inputs and source-
proxy rows, link checks, summary/card and manifest. All seven files replay byte-
identically. Original18delivery files,6,969unknown staging cells,33SCRUBDproxies,
prior groups/audits and annotations are unchanged. Declared raw upstream hashes
and retained-object fingerprints are distinguished; no fresh upstream blob or
underlying audit-report verification is claimed. Source payloads remain local.

The exporter/diagnostic join and regression fixtures refuse unsupported mappings,
ambiguous overload/anchor relations, patched promotion, missing-to-negative
conversion, extra prose model fields, gold/eligibility promotion and training.
The existing masked loss and SCRUBD-only loader are unchanged; a reviewed multi-
source protocol remains necessary. Detailed scope/reproduction and definitions
are in `docs/FUNCTION_COINFABRIK_SUPERVISION_20261105.md`.
Applicable tests, both integrity validators, fresh isolated full TeX/BibTeX/visual
checks, backed-up input-matched PDF refresh and final/staged privacy/Git evidence
are recorded in the run report. Next target source quality/alignment on these
cases and missing qualified contexts, plus independently reviewed grouped
evaluation and matched baselines, without requiring blanket double relabeling
or student delivery. Manuscript scientific claims and automation are unchanged.

## 5 November 2026, 11:00 UTC — measured CoinFabrik role-pair input limits

Continued from locally validated a0731cd; the existing GitHub403publication hold
remains, with no push/account/remote retry. Reused the prior filename inventory
instead of repeating the eight missing-context lookup or110000path/V6 scans.
Workspace `scafe_iteration_20261105_1100UTC/run.json` predeclared all643CoinFabrik
candidates and exact project/role-stripped-path/qualified-symbol pairing at the
unchanged pin, plus byte and literal-preserving heuristic-token comparisons.
The shared compute lock initially conflicted; no comparison ran until available.

Result:272unique pairs/544records,99unpaired records, no ambiguous buckets.
131pairs48.16%have byte-identical current annotation-v2 inputs;149pairs54.78%
have identical literal-preserving heuristic tokens. The latter is not actual
CodeBERT-tokenizer equivalence. All272already share evidence/strict components;
no grouping/quarantine/split changes. Opposing role-derived targets on the131
exact-input pairs would be indistinguishable to a deterministic code-only model;
no such labels were assigned and this is not a measured label-error rate.
Contextual repairs can occur outside the retained function.123different-token
pairs remain unverified repair candidates, never automatic clean negatives.

All three previously source-positive functions have changed unique counterparts.
The Venus swapVAIForStable view moves its state decrement before fee transfer,
while both views retain nonReentrant. The other swap retains incoming transfer
before state update despite other changes; textual change does not verify that
source rationale's repair. Neptune _setPool adds validation/whole-struct assignment,
not proof of resolved authority/caller-policy concerns. These view observations
are not independent adjudication, contextual exploitability or negative labels.

Private five-file versioned derivative:
`scafe_function_dataset_pilot/derivatives/coinfabrik_pair_evidence_v1_20261105T1100UTC/`.
It binds original delivery, three parent manifests, generator/helpers and output
hashes; emits metadata/hashes only. All18delivered originals, prior derivatives,
33SCRUBD/3CoinFabrik source cells,6,969unknown harmonized cells and zero gold/
eligible/split counts remain unchanged. No model/training/inference or gain.
Scientific value is measured support for rejecting blanket patched negatives;
software value is deterministic pair evidence with ambiguity/literal regressions.

See `docs/FUNCTION_COINFABRIK_PAIRS_20261105.md` for method, scope and fixed-three
observations. Focused tests, both integrity validators, byte replay, independent
checks, fresh isolated full TeX/BibTeX/visual build, backed-up input-matched PDF
refresh and reviewed local Git outcome are recorded in the run report; publication
remains held, not successful. Manuscript claims and automation are unchanged.

Current critical path stays milestone2: source-quality/target-alignment decisions
and separate independent grouped evaluation, not repeated inventories/tools.
Small next deliverable: finding-specific context/adjudication for changed pairs,
with a declared source-supervised eligibility decision and precise remaining
unknowns. Existing evidence is insufficient for verified negatives; review/context
availability controls timing, so no invented completion deadline or training claim.
Adapter/masked learning are implemented; baselines/CodeBERT remain downstream of
quality/evaluation/grouped-cohort/license/compute gates.

## 5 November 2026, 12:00 UTC — native-UX-supported typed discarded-send cells

Continued from local a7dbc0d, with no repeat of the completed272pair/V6/110000path
analyses. SCAFE lock ownership and active work were checked; shared-compute
conflicts were deferred. Fixed all11prior bare direct-send sites in8qualified
native-UX-positive functions/4retained contexts before compiler evaluation.
Workspace `scafe_iteration_20261105_1200UTC/run.json` binds this frame and inputs.

New separate weak-proxy definition:
`source-UX-supported-compiler-typed-discarded-address-send-result-v1`.
Compiler AST proves exact unique owner/function/UTF-8 span, native address.send
receiver/member type, bool result and direct ExpressionStatement disposal; exact
prior member anchors and literal-preserving call retention in current code-only
views are checked. All11sites pass, supporting8positive unchecked-call proxy
cells where source derivatives previously had0on this axis. Native UX alone is
not mapped wholesale: creation/high-level/other exception claims remain distinct.
110native rows retain39UXpositive/32negative/39unknown claims. Only8positive
pattern/source cells are observed; no negative targets. All8have existing-corpus
overlap/group restrictions; they are not independent projects or evaluation truth.

Initial0.4.24accepted1context/rejected3byexplicit0.4.25pragmas. Before replacement
AST interpretation, an evidence-backed amendment fixed already-installed0.4.25
for the entire unchanged4context frame, retaining original errors/warnings and
never patching source/pragmas or ignoring errors. All4pass,48warnings perpass.
A preparatory CRLF-normalizing read failed its raw hash assertion before source
compilation; byte-preserving decoding corrected the preparation, not original data.
A preliminary replay differs only in one compiler AST's contractDependencies
integer-list ordering and corresponding manifest hash; both raw versions remain
unchanged/hashed, all six derived decision/input/report files match. Final guarded
export/replay matches20/20files byte-for-byte; raw compiler nondeterminism is still
disclosed, not hidden by selecting a successful attempt.24small standard-JSON
compilations total, no deployed execution/model/training/inference or metric gain.

Private20fileversionedrelease:
`scafe_function_dataset_pilot/derivatives/scrubd_send_source_supervision_v1_20261105T1200UTC/`.
Original18deliveryfiles,6,969unknown harmonized cells,33SCRUBDRE/3CoinFabrik
proxies and zero independent reviews/eligible/split counts remain unchanged.
Source+compiler weak agreement is not independent gold or a label-accuracy estimate;
never evaluate correctness against the same extraction rule. The loader refuses
training and old namespaces are not silently migrated. Exact reproduction, failed
attempts, limitations and native meanings are documented in
`docs/FUNCTION_SCRUBD_SEND_20261105.md`. Source/compiler payloads remain private.

Scientific/dataset value: measured supported source-supervision coverage on a
second narrow axis, not detector performance. Software value: guarded exporter
and typed-span/code-retention boundary regressions, reusing existing source/call
and token evidence rather than a new scan/queue. Required tests, both validators,
fresh isolated full TeX/BibTeX/visual checks and backed-up input-matched PDF refresh,
reviewed local commit and held-publication state are recorded in the run report.
Next milestone2criticalstep: targeted quality/target-alignment decision and declared
source-supervised eligibility, independent grouped evaluation and license gates;
then matched baselines and gated CodeBERT. No new blanket manual-review rule,
student dependency, publication promise, manuscript scientific claim or automation
change. The existing403publicationhold remains with no push/auth/remote retry.

## 6 November 2026, 09:00 UTC — versioned native-UX send/call coverage

Resumed from local cc2d33b after the prior disconnected read; both repositories
were clean and no interrupted write/owned lock existed. Minimal process-name/PID
checks only, refreshed input hashes, and exclusive task/shared-compute coordination.
No repeat of pair/V6/source inventories or prior send compiles. Predeclared all14
qualified nativeUX-positive bare send/call sites in10functions/6contexts using the
existing source/call release, before compiling new contexts.

Explicit v2 target scope: native-address send OR legacy call discarded bool result,
supported by native UX=1. Ten weak-positive cells/14typed sites, preserving8prior
functions and adding2unique functions: POOHMO.endRound and TwoCoinsOneMoonGame.withdraw.
Participant.processPayment gains call evidence but remains one observed cell. Do
not sum v1/v2 targets. Four cached compiler bundles are checked against pinned
ancestry, exact source/settings/compiler/input/output/error identity and reused;
only2new contexts compile per pass with unchanged installed0.4.25settings.
Exact owner/function/member UTF-8 spans, native address/barecall types, bool outer
result, direct ExpressionStatement and literal-preserving input retention are
required. Only typed single gas/value wrappers are allowed; guards/assignments,
custom calls and tuple-return variants remain unsupported/unobserved.

All10retain existing-corpus/group quarantine, not independent test truth. The new
POOHMOUXevidence does not resolve its separate RErationale issue; MonkeyScam and
CoinFabrik contextual/repair questions remain unchanged. No negatives, independent
gold, eligibility, splits, training/inference or score gain. Original18delivery
files,6,969canonical unknown cells,33RE/3CoinFabrik claims and send-v1release remain
unchanged. Default v1 protocol/definition is retained; explicit prior-layer input
selects separately named v2. Native110UXclaims39positive/32negative/39unknown remain
preserved. Source/compiler agreement is weak supervision, not an independent oracle.

Private26filederivative:
`scafe_function_dataset_pilot/derivatives/scrubd_lowlevel_source_supervision_v2_20261106T0900UTC/`.
Final/replay are26/26byte-identical; prior raw-compiler dependency-order variability
remains disclosed. An initial workspace protocol-variable shadowing error failed
closed after2new context compiles; diagnostics preserved and no partial target
release installed. Final/replay add4new compiles,total6; cached4contexts reused
eachpass. Source/compiler payloads stay private. Exact method, scope, regression
controls and reproduction are in `docs/FUNCTION_SCRUBD_LOWLEVEL_20261106.md`.

Dataset value:2new source-supported functions/3call sites under an explicitly
broadened target, not classification accuracy. Applicable tests, both validators,
fresh isolated full TeX/BibTeX/visual checks, verified backed-up local PDF refresh,
reviewed task-owned local commit and publication-hold state are in the run report.
Next critical milestone2decision: targeted quality/target-alignment and justified
source-training eligibility, real negative-class support, provenance/license and
independent grouped evaluation. Baselines/CodeBERT remain gated, no blanket full-
corpus double-review rule or student prerequisite. No manuscript claim, automation
setting, source original or shared-feature target changed; no push retry.

## 6 November 2026, 10:00 UTC — conservative source-negative local contrast

Continued from local commit e767edc with clean worktrees and safe PID/executable-
name resource checks. Held the exclusive task lock and bounded shared compute
leases. Reused existing evidence without old scans, pair audits or repackaged
queues. The fixed frame contains all seven localized native UX=0 functions in
six existing contexts. The admission policy followed source-visible inspection
of all seven snippets, before compiler decisions; it is not blinded or random
sampling, and does not estimate label quality.

The explicit v3 local-result target preserves ten v2 positives and adds one
source-specific weak negative: Wallet.confirm. Evidence combines native UX=0,
exact compiler owner/function/UTF-8 spans, a native boolean call checked directly
by builtin require, complete local target-call enumeration, guard retention in
the code-only view, and no unresolved alias, inline-assembly or input conflict.
An observed zero has a true mask; unknowns remain null with false masks. Scope
is the named function body, excluding modifiers, callees, creation helpers and
global CLEAN claims. Older releases and canonical labels remain unchanged. The
diagnostic loader refuses training.

The seven admission outcomes are one supported checked contrast, two no-call
unknowns, three other/nonreverting-handling unknowns and one compiler-failed
unknown. WalletLibrary.confirm's unchanged full context has legacy assembly
declaration/stack errors under 0.4.25; it stays unknown. No source or pragma was
patched and no error ignored. Other cases are not declared incorrectly labeled
or unsafe. The POOHMO and MonkeyScam RE concerns remain unresolved.

Five of six contexts compile. One immutable MonkeyScam AST bundle is reused
after source/compiler/semantic-settings checks, preserving its AST-plus-bytecode
output selection verbatim. Five new contexts compile per pass. Two prototype
and two final passes total 20 small standard-JSON invocations, including four
attempts on the same failing context; no model, training or inference ran. A
pre-release inline-assembly completeness refinement left the frame and outcome
unchanged. All 97 warnings are retained in each full six-context report. The
final 26 files replay byte-for-byte. Prototype raw AST externalReferences order
varied while typed bindings and all six decision/input/report files matched;
both raw versions and their manifest hashes remain preserved. This is not a
universal raw-compiler determinism claim.

The private 26-file derivative is
`scafe_function_dataset_pilot/derivatives/scrubd_local_result_source_supervision_v3_20261106T1000UTC/`.
Support is ten positive, one negative and 99 unobserved local target cells among
110 native rows. All eleven observed rows retain existing-corpus/group quarantine;
they are not fresh test data. The 18 delivery files, 6,969 unknown canonical cells,
33 RE proxies, three CoinFabrik claims and native UX counts (39 positive, 32
negative, 39 unknown) are unchanged. Independent gold, eligibility and split counts
remain zero. No absent-label, patched or synthetic negatives, independent
adjudication, label-accuracy estimate or performance gain were introduced.
Source/compiler agreement must not become its own evaluation oracle.

Dataset value is the first tightly scoped source-supported weak negative in the
local send/call-result layer, not classifier readiness. Software value is complete-
body strong-check admission, code-retention/mask/unknown boundaries and conservative
cache reuse. Method and limits are in
`docs/FUNCTION_LOCAL_RESULT_CONTRAST_20261106.md`. Tests, both validators, fresh
isolated full TeX/BibTeX/visual checks, backed-up input-matched PDF refresh and
reviewed local Git state are recorded in the run report. No push retry or automation
change occurred. The smallest next decision is qualified targeted acceptance or
rejection of this local-body contrast/definition, followed by legitimate negative
diversity, provenance/license checks and a separately reviewed grouped evaluation
cohort. No blanket double review of every training function or student dependency
is imposed. Matched baselines and CodeBERT remain gated.
