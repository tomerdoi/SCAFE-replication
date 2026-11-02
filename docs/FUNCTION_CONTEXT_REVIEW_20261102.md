# Context-enriched review release — 2 November 2026, 14:00 Israel

This separately versioned package makes four previously snippet-only review
records inspectable with full-source evidence and an explicit context-sufficiency
question. It is not a completed annotation dataset or a training-ready release.
The original 24-function packet and 21 recovered raw source texts remain immutable.

## Measured progress and scope

| Component | Earlier review packet | New context-sufficiency package |
|---|---:|---:|
| Records with bundled full-source review views | 0 | 4 |
| Bundled comment-masked source documents | 0 | 4 |
| Category-specific context assessments per reviewer | not provided | 12 |
| Completed independent reviews | 0 | 0 |
| Verified dependency closures / promoted pattern labels | 0 / 0 | 0 / 0 |
| Training-eligible rows / assigned splits | 0 / 0 | 0 / 0 |

Selection is exactly the intersection of the existing 24-function review queue
and records with unique hash-checked fragments in the 13:00 recovery derivative:
three SCRUBD functions and one CoinFabrik function. This availability-biased
feasibility subset is not a probability sample or independent evaluation cohort.
Existing-corpus overlap quarantine remains in the coordinator lineage. Overall
verified raw context coverage remains 136/2,323 candidates; no new source recovery
or reviewed-label coverage is claimed by this iteration.

The four views mask 118 comment spans and expose one import directive as an
unresolved dependency hint. Every record still has unverified dependency closure;
zero imports does not establish closure or sufficient semantic context. Relevant
modifiers, callers/callees, deployment behavior, specifications and compiler
semantics may still be missing. Original license states and project identities
are retained, not inferred from availability or reviewer agreement.

## Representation and lineage

`prepare_context_review.py` verifies the 18 delivered hashes, both input release
manifests and all their declared artifacts before processing. Each selected raw
context and raw function span is checked against pinned hashes. Source comments
are lexed with literal/escape awareness and replaced by spaces while retaining
Unicode character offsets and CR/LF positions. All code and string literals,
including URL/comment delimiters inside literals, are preserved. Import directives
are lexically inventoried outside comments/literals; no dependency is fetched,
resolved, executed or automatically declared irrelevant.

The initial exact-legacy-equality criterion failed for one function: the old pilot
had replaced 15 carriage returns in comments with spaces. The amended, explicitly
recorded acceptance boundary permits **only** raw CR versus legacy space at
verified comment positions/endings. All other differences fail closed. Three
function views remain identical; one has these 15 recorded comment-line-ending
differences. Both legacy and new function hashes and the difference count are
retained. This is a new review representation, not an overwrite of legacy model
inputs, checkpoint lineage, scores or annotations.

Masking removes potentially useful NatSpec/policy statements along with annotation
cues. Reviewers must request independently sourced specification evidence when
needed. Source names/claims, original identities and raw evidence stay with the
coordinator; code identifiers, literals and import paths can still reveal origin,
so perfect blinding is not claimed. Readable cards use robust code fences. All
embedded source remains data, never instructions to execute.

The output manifest binds the generator checksum, original delivery manifest,
prior review manifest, recovery manifest and every generated artifact. Old inputs
are rechecked before output creation; existing directories are refused. Artifact
paths must remain contained and files/submissions are bounded at 16 MiB. Two
complete preparation replays must be byte-identical. No raw or masked context
source, candidate data or reviewer packet is added to Git or uploaded.

## Concrete review action and unchanged truth gates

Local package:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/context_review_v1_20261102T1400/`

Give two qualified independent reviewers the instructions, `function_cards.md`,
`context_documents/` and separate copies of `reviewer_A.jsonl`/`reviewer_B.jsonl`.
Keep `coordinator_key.jsonl`, original claims/raw sources and the other review
outside first-pass work. No reviewers were contacted and no submissions have been
promised or completed. Use stable pseudonyms and save submissions as new files.

For EACH of reentrancy, unchecked external call and authorization, reviewers
propose whether the supplied evidence is sufficient to assess **contextual
suspiciousness** under stated assumptions: sufficient, insufficient or unknown.
These are context assessments, not positive/negative pattern labels or exploit
judgments. Sufficient proposals require exact document-bound evidence spans,
explicit assumptions and dependency reasoning, with no declared missing context.
Insufficient proposals require concrete missing evidence. Reasons are required
for unknowns too. Document offsets are zero-based Unicode characters, end exclusive.

Comparison rejects incomplete/duplicate/same-reviewer packets, immutable input or
label changes, invalid evidence, and unsupported sufficiency proposals. Every
cell, including agreements, enters a context-adjudication queue. Different IDs
cannot establish reviewer independence or expertise; agreement does not establish
truth or sufficient context. The tool always leaves `context_gate_passed=false`,
all three pattern labels unknown, eligibility false and splits unassigned. A
separately qualified independent context-adjudication and pattern-annotation
protocol remains necessary. The earlier snippet-only forms cannot be silently
upgraded by pointing them at these files.

## Commands and evidence

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/prepare_context_review.py prepare \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --recovery-dir /absolute/path/local_context_v1_20261102T1300 \
  --prior-review-dir /absolute/path/review_pilot_v1_20261102T1200 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION

PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/prepare_context_review.py compare \
  --kit-dir /absolute/path/context_review_v1_20261102T1400 \
  --review-a /absolute/path/completed_A.jsonl \
  --review-b /absolute/path/completed_B.jsonl \
  --output /absolute/path/NEW_context_comparison.json
```

The task workspace `scafe_iteration_20261102_1400/` records the recovered interruption
state (no startup artifacts or active run), initial/amended acceptance criterion,
masking compatibility diagnostic, replay results, tests and build evidence. The
12 new regression tests cover safe comment/literal/import handling, Unicode/CRLF
preservation, narrow legacy CR compatibility, replay/original preservation,
artifact containment, completed-review/evidence/assumption gates and prevention of
label promotion even when both reviewers propose sufficiency. All 74 focused tests and both integrity validators passed; full pytest remains
unavailable and was not installed or run. Fresh paper-build/PDF-refresh evidence
is recorded in the task workspace before commit. No training, accuracy gain or submission readiness is
claimed. Source/license, project/clone, frozen evaluation and independent label
review gates remain separate from this review-tooling advance.
