# Comment-control extraction experiment, 10 November 2026 12:00 UTC

## Fixed question, admitted inputs and early gate

Test whether ordinary comments change intended code counters in the frozen primary
extractor while non-comment code, strings, character positions and CR/LF stay fixed.
The cohort was fixed before measurement: all 17 previously qualified CoinFabrik
MIT/BSD contexts and two newly bound ReentrancyStudy MIT contexts, 19 unique source
hashes / 216,879 bytes. No new acquisition, labels, reserved outcomes, synthetic
programs, model fit or scoring. These are inspected development sources, not 19
independent projects or fresh confirmation. No statistical population claim.

The immutable predeclared protocol records every input/parent hash, the original
244-column feature schema, 14 named focus counters, budgets (12s/source, 180s total),
shared-compute lease and continue/stop gates. Continue only if a focus counter
changes on a verified identity-preserving pair; otherwise no bug/fix claim. Any
identity or relational-negative-control failure blocks a validated remedy claim.
Expected style changes were explicitly separated from code-counter changes.

Treat comments as an experimental preprocessing view, not new training examples:
only in-memory masking, no transformed source corpus persisted. The literal-aware
scanner blanks comment characters, retaining strings, escapes, all non-comment
characters, character offsets and every CR/LF. Unicode comment byte lengths may
differ; no byte-offset preservation claim. Unterminated quotes/block comments and
oversized inputs reject. Independent existing comment-remover/tokenizer checks
confirm identical lexical code/literal streams; this is not compiler equivalence.

## Measured result and implemented bounded remedy

All 19 pairs passed token, literal, character-position and line-break checks.
The existing comment-aware relational feature negative control changed on 0/19.
The frozen extractor changed at least one of 244 features on 19/19 sources; 87
feature names changed somewhere. That includes intentional style/token effects,
ratios and dependent summaries: it is NOT a count of 87 semantic defects.

| Predeclared focus counter | Sources changed (of 19) |
| --- | ---: |
| n_functions | 10 |
| n_contracts | 10 |
| n_assignments | 6 |
| n_events | 2 |
| n_require_assert | 2 |
| n_delegatecall | 1 |
| Other eight focus counters | 0 |

In total 14/19 source pairs change a focus code counter, 31/266 source-counter
comparisons. A real-source witness (hash e50347eee757ac3e238257bd90034370f02207bf467796bb7611254fe506de27)
has n_delegatecall and uses_delegatecall both 1 before masking and 0 after: the
matching word occurs only in a comment, not the retained code-token stream.
No claim of absolute AST correctness, exploitability or vulnerability truth follows.

Added src/ml/comment_feature_control.py as a separately versioned OPT-IN API.
It masks comments before calling the unchanged legacy extractor, returns features
separately from raw/processed-source and implementation hashes, version, file count,
seed/tolerance and explicit incompatibility metadata. No address argument, network,
registry enrichment, training, scoring or cache write is performed. The original
244-feature extractor, sanitizer, cached CSVs, trained models and paper tables are
untouched. Legacy extractor SHA-256 remains
c2d5e5b82d986816c076f4adc8b49377f18b99ec02ca5de7d614e33b736d5bc3.

On the same 19 pairs, the new API has 0/19 focus-counter dependencies and 0/266
changed focus comparisons; all returned features are invariant to applying the
same comment preprocessing first. All 244 values match direct masked extraction
within declared numerical tolerance. This is a mechanical robustness/correctness
improvement, NOT an accuracy/generalization gain or trained-method comparison.
Comment/style features are also changed, so this view is not a drop-in replacement
for existing features/models. Strings remain available and other regex/function/
AST approximations remain; no claim that every lexical confound was removed.
The 0/19 result is the fixed original-versus-masked pair comparison, not arbitrary
comment-edit invariance. Preserved comment-origin line breaks can still affect
LOC/style fields when comment layout changes.

## Failures, replay and numerical limits

The initial runner mistakenly passed dictionaries instead of keys to Counter.update;
it failed aggregation after the first source. That invalid attempt is retained,
not counted as a scientific result. Corrected the runner without changing cohort,
focus counters or gates. A later cross-process exact comparison exposed four
sv_fanout_entropy differences no larger than 9e-16; no focus integer outcome changed.
An explicitly dated technical amendment fixed PYTHONHASHSEED=0 and finite numeric
comparison at relative/absolute 1e-12 for final artifacts. Earlier attempts remain.
No model threshold, seed search or outcome-based exclusion was performed.

The final diagnostic and remedy both completed; offline independent recomputation
checks all 19 records, counter deltas, unchanged source/parent hashes and the
comment-only delegatecall witness. The API carries implementation/input/view hashes
to prevent silent mixing with legacy artifacts. Ten software regression tests cover
quoted delimiters/escapes, Unicode/CR/LF, token separation, EOF, malformed/bounded
input, idempotence, the counter regression and metadata separation. They are test
fixtures only, never research model inputs. Full validation results are checkpointed.

Private derivative: Desktop/D_local/func-dataset/scafe_function_dataset_pilot/
derivatives/comment_feature_control_v1_20261110T1200UTC. It retains protocol/amendment,
source lineage, initial-attempt status, numeric records, source hashes, replay and
remedy evidence; no raw or masked source body is published through Git. Model/data
eligibility, gold and frozen evaluation splits added: zero. Existing source-use,
real-only, privacy, shared-resource and remote holds are unchanged.

## Next decisive outcome

Do not repeat this invariance experiment or call the 14-to-0 result classification
improvement. Next bounded session: establish an exact separately versioned feature-
cache/lineage plan for a matched existing-corpus development comparison of original
and comment-controlled views, with common groups, fair style/nuisance baselines,
metrics, uncertainty and stop criteria fixed before outcomes. Use only admitted
real inputs and justified native target scope; the stopped alarm-source route
remains stopped. If the required source, target/evaluation or resource gate is
absent, report that exact dependency and do not train or reuse an old model on the
new view. No full-corpus overwrite or expensive rebuild is authorized implicitly.

A few-week submission package remains conditional on a substantive matched result
and qualified independent evaluation. This experiment alone does not establish
novelty or Q1 readiness. Distinguish SCAFE's method/control contribution from the
data-centric intervention paper. All required tests, both integrity validators,
fresh direct full TeX/BibTeX build, visual checks, input-matched PDF refresh, final/
staged review, local Git and own-lock states are recorded in the run checkpoint.
Manuscript scientific claims and automation settings are unchanged here.
