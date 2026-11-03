# Versioned inline audit-prose filter — 3 November 2026, 10:00 Israel

This iteration removes a narrowly identified source-claim cue from candidate
**model views**, preserving all original records and source evidence. It addresses
non-comment audit prose that remained visible after the producer's comment mask.
It does not establish original contract correctness, gold labels or model accuracy.

## Observed data-quality change

| Property | Parent candidate view | New versioned view |
|---|---:|---:|
| Candidate records | 2,323 | 2,323 |
| Declared inline audit-marker occurrences | 7 | 1 |
| Records changed by the narrow rule | 0 | 6 |
| Lexical-signature successes | 2,321 | 2,323 |
| Literal-scanner failures | 2 | 0 |
| Exact/provenance constraint components | 1,326 | 1,324 |
| Strict template sensitivity components | 1,228 | 1,226 |
| Completed independent reviews / training eligibility / splits | 0 / 0 / 0 | 0 / 0 / 0 |

The original two scanner failures contain apostrophes in non-code audit prose,
after a completed statement. The same declared form occurs in four more candidate
records that did not trigger the literal scanner. Six tails satisfy the filter;
all are reported with record IDs, exact offsets and parent-span hashes. Removing
these source-claim cues makes the two failed inputs tokenizable under the existing
heuristic recipe and reveals net additional token-equality/template constraints.
It does not certify Solidity compilation or semantic equivalence to an independently
retrieved original contract. No source-claim text becomes a suspicious-pattern label.

One marker follows a block opener instead of a semicolon. The predeclared rule
leaves it untouched and sets `representation_review_required=true` with an explicit
residual-marker flag. Its stable ID is
`b407a429c804fffa264515bba167e0c7bcf394622854252df70c3bb0baf50db8`.
A lexical-signature success for this record is **not** a source-validity or readiness
claim. Other cue types may remain outside this narrow audit; absence of flags is
not a compiler, annotation or complete cue-removal certificate.

## Exact acceptance rule

The filter recognizes `<- audit-issue` and `<- @audit-issue`, case-insensitively,
only outside quoted literals/comments and immediately after a same-physical-line
statement semicolon, allowing horizontal whitespace. For example, in a controlled
fixture `x(); <- @audit-issue explanation`, only the marker and remaining line tail
are replaced by spaces. CR/LF positions, Unicode character offsets and all text
outside the exact span stay unchanged. Every changed span is reversible using the
unchanged parent view and its hash.

The scanner handles quoted strings and escapes before recognizing marker syntax.
It preserves genuine comparison expressions, markers in strings/comments and
unanchored markers. It never guesses through an unterminated literal or block
comment. An unhandled/residual marker or scanner problem is explicitly flagged.
This is a data-representation rule, not a security sanitizer or a vulnerability
classifier, and it does not insert a missing guard or modify program logic.

The two known failing records motivated the bounded repair hypothesis; the rule
and success criteria were recorded before applying it across the cohort. No model
outcome, held-out score or target-label optimization selected the rule. The broader
seven-marker inventory and the one unhandled case are reported, not hidden by an
expanded post-hoc acceptance rule.

## Model-input schema and recomputed constraints

The new schema keeps stable record/source/revision identities, original provenance
and license/context fields, `parent_model_code_sha256`, `model_code_sha256`, the
parent candidate-fragment hash and `view_protocol`. `model_code` is the only intended
code input. Parent/source metadata and masking/audit records are lineage, not model
features or gold targets.

Every category retains `pattern_labels=null` and an explicit
`label_observation_mask=false`. These masks mean **unknown/unobserved**, never a
negative class. All records remain unreviewed, training-ineligible and unsplit.
Legacy producer `token_sha256`, template hashes and other stale representation
summaries are deliberately omitted instead of being copied onto changed code.

The exact/provenance and template graphs are recomputed with the frozen grouping
implementation and comment-masker identities. Token and template spanning edges
increase from 456/578 to 458/580; the component counts decrease by two in each graph.
These remain heuristic constraints with the previously documented semantic-clone,
project-provenance and overmerge limits. The 136 direct local-context overlap seeds
and conservative 255/307 quarantine scopes are unchanged. Existing inspected sets
remain exploratory. The prior graph release is not overwritten, and old features,
checkpoints, scores or aggregates cannot be silently mixed with this new view.

## Local artifact and reproduction

New local derivative, excluded from Git:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/annotation_view_v1_20261103T1000/`

- `function_model_views.jsonl`: all 2,323 versioned code views and observation masks.
- `view_audit.jsonl`: per-record before/after signatures, residual flags and hashes.
- `mask_events.jsonl`: six exact masking spans, reasons and parent-span identities.
- `groups/`: newly computed memberships, typed edges, components and summary.
- `summary.json`, `manifest.json`: complete counts/limits, generator/grouping/masker
  identities, parent delivery/group/recovery identities and every artifact hash.

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/sanitize_function_annotations.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --baseline-groups-dir /absolute/path/constraint_groups_v1_20261102T1500 \
  --recovery-dir /absolute/path/local_context_v1_20261102T1300 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The tool verifies all 18 delivered hashes, baseline-group implementation/input
bindings and actual recovered-context seed hashes. It rechecks both parent releases,
refuses existing output or writes inside original/input release directories, and
bounds processing to 5,000 records/20 million total characters/one million per
function. Final artifacts reproduce byte-for-byte. Originals and prior releases
remain unchanged; no candidate code, source claims, reviewer material or dataset
artifact is committed or uploaded. Only tooling/tests/documentation enter Git.

Eleven new regression tests cover strings/comments/comparisons, exact anchoring,
escapes, Unicode/CRLF, multiple markers, unclosed literals, before/after signature
behavior, unknown observation masks, stale-field exclusion, input bounds, tampering,
original preservation and identical replay. All 94 focused tests and both integrity validators passed. Full pytest remains
unavailable and was not installed/run. Fresh isolated paper build/visual/PDF-refresh
evidence is recorded in `scafe_iteration_20261103_1000/`.

Next resolve the remaining block-opener case from original evidence and independently
review source/context sufficiency and category labels. License/provenance, project/
clone and frozen-evaluation gates remain. Neither tokenizability nor source-claim
filtering makes this dataset ready for CodeBERT fine-tuning. No training or accuracy
improvement was measured, and manuscript scientific claims remain unchanged.
