# Call-site result evidence for category review — 3 November 2026, 14:00 Israel

The localized source contexts now have concrete, hash-bound call-site evidence
and a coordinator review queue. All 139 qualified named functions were examined;
28 target-member sites occur in 21 functions. Fourteen sites in ten functions have
a conservative bare-expression result-discard shape. These are review candidates,
not verified unchecked external calls or new training labels.

| Observed source property | Count |
|---|---:|
| Functions analyzed / total staging candidates | 139 / 2,323 |
| Functions with observed target-member sites | 21 |
| Sites spelling `call` / `send` | 14 / 14 |
| Direct / legacy gas-value / modern-options invocation shapes | 15 / 10 / 3 |
| Bare-expression result-discard candidates | 14 sites, 10 functions |
| Lexical guard-expression contexts | 7 sites |
| Assignment contexts | 3 sites |
| Ambiguous/other result contexts | 4 sites |
| Independent reviews / training eligible / assigned splits | 0 / 0 / 0 |

The 21-function queue includes all functions with observed sites, prioritizing the
ten with bare-result candidates and retaining the other result-use contexts as
comparators. It is a descriptive convenience queue, not a prevalence sample.
Guards, assignments and returns do not create verified safe/hard-negative labels.
The ten candidates are not ten confirmed vulnerabilities.

## Predeclared scope and meaning

Before evaluation, workspace `scafe_iteration_20261103_1400/run.json` fixed the
139-function cohort and target spellings: `call`, `delegatecall`, `staticcall`,
`callcode`, `send`. Only the first and last occur here. `transfer` is deliberately
excluded because token-transfer and Ether-transfer semantics cannot be resolved
by spelling. Even the included member names are not typechecked EVM operations:
receiver types, versions and exact semantics remain unverified.

Comments are masked with the existing frozen helper and literal tokens cannot
create call sites. Balanced direct, modern option-brace and bounded legacy
`.gas(...).value(...)(...)` shapes are recognized. Unsupported invocation syntax
remains explicit. A bare-result candidate requires a complete standalone statement,
a supported simple receiver, and no assignment, guard, return or trailing expression.
Indexed/complex receivers and compound-flow ambiguity can conservatively remain
unclassified, including genuine ignored-result statements. Counts are not recall
or correctness measurements against an independent annotation set.

Each site records member, invocation-suffix and optional statement-window spans,
raw and masked hashes, absolute Unicode offsets, comment-masked excerpts and
receiver-type uncertainty. The invocation suffix begins at the member word;
the enclosing full function/context is retained by immutable reference. A
`statement_window_span` is a bounded lexical evidence window, not an AST statement:
guard/compound-flow windows can contain several statements. Inspection of all
28 excerpts prompted this field-name clarification; no acceptance rule, site
count or disposition changed after seeing the results.

Lexical guard spellings (`require`, `assert`, `if`) do not prove a failure is
handled on every path. Assignment does not prove a result is later checked;
returning a value would not prove its caller uses it. No CFG, type, alias, dataflow,
reachability or dependency analysis is claimed. This is category-specific source
evidence for independent judgment, not a rule-generated gold-label shortcut.

The reference-to-gap review's P1 context-aware question explicitly distinguishes
unhandled calls from checked/authorized wrappers (`docs/SCAFE_REFERENCE_REVIEW_20261101.md`,
context mechanisms and GoLiSA pp.5–6/GPT4Detection pp.12,18). This step makes the
review question concrete while retaining the distinction between code observations,
contextual suspicion and supported vulnerability evidence.

## Versioned deliverable and review action

New local derivative:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/call_evidence_v1_20261103T1400/`

Five files provide per-function analysis status, call-site evidence, the complete
21-function review queue, summary and manifest. Resolve `context_artifact` against
the immutable hash-bound `content_context_v1_20261103T1200` release; the selected
function spans come from `scope_mappings_v1_20261103T1300`. All five files reproduce
byte-for-byte. Original inputs, current model views and earlier releases remain
unchanged; no raw source or dataset payload is staged or uploaded.

For review, join `review_queue.jsonl` site IDs to `call_site_evidence.jsonl`, inspect
the exact parent function/context, and establish receiver types and call semantics,
result/failure propagation, relevant modifiers/callers/callees and missing dependencies.
Record independent category decisions and evidence through a separately versioned
annotation/adjudication protocol. No reviewers were contacted. This coordinator
evidence is not guaranteed source-blinded annotation material: identifiers,
literals, selection and parent context can reveal origin. Do not count queue
construction or these source observations as completed independent review.

Missing-context records (2,178) and unsupported special functions (six) have null
analysis counts rather than negative labels. An analyzed function with zero observed
target spellings is not clean or negative for any category. Every category label
remains null with an all-false observation mask; human-review status stays unreviewed.
All three-category/legacy-eleven-output, qualifier, context, license and corpus
limitations persist. Source coverage stays 145, direct overlap stays 145 and
conservative quarantine stays 264/316. There is no new holdout or CodeBERT training
eligibility, no inference/training and no model-accuracy/Q1-readiness claim.

## Reproduction and safeguards

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/extract_function_call_evidence.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --scope-release-dir /absolute/path/scope_mappings_v1_20261103T1300 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

Only existing local contexts are read; there is no source execution, corpus rescan,
install, download or network/model use. The output must be new. Reviewed/observed
labels, eligibility or split assignments require explicit migration and are refused.
Ten new regression tests cover invocation versions, disposition distinctions,
comments/literals, excluded forms, nested calls, unsupported structures, Unicode/CRLF
and absolute hashes, deterministic replay, immutable parents and null unanalysed
counts. Independent checks bind every excerpt to its exact original source, inspect
all 28 site windows, verify counts/queue membership and unknown labels, and preserve
parent scope/context/group evidence. Applicable tests, both integrity validators,
fresh full paper compilation/visual/PDF refresh and final/staged Git review must
pass before commit. Exact outcomes are in the current workspace run report.
