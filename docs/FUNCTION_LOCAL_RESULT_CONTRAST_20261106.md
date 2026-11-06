# Scoped source-negative result-handling contrast — 6 November 2026, 10:00 UTC

## Decision and measured coverage

Added one **source-specific weak negative**, `Wallet.confirm`, for the narrowly
specified **local named-function-body native send/call result-discard** target.
It has an explicit retained native UX=0 claim and an exact compiler-typed native
call whose result is directly checked by builtin `require`. The complete typed
body enumeration finds one target call and no unresolved target/assembly forms;
the guard remains in the current code-only view. This is not global CLEAN,
absence of all unhandled exceptions, audited security or independent gold.

The explicit v3 layer preserves the ten v2 positive functions and adds this one
weak contrast: **10 positive / 1 negative / 99 unobserved** function cells among
110 native SCRUBD candidates. Native UX counts remain 39 positive/32 negative/
39 unknown. Other target categories stay unobserved; the original 6,969 canonical
cells, 33 RE proxies, three CoinFabrik claims and all older releases are unchanged.
Do not sum the versioned layers or call these independently reviewed labels.
All eleven observed rows retain existing-corpus/group quarantine. Ten-to-one
support with one negative context is insufficient for a defensible classifier
comparison or source-quality estimate; eligibility, evaluation and splits stay zero.

| Fixed seven-function native UX=0 frame | Outcome |
|---|---:|
| Explicit strong checked contrast | 1 |
| No local target call, kept unknown | 2 |
| Other/nonreverting handling, kept unknown | 3 |
| Full-context compiler failure, kept unknown | 1 |

The other conditional cases (`MonkeyScam.endRound`, two contexts, and
`CB_BANK.Collect`) are **not** declared incorrectly labeled or unsafe. The narrow
rule does not decide their nonreverting/accounting behavior. No-call cases
(`MonkeyScam.withdraw`, `GAME.startRaffle`) do not become easy negatives merely
from absence. `WalletLibrary.confirm` remains unknown: the original full context
fails 0.4.25 with an unresolved legacy assembly label and unbalanced assembly
stack. Both errors are retained, not patched or ignored. A historically compatible
compiler/context or qualified contextual review is needed for that candidate.
The separately recorded POOHMO/MonkeyScam RE rationale questions are unchanged.

## Target and admission policy

Protocol: `scrubd-source-local-native-result-binary-proxy-v3`.
Definition: `source-UX-supported-local-native-result-discard-with-strong-checked-contrast-v3`.
This is an explicit source/weak namespace with deliberately partial coverage,
not an automatic migration into canonical suspicious-pattern or legacy eleven
labels. Positive evidence is inherited from the hash-bound v2 parent. A negative
needs native UX=0, at least one typed local target call, complete function-body
coverage, and **every** target call immediately checked by one of:

- Direct builtin `require(call)`; a user-defined function named require is excluded.
- `if (!call)` with an unconditional throw or builtin revert as its sole failure
  action; arbitrary, nested or nonreverting fallback statements are excluded.

Require exact original context/function/UTF-8 byte bounds, qualified compiler
owner, native address and send/barecall bool types, verified call/guard spans,
current model-view guard retention and no same-input conflict with positives.
Unknown native aliases and inline assembly within the function body fail closed.
Calls in modifiers, callees and creation helpers are outside this local target;
no transitive or global safety conclusion is made. Unsupported/absent/ambiguous
annotations are never negative. An observed zero has a true observation mask;
an unknown zero placeholder has a false mask. The new diagnostic loader refuses
training regardless of available binary support.

The admission rule was fixed after a **source-visible inspection of all seven
snippets**, before compiler decisions. It is conservative case construction,
not a blinded/random audit, label-noise estimate or independent evaluation.
No two fresh independent reviews of every training example are required, but
quality/adjudication and separately qualified independent evaluation remain gates.
Never evaluate correctness against this extraction/admission rule as its oracle.

## Compilation, reuse and replay

Six fixed contexts: five compile successfully and one fails as described above.
Reuse one immutable MonkeyScam 0.4.25 AST bundle after checking raw source,
compiler checksum, optimizer disabled/runs=200, Byzantium and complete AST request.
Its original output-selection superset (AST plus bytecode) is preserved verbatim,
not represented as an AST-only compiler input. Only five remaining contexts are
newly compiled per pass, using the same installed checksum-pinned 0.4.25 binary,
AST-only output, no imports/source patching/dependencies or deployed execution.
All diagnostics are retained, including 97 warnings across the six-context report.

Two prototype passes and two final passes made 20 small standard-JSON compiler
invocations total, including four repeated failures of the same unsupported raw
context, not four distinct failing candidates. The final pair followed a pre-
release fail-closed refinement for inline assembly; the frame/outcome did not
change. One cached bundle is reused per pass, with no recompilation of that source.

All final 26 files replay byte-for-byte. A prototype replay differed in ordering
of identical compiler AST assembly `externalReferences` entries and the resulting
manifest hash; raw outputs/hashes are preserved. All six decision/input/report
files were identical. Comparison checks permit only those exact-shape entry-list
orders and prior `contractDependencies` integer ordering, never different bindings,
source spans, types or decisions. No universal raw compiler determinism claim.

Private 26-file derivative:
`scafe_function_dataset_pilot/derivatives/scrubd_local_result_source_supervision_v3_20261106T1000UTC/`.
It contains all 110 code-only inputs/proxy rows, seven admission records, compiler
reports/bundles, data card, summary and provenance manifest. Source/compiler/private
review payloads stay local and are not staged in Git. Parent delivery/source/v2/
context/compiler manifests, generator/helpers and output hashes are bound.

Reproduce `scripts/adapt_scrubd_local_result_labels.py --help` with the explicit
pilot, source layer, v2 positive layer, raw context release, historical compiler
release, installed compiler binary and a new versioned output directory. Existing
outputs/originals are never overwritten. Focused tests and both integrity validators,
fresh full isolated paper build/visuals and backed-up input-matched PDF refresh,
reviewed local commit and held publication status are recorded in the run report.

## Smallest next readiness decision

Have the source-quality/target-definition review accept or reject this precise
local-body contrast and its limits, including the out-of-scope creation/callee/
modifier behavior. Expand legitimate negative diversity only with supported
source/context evidence, not all 32 UX=0 rows or patched folder membership.
The independently reviewed/adjudicated grouped evaluation cohort, license/
provenance eligibility, frozen splits and matched baselines remain missing gates
before CodeBERT. This run adds source-class coverage, not independent truth,
training readiness, accuracy or a publishable-result guarantee.
