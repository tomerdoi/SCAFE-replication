# Compiler-version call evidence — 4 November 2026, 13:00 UTC

The compiler part of the earlier MonkeyScam rationale concern is now narrowed by
actual code generation. In the unchanged recovered caller, all three PlayerBook
calls compile to **CALL**, including the locally `view`-declared getters, with
installed Solidity 0.4.24, 0.4.25 and 0.4.26 under the declared settings. A separate
same-source control switches its view call to STATICCALL with 0.5.3. Thus the view
keyword alone is not a compiler-level contradiction of the original source
rationale under these old builds. This corrects the version-insensitive limitation
of the earlier preliminary AI flag; it does not validate the RE label or an exploit.

## Question, fixed matrix and lineage

Before compilation, workspace `scafe_iteration_20261104_1300UTC/run.json` fixed the
candidate, three exact PlayerBook call expressions, actual versions 0.4.24–0.4.26,
control versions including 0.5.3, optimizer disabled (runs=200), Byzantium EVM,
call-site evidence rules, input/compiler hashes and small offline CPU budget.
No solc-select version setting was changed, compiler installed or model run.
The project virtual environment already contained the direct compiler binaries;
their hashes and `--version` outputs are retained. All executions used those
explicit binaries, not a wrapper with possible download behavior.

The actual source is the original hash-bound full context for
`MonkeyScam.determinePID`, with raw function span preserved. It is not patched,
reformatted or converted to compile with 0.5.3. Only 0.4.24–0.4.26 compile the
actual context. The two-call minimal control is identical across four compiler
versions, has no dataset role or vulnerability labels, and is not a replacement
for the actual source. No contract bytecode is executed.

[Solidity's v0.5.0 change notes](https://docs.soliditylang.org/en/latest/050-breaking-changes.html)
document the CALL-to-STATICCALL change for external view/pure calls on Byzantium
or later. This is language-version evidence, not a new vulnerability report.
The existing reference-to-gap review's P1 compiler/context limitation motivated
this concrete check; no new paper-performance or novelty claim is made.

## Results

| Input and compiler | getPlayerID | getPlayerName | getPlayerLAff |
|---|---|---|---|
| Original caller, 0.4.24 | CALL, PC 3877 | CALL, PC 4069 | CALL, PC 4261 |
| Original caller, 0.4.25 | CALL, PC 3877 | CALL, PC 4069 | CALL, PC 4261 |
| Original caller, 0.4.26 | CALL, PC 3877 | CALL, PC 4069 | CALL, PC 4261 |
| Minimal control, 0.4.24/0.4.25/0.4.26 | CALL, PC 335 | CALL, PC 506 | not included |
| Same minimal control, 0.5.3 | CALL, PC 335 | STATICCALL, PC 504 | not included |

All seven unique build cases exited successfully without compiler errors. Each
actual build retains four original warnings: two address-checksum warnings,
one unused parameter and one stricter-mutability suggestion. The control builds
have no warnings. No warning was fixed or hidden, and all output is preserved.
The seven-case matrix was run in initial/prototype/final preparations and replay
checks (35 compilation invocations in total); all seven cases and all attempts
are retained. This remained a small offline CPU run, with 30-second subprocess
limits and shared-compute lock sections; no model/corpus work or training occurred.

These are mapped call instructions, not raw opcode-byte counts. The original
call sites are at source lines 1056–1058. Each result is bound to exactly one
CALL-family instruction whose compressed source-map range lies wholly inside the
unique exact call expression within the selected function. Source locations use
UTF-8 **byte** offsets; instruction offsets account for PUSH data lengths. Generated
source IDs and unmapped metadata tails cannot supply candidate call evidence.
Ambiguous/missing maps, imported sources, malformed/truncated code and compiler
errors fail closed. [Solidity's source-map specification](https://docs.soliditylang.org/en/latest/internals/source_mappings.html)
explains these offset and compressed-field conventions.

## What this resolves and what remains unknown

The earlier AI audit observed a view/non-view declaration difference and required
compiler/callee review before drawing a mechanism conclusion. The compiler part
has now been tested: pre-0.5 view syntax does not establish STATICCALL protection
in these recompiled callers. A qualified reviewer should no longer use that
syntax difference alone as a reason to reject the source's getPlayerName rationale.

A CALL does not prove the callee calls back, that an unsafe state is reachable,
or that the source RE=1 assessment is correct. The deployed caller's exact
compiler/settings/bytecode were not verified; the deployed PlayerBook code,
callback behavior, caller guards and cross-function state remain unverified.
The smallest remaining decision is whether the actual callee and reachable state
support the claimed mechanism under the deployment's configuration. If unavailable,
retain unknown/context-required independent truth. No source label is flipped,
no AI observation becomes independent adjudication, and CALL/STATICCALL must not
be converted into automatic RE labels or circular evaluation truth.

The POOHMO admin-versus-flushDivs concern was not re-audited and remains unresolved.
All 110 native RE claims, 33 source proxies, masks, code views, group/quarantine
constraints and original/prior audit files remain unchanged. Independently
reviewed/adjudicated labels, training/evaluation eligibility and assigned splits
remain zero. No CodeBERT fitting, inference, accuracy gain or readiness promotion.

## Private local artifact and reproduction

New versioned derivative, never staged in Git:

`scafe_function_dataset_pilot/derivatives/compiler_call_evidence_v1_20261104T1300UTC/`

Its 28 files preserve config, four compiler-version outputs, seven JSON inputs,
seven full JSON outputs, stderr streams, report and manifest. They include source
and bytecode payloads and stay local. All 28 final release files reproduce byte
for byte. Hashes bind original delivery, source/context parents, exact compiler
binaries, config, generator and helpers. Old source/AI artifacts are immutable;
this is a separate conditional compiler-evidence supplement, not a reviewed-label
release. Reproduce under the task/compute locks with installed compilers and a new
output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/probe_function_compiler_calls.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --source-release-dir /absolute/path/source_supervision_v1_20261104T1000UTC \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --compiler-root /absolute/path/existing/solc-select/artifacts \
  --config /absolute/path/compiler_call_evidence_v1_20261104T1300UTC/probe_config.json \
  --output-dir /absolute/path/NEW_VERSION
```

Decoder regressions cover PUSH bytes/tails, inherited fields, UTF-8 byte offsets,
function scope, ambiguity/generated sources, imports and retained fatal diagnostics.
Applicable tests, both integrity validators, fresh full isolated manuscript build,
visual checks, input-matched backed-up PDF refresh and reviewed Git publication
remain mandatory; actual evidence is in the current run report. Manuscript
scientific claims are unchanged. Next obtain supported deployed callee/caller
context or qualified adjudication, and continue genuinely independent source
completion as available; no blanket double review of every training row is required.
