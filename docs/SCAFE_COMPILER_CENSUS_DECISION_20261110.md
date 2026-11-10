# Exact-compiler census decision, 10 November 2026 09:00 UTC

## Question and fixed gate

Can the existing source-only named implemented member census be used as an
AST-checked review frame on the four already admitted MIT CVE source snapshots?
Selection was all four qualified development sources from the prior batch, without
selecting on compiler outcomes. The protocol was recorded before compilation at
09:23 UTC (its exact timestamp and SHA-256 are in the private protocol). The fixed
utility screen required at least three exact-version successful snapshots, at least
50 reference functions, >=95% precision/recall and no unexplained boundary errors.
This is a structural utility screen, not a statistical power calculation, an
independently adjudicated vulnerability benchmark, or a publication novelty gate.

The reference unit is a named implemented function declared inside a contract,
library or interface in the selected source file, with exact UTF-8 byte start/end
and owner/name. Constructors (including legacy named constructors), fallback,
receive, modifiers, abstract declarations and free functions are excluded and
counted separately. Inherited functions are not duplicated into every child owner.
The source-only census uses the existing comment/literal-aware scope helper. The
legacy baseline uses the retained raw-brace regex recipe; both use the same lexical
owner-attribution helper. The reference is the compiler AST, never used to correct
baseline outputs. Native compiler geometry does not establish semantic labels.

## Exact versions and measured outcome

| Snapshot | Explorer compiler version | Exact binary | Result |
| --- | --- | --- | --- |
| CVE-2025-57247 | 0.8.26+commit.8a97fa7a | unavailable | Not compiled |
| CVE-2025-56207 | 0.8.17+commit.8df45f5f | unavailable | Not compiled |
| CVE-2023-47033 | 0.8.19+commit.7dd6d404 | unavailable | Not compiled |
| CVE-2021-3004 | 0.8.0+commit.c7dfd78e | cached direct binary | AST comparison passed |

The successful snapshot has 40 reference units. Both methods predict 40 and match
all 40 exact owner/name/byte-span tuples: precision 40/40 and recall 40/40, with zero
extra/missed units. Excluded AST nodes: 43 abstract declarations, two constructors
and two modifiers. No compiler diagnostics, exit 0. A second local replay produced
byte-identical compiler output and summary. This is one source snapshot, not 40
independent projects; no confidence/generalization claim is justified. Compiler
binary SHA-256 is c7c3ff484d2dd69350fc182d44ecf6057ff2885b96a1c3990ca362e9c8325335.
AST output SHA-256 is 95bceba38143be93306f61e6f838ae73daa1f6ef245e6ced58da2c762701ada6.

The screen FAILS: 1/3 required snapshots and 40/50 required reference functions.
No pooled score is reported for the three uncompiled cases. Do not replace their
compilers, rewrite pragmas or silently exclude them from the gate. AST-backed
geometry coverage rises from 0 to 40 of the old 176 lexical census entries (22.7%),
and 0 to 1 of four snapshots; the other 136 entries remain lexical-only. That is
not semantic-review, training-eligibility or independent-gold coverage.

## Tooling correction and route decision

Earlier notes saying 'no compiler installed' were based on an incomplete standard-
path probe. They are superseded: direct Solidity binaries already exist in the
project venv/.solc-select/artifacts cache. The three exact versions above are
absent there. This run used only cached 0.8.0, without changing the version selector,
installing/downloading software or executing contracts/PoCs. Standard JSON requested
AST only, inline sources, no code generation; the compiled source has no imports.
Input artifact/source/compiler hashes were checked. Shared compute was exclusively
leased, compilation bounded to 30 seconds. No new dataset acquisition took place.

The result validates the one small source frame, but baseline parity provides no
extractor advantage. STOP treating this experiment or parser as a stand-alone
research contribution, and do not expand it just to fill a count. The current-code
classifier route is still NO-GO: audited source-version applicability and qualified
independent evaluation remain missing. Historical report links cannot fill these
gates. No code/model improvement, vulnerability accuracy, labels, negatives,
independent gold, training eligibility or splits are claimed. Original pilot and
source deliveries are unchanged. Existing real-only, source-use, grouping,
scientific-quality and publication holds remain.

## Next bounded outcome

Before another implementation campaign, one existing session should make an
admission decision on a concrete real-public supervised task: exact target meanings,
usable source versions, allowed use and a genuinely qualified held-out evaluation
frame. Reuse inspected source metadata; stop/pivot if that frame cannot be evidenced.
Do not make another broad inventory or compiler-install request the default path.
The current evidence does not support promising a few-week Q1 submission package.
A decisive learner comparison can be estimated only after that admission; model
choice remains open, and source training labels may be reused with targeted auditing
rather than double manual relabeling of every training function. Keep SCAFE distinct
from the data-centric intervention study.

## Private reproducibility and validation boundary

Versioned derivative: Desktop/D_local/func-dataset/scafe_function_dataset_pilot/
derivatives/compiler_census_v1_20261110T0900UTC. It contains the predeclared protocol,
observed summary/reference geometry, local replay recipes and file hashes; no pilot
or source body is published in Git. Workspace stage is
Documents/Codex/2026-11-01/task/scafe_iteration_20261110_0900UTC. The recipes use the
recorded authorized local paths and cached binary, not an automatic acquisition.
Obtain the task run lock before replay; the runner separately leases shared compute.
Failed/missing cases remain explicit. This note reports the scientific experiment;
mandatory project tests, both integrity validators, fresh manuscript build/visual
checks, PDF refresh and local Git outcome are recorded in the final run checkpoint.
Full pytest is not claimed. Manuscript scientific claims are unchanged.
