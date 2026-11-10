# Independent compiler count reference, 10 November 2026 13:00 UTC

## Question, selection and fixed units

The prior 19-source diagnostic establishes comment dependence and an opt-in view,
not absolute structural correctness or classification value. This session asks
whether that view agrees better with an independent exact-version compiler AST
reference. It does not repeat the old preprocessing-invariance or CVE census test.

Before compilation, selected all two already-bound MIT source contexts with exact
publisher compiler metadata/cache availability. Both use Solidity
0.6.12+commit.27d51765. They are previously inspected development sources, not a
fresh confirmation set, independent project sample or population study. No new
source acquisition, target labels, reserved outcomes, generated programs, model
fit/scoring or source execution. Existing classifier/evaluation gates remain.

The immutable protocol fixes inputs/hashes, versions, units, resource budget and
an early continue gate: aggregate absolute count error must decrease, with no
source/counter regression. Otherwise report a negative result; do not tune units
or select cases after outcomes. Four primary reference units are:

- n_contracts: ContractDefinition with contractKind=contract, including abstract
  contracts, excluding libraries and interfaces.
- n_functions: named FunctionDefinition kind=function, including abstract
  declarations, plus FunctionTypeName expressions. This is the explicit function-
  keyword-construct proxy, NOT a count of executable functions. Named declarations
  and function types are also reported separately; no silent unit conversion.
- n_events: EventDefinition.
- n_modifiers: ModifierDefinition.

Constructors, fallback/receive, libraries and interfaces are separate auxiliaries.
This narrower definition avoids mistaking a keyword proxy for semantic function
or vulnerability truth. Eight software-only regression tests cover these cases,
unsupported AST shapes/kinds, nested source units and bounded traversal.

## Exact tool/input lineage and bounded execution

Both original source snapshots were already MIT-qualified and pinned at
InPlusLab/ReentrancyStudy-Data revision
e3e089c4d119ac4de1398345c525f06dc4f2ef96. Parent source hashes/Git bindings and all
original notices remain unchanged. No new rights adjudication or promotion of
native false-alarm claims. Native labels are irrelevant to this structural target.

Used the installed direct binary venv/.solc-select/artifacts/solc-0.6.12/solc-0.6.12,
SHA-256 05ad8afa83df3b51d36fe9a84ea4467b3ed17585c903946985d6e2cd5e95685a.
Its --version matches the full metadata build. No wrapper/version-selector change,
install, model/corpus download, network or paid compute. Each bounded source has no
imports; standard-json embeds source.sol and requests AST only, optimizer disabled.
Compiler/source hashes, exits and output limits are retained; no bytecode request,
contract/constructor execution, PoC or downloaded script execution.

An exclusive shared-compute lease covers the compiler and paired extractions.
Budget: 20 seconds/compiler case, 120 seconds total, maximum 8 MB stdout/case.
Both cases compile with exit 0 and no diagnostics; measured comparison duration is
3.70 seconds. PYTHONHASHSEED=0, original extractor and opt-in view hashes are fixed.
The original, comment-controlled and compiler reference use the same source unit,
not a new training set. The new view remains incompatible with old models/caches.

## Measured comparisons

| Source hash prefix | Counter | AST reference | Original | Comment-controlled |
| --- | --- | ---: | ---: | ---: |
| 959aa4c2a34d | n_contracts | 2 | 5 | 2 |
| 959aa4c2a34d | n_functions | 130 | 136 | 130 |
| 959aa4c2a34d | n_events | 3 | 3 | 3 |
| 959aa4c2a34d | n_modifiers | 0 | 0 | 0 |
| decbea9b6b91 | n_contracts | 1 | 1 | 1 |
| decbea9b6b91 | n_functions | 23 | 27 | 23 |
| decbea9b6b91 | n_events | 9 | 9 | 9 |
| decbea9b6b91 | n_modifiers | 0 | 0 | 0 |

Aggregate absolute count error: 13 to 0; exact matches: 5/8 to 8/8; regressions: 0.
The prespecified continue gate passes on this two-source diagnostic. It is not
13 distinct defects, an accuracy percentage, population generalization, significance
or model improvement. Both sources have zero function-type expressions, so the
named declaration and primary proxy counts coincide here. They also have three
constructors total (excluded), 11 interfaces and two libraries (excluded from
n_contracts). Do not call all 153 named declarations executable functions: abstract
interface declarations are included. Compiler-reference agreement for these four
fields does not validate all 244 features, vulnerability mechanisms, source-line
localization, runtime behavior or every possible comment edit.

## Delivered reference and limits

Added src/evaluation/ast_counter_reference.py: bounded iterative traversal of one
compact SourceUnit, explicit modern node kinds, separate auxiliaries and no
security-label/model-admission output. It rejects unsupported/legacy roots or
function/contract kinds and nested source units rather than guessing or merging
imports. It consumes AST metadata only and does not launch compilers or source.

Private derivative: Desktop/D_local/func-dataset/scafe_function_dataset_pilot/
derivatives/compiler_count_reference_v1_20261110T1300UTC. It preserves protocol,
source/compiler/output hashes, AST outputs, independently reproducible counts and
numeric comparison. Raw compiler input bodies remain in the private task workspace;
source originals/notices stay in the parent release. No source/AST payloads or raw
addresses are published through Git. No semantic security labels, training/evaluation
eligibility or frozen splits added. Existing opt-in extractor, legacy source recipe,
sanitizer, datasets, models and primary paper tables are unchanged.

The independent checker recounts saved ASTs without the new helper and verifies
all four units, exact source/compiler/input/output hashes and all eight errors.
This is an independent reference algorithm, NOT an independent source cohort or
human vulnerability adjudication. No full-suite or compiler proof beyond the
recorded cases is implied.

## Next checkpoint

The opt-in view now has preprocessing and limited absolute-count evidence. Next
bounded session should establish the separately versioned source/feature-cache and
matched existing-corpus development-comparison admission plan, with native target
scope, common groups, style/nuisance baselines, uncertainty and resource limits.
Do not repeat these completed diagnostics unless relevant inputs change. Do not
feed new features to old models, bypass source/evaluation/training gates, overwrite
shared caches or reopen the stopped alarm-source learner route without new evidence.

This does not establish classifier benefit, novelty, two distinct Q1 contributions
or a submission-ready date. All real-public-only, privacy, compute/lock, validation,
fresh full build/visual/PDF refresh and specific remote holds persist. Applicable
tests, both integrity validators, PDF and local Git/own-lock evidence are recorded
in the final run checkpoint. Manuscript scientific claims/automation are unchanged.
