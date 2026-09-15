# SCAFE Upgrade Plan: Fast Compilation-Free Function-Level Vulnerability Detection

## 1. Executive summary

Upgrade SCAFE from a contract-level, vulnerability-agnostic risk scorer into a fast, compilation-free, function-level vulnerability detector and localizer.

The upgraded system should combine:

1. the existing SCAFE handcrafted features;
2. Tree-sitter Solidity parsing;
3. a lightweight semantic layer built over the syntax tree;
4. vulnerability-specific relational rules;
5. a small classical classifier for combining exact and uncertain evidence.

The intended output is not only a probability. For each finding, return the vulnerability type, affected function, relevant lines, confidence, and an evidence path such as:

```text
attacker-controlled parameter
-> external call target
-> external interaction
-> related storage write after the call
```

The primary research hypothesis should be:

> Lightweight, compilation-free relational analysis can approach strong learned-model accuracy for pattern-oriented vulnerabilities while improving source coverage, speed, interpretability, and cross-dataset robustness.

Do **not** claim universal state of the art before matched experiments. Complex protocol and business-logic vulnerabilities will remain outside the reliable scope of this design.

---

## 2. Current SCAFE and motivation

Current SCAFE performs binary, contract-level, vulnerability-type-agnostic triage using:

- 257 structural source features;
- an optional 87 identifier/context features;
- CPU-based tree classifiers;
- no compiler, bytecode, GPU, hosted API, or third-party detector.

Its main limitations are:

- aggregate contract counts dilute a vulnerability located in one function;
- most features describe the presence of constructs, not relationships between them;
- the semantic/context stream is primarily lexical and heuristic, not full program semantics;
- approximate regular-expression parsing cannot reliably represent nested control flow;
- there is no direct function/line localization;
- cross-dataset results show substantial distribution and label shift.

The upgrade should preserve SCAFE's low-dependency and interpretable design while adding the minimum program analysis needed to represent vulnerability mechanisms.

---

## 3. Target architecture

```text
Solidity source
    |
    v
Tree-sitter Solidity parse tree
    |
    v
Scope-aware symbols + lightweight SCAFE IR
    |
    +--> per-function CFG
    +--> approximate call graph
    +--> modifier expansion
    +--> storage read/write facts
    +--> reaching definitions / def-use
    +--> taint propagation
    +--> function summaries
    |
    v
Vulnerability-specific relational detectors
    |
    +--> deterministic high-confidence findings
    +--> relational numeric/categorical features
    |
    v
Small per-vulnerability classifier
    |
    v
Function/line findings + contract-level aggregation
```

Use one classifier per vulnerability type rather than one universal vulnerable/clean head. Different vulnerability classes require different evidence and decision thresholds.

For function `f`, use three feature scopes:

```text
X(f) = local_function_features
     + callee_summary_features
     + enclosing_contract_context
```

This is necessary because protection may live in modifiers, inherited contracts, callers, or internal callees.

---

## 4. Parser and dependency stack

### Minimal Python-facing stack

- `tree-sitter`: parsing runtime;
- `tree-sitter-solidity`: Solidity grammar;
- `networkx`: initial CFG algorithms, dominance, reachability, and strongly connected components;
- `scikit-learn`: classifiers, calibration, metrics, and ablations;
- optionally `xgboost` if it shows a reproducible gain.

Optional components:

- `rustworkx` if profiling proves graph operations are a bottleneck;
- `z3-solver` only for bounded confirmation of already suspicious paths;
- `semgrep` as an optional external pattern stream and comparison, not a core dependency;
- Slither/Wake as compiler-backed baselines or higher-fidelity optional backends.

Tree-sitter supplies a concrete syntax tree and source ranges. It does **not** supply resolved types, a CFG, call graph, def-use graph, taint results, inheritance resolution, or storage aliases. Those must be implemented above it.

Pin exact versions in a lockfile, prefer binary wheels, record licenses, and run dependency auditing. Treat Solidity input as untrusted: cap file/project size, bound recursion and analysis iterations, never execute source, never resolve imports outside the allowed project root, and do not pass source-derived strings through a shell.

---

## 5. Lightweight intermediate representation

Normalize relevant syntax into a small IR rather than carrying the full Tree-sitter tree through every analysis.

Suggested operations:

```text
READ_LOCAL(variable)
WRITE_LOCAL(variable, expression)
READ_STATE(location)
WRITE_STATE(location, expression)
INTERNAL_CALL(callee, arguments, result)
EXTERNAL_CALL(target, value, payload, result)
DELEGATECALL(target, payload, result)
STATICCALL(target, payload, result)
BRANCH(condition)
REQUIRE(condition)
REVERT
RETURN(value)
LOCK_SET(location)
LOCK_CLEAR(location)
UNKNOWN_OPERATION
```

Every IR operation should retain:

- file, contract, and function identity;
- start/end byte offsets;
- start/end lines and columns;
- original syntax node type;
- confidence/provenance;
- enclosing branch/loop/modifier context.

Use explicit uncertainty states:

```text
EXACT_SYNTAX
RESOLVED_LOCAL
RESOLVED_PROJECT
MAY_ALIAS
UNRESOLVED_CALL
UNRESOLVED_IMPORT
UNKNOWN_DYNAMIC_TARGET
```

Do not convert missing information into a negative finding. Use `TRUE`, `FALSE`, and `UNKNOWN` internally and expose resolution/coverage features to the classifier.

---

## 6. Semantic-lite analyses

### 6.1 Symbols and scopes

Create indexes for:

```text
project -> source files and imports
contract -> parents, state variables, functions, modifiers
function -> parameters, locals, modifiers, calls
```

Resolve identifiers in lexical order: local variable, parameter, state variable, inherited declaration, unresolved. Resolve internal functions using contract, name, arity, and declared argument types where syntactically available. Keep a conservative candidate set when overload or inheritance resolution is ambiguous.

### 6.2 Control-flow graph

Create basic blocks and edges for:

- sequential statements;
- `if`/`else`;
- `for`, `while`, and `do while`;
- `break` and `continue`;
- `return`, `revert`, `require`, and `assert`;
- `try`/`catch`;
- modifier prefix, `_` continuation, and suffix.

Compute reachability, immediate dominators, post-dominators, loops, and strongly connected components. Do not enumerate every execution path.

### 6.3 Abstract storage locations

Represent storage at useful but conservative precision:

```text
owner
implementation
balances[msg.sender]
balances[param#0]
balances[*]
users[*].credit
```

Locations overlap when the same base variable and symbolic keys may unify. Unknown keys produce `MAY_ALIAS`, not `NO_ALIAS`.

### 6.4 Def-use and reaching definitions

A definition assigns a value; a use reads it. Propagate possible reaching definitions through the CFG. At a branch merge, union the definition sets. Use integer identifiers and bitsets where possible.

Example:

```text
external parameter `recipient`
-> definition `target = recipient`
-> use of `target` as delegatecall receiver
```

### 6.5 Taint analysis

Initial attacker-controlled sources include:

- parameters of public/external functions;
- `msg.sender`, `msg.value`, and decoded calldata;
- values returned by unknown external calls;
- storage writable through an insufficiently protected external entrypoint.

Sensitive sinks include:

- external/delegatecall targets;
- transferred value and recipients;
- sensitive storage writes;
- authorization decisions;
- loop bounds;
- destructive or upgrade operations.

Propagate taint through assignments, casts, tuple operations, expressions, function arguments/returns, and storage reads/writes. Record data taint separately from control dependence.

Guards such as a trusted allowlist can constrain taint. Weak conditions such as `target != address(0)` must not be treated as trust sanitizers.

### 6.6 Function summaries

For every function/modifier, compute:

```text
may_read_state
may_write_state
must_write_state
internal_calls
external_calls
delegatecalls
parameter_to_sink_relations
returned_taint
authorization_guards
reentrancy_locks
normal/reverting exits
unresolved facts
```

Substitute caller arguments into callee summaries. Process the call graph bottom-up; for recursive components, iterate summaries to a bounded fixed point.

---

## 7. Initial vulnerability relations

### 7.1 External call before related state write

For each external call `c` and relevant storage location `L`, report risk when:

```text
L influences payment amount, target, permission, or eligibility
AND L is not definitely protected on every path before c
AND a related write may occur after c
```

This represents a checks-effects-interactions violation. A write before the external call is generally safer; the suspicious pattern is an external interaction while relevant state is still stale.

Retain evidence: relevant read, external call, later related write, path/distance, alias confidence, and lock status.

### 7.2 Untrusted call target

Taint the receiver of `call`, `delegatecall`, `staticcall`, interface calls, and value/token transfers. A target is high risk when it depends on an attacker source and no meaningful dominating constraint exists.

Treat constants, `address(this)`, and protected immutable/configuration values as stronger trust evidence. For storage-held targets, inspect every available setter and its authorization.

### 7.3 Reentrant entrypoint reaches the same state

At each untrusted external interaction:

1. calculate relevant storage still exposed at the call;
2. model an adversarial callback;
3. consider accessible public/external/fallback/receive/callback entrypoints;
4. use transitive summaries to test whether they may read or write overlapping state;
5. exclude only entrypoints provably blocked by an active, correctly implemented lock.

Support same-function and cross-function reentrancy. Inspect modifier bodies; never infer protection only from a modifier name.

### 7.4 Authorization guard dominates sensitive sink

Define sensitive sinks, including ownership/role changes, upgrades, destructive operations, arbitrary calls, withdrawals, minting, pausing, and critical configuration writes.

A guard protects a sink only when:

- its successful edge dominates the sink;
- failure cannot reach the sink;
- its predicate meaningfully constrains the caller against a trusted authority;
- the authority/role configuration is not freely writable.

Distinguish real authorization from unrelated `require` statements and weak checks.

### 7.5 Unchecked call on some path

Track the success result of low-level calls and `send`. Under the abstract condition `success = false`, determine whether any CFG path can reach a normal exit or sensitive state effect without reverting, returning failure, or otherwise propagating the error.

Classify separately:

```text
ignored
assigned but unused
used but not as a guard
handled on some paths
handled on all paths
propagated to caller
unresolved
```

Do not treat `transfer()` as equivalent to `send()`; the former normally reverts on failure.

### 7.6 Tainted value reaches delegatecall

Track three separate delegatecall dimensions:

```text
delegate_target_tainted
delegate_selector_tainted
delegate_arguments_tainted
```

Target taint is the strongest generic signal. Tainted payload is common in legitimate proxies and should not alone imply vulnerability. Inspect implementation setters, authorization, allowlists, immutable assignments, initialization protection, and selector restrictions.

---

## 8. Feature design

Do not reduce each relation to one boolean. Emit richer features per function and vulnerability type:

```text
relation_true_count
relation_unknown_count
minimum_source_sink_distance
number_of_distinct_state_locations
number_of_reachable_entrypoints
guard_strength
guard_dominates
effective_lock_present
target_taint_source_kind
may_alias_count
unresolved_call_count
resolved_call_fraction
missing_import_count
analysis_confidence
```

Preserve existing SCAFE features at two scopes:

- local function features;
- enclosing contract context.

Add normalized structural motifs:

- AST parent-child n-grams;
- IR operation-sequence n-grams;
- CFG edge/path motifs;
- normalized subtrees with identifiers/literals abstracted;
- source-to-guard-to-sink relation motifs.

A fast sparse token/character n-gram linear model is also a useful low-cost baseline or auxiliary stream, but it must be tested under project-family-disjoint and cross-dataset splits because lexical features can memorize projects and dataset conventions.

---

## 9. Findings and output schema

Every finding should be independently auditable:

```json
{
  "vulnerability_type": "reentrancy",
  "contract": "Vault",
  "function": "withdraw(uint256)",
  "probability": 0.91,
  "rule_confidence": "high",
  "analysis_confidence": 0.84,
  "primary_line": 42,
  "evidence": [
    {"line": 40, "kind": "relevant_storage_read", "location": "balances[msg.sender]"},
    {"line": 42, "kind": "untrusted_external_call"},
    {"line": 45, "kind": "related_storage_write", "location": "balances[msg.sender]"}
  ],
  "protections": {"dominating_guard": false, "effective_lock": false},
  "uncertainty": ["unresolved_external_target_type"]
}
```

Contract-level risk can be derived from function probabilities, for example by maximum risk or calibrated noisy-OR. Keep the function findings as the primary output; contract aggregation must not discard evidence.

---

## 10. Labels and dataset policy

Function-level learning requires genuinely located labels.

- DAppSCAN audit records may provide file, category, function, and line fields, but function may be unavailable for some findings. Verify the actual released records and taxonomy before use.
- Empirical-Analysis includes manually located line-level annotations that can be mapped to enclosing functions.
- DIVE's released labels are primarily contract-level and should not be silently converted into precise function labels.

Important negative-label rule:

> A function without an annotation inside a vulnerable contract is not automatically clean.

Use verified-clean contracts for high-confidence negative functions. Treat unannotated functions in vulnerable contracts as unknown unless dataset documentation justifies a negative label. Consider positive-unlabeled training or exclude unknown functions from supervised loss.

Map finding lines to the smallest enclosing function or modifier. Keep contract-level/unmapped findings for contract evaluation, not fabricated function labels.

Normalize vulnerability taxonomies before training and initially target only types with enough high-quality located positives.

---

## 11. Evaluation plan

All methods must use identical samples, labels, source coverage, and splits.

### Baselines

1. existing SCAFE adapted to function scope;
2. Tree-sitter syntax features only;
3. token/character n-gram linear model;
4. SCAFE plus local CFG relations;
5. SCAFE plus def-use/taint;
6. full SCAFE plus interprocedural summaries;
7. Slither/Wake on the compilable subset;
8. a reproducible function-level neural baseline such as a code Transformer;
9. a reproducible graph-based detector where task and labels match.

### Required ablation sequence

```text
SCAFE only
-> + Tree-sitter syntax
-> + symbols and modifier expansion
-> + local CFG/dominance
-> + def-use and taint
-> + interprocedural summaries
-> + optional lexical/motif stream
```

### Metrics

Report per vulnerability type:

- PR-AUC and ROC-AUC;
- F1, precision, recall, and MCC;
- false positives per 1,000 functions;
- top-k localization recall or line-distance accuracy;
- calibration error/Brier score if probabilities are claimed;
- percentage of files/contracts successfully analyzed;
- median and p95 time per contract;
- peak memory and contracts per second.

Use project/source-family-disjoint splits and cross-dataset transfer. Deduplicate normalized sources and clones before splitting. Thresholds and calibration must be selected using training/validation data only.

Run adversarial unit tests for malformed/deeply nested Solidity to verify bounded runtime and memory.

---

## 12. Realistic performance expectations

The approach is most likely to be competitive for explicit, pattern-oriented classes:

```text
unchecked low-level calls
direct attacker-controlled delegatecall
tx.origin authorization
local access-control omissions
timestamp-to-critical-sink dependence
same-function reentrancy
```

It will be harder to match compiler-backed or semantic-heavy methods for:

```text
cross-function reentrancy with complex aliases
inheritance and proxy edge cases
oracle/economic manipulation
multi-contract protocol invariants
general business-logic vulnerabilities
```

Success should be framed as a Pareto improvement, not necessarily the highest pooled F1:

- competitive per-type function-level detection;
- compilation-free operation;
- high source coverage;
- low CPU latency;
- interpretable evidence;
- stronger cross-dataset robustness.

---

## 13. Implementation roadmap

### Phase 1: parsing and inventory

- Pin Tree-sitter and the Solidity grammar.
- Parse the corpus and record error-node rate, file size, parsing latency, and syntax coverage.
- Replace regex-based function/statement boundaries with Tree-sitter nodes.
- Preserve exact source mappings.

**Acceptance:** deterministic parse artifacts; bounded input handling; regression corpus covering Solidity versions in the datasets.

### Phase 2: local function analysis

- Build the lightweight IR and per-function CFG.
- Implement symbols/scopes, local reaching definitions, local taint, dominators, and modifier expansion.
- Implement unchecked-call, direct delegatecall-taint, and local call-before-write detectors.

**Acceptance:** unit tests include positive, protected, branch-dependent, malformed, and ambiguous examples; every finding includes evidence lines.

### Phase 3: interprocedural analysis

- Build best-effort import/inheritance/call indexes.
- Implement function/modifier summaries and bounded fixed-point propagation.
- Add storage setter provenance, cross-function reentrancy, and generic guard-to-sink analysis.

**Acceptance:** explicit unresolved states; recursion terminates; missing imports never become implicit safety evidence.

### Phase 4: learning and calibration

- Create per-function feature vectors combining SCAFE, syntax, relational, context, and uncertainty features.
- Train per-vulnerability heads.
- Calibrate using validation data only and produce contract aggregation.

**Acceptance:** reproducible manifests, predictions, thresholds, seeds, and feature inventories.

### Phase 5: matched evaluation

- Run all ablations and baselines under identical splits.
- Audit duplicates, label provenance, compiler coverage, and failure modes.
- Report per-dataset and per-type results rather than one pooled headline.

**Acceptance:** every headline table is regenerated from frozen artifacts and integrity checks.

---

## 14. Non-goals and cautions

- Do not enumerate all paths; use reachability, dataflow joins, summaries, and fixed points.
- Do not describe Tree-sitter output as compiler-equivalent semantics.
- Do not call identifier/context features complete semantic analysis.
- Do not treat an unresolved call, alias, parent, or import as safe.
- Do not infer vulnerability protection solely from modifier or variable names.
- Do not use third-party detector findings as training labels without a clearly disclosed weak-label experiment.
- Do not feed Slither detector verdicts into the primary model while claiming independent compilation-free detection.
- Do not label every unannotated function in a vulnerable contract as clean.
- Do not use random record-level splits that place clones/project families on both sides.
- Do not claim universal SOTA from results on only locally recognizable vulnerabilities.

---

## 15. Suggested research positioning

Function-level classification, expert vulnerability patterns, and structural/semantic graph features already exist in prior work. The defensible novelty is the combination of:

- compilation-free and error-tolerant parsing;
- interpretable fixed relational features;
- lightweight CPU-only static analysis;
- function/line evidence;
- vulnerability-specific classical models;
- explicit uncertainty and parser-coverage measurements;
- strict source-family-disjoint and cross-dataset evaluation.

Suggested positioning:

> SCAFE-SA is an interpretable, compilation-free, function-level vulnerability localization framework that combines lightweight relational program analysis with classical learning. It investigates which vulnerability classes can be detected competitively without compiler IR, graph neural networks, or large language models.

---

## 16. Immediate next task for the implementation agent

Build a narrow end-to-end prototype for two vulnerability classes:

1. unchecked low-level calls;
2. attacker-controlled delegatecall targets.

These provide the cleanest validation of the architecture. Implement parsing, local CFG, reaching definitions, taint, evidence output, tests, and runtime instrumentation before adding reentrancy. Compare:

```text
regex SCAFE
vs Tree-sitter syntax rules
vs CFG/def-use/taint relations
vs combined classical classifier
```

Only proceed to cross-function reentrancy after local control/data-flow correctness and label quality are demonstrated.

