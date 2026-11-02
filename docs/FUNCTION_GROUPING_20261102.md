# Function split-constraint registry — 2 November 2026, 15:00 Israel

This iteration builds reproducible hold-together constraints across all 2,323
function candidates, rather than another review package. It measures how exact
code/provenance and identifier-template constraints affect prospective evaluation
units and conservative existing-corpus quarantine. It assigns **no dataset splits,
new project identities, labels or training eligibility**.

## Predeclared graphs and measured results

| Property | Exact/provenance evidence graph | Strict template sensitivity graph |
|---|---:|---:|
| Candidate records covered | 2,323 | 2,323 |
| Connected components | 1,326 | 1,228 |
| Non-singleton components | 195 | 212 |
| Largest component | 68 | 170 |
| Components spanning sources | 1 | 4 |
| Components joining multiple recorded project anchors | 0 | 3 |
| Direct hash-verified local-context overlap seeds | 136 | 136 |
| Conservative quarantine after component propagation | 255 | 307 |
| Additional propagated quarantine candidates | 119 | 171 |

The evidence graph holds together exact model-input bytes, independently recomputed
lexical-token equality, source-qualified project metadata, source/revision-qualified
recorded addresses and pinned producer context hashes. These evidence types are
explicitly distinguished in the edge registry: some are verified input identities,
others are source-asserted metadata, **not verified true projects**. A pinned context
hash does not mean its source bytes were independently recovered for every record.

The strict sensitivity graph adds consistent identifier-renaming template equality,
preserving literal values, curated keywords/types and member selectors. Its three
multi-project components contain 170, 150 and 109 records and respectively four,
three and two recorded project anchors. Template relationships can reflect common
boilerplate or different callees/types, not shared provenance or equivalent behavior.
These are candidate split constraints to inspect, not certified semantic clones or
proof of leakage under a particular unrun split. Using project names alone would
not enforce these stricter template constraints. Neither graph is a comprehensive
near-clone audit; edits to constants, structure or dependencies can remain unlinked.

The 136 direct overlap seeds retain the earlier exact raw-context hash evidence.
The 255/307 totals are conservative exclusions propagated through each graph, **not
255/307 independently confirmed raw-source matches**. Cross-component relationships
and historical training/test membership remain unassessed. No recovered-code or
model-performance gain is inferred from propagation.

## Source and representation quality findings

All 1,680 missing original project IDs remain missing; the 15 source-qualified
project metadata anchors are retained without inventing new identities. Recorded
addresses provide hold-together hints, not full project or chain provenance.

There are 1,939 distinct raw model-input hashes. Lexical signatures were computed
for 2,321 records, yielding 1,865 token signatures and 1,743 identifier-template
signatures. Two HF-source records raise `Unterminated source literal` under the
new recipe and need representation review:

- `1da24dddb7fb803bd71f81364797ab4ff2f90770e9706327a06fce91cbb05662`
- `f86fe72988b2d26294498afbb5d195a12bf0c4105339707243afd67b09521840`

They remain in the inventory with explicit lexical-error status and available
byte/provenance constraints; no token/template edge is guessed for them. This is
not an independently verified Solidity compiler rejection. Do not treat absent
family edges as evidence that these records are independent or evaluation-ready.
The original snippets are not silently repaired, discarded or relabeled.

The frozen heuristic token recipe masks comments with literal/escape awareness,
ignores whitespace outside literals, preserves string/numeric contents and uses
maximal operator alternatives. Identifier normalization preserves binding patterns
within each function and keeps member selectors. It is not a compiler lexer,
AST/type resolver, semantic equivalence test or vulnerability detector. Tests
specifically retain distinctions such as string whitespace, constant values,
member selectors and increment versus separated operators. Source claims and
pattern labels do not determine grouping; changing source-claim descriptors or
row order leaves the graph result unchanged.

## Versioned local artifacts and reproduction

Local output, excluded from Git:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/constraint_groups_v1_20261102T1500/`

- `candidate_groups.jsonl`: every stable ID, both component IDs, signature/error
  status, original provenance, direct versus propagated quarantine and unchanged
  unknown labels, false eligibility and unassigned split.
- `constraint_edges.jsonl`: deterministic spanning edges with evidence type and
  hashed constraint key; strict-only edges are marked separately.
- `components.jsonl`: sorted members, metadata anchors, source counts, direct
  overlap seeds and conservative quarantine for both graphs.
- `summary.json`, `manifest.json`: measured results, protocol/limits, generator and
  comment-masker hashes, original/recovery manifest identities and artifact hashes.

From `sc_review` with existing Python dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/group_function_candidates.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --recovery-dir /absolute/path/local_context_v1_20261102T1300 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The script verifies the 18 delivered originals and recovery release, binds each
seed to its original pinned context and actual context-file hash, and rechecks
inputs before creating a new output directory. It refuses existing output and
writes into original/input release directories. The run is bounded to 5,000
records and 20 million input characters. Stable component IDs hash sorted member
IDs. All five final artifacts reproduce byte-for-byte on a second run. Original
pilot files and all prior derivatives remain unchanged; no dataset/code snippets,
source files or reviewer material are uploaded or force-added to Git.

## Readiness implication and next gate

Before this run there was no integrated versioned constraint membership for all
candidates; now all 2,323 have both graph memberships and auditable edge lineage.
This is a measured evaluation-integrity/dataset-readiness advance, not new gold
labels or a ready training benchmark. Raw context coverage remains 136/2,323;
completed independent annotations, training eligibility and frozen splits remain
zero. Suspicious patterns are not verified exploitable vulnerabilities.

Next inspect the template bridges and their overmerge risk, recover the two
flagged representations from original evidence, and resolve unknown provenance.
Prespecify how valid project/clone constraints and conservative quarantine govern
future grouped development/test cohorts, without choosing a favorable graph using
model results. Existing inspected external corpora cannot become fresh confirmation
sets. Independent context/label adjudication, source licensing, partial-label
observation masks and fair lexical/nuisance baselines still precede CodeBERT training.
The reference-to-gap review's project/template separation and nuisance-control
requirements motivate this step; publication is not proof that either graph is
scientifically sufficient.

Nine new regression tests cover literal/operator/selector preservation, consistent
renaming, source-qualified metadata, conservative propagation, order/claim
independence, malformed inputs, missing lexical edges, original preservation and
identical replay. All 83 focused tests and both integrity validators passed; full pytest remains
unavailable and was not installed/run. Fresh isolated paper build/visual/PDF-refresh
evidence is recorded in task workspace `scafe_iteration_20261102_1500/`. The manuscript's
scientific claims are unchanged. No classifier accuracy or Q1-readiness claim is made.
