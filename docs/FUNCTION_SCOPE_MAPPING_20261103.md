# Contract-qualified lexical function mapping — 3 November 2026, 13:00 Israel

The three records with identical function fragments in multiple contracts now each
have one lexical owner-compatible span under their explicit producer-qualified
identifier. This is a conditional provenance/context mapping, not independent
validation of the producer's identifier or a suspicious-pattern label. All original
candidate spans, raw contexts and prior releases remain immutable.

## Whole-cohort outcome

The protocol was fixed before examining source scopes and applied to every one
of the 145 context-covered candidates, not just the three ambiguous cases.

| Property | Prior evidence | New mapping evidence |
|---|---:|---:|
| Candidates / exact source contexts | 2,323 / 145 | unchanged |
| Context documents | 26 | unchanged |
| Unique raw fragment matches | 142 | unchanged |
| Multiple identical raw fragment matches | 3 | all original alternatives preserved |
| Qualified lexical scope assessment | unassessed | 145 assessed |
| Unique named-function/owner-compatible mappings | unassessed | 139 |
| Previously ambiguous records with one owner-compatible alternative | unassessed | 3 |
| Unsupported special function forms | unassessed | 5 constructors + 1 receive |
| Independent reviews / training eligible / actual splits | 0 / 0 / 0 | unchanged |

In one exact pinned source file, `RC` begins at line 168 and `RCpro` at line 360.
`RC.giveToken` maps to the exact fragment at character 11,315 (line 314),
`RC.refundEther` to 12,328 (line 345), and `RCpro.refundEther` to 19,566 (line 549).
The two refund functions have identical original bytes, so choosing the first
hash match would mis-map the RCpro record. Contract qualifiers supply an explicit
additional source claim, and independently recorded lexical declaration spans
show which alternative agrees. Qualifier truth is not independently established.

There is no increase in raw-fragment uniqueness: 142 stays 142. The new fact is
that three otherwise ambiguous alternatives can be distinguished conditionally
by the producer qualifier. Six other covered records are unsupported by this
named-function-only rule. They are not demonstrated mapping errors or bad labels.
The first diagnostic grouped them with generic unsupported/conflict outcomes;
inspection showed five constructor qualifiers and one receive qualifier. Only
the diagnostic status was clarified; the acceptance rule and 139 selected records
were not broadened after seeing the results.

## Conservative scope rule

The new mapper consumes existing pinned source contexts and their exact original
function spans. It masks comments with the frozen helper, preserves literal tokens
and Unicode offsets, and checks global brace balance. It records top-level
contract/library/interface declaration spans. An accepted candidate must start at
a real named-function token, lie at direct declaration-member depth, match the
source-provided function name, have its complete body end at the original span
boundary, and have exactly one compatible enclosing declaration with the claimed
contract name. Duplicate declaration names, multiple compatible alternatives,
unsupported qualifiers, malformed/complex headers, nested function-like text,
wrong owners and incomplete scope evidence remain unresolved. Constructor,
receive and fallback forms are deliberately outside this version's rule.

This is a lexical scope recognizer, not a Solidity parser/compiler, inheritance
resolver or overload oracle. Contract qualifiers remain source metadata. A lexical
mapping does not establish authorization policy, call reachability, dependency
closure, vulnerability evidence or safe negative labels. Comments/literals that
look like declarations cannot create an owner, and no first-match choice is used.
Future AST/inheritance support requires a new validated protocol, not silent
promotion of these mappings to semantic truth.

The reference-to-gap review's context/dependency P1 question motivates retaining
explicit enclosing-contract evidence (see `docs/SCAFE_REFERENCE_REVIEW_20261101.md`,
GoLiSA pp.5–6 and GPT4Detection pp.12,18). This step adds localized evidence for
future review; it is not a model, baseline comparison or manuscript accuracy gain.

## Immutable deliverable and gates

New local derivative:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/scope_mappings_v1_20261103T1300/`

Four files contain the 2,323 scope links, 26-document declaration index, summary
and manifest. `context_artifact` paths refer to the immutable, manifest-hash-bound
`content_context_v1_20261103T1200` input release, not to new copied source files.
The manifest binds the original delivery, parent context release, generator and
frozen helpers. Scope links preserve all original alternatives and separate the
qualifier claim from lexical owner evidence and any selected conditional mapping.
No raw source, reviewer payload or dataset derivative is uploaded or staged in Git.

All 18 delivered originals, prior context/group/model-view releases and reviewer
packets remain unchanged. Context coverage stays 145, direct corpus overlap stays
145 and conservative quarantine stays 264/316; these are not fresh holdouts.
All labels remain unknown with unobserved masks, no eligibility and no actual splits.
The three categories are not the legacy eleven model outputs. Independent context
and category review, provenance/license checks, project/clone-separated cohorts
and fair lightweight baselines still precede CodeBERT training. No inference,
training, installation, downloads or source execution occurred.

## Reproduction and validation

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/map_function_scopes.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The output directory must be new. Only the existing 26 context documents are
analyzed; no corpus rescan occurs. All four release files reproduce byte-for-byte.
Independent checks verify original span/hash preservation, every chosen function
and declaration boundary, the three explicit selections, six unsupported forms,
unknown labels/masks and unchanged parent contexts/group artifacts.

Ten new regressions cover selecting a second identical fragment; fake scopes in
comments/literals; Unicode/CRLF; repeated owners and multiple matches; wrong names,
qualifiers and special forms; nested text; unbalanced/complex headers; exact body
boundaries; declaration kinds; replay/parent preservation; and refusal to reset
reviewed/tampered inputs. Applicable tests, both integrity validators, fresh full
paper compilation, visual checks, backed-up PDF refresh and final/staged Git
review are mandatory before commit. Exact results live in task workspace
`scafe_iteration_20261103_1300/run.json`. No Q1-readiness claim is made.
