# Guarded if-block annotation view v2 — 3 November 2026, 11:00 Israel

This opt-in extension resolves the one marker left by the 10:00 narrow audit-prose
filter. It creates a separate lineage-bound view and preserves the previous release.
It does not certify original contract/compiler validity or supply training labels.

## Measured change relative to v1

| Property | v1, 10:00 release | v2, 11:00 release |
|---|---:|---:|
| Candidate records | 2,323 | 2,323 |
| Declared marker occurrences remaining | 1 | 0 |
| Changed code views relative to v1 | — | 1 |
| Earlier six semicolon masks preserved | 6 | 6 |
| Total masked tails relative to delivered originals | 6 | 7 |
| Lexical-signature successes | 2,323 | 2,323 |
| Exact/provenance components | 1,324 | 1,324 |
| Strict template components | 1,226 | 1,226 |
| Independent reviews / training eligibility / splits | 0 / 0 / 0 | 0 / 0 / 0 |

The affected stable ID remains
`b407a429c804fffa264515bba167e0c7bcf394622854252df70c3bb0baf50db8`.
Its explicit source-claim marker follows an isolated same-line `if (...) {` header.
The condition, opening brace, following statements and closing braces remain intact.
Only the recorded annotation tail becomes spaces. This is a versioned source-view
proposal, not proof of equivalence to independently retrieved original contract
bytes or a verified vulnerability claim.

## New acceptance boundary and preserved v1 behavior

Default behavior still uses the v1 post-semicolon rule. V2 requires BOTH
`--include-if-block-tails` and a hash-verified `--prior-view-dir`. Before extending
anything, the tool regenerates all 2,323 default v1 code views from the delivered
originals and compares them with the immutable parent release. It refuses changed
ancestry, a different protocol, or prior supervised/reviewed annotations without
explicit migration. A new code change is allowed only for a record already flagged
for a residual marker in that parent. Every output binds original and direct-parent
view hashes as well as the protocol version.

For the new anchor the same physical line must begin, after whitespace/comment
masking, with a single complete `if` header followed by its block-opening brace.
Literal-aware tokens and balanced parentheses support nested calls and parentheses
inside strings. Other statement headers, function declarations, incomplete headers,
`else if` and extra structural tokens are rejected by this deliberately narrow
recognizer. This is a lexical shape check, not Solidity type/AST/semantic validation.

The marker must be outside literals/comments after that header. Its remaining line
tail may contain no brace or semicolon; otherwise it is preserved for review to
avoid discarding possible code. Original comparison expressions, literal markers,
comments and code on ambiguous same-line tails are not masked. CR/LF and Unicode
character offsets are retained. The old semicolon rule remains unchanged and its
six prior code views are byte-identical. Old v1 artifact manifests still pin their
historical generator checksum; use that historical source revision for full
byte-identical reproduction, rather than relabeling old manifests as v2.

## Evidence, schema and limits

The new local derivative is:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/annotation_view_v2_20261103T1100/`

It contains the same model-view/audit/event/group artifact layout as v1. Seven
span events distinguish six `statement_semicolon` anchors and one
`complete_if_block_opener`. The manifest binds the original delivery, baseline
groups, recovered-context release, direct parent v1 view, generator and frozen
helper checksums. Every model view records `prior_view_model_code_sha256` alongside
its original-parent and new-code hashes. No stale producer signatures are copied;
all grouping signatures and constraints are recomputed.

No marker of this declared `<- [@]audit-issue` form remains in this release. That
statement is scoped to the audited pattern, not all possible source-claim cues.
Zero representation flags does not establish source validity, sufficient context,
reviewer truth or training readiness. All categories remain null with all-false
observation masks: unknown/unobserved, never negative. No project identity, license
clearance, independent annotation, eligibility or split is inferred.

Raw-context coverage stays 136/2,323. Direct corpus overlap remains 136 and the
conservative 255/307 quarantine scopes remain unchanged. Project/template/clone
limitations, context sufficiency, independent category adjudication and frozen
cohort gates still precede CodeBERT training. No model inference/training or
accuracy improvement is claimed, and manuscript scientific claims are unchanged.

## Reproduction and validation

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/sanitize_function_annotations.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --baseline-groups-dir /absolute/path/constraint_groups_v1_20261102T1500 \
  --recovery-dir /absolute/path/local_context_v1_20261102T1300 \
  --prior-view-dir /absolute/path/annotation_view_v1_20261103T1000 \
  --include-if-block-tails \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

All final artifacts reproduce byte-for-byte on a second preparation. Original
records, all earlier derivatives and existing source/model inputs remain intact.
Outputs are local only; no candidate code or reviewer/source payload is force-added
to Git or uploaded. The first staged validation selected the old repository module
because of import precedence, rejected the new keyword before processing, and
created no output. Corrected staged precedence now asserts the loaded file path;
this tooling issue and resolution are recorded rather than counted as research.

Five new regressions cover nested/literal parentheses, explicit opt-in, preserved
conditions/newlines, rejection of structural tails and other headers, mandatory
parent lineage, reproduction of v1 views and refusal to reset prior human labels.
All 99 focused tests and both integrity validators passed; full pytest remains
unavailable and was not installed/run. Fresh paper build/visual/PDF-refresh
evidence is recorded before commit. Evidence and exact outcomes are in task workspace
`scafe_iteration_20261103_1100/`; unresolved scientific readiness gates remain explicit.
