# Localized CoinFabrik source supervision — 5 November 2026, 10:00 UTC

Added a separate, reproducible source-finding proxy layer for three existing
functions, including the first source-specific authority-policy observation in
this workstream. Original labels, code views, source claims and the 33 SCRUBD RE
proxies are unchanged. These new observations are **retained upstream dataset
claims**, not independently adjudicated patterns, confirmed vulnerabilities,
training-ready examples or detector-performance evidence.

## Fixed frame and category meanings

Before export, workspace `scafe_iteration_20261105_1000UTC/run.json` fixed the
643 CoinFabrik standalone candidates, 171 retained finding objects, source/category
crosswalk, exact span/line-anchor gates and small offline budget. Revision remains
`efd709441987a28bd7220d79b2d872f8c7cfa8d9`; no upstream request, new corpus/model,
software installation, student delivery or model fitting was used.

Only these source-specific definitions are allowed in v1:

| Cell | Source crosswalk | Meaning and boundary |
|---|---|---|
| reentrancy | Explicit source class Reentrancy and integer CWE 367/696/691/280 | The upstream dataset reports a reentrancy/order/mitigation concern attributed to this localized function; not independently established susceptibility, pattern truth or exploitability. |
| authorization | Source Access Control/Authorization and integer CWE 284/285 | The upstream dataset reports an authority/privilege-policy concern attributed to this localized function; excessive privilege is not proof of a missing guard or unauthorized execution. |
| unchecked_external_call | No mapping | Unknown throughout v1; unchecked-transfer titles or missing findings are not silently turned into labels. |

Each observed cell has an explicit target-definition ID. The CoinFabrik RE-order
namespace is distinct from SCRUBD's source-assessed susceptibility namespace and
from the three canonical annotation axes/legacy eleven outputs. A reviewed
multi-source target-alignment protocol is required before combining them in a
learner. No source-CWE rule is used to manufacture independent evaluation truth.

## Evidence gates and measured export

A positive requires the vulnerable role, pinned source/project/path agreement,
retained finding link, exact whole-context/raw function hashes, unique qualified
lexical owner/symbol and a positional source line anchor wholly inside that
function. Empty `()` in a source pointer is treated as symbol notation, not proof
of a zero-argument type signature. Overloaded or ambiguous symbols are refused.
Contract-only references, missing context or unsupported definitions stay unknown.
Full dependency closure/compiler semantics are not established by this linkage.

| Function | Source proxy | Finding references and raw-source anchors |
|---|---|---|
| PegStability::swapVAIForStable | RE/order concern | venusProtocol00 finding 1; line 163 inside function lines 151–168 |
| PegStability::swapStableForVAI | RE/order concern | Same finding; line 189 inside function lines 176–198 |
| LiquidityGaugePoolController::_setPool | Authority/privilege-policy concern | neptune00 findings 0 and 3; lines 18–68 match the function span |

Three finding objects support four edges to three function/category cells across
two projects and two full contexts. The two Neptune findings do not count as two
independent labels or two target cells. All three functions have verified context
presence in existing SCAFE corpora and retain overlap/group quarantine; none is a
fresh confirmation example. Known project IDs do not establish clone independence.

The 643-row export has 312 patched-unobserved functions, 266 without a specific
finding link, 62 linked but unmapped and three mapped. Per-link checks record
83 unsupported-category edges, 72 patched-role edges, eight edges without qualified
exact context/span and four mapped support edges. These edge counts are not
annotation accuracy, false-positive rates or independent samples. Observed source
positive cells rise from no CoinFabrik target layer to three (RE two, authority
one); observed negative cells remain zero. Training/evaluation eligibility,
independent gold and assigned splits remain zero. Source labels are not added to
the original 6,969 unknown harmonized staging cells.

## Limits that affect interpretation

Both Venus functions visibly carry `nonReentrant`. The source finding's ordering
concern must not be read as proof that reentry defeats that guard; dependencies,
callers and context still need targeted quality review. Neptune's `_setPool` is
internal: its findings concern caller authority, privileged parameter changes and
policy/initialization assumptions, not proof of an exposed unauthenticated setter.
Those semantics motivated a distinct authority-policy proxy instead of a missing-
authorization label. Source class/severity/CWE does not supply calibrated confidence.

The delivered finding objects and declared upstream file hashes/revision are
preserved separately. Per-object fingerprints verify retained structured content;
no fresh upstream raw JSON blob or underlying audit-report authentication is
claimed. Original source/license eligibility and semantic quality remain gates.
The earlier POOHMO/MonkeyScam concerns and compiler-version qualification are
unchanged. No student response is required for continued collection, and no two
fresh independent reviews of every training function are imposed.

## Artifacts and reproduction

Private local derivative, not staged or uploaded:

`scafe_function_dataset_pilot/derivatives/coinfabrik_source_supervision_v1_20261105T1000UTC/`

Seven files preserve 643 source-proxy rows, 643 separate code-only inputs, per-link
checks, all 171 original finding objects byte-for-byte, summary, data card and
manifest. All seven final files replay byte-for-byte. Manifests bind delivered
originals, context/scope/view ancestors, generator/helpers and outputs. Every model
input is identical to annotation-v2; finding descriptions, rationales, categories,
labels and metadata are not appended to code. Residual code/style cues still need
matched nuisance/leakage controls before any model claim.

`scripts/adapt_coinfabrik_source_findings.py` provides a checked diagnostic join.
It preserves null/false observation masks, accepts only supported source-positive
cells and refuses training, gold/eligibility promotion or use of this namespace
with the SCRUBD-specific loader. The existing generic masked loss is unchanged;
no model/tokenizer/checkpoint/optimizer is loaded. A future multi-source migration
must retain per-cell meanings/masks and pass all existing readiness gates.

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/adapt_coinfabrik_source_findings.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --scope-release-dir /absolute/path/scope_mappings_v1_20261103T1300 \
  --view-dir /absolute/path/annotation_view_v2_20261103T1100 \
  --output-dir /absolute/path/NEW_VERSION
```

Regression fixtures cover category/CWE meaning, pointer/anchor correspondence,
overload refusal, raw span/context/owner mismatches, patched/unmentioned unknowns,
positive-only masks, code/prose separation and readiness/namespace refusal.
Applicable checks, both integrity validators, fresh full paper compilation/visual
inspection, input-matched backed-up PDF refresh and reviewed Git publication are
required; actual outcomes are in the workspace report. Manuscript scientific
claims and detector scores are unchanged.

Next assess target alignment and source quality on these concrete cases, retaining
explicit guard/internal-helper uncertainty, and pursue the eight context-missing
support edges where local evidence permits. Independently adjudicated grouped
evaluation, justified negatives/class supports, licensing, frozen splits, matched
simple baselines and a reviewed multi-source protocol still gate CodeBERT fitting.
