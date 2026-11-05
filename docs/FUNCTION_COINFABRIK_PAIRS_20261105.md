# CoinFabrik retained role-pair evidence — 5 November 2026, 11:00 UTC

## Decision and measured outcome

Do not create function negatives merely from `patched` membership. Under a
predeclared exact metadata pairing rule, 131/272 unique vulnerable/patched pairs
have **byte-identical current classifier inputs**. This is 48.16% of pairs, not
a measured label-error rate: no opposing labels were assigned. A deterministic
code-only classifier cannot distinguish the two endpoints of those exact pairs
if opposing targets were imposed. A contextual repair can still occur elsewhere.

| Measure | Count |
|---|---:|
| CoinFabrik candidate frame | 643 (331 vulnerable-role, 312 patched-role) |
| Unique role pairs / paired records | 272 / 544 |
| Unpaired records / ambiguous buckets | 99 / 0 |
| Byte-identical pairs | 131 |
| Literal-preserving heuristic-token-identical pairs | 149 (54.78%) |
| Different-token pairs | 123 |
| Previously source-positive functions with unique counterparts | 3 |
| Source-positive pairs with changed bytes and tokens | 3 |
| Pairs already sharing both grouping components | 272 |
| New labels / verified repairs / eligible records / splits | 0 / 0 / 0 / 0 |

The 18 token-identical but byte-different pairs are a whitespace-insensitivity
measurement, not actual CodeBERT-tokenizer equivalence or semantic equivalence.
The 123 different-token pairs are **unverified repair candidates**, not negatives.
No accuracy, training, inference or journal-readiness improvement was measured.
The useful advance is a reproducible dataset-selection decision and measured
limits of role-derived function supervision, rather than another source inventory.

## Frozen frame and method

Workspace `scafe_iteration_20261105_1100UTC/run.json` records the hypothesis,
frame, rules, budget and input hashes before comparison. Source revision remains
`efd709441987a28bd7220d79b2d872f8c7cfa8d9`. All 643 candidates are considered;
no favorable subset or fuzzy/header alignment is selected.

Pair by exact project ID, relative source path with only its second role segment
removed, and qualified function identifier, at one recorded source pin. Require
exactly one vulnerable and one patched endpoint. Reject missing counterparts,
duplicate IDs, mixed revisions, malformed paths and ambiguous qualified symbols
(including unresolved overloads). Metadata pairing is not proof that a patch
fixed a particular finding or that the source was independently authenticated.

Compare current annotation-v2 view bytes and a frozen heuristic regex token
sequence preserving identifiers, literal contents, numbers and operators. Ignore
whitespace outside tokens only. This is neither a compiler lexer/AST, normalized
semantic representation nor CodeBERT tokenization. Unclosed string literals fail
rather than silently dropping content. Parents, code hashes, original delivery,
source-layer diagnostic schema and existing grouping hashes are checked. A pair
must already share both evidence and strict-template components; no split or
quarantine relaxation is performed. The frame includes all source claims and
unlabeled context functions; its percentages do not estimate a population error
rate or independent semantic label quality.

## Fixed three-function source-positive subframe

These are the three positives already exported at 10:00, not newly discovered
labels. View-level diffs are private local evidence; they do not establish full
patched-source/dependency/deployment semantics.

- `PegStability::swapVAIForStable`: the retained patched view moves the
  `vaiMinted` decrement before the fee transfer. Both views retain `nonReentrant`.
  This is a localized order change consistent with inspecting the retained
  Check-Effect-Interaction claim, not proof of exploitability or a category-wide
  negative. Oracle, helper, token and caller behavior remain unverified.
- `PegStability::swapStableForVAI`: both views retain `nonReentrant`, and both
  perform the incoming `safeTransferFrom` before updating `vaiMinted`. Other
  changes include helper/token references, oracle update, return value and
  error handling. Text change does not establish that the specific source
  rationale is resolved. Preserve that uncertainty.
- `LiquidityGaugePoolController::_setPool`: the patched view adds input checks
  and replaces conditional field assignments with `_poolInfo = args`. This
  changes initialization behavior but does not prove resolution of excessive
  authority, caller privileges or documentation concerns. Authorization is
  still the separate source authority-policy meaning, not missing-guard truth.

Finding IDs: `venusProtocol00/vulnerable/findings.json:1` and
`neptune00/vulnerable/findings.json:0,:3`. The two Neptune findings remain one
observed function/category cell. Native source claims, partial masks and unknown
patched categories are unchanged. AI inspection is not independent adjudication.

## Artifacts and reproduction

Private five-file derivative:
`scafe_function_dataset_pilot/derivatives/coinfabrik_pair_evidence_v1_20261105T1100UTC/`.
It contains pair hashes/metadata, unpaired metadata, summary, data card and manifest;
no copied code or audit-description model payload. Parent originals, source
supervision and independent review/eligibility remain unchanged. Source payloads
and private diffs are not committed or uploaded.

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/measure_coinfabrik_pairs.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --view-dir /absolute/path/annotation_view_v2_20261103T1100 \
  --context-release-dir /absolute/path/content_context_v1_20261103T1200 \
  --source-layer-dir /absolute/path/coinfabrik_source_supervision_v1_20261105T1000UTC \
  --output-dir /absolute/path/NEW_pair_evidence
```

Use only a new versioned derivative or separate workspace output. The manifest
binds generator/helpers, original delivery, parent manifests and output hashes.
Reproduction and current tests/build/PDF/Git status are recorded in the run report.
Neither this command nor its output assigns labels or permits training.

## Next critical dependency

For the changed source-positive cases, establish finding-specific contextual
repair evidence or qualified adjudication before proposing matched negatives.
The retained views alone do not satisfy that gate. In parallel, source-supervised
training protocol still needs measured label quality, ambiguity handling and
license/provenance eligibility; evaluation needs separate independent review and
frozen project/clone-aware cohorts. Reuse the existing adapter/masked loss and
these measured exclusions rather than add another queue or rescan unchanged
files. No blanket double manual review of every training row is required.
Matched baselines and bounded CodeBERT remain gated on those outcomes.
