# POOHMO source-quality concern: compiler-resolved caller guards

## Concrete result — 8 November 2026, 09:00 UTC

The earlier source-visible review flagged an omitted low-level value call in the
RE0 rationale for POOHMO.endRound but explicitly lacked exhaustive local caller
coverage. This bounded follow-up resolves that local source question using the
already saved compiler AST; it does not repeat the eight-case audit, train a model,
recompile Solidity, fetch code or adjudicate the source label.

All three resolved callers of the private endRound function are buyCore,
reLoadCore and withdraw. Reverse graph closure and independent forward path
traversal agree on eight externally invocable entry paths:

| Entry | Local path to endRound | Explicit entry modifier |
|---|---|---|
| fallback, buyXid, buyXaddr, buyXname | buyCore → endRound | isHuman on all four |
| reLoadXid, reLoadXaddr, reLoadXname | reLoadCore → endRound | isHuman on all three |
| withdraw | endRound | isHuman |

The modifier binds its address alias to msg.sender and requires equality with
tx.origin before the body placeholder; its code-size restriction is also visible.
All three direct call sites have an immediately preceding same-block assignment
of the current round's ended flag to true. Their line ranges are 492–493,
831–832 and 887–888 in the original context. The guard is at lines 197–205.
All three AST Identifier references to endRound are accounted for by those calls.
The inherited events-only contract adds no function or modifier edges. Internal
library call declarations were included, not silently dropped as unresolved.

## Scientific interpretation and boundary

This supplies compiler-resolved local guard coverage beyond the earlier lexical
observations. The omitted flushDivs call still exists before later state writes,
but that ordering alone is insufficient evidence of a wrong RE0 label. Any claimed
callback mechanism must address these caller restrictions and round markers.
The source comment's omission and the source label's correctness remain separate
questions. This is not a proof that RE0 is correct, a repaired negative, a global
safety certificate or independent evaluation truth.

The graph is limited to resolved calls/modifiers in the retained source unit.
It is not a runtime feasibility proof and does not model external callback edges,
callee implementations, deployment/compiler equivalence, cross-function state or
read-only effects. Guard presence alone must not become an automatic RE label.
The tx.origin-related condition is evidence to analyze, not security guidance to
introduce origin-based authorization into new software. Native labels, code-only
views, observation masks, overlap groups, eligibility and splits remain unchanged.
Qualified adjudication remains pending; no row was promoted or label flipped.

## Reproducibility and checks

Private release:
`derivatives/poohmo_caller_guard_evidence_v1_20261108T0900UTC/`.
It contains call edges, the case evidence report, analysis script and manifest.
Source locations are UTF-8 byte spans with raw excerpt hashes and original line
numbers. The manifest pins delivery/candidate/context and saved compiler artifacts
plus helper implementations. The exact original context, not a masked alternative,
matches the saved compiler input. AST endRound span hashes match the delivered
candidate. The earlier compiler artifact manifest is fully verified before reuse.

The analysis includes source-unit library declarations, checks all target
references, validates exact guard/marker AST forms, compares forward and reverse
reachability, and replays its two evidence files byte-for-byte. Initial analysis
assumptions about masked input and an owner-only graph failed and are preserved
in the workspace; the corrected final analysis requires original input identity
and includes resolved library declarations. No failed attempt is a label result.

## Next decisive dependency

Ask qualified adjudication to address one precise mechanism: can the actual
constructor-supplied callee, under the deployment's configuration, exploit a
reachable cross-function/read-only effect despite the documented guards and
pre-call round marker? If callee/deployment evidence is unavailable, retain
context-required truth. Do not rescan the old corpus or regenerate this graph
as new progress. This resolves a bounded local-context question; it does not
remove the separate evaluation, source-version, license, grouping or training gates.
