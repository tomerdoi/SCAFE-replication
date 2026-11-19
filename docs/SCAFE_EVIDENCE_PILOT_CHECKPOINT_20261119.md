# SCAFE-Evidence paired pilot checkpoint — 19 November 2026

## Decision

The retrieval hypothesis passed its prespecified feasibility gate after a
transparent corrective replacement, but the evidence is a **pilot**, not a
submission-ready Q1 evaluation.  The legacy risk-score manuscript must not be
rewritten as if seven project groups establish population-level reliability.

The candidate contribution remains:

- answer one compiler-checkable defensive fact from partial Solidity source;
- retrieve exact source rather than synthesize missing code;
- return `unknown` instead of guessing when evidence is missing or ambiguous;
- emit a replayable source-hash and declaration-span certificate; and
- measure evidence bytes against ordinary repository traversal.

It is not a vulnerability detector and modifier resolution is not proof of
correct authorization.

## Frozen question and novelty boundary

The first fact is the declaration referenced by a named modifier invocation.
Repoformer and DraCo already establish selective repository retrieval, TRACE
already analyzes repository-level Solidity access control, and compiler/Slither
name binding is established.  The novelty claim is therefore limited to
fail-closed evidence sufficiency and replayable exact-source certificates for
defensive Solidity analysis.

The prespecified feasibility protocol is
`docs/SCAFE_EVIDENCE_PROTOCOL_20261119.md`.  It requires zero observed wrong
answers, at least 90% coverage overall and 80% in each project, at least 20%
median byte saving versus strong traversal, a positive project-bootstrap
interval, at least 20% cross-source queries, complete certificate replay, and a
gain that remains after common-library targets are excluded.

## Corpus and admission

Five development projects and two qualified confirmation projects are retained:

- development: Coinbase Smart Wallet, PRB Proxy, DN404, ERC-6900 Reference
  Implementation, and Polymarket CTF Exchange;
- confirmation: Multibridge and WebAuthn Owner Plugin.

All are pinned non-fork repositories.  Root licenses are MIT except Multibridge
(GPL-3.0) and WebAuthn Owner Plugin (GPL-3.0 root with AGPL-3.0 first-party
files); original notices remain intact.  First-party SPDX variants include MIT,
CC0-1.0, AGPL-3.0-only, GPL-3.0-only, and MIT OR Apache-2.0.

The exclusion-only screen rehashed all 21,457 reserved DAppSCAN source files
(107,970,249 bytes) and 145,734 named function occurrences without loading
labels or findings.  Exact-file and exact/token/template function matches are
excluded before method evaluation.  A non-match is not proof of family
independence.  Eight development queries were excluded under this rule.

Worldcoin is retained as diagnostic evidence but is not counted as qualified
confirmation: `src/VestingWallet.sol` carries an `UNLICENSED` notice.  Its first
confirmation run also exposed a real fail-closed limitation: two `onlyOwner`
queries returned `unknown` because duplicate import paths to the same OpenZeppelin
declaration were treated as distinct.  The result was not overwritten.

The general duplicate-path defect was fixed by deduplicating identical
declarations by source path, declaration name, and source offset.  Genuinely
distinct declarations still produce `unknown`.  A corrective freeze was made
before selecting or evaluating the rights-cleared WebAuthn replacement.
Multibridge was not rerun.

## Development result

Input: five project groups, 51 eligible queries after eight pre-method overlap
exclusions; 18/51 (35.3%) are cross-source.

- file-only lookup: 33 correct, 0 wrong, 18 unknown; 64.7% coverage;
- SCAFE-Evidence: 51 correct, 0 wrong, 0 unknown; 100% coverage;
- strong traversal: 51 correct, 0 wrong, 0 unknown; 100% coverage;
- median evidence: 15,566 bytes versus 142,460 bytes;
- median saving: 89.1%;
- mean project-saving bootstrap interval: 28.9% to 84.0%;
- every certificate replayed;
- the saving remains when common-library targets are removed.

DN404 has zero per-project saving, and must remain visible rather than be hidden
by the aggregate.

Exact result:
`data/derived/scafe_evidence_development_pilot_v4_20261119.json`.

## Qualified confirmation result

The qualified confirmation combines the retained Multibridge cohort with the
once-evaluated corrective WebAuthn replacement.  It does not include Worldcoin.

Input: two project groups, 32 eligible queries; 17/32 (53.1%) are cross-source.

- file-only lookup: 15 correct, 0 wrong, 17 unknown; 46.9% coverage;
- SCAFE-Evidence: 32 correct, 0 wrong, 0 unknown; 100% coverage;
- strong traversal: 32 correct, 0 wrong, 0 unknown; 100% coverage;
- median evidence: 12,567 bytes versus 41,876 bytes;
- median saving: 70.0%;
- project median savings: 58.3% and 92.1%;
- all 32 certificates replayed;
- on 19 non-common-library queries, median saving remains 70.8%.

Exact summary:
`data/derived/scafe_evidence_qualified_confirmation_20261119.json`.

Across qualified development and confirmation there are seven project groups,
83 eligible queries, 35 cross-source queries (42.2%), zero observed wrong
answers, and complete certificate replay.  These are descriptive pilot counts,
not proof of zero error or broad generalization.

## Correctness repairs retained in lineage

The development process exposed and corrected:

1. compiler `src` spans are UTF-8 byte offsets, while the first parser used
   Python character offsets;
2. Solidity symbols can be reached through import re-export files;
3. unique declaration resolution does not require committing to a C3 order,
   while multiple distinct inherited declarations must remain unknown; and
4. identical declarations reached through multiple import paths are not
   genuine ambiguity.

Earlier invalid or failed outputs remain in ignored local derivatives.  The
qualified summaries bind the accepted inputs by SHA-256.

## Reviewer-facing limits

This pilot does not yet support a full Q1 empirical claim:

- only seven qualified project groups and two confirmation groups;
- correlated queries are not independent samples;
- the selected method returned no `unknown` on the qualified natural queries,
  so fail-closed behavior is covered by software controls and the retained
  Worldcoin diagnostic, not a broad real-project abstention cohort;
- import-closure and full-source cost reporting still need explicit separate
  rows even where their implementation coincides with strong traversal;
- only modifier-declaration resolution is evaluated;
- project-family independence is screened conservatively, not proven; and
- the two-project confirmation interval is descriptive, with endpoints driven
  by the two observed projects;
- pinned source/compiler inputs, exact JSON freezes, and result artifacts remain
  local and ignored; the current Git diff alone is not a self-contained public
  reproduction package.

## Shortest credible route to a full paper

Do not polish the legacy manuscript.  First extend the exact same CPU-only
evaluation to roughly 15–20 independent, rights-qualified projects, reserve at
least six untouched project groups, and report project-level uncertainty.
Add a predeclared partial-source stress cohort that removes required imports,
introduces genuine ambiguity, and tampers with certificate evidence so that
`unknown` and replay rejection are measured rather than merely asserted.
Report file-only, import-closure, strong traversal, selected retrieval, and
full-source upper-bound costs on identical queries.  A second structural fact
should be attempted only if it reuses the same evidence graph and remains
compiler-checkable.

All current experiments complete in seconds once sources and four verified
compiler binaries are present.  A 15–20-project CPU evaluation remains a
same-day Mac experiment; source qualification and manuscript rewriting, not
compute, are the critical path.

Only after that gate passes should the manuscript be replaced end to end.  A
feasibility-paper rewrite now would be honest but materially weaker than the
Q1 goal.

## Validation checkpoint

After the final resolver correction and checkpoint update:

- 31 focused modifier-scope and resolver tests passed;
- `validate_experiment_integrity.py` passed;
- `validate_submission_artifacts.py` passed;
- `git diff --check` passed;
- IDE lint reported no errors in the changed Python files; and
- an isolated `latexmk` build of the unchanged manuscript completed with
  BibTeX and produced a 40-page PDF.  Existing underfull-box warnings remain;
  no manuscript input or checked-in PDF was changed.

These checks establish software and artifact consistency, not scientific
generalization or submission readiness.
