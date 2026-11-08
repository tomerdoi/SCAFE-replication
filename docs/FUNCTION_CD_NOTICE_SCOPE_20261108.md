# Retained CD notice-scope evidence

## Decision — 8 November 2026, 12:00 UTC

A bounded source inspection resolves a specific possible shortcut in the fixed
33-function real-CD cohort: notices in bundled code do not, by themselves, establish
coverage for all selected functions in the same flattened file. The task's
allowed-use evidence gate remains unresolved for all 33 rows. This is an evidence
status, not a legal opinion that training is prohibited or allowed.

Across 18 exact retained contexts, 16 have no hits under the recorded marker
search and two have scope-limited notices. Three selected functions reside in
those two notice-bearing contexts; none contains a detected notice inside its
selected function span. Absence of a marker does not prove absence of a license,
permission or another lawful basis. No new use permission is inferred either way.

| Retained context / selected function | Observed scope evidence | What remains unestablished |
|---|---|---|
| a270aa70… / GAME.startRaffle, lines 1493–1532 | MIT notice at lines 25–41 within the explicitly named solidity-cborutils component, lines 23–245; later helper-specific MIT notices at 1091/1114/1143 | Coverage for the selected GAME function or the whole flattened contract |
| a433beab… / KittyBreeding.giveBirth, 1020–1063; KittyCore.withdrawBalance, 2000–2008 | Notices at 678/700 explicitly refer to individual methods and Apache licensing | Coverage for these two later selected functions or the whole flattened contract |
| Other 16 contexts / 30 selected functions | No marker hits under the declared bounded search | Source-specific allowed-use evidence remains missing from this inspection |

The delivered SCRUBD source manifest contains zero retained explicit LICENSE/
LICENCE/COPYING entries. This is a statement about retained artifacts, not an
exhaustive search or a claim about the current upstream repository. Public
availability, source-documented real-world origin and the paper's license are not
accepted here as evidence resolving code plus dataset-annotation use coverage.
No upstream terms, new corpora or missing acquisition caches were downloaded.

## Method and reproducibility

Private release:
`derivatives/CD_notice_scope_evidence_v1_20261108T1200UTC/`.
Four artifacts contain 18 context evidence rows, 33 function links, report and
manifest. Raw source remains in its immutable parent. Marker hits have exact UTF-8
byte spans, original line numbers and raw hashes; selected function hashes are
rechecked against the parent cohort. The report records the full search expression.
It includes SPDX, license/licence, copyright, redistribution, permission-grant,
GNU, Apache, Mozilla, public-domain, Creative Commons and CC0 wording. This is a
bounded lexical search plus explicit notice-scope inspection, not a complete
legal or semantic license detector. An initial narrower wording missed the two
Apache-method notices; the broader final result is preserved rather than hidden.

Two invocations reproduce all four artifacts byte-for-byte. A separate check
revalidates every marker's raw bytes/line and all 33 selected function spans.
Nothing is promoted: allowed-use status stays unresolved, source labels and
origin decisions are unchanged, and training/evaluation eligibility and splits
remain false/empty. The 33 source-documented real-origin decisions are preserved;
this independent rights-evidence layer does not revoke or strengthen that tier.
No model training/scoring, new examples or code changes are introduced.

## Exact next dependency

Find source-specific terms or other appropriate documentation covering the intended
use of the selected contract code and dataset annotations, with clear component/
function scope and provenance. Alternatively, progress with another already
authorized real public corpus whose applicable terms are documented. Do not
propagate a library/method notice to unrelated functions, invent a license, treat
missing notices as a prohibition, or silently clear the task's gate. No external
outreach, new download permission or legal determination is authorized by this
note. Reuse this scoped result; repeating the same marker scan is not progress.
The independent evaluation, target-quality, grouping and CodeBERT gates remain.
