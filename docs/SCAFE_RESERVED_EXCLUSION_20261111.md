# Reserved DAppSCAN function exclusion — 11 November 2026

## Outcome and protocol

Continued from the unified 2,323-function release rather than rebuilding it. The
latest user reserves DAppSCAN from function-model training and training-label
construction. Old corpus-overlap seeds pointed to Wild, DIVE and FORGE, not exact
DAppSCAN membership; do not reinterpret their absence as clearance.

Predeclared all candidate functions and all local DAppSCAN Solidity source files,
30,000-file / 350-MB / 2-MB per-file / 240-second bounds, fixed exact-byte,
comment/whitespace-token and identifier-template comparisons, plus conservative
existing component/source-project propagation. No new source requests, reserved
labels/outcomes, contract execution, compiler/model call, generated examples or
source-rights change. Named implemented direct members only: lexical extraction
omits special/abstract forms and does not establish compiler coverage or semantics.

Hashed all 21,457 local Solidity files / 107,970,249 bytes and examined 145,707
named implemented function occurrences in the initial 25.74-second scan. One file
failed UTF-8 decoding; the targeted predeclared strict cp1252 recovery, without
replacement/discarded bytes or rewriting the source, recovered another 27 functions
and no additional candidate matches. The initial failure is retained, remaining
unparsed files zero; supported syntax limits still apply. Final examined function
occurrences: 145,734, not unique functions or independent projects.

| Candidate status | Count |
| --- | ---: |
| Direct reserved content match: exact bytes or normalized tokens | 429 |
| Template-only heuristic quarantine | 156 |
| Additional conservative related-group quarantine | 477 |
| No detected match; unresolved, not admitted | 1,261 |

35 candidates have an exact function-byte match; 429 have a normalized-token match.
585 have some template match, including direct matches: these columns overlap and
must not be added as distinct functions. Common code can cause cross-project
matches. Direct content exclusion is not proof of shared project ancestry; template
and source-project/component propagation are explicitly heuristic and can overmerge
boilerplate. The original matches and reasons remain separate from group propagation.

49 of the 53 existing source assertion cells are on unresolved/no-match records;
four are on conservative related-group records. No source assertion was relabeled
or silently discarded. Zero new semantic/AI labels, zero training/AI-assessment
admissions, zero independently reviewed gold or frozen splits, and zero model runs.
The 1,261 nonmatches are not known safe, label-negative, rights-cleared, or proven
independent from DAppSCAN. There is no classifier performance or Q1-readiness gain.

## Delivered code and private evidence

Added scripts/function_reserved_overlap.py and eight software-only regression tests.
The helper exposes offset-preserving named-member fragments, frozen fingerprint
comparisons and fail-closed exclusion status; it never admits training on no match.
Tests cover special/abstract exclusion, comments/literals, overloads and owners,
malformed source, token/template sensitivity and exclusion-versus-clearance rules.

Private versioned pilot derivative:
reserved_dappscan_exclusions_v1_20261111T0900UTC. It contains candidate exclusions,
source-file hash inventory, candidate fingerprints, initial failure/targeted encoding
recovery, summary, lineage, README and manifest. Original source bytes, prior unified
release and delivered pilot are unchanged. Public Git candidates include generic
code/software fixtures and this method note only, not source inventories/payloads.

## Next decision and preserved holds

First narrow source-use and project/version/near-clone evidence for a small real
candidate cohort among the 1,261 unresolved functions. CoinFabrik has 96 nonmatching
candidates in DeXe00, venusProtocol00 and neptune00; names alone are not clearance.
In particular, DAppSCAN has a Venus audit directory: investigate genuine repository/
revision links before any family decision, not a substring-based safe or unsafe
assertion. SCRUBD has 105 nonmatches and HF 1,060, but original label quality/rights,
real origin and context remain required. Separate source claims from qualified AI
assessments and independent gold. When a real cohort meets its declared source-use
and overlap gates, assess missing cells with evidence and explicit label origin;
do not duplicate the completed exclusion scan without changed inputs or a specific
regression. No blanket fresh independent review of every training function is needed.

The offline Sherlock convenience screen admitted no functions; sponsor labels,
revision/source-use and older Notional family links are not resolved. Do not retry
the restricted revision route or use this small screen to justify generation. Real
functions remain first; any generated training-only fallback still needs a concrete
unfilled gap and the latest explicit conditions. All validation/testing stays real.

The earlier four staged consolidation files remain byte-for-byte intact. The prior
local-commit denial is preserved: no retry, indirect/unrelated commit or push around
it. The GitHub 403 and other source/access holds remain. Current work is local and
uncommitted; new files are not staged. Applicable tests, both integrity validators,
fresh full direct TeX/BibTeX build, visual checks, safe verified PDF refresh, diff/
privacy review and own-lock release are recorded in the task checkpoint. The paper
claims and automation/schedule are unchanged. A validated exclusion overlay is not a
trained function model, completed AI labeling or publication-ready evidence package.
