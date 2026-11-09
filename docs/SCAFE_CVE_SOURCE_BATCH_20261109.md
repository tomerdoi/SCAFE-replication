# Bounded CVE source qualification and split constraints — 9 November 2026

## Outcome

The separate CVE derivative now has four explicit-MIT source files and five
source/scope/line-bound publisher locations, up from one file and one location.
This is a concrete source-dataset advance, not independent vulnerability truth,
training eligibility or a model result. The original 2,323-candidate pilot and
all canonical labels remain unchanged. No example is promoted to a safe negative.

| Qualification measure | Previous intake | Current bounded release |
|---|---:|---:|
| Explicit-notice source files with exact locations | 1 | 4 |
| Bound source-claim locations | 1 | 5 |
| Independently qualified gold / training-eligible records | 0 / 0 | 0 / 0 |

The five records cover four CVE IDs; one claim has two located functions. File,
case, function, project and independent group counts must not be conflated.

## Predeclared acquisition and actual outcome

At pin `1128b7aae666df541a5ffcf56782df17113c3d36`, descending development case
IDs selected at most 12 unique artifacts, each at most 128 KiB and at most 1 MiB
total. Expected bytes were 370,810. One previously verified artifact was reused;
11 new artifacts transferred 308,311 bytes. All received sizes, Git blobs and
SHA256 receipts matched. No reserve values, report bodies or new model were used.

One request initially timed out before transferring bytes. Its absent completed
and partial outputs were checked; other selected requests continued. One later
bounded retry succeeded without increasing the original request/data budgets;
the original failure remains in the receipt history. No failed outcome is hidden.
All unknown-use sources initially entered quarantine; no downloaded code executed.

## File qualification and structured source resolution

Four target source files have matching MIT SPDX and explorer MIT declarations.
One artifact carries a JSON bundle rather than plain source. The builder resolves
its exact publisher source_path (`Dependencies.sol`) and inspects that file's own
notice and scope; it does not regex-classify escaped JSON as Solidity source.
All four selected files have no import directives. This is not a dependency,
compiler, whole-project rights or independent deployment verification claim.

Eight artifacts do not meet this release's explicit source-notice tier. One has
explorer MIT metadata without a matching direct file notice; seven have absent or
nonaffirmative evidence. This is our conservative cohort rule, not a legal finding
that missing SPDX means prohibited use or that other evidence cannot qualify it.
Their original bytes/evidence remain quarantined and unmodified.

The publisher's original location annotations have separate CC BY 4.0 terms;
embedded snippets/narratives and third-party material do not inherit that grant.
Attribution and pinned notices from the parent intake are retained. Public source:
https://github.com/smartbugs/CVE-Smart-Contracts

## Actual scope and leakage evidence

Existing comment-aware token/scope helpers were reused and hash-pinned. The
legacy brace-only extractor was not used as an authoritative census. The qualified
files yield 176 declared named implemented lexical function spans, all distinct
by raw hash, 173 token hashes and 166 identifier-family hashes. This census omits
special/abstract forms and does not prove inherited callable-function coverage or
a complete fair review universe. No Solidity compiler is installed; no AST,
compilation, runtime, exploit or parameter-type semantic verification was claimed.

Five publisher locations match a unique declared contract, function name and exact
start/end line/body span. Signatures remain source claims; type semantics remain
unverified. Full contexts/notices, code-only fragments, source/artifact/function
hashes, provenance, partial masks and false eligibility are retained. No native
claim becomes an eleven-output or three-category label merely from a location.

Three shared token fingerprints (`name`, `symbol`, `_msgData`) link two otherwise
different source files. Both token and identifier-family split constraints yield
three connected components across four sources. These are conservative leakage
constraints, not three verified independent projects. Whole-source family hashes
alone would miss this function reuse. Identifier-family constraints can overmerge
and are retained as sensitivity evidence rather than a semantic-clone oracle.

Four census functions match stored exact raw hashes in the initial pilot; none of
the five located targets do. That does not establish untouched evaluation data:
formatting, templates, near-clones, project relations and historical train/test
membership remain unresolved, and these files have now been inspected. No random
per-function split may ignore the known shared code. No split was assigned.

## Decision and next deliverable

Do not fit a model on this mini-cohort: five target locations and only three known
constraint components are inadequate for the intended multi-project comparison.
The next outcome must close genuinely broader admitted source/group coverage and
a fair candidate universe, or explicitly pivot/narrow the research design. Reuse
this acquisition, scope check and clone evidence; do not repeat it as a new audit.
Original author taxonomy labels may be reused only through justified native
mappings with provenance/partial masks and targeted review, not mandatory double
manual relabeling of every training function. Independent qualified evaluation,
matched simple/nuisance baselines, uncertainty and real-only rules remain gates.
The conditional manuscript target is still blocked by data qualification.

Private versioned release:
`derivatives/cve_source_batch_v1_20261109T1200UTC/` in the authorized pilot.
It preserves the bounded inputs, quarantined originals, qualification output,
source-group constraints, hash-overlap result and replay/helper lineage. Only
this aggregate method note and ledger are staged. No corpus, source payload,
ignored PDF or data from the other paper is published. Required tests, fresh
manuscript build/visual/PDF refresh, local commit and lock outcomes are recorded
in the checkpoint; the existing remote push and Code4rena holds remain unchanged.
