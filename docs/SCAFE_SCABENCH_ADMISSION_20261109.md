# ScaBench source-location admission screen — 9 November 2026

## Outcome and decision

A new bounded real-public-data candidate was acquired for metadata inspection,
not training or evaluation admission. ScaBench (`scabench-org/scabench`) is distinct
from the error-injected SC-Bench corpus. The publisher describes real audit-contest
projects; this is a source claim, not independently established finding truth.
The pinned publisher repository declares MIT terms. Underlying source files,
imports and original reports require separate applicable-terms review.

The 1,982,255-byte curated JSON has SHA256
`542b2fd605845a9939a0a89767877544fc1870a8e9a64dfb236fa20e203681a6`
and verified Git blob `eaf0da124bcc4fc66021ff1cc12a704c9cdab3f6` at publisher
commit `eec0020939a47bbc06a98a7348a90908b96f4b4d`. No code archive,
contract source, model, scorer, baseline outputs or installer was acquired or run.
No original delivery or old label was changed.

## Prospective inspection boundary

Before acquisition/description inspection, a saved rule sorted project IDs by
SHA256 of a fixed salt plus ID, reserving floor(0.4*N). The 31 projects produce
19 development and 12 provisional reserve projects. Reserve finding content,
counts, titles, locations and descriptions were not inspected or summarized.
The allocation is not a frozen qualified evaluation split. Project/provenance
and source-clone overlap remain unverified; old-source overlap must be quarantined.
Reading or hashing the raw corpus does not establish untouched independent truth.

## Measured development feasibility

The 19 development projects contain 313 finding records. Only 12 of their 19
codebase references are full immutable commit hashes; seven are mutable/non-full
references. Ninety-five findings contain an immutable GitHub line link, but only
37 findings across seven projects have a link matching their declared repository
and exact commit. Those yield 62 distinct finding/location links. The strict
Solidity subset is 24 findings, 41 links, four projects; remaining links point to
Rust and Move. Links are candidates, not verified function boundaries or evidence
that the referenced revision contains the claimed problem. Markdown descriptions
are used only to recover provenance/location metadata and must never become model
inputs. Unmentioned functions are not negative examples.

The previous provisional 30-function/10-project screen is not passed: these are
finding links rather than verified functions, and exact matching project coverage
is seven overall/four for Solidity. Do not pool languages or inspect reserve
findings to make the screen pass. No new reviewed labels, training eligibility,
model fit, benchmark score or qualified confirmation was produced.

## Next decisive step and stop criterion

Within the next bounded session, choose a small development project with immutable
anchors; review its source/report terms and original finding provenance, then
acquire only explicitly admitted source files to verify exact function mapping and
the complete candidate review universe. Recover mutable references only with
original contest/version evidence, never by silently substituting today's main.
If the admitted development pool cannot support adequate independent project
coverage and a fair review universe, pivot the task/data instead of adding proxy
labels or prolonging metadata audits. Qualified evaluation, clone grouping, matched
baselines, uncertainty and real-public-only restrictions remain. Existing three
outcome milestones are conditional; no few-week or Q1 acceptance promise is added.

## Evidence and reproducibility

Private derivative: `derivatives/scabench_metadata_admission_v2_20261109T0900UTC/`
under the authorized function-dataset pilot. It retains pinned metadata, retrieval
manifest, pre-acquisition reserve rule, project roles, metadata-only analysis and
its results. The extractor accepts only HTTPS GitHub blob URLs with 40-hex commit,
positive line ranges and exact declared repository/revision match. Three assertion
checks reject mutable references/reversed ranges and accept a valid span. Initial
path-component validation was corrected; replay also caught unstable set iteration
ordering. Version 2 sorts links deterministically. Version 1 is retained as the
failed byte-replay preparation, not the final derivative. Counts were unchanged. Replaying analysis must use the saved role map, not reallocate it.
The corpus and private derivatives are not staged in Git. No data is admitted to
training/evaluation by this metadata screen. Scientific value is a new bounded
source candidate and a quantified qualification decision, not accuracy improvement.

Publisher: https://github.com/scabench-org/scabench
Pinned metadata: https://github.com/scabench-org/scabench/blob/eec0020939a47bbc06a98a7348a90908b96f4b4d/datasets/curated-2025-08-18/curated-2025-08-18.json
