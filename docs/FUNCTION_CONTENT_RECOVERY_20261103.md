# Filename-independent exact context recovery — 3 November 2026, 12:00 Israel

A bounded local content-hash search recovered five pinned whole-source texts
missed by the earlier filename prefilter. These provide context for nine additional
SCRUBD candidates. Six have a unique original-fragment match; three have multiple
identical named-function fragments and remain explicitly ambiguous. This advances
source evidence and overlap control, not independent labels or model accuracy.

| Measure | Previous recovery | Content extension |
|---|---:|---:|
| Pinned context documents | 21 | 26 |
| Functions with exact whole-source context | 136 / 2,323 | 145 / 2,323 |
| Unique exact original-fragment localization | 136 | 142 |
| Ambiguous identical original-fragment localization | 0 | 3 |
| Direct existing-corpus context-overlap seeds | 136 | 145 |
| Evidence-graph conservative quarantine | 255 | 264 |
| Strict-template conservative quarantine | 307 | 316 |
| Independent reviews / training eligible / assigned splits | 0 / 0 / 0 | 0 / 0 / 0 |

Whole-context coverage increases from 5.85% to 6.24%; unique localization reaches
6.11%. The unchanged current v2 code views still have 1,324 evidence components
and 1,226 strict-template components. This release changes seed evidence, not
function code, labels or grouping algorithms. All three ambiguous mappings retain
all exact spans and cannot be treated as uniquely localized reviewed examples.

## Predeclared bounded experiment

The hypothesis and acceptance/budget gates were recorded before evaluation in
workspace `scafe_iteration_20261103_1200/run.json`. Search existing authorized local
Solidity files without assuming their basename matches the producer path. Only
exact SHA-256 of UTF-8-sig-decoded whole text can recover a producer-pinned context;
then separately check the original raw function's character length and SHA-256 at
its source-provided offset or named-function candidates. Similar filenames,
whitespace, token signatures and model outputs cannot establish context identity.

The fixed budgets were at most 2 GiB and 90,000 read files, at most 16 MiB/file and
500,000 listed paths. Sorted corpus queues are visited round-robin; within each,
relative paths are sorted. `paper_benchmarks` is separated by benchmark. The scan
listed 212,823 Solidity paths and stopped at the 90,000-file limit after reading
1,071,278,311 bytes. One invalid UTF-8 source was inventoried but could not match.
Default `rg` ignore rules and explicit dependency/cache directory exclusions apply.
This is a reproducible bounded search, not an exhaustive disk/corpus search.

The five new text matches occur in DIVE (two) and local paper-benchmark corpora
(three). Thus these contexts must carry direct existing-corpus quarantine. Their
presence does not prove precise historical training/test membership, but they
cannot be described as a new independent holdout. The nine newly direct seeds
expand both conservative quarantine scopes by nine; propagated membership in the
other records is still a constraint, not additional independently confirmed overlap.

This addresses the context and cohort-separation gaps in the reference-to-gap
review (`docs/SCAFE_REFERENCE_REVIEW_20261101.md`, P1 rows; context/dependency
mechanisms discussion with GoLiSA pp.5–6 and GPT4Detection pp.12,18). Exact files
are evidence to inspect, not proof of dependency closure, exploitable behavior or
an intended access policy. No claim of new architecture or Q1 readiness is made.

## Immutable lineage and deliverable

New local derivative:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/content_context_v1_20261103T1200/`

It includes all 26 context texts, combined candidate links and source evidence,
a partitioned checked-file inventory, current-view group memberships/constraints,
a before/after summary and a recovery manifest. The manifest binds the original
18-file delivery, prior recovery, current annotation-view-v2 release, generator
and helper hashes. Every inventoried source is rehashed before release; existing
parent artifacts are reverified. Original/earlier releases are preserved, including
immutable model views and reviewer packets. No raw source or dataset payload is
staged in Git or uploaded. The derivative is a context/constraint release, not a
new model-view, review submission, adjudication or training export.

All unified category labels remain unknown. Unknown is never negative. Existing
human annotation state, observed masks, eligibility or split assignments require
explicit migration and are refused by this generator. No project ID, license
clearance or semantic truth is inferred from a source match. The three-category
schema is not the legacy eleven model outputs. CodeBERT remains gated by context
sufficiency, independent category review, provenance/license controls, clone/project
constraints and a frozen legitimate evaluation cohort with fair simple baselines.

## Reproduction and validation

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/extend_function_context.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --raw-root /absolute/path/sc_review/data/raw \
  --prior-recovery-dir /absolute/path/local_context_v1_20261102T1300 \
  --view-dir /absolute/path/annotation_view_v2_20261103T1100 \
  --max-bytes 2147483648 --max-files 90000 \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The output directory must not already exist. No source is executed and no network,
model inference, training, install or download is used. Reproduction requires the
pinned original delivery, unchanged local inventory and exact generator/helpers.

The first staged run caught universal-newline normalization when copying a prior
CRLF context: the stored hash check failed and no release manifest was produced.
Its partial workspace output is retained; raw/delivered inputs were unchanged.
Copying exact UTF-8 bytes fixed this, with a dedicated CRLF regression. Direct
scanner tests also exposed unresolved macOS `/var` aliases in test roots; public
entry points now resolve roots consistently, without weakening containment.

Eight new regression tests cover renamed/BOM aliases, near-match rejection,
invalid UTF-8, deterministic byte/file budgets, corpus round-robin order,
symlink/changed-input rejection, CRLF source preservation, no annotation promotion,
quarantine, byte-identical fixture replay and refusal of reviewed/tampered parents.
Whole-corpus replay, independent span/group checks, applicable tests, both integrity
validators, a fresh full paper build and visual/PDF-refresh checks are required
before commit. Exact final validation and Git evidence are recorded in the current
workspace run report. Manuscript scientific claims remain unchanged.
