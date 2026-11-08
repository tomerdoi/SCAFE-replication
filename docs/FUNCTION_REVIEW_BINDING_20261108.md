# Retained SCRUBD review-provenance binding

## Deliverable — 8 November 2026, 08:00 UTC

The versioned exporter `scripts/bind_function_review_provenance.py` now provides
lossless delivered-record links for all 110 standalone SCRUBD candidates. Each
has a unique matching record in the 746-row native SCRUBD staging subset, with
identical complete retained content and source revision. Exact JSONL file hashes,
1-based line numbers and line hashes (excluding line terminators) bind the links
to the delivery. Native metadata, function/model hashes, source identity, claimed
context hash and claimed upstream annotation-file hash are retained separately.

The native role flags are 71 student-involvement and 39 non-student-involvement
values. They do not identify reviewers or certify expert adjudication for a row.
Native RE support remains 52 positive/58 negative; UX is 39 positive/32 negative/
39 unknown. Unknown UX cells have false observation masks; no negative is created.
These are native source meanings, not the unified three-category pattern labels.

## Evidence boundary and decision

The [upstream protocol evidence](FUNCTION_UPSTREAM_REVIEW_EVIDENCE_20261107.md)
is referenced by each link as source-protocol evidence. All links explicitly keep
retained-revision review coverage unverified and fresh evaluation not established.
The revision claim is `9dba1928d4aea9a5d27014e5f31a8ed0b703f711`. The delivery's
claimed annotation-file SHA-256 is
`b642ff45b580466f1c45ce48d42076c90f929498bde6fc07ba2a30a0f3d3e9d7`;
the exact original CSV bytes were not available to this run, so that hash was not
independently reverified. This is not a new source-version ancestry result.

Review-provenance link coverage increases from no dedicated bound release to
110/110 retained candidates. The underlying metadata already existed; no new
functions, source labels, qualified reviews or evaluation eligibility were added.
Certified record-level adjudication coverage at the retained revision remains zero
in our accepted artifacts. This zero is a missing evidence binding, not proof that
upstream review did not happen. The existing 33 mapped RE proxies and all native
claims, original files, masks, split/eligibility fields and overlaps are unchanged.
No source role flag, AI observation or protocol citation is promoted to gold.

## Reproduce and verify

```sh
python3 scripts/bind_function_review_provenance.py --pilot PILOT --output NEW_VERSIONED_OUTPUT
```

The private pilot release is
`derivatives/review_provenance_links_v1_20261108T0800UTC/` with three artifacts:
`review_provenance_links.jsonl`, `report.json`, `manifest.json`. The manifest pins
all four original input files and implementation bytes, and hashes the output
links/report. Two independent invocations produced byte-identical artifacts.
A separate anchor check verified each emitted line/hash against the delivered
JSONL and every native field against the original candidate. Unit tests reject
missing/duplicate identities, altered metadata, contradictory native labels,
wrong revisions, synthetic-source inheritance and silent eligibility overwrites.
No model code or source rationale is copied into the derivative, and no pilot
artifact is staged in Git. Existing source rationale comments are only hashed.

## Smallest remaining dependency

Record linkage is complete for the retained candidates. Do not repeat it unless
inputs change. To qualify upstream review for an agreed evaluation target, obtain
or verify coverage evidence for the retained revised labels (including correction
coverage), then apply targeted adjudication and existing grouping, overlap and
licensing gates. A role bit or matching aggregate count cannot supply that evidence.
Existing inspection cannot be made fresh confirmation by this exporter. If source
coverage evidence stays unavailable, qualified targeted adjudication of an explicitly
chosen cohort remains the direct alternative; no blanket fresh double review of
all source-supervised training functions is required.

After that concrete input exists, one bounded ingestion/decision session is the
conditional next milestone. Review/evidence turnaround is unknown. No training,
baselines, model download, performance gain or readiness claim accompanies this
release. Function-to-contract integration remains downstream of credible evaluation.
