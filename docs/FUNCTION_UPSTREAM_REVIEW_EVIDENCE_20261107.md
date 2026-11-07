# SCRUBD upstream review evidence and evaluation decision

## Question and primary evidence — 7 November 2026, 10:00 UTC

Can existing upstream review satisfy part of the evaluation evidence requirement,
without independently relabeling every source-supervised training function?

The [authors' v1 paper, III-C](https://arxiv.org/html/2412.09935v1#S3.SS3)
reports 63 student annotators, at most three per function, expert-author
validation and individual meetings to resolve disagreements. Some RE annotations
were made directly by the expert. Selection favored tool-flagged functions.
The [current official README](https://github.com/sujeetc/SCRUBD#version-history)
records later label corrections. Its `is_student` column denotes student
involvement; it does not certify per-record independent adjudication.

These are upstream protocol claims, not a new semantic review performed here.
The paper's original RE counts differ from the retained later-label counts.
The original pin remains unchanged; exact tag ancestry and the review coverage
of subsequent corrections are not established by this session. No restricted
pinned-tree/tag route was retried and no corpus or new revision was downloaded.

## Decision and scope

Recognize **documented upstream expert review at source-protocol level**. The
pilot's zero independent-review fields mean no qualifying record-level evaluation
review has been ingested into our declared protocol; they must not be described
as proof that upstream functions were never human reviewed. This distinction
avoids imposing unnecessary fresh manual annotation on every training record.

No source label, mask, target definition, original file, saved eligibility or split
is changed. This evidence is stronger than a bare source label but does not
establish that every retained revised row meets the agreed evaluation protocol.
Nor does it resolve POOHMO.endRound or MonkeyScam.determinePID, prove label accuracy,
turn vulnerability labels into the three suspicious-pattern categories, or make
already-inspected examples fresh confirmation data. Source annotation selection
also limits population-generalization claims. Report benchmark-conditional results
if this source later becomes an accepted exploratory evaluation set.

## Smallest next deliverable and acceptance checks

Before commissioning fresh review, assess whether available upstream evidence can
qualify a bounded source-defined cohort under an explicitly accepted target:

1. Bind each proposed row to retained source revision, function/span, native label,
   context and reviewer/adjudication evidence (or a clearly delimited protocol
   coverage claim). Resolve changed-label coverage rather than infer it from
   `is_student`. Missing evidence remains unknown; do not invent reviewer logs.
2. Accept or reject that evidence against the declared qualified evaluation
   protocol and target, with targeted adjudication for unresolved cases. Upstream
   review can be reused when it actually meets that protocol; no blanket new
   double review of all training rows is required.
3. Apply existing project/clone, overlap, licensing and frozen-split checks.
   Prior inspection cannot be undone by a new metadata tier. If an exploratory
   source benchmark is supportable, distinguish it from fresh confirmation.

Conditional effort: one bounded ingestion/decision session after the specific
coverage evidence and target agreement exist. Evidence availability and qualified
adjudication turnaround are unknown. Continue independent useful construction;
no model training or implied acceptance merely because this note exists.

## Reproducibility and result

The run pins hashes of the delivered manifest, source manifest, candidate file and
source-adapter manifest/targets in its private workspace. The source SHA retained
in the manifest is `9dba1928d4aea9a5d27014e5f31a8ed0b703f711`. Public evidence was
read on 7 November 2026; live README content is not a pinned historical review log.
No private dataset content or raw annotation comments are included in this note.

This is an evidence/eligibility decision, not increased label coverage, completed
adjudication, a trained classifier or an accuracy gain. Reuse it; do not repeat
this protocol lookup unless relevant evidence changes.
