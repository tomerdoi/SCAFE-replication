# Source-target cohort decision — 7 November 2026, 09:00 UTC

## Concrete decision

**Do not start training or claim fresh independent evaluation from the current
releases.** More variants of the same inspected proxies cannot satisfy the missing
independent-evaluation gate. The next critical deliverable is agreement on the
target and qualified evaluation evidence, rather than another parser-specific
label expansion. This is a pinned readiness decision, not a new loader or a new
training restriction. Existing fail-closed loaders and all objectives remain.

Only latest, nonduplicated releases are considered: SCRUBD RE v1, CoinFabrik v1
and local-result v3. Local v1/v2 are parents, not extra examples. The resulting
47 observed weak-label cells cover **36 functions and 35 distinct code inputs**,
across **four target definitions**. They are not 47 independent examples.
All observed cells retain direct existing-corpus overlap and both quarantine
flags; independent gold, training/evaluation eligibility and split fields remain
zero. No original label, code, mask, group or eligibility field was changed.

| Separate target | Positive / negative rows | Evidence groups + / - | Strict-template groups + / - | Unique code inputs |
|---|---:|---:|---:|---:|
| SCRUBD source-assessed RE susceptibility | 14 / 19 | 11 / 12 | 10 / 11 | 32 |
| CoinFabrik reported RE/order concern | 2 / 0 | 1 / 0 | 1 / 0 | 2 |
| CoinFabrik authority-policy concern | 1 / 0 | 1 / 0 | 1 / 0 | 1 |
| Local native-result discard/checked contrast | 10 / 1 | 6 / 1 | 6 / 1 | 11 |

The SCRUBD RE target has 17 evidence components (six containing both classes)
and 16 strict-template components (five containing both classes). It passes the
necessary class-count bound below, but that does not establish label quality,
a legitimate split, statistical power, eligibility or independent evaluation.
The other three targets fail that counting bound. All four lack an independently
reviewed eligible cohort in their current saved fields. No exact same-input
opposing-label conflicts were found within an individual target definition.

## Counting proof and meaning boundary

For three disjoint partitions each to contain a positive and a negative under
whole-group assignment, each class needs support in at least three distinct
groups. A single negative group can be placed in at most one partition. Thus
the local-result release cannot by itself provide that three-part design; it
lacks at least two additional distinct negative groups even for this **necessary
counting condition**. Two extra groups would not establish adequate sample size,
review quality, fresh evaluation or permission to train. No split was assigned.
Different reviewed designs may use external evaluation cohorts; this calculation
does not prescribe a new learner design or replace the existing task gates.

CoinFabrik's two RE functions share one group and have no negative support;
the authority target has one positive group and no negatives. Copying rows or
using patched/unknown examples as negatives cannot fix those deficiencies.

Two definitions share the category key `reentrancy`: source-assessed susceptibility
and source-reported order/mitigation concern. Their IDs and declared meanings are
not automatically equivalent. A justified crosswalk may be possible after review;
none is implemented here. Blind pooling would display 35 RE cells without resolving
that target difference. The local-result target also remains a narrow body-local
weak proxy, not canonical absence of all suspicious patterns or global CLEAN.

Overlap/quarantine is reported as a barrier to **fresh evaluation** from these
selected rows. This does not invent a blanket ban on using overlapping corpora
for source-supervised training under a reviewed protocol. Previously inspected
external sets cannot become fresh confirmation data merely through relabeling.
Any retrospective exploratory evaluation must be described as such and satisfy
the existing review/group/eligibility rules; this snapshot approves no exception.

## Next milestone and exact dependency

1. Agree the first evaluation target and source-quality protocol. SCRUBD RE is
   the only current target with both-class group-count support, so it is a
   candidate first gated source-supervised pilot. Its susceptibility meaning is
   not automatically the user's canonical suspicious-pattern task. This is a
   prioritization proposal, not a changed objective or permission to train.
2. Obtain qualified, independently reviewed/adjudicated evaluation labels with
   full source/context, exact function IDs/spans, reviewer/adjudication provenance,
   unknown handling and group membership. Current releases contain none eligible.
   For fresh confirmation, the agreed cohort must also satisfy the existing
   provenance/overlap restrictions. Preserve the two open RE rationale/context
   questions; the existing eight-case source-quality pilot is not completed gold.
3. Resolve licenses/provenance and source-quality eligibility, freeze legitimate
   group memberships, then run matched simple baselines before CodeBERT. Do not
   silently merge definitions or infer source/weak eligibility from this report.

The smallest missing input is an agreed target plus qualified independent review/
adjudication evidence for the evaluation cohort, not two new reviews of every
training function. A local path to complete review/context files would permit
bounded ingestion and validation without uploading data. Reviewer turnaround
is unknown. Once complete inputs exist, target one existing bounded session to
validate/match them and update affected cohort gates; do not invent a calendar
promise or repeat this unchanged readiness analysis while waiting.

The dataset/CodeBERT/contract-score objectives remain unchanged. Continue genuinely
independent authorized construction or source research while dependencies are
missing; the schedule is not paused. Do not treat another weak-label profile,
queue or audit as completion of independent evaluation.

## Evidence and validation scope

Private three-file decision snapshot:
`scafe_function_dataset_pilot/derivatives/cohort_readiness_decision_v1_20261107T0900UTC/`.
It contains a per-definition decision, 47 metadata-only observed-cell records,
and a manifest binding delivery, parents, generator/helpers and outputs. It is
explicitly **not training data** and contains no code payload. The reproducible
workspace analysis is `scafe_iteration_20261107_0900UTC/assess_cohort_readiness.py`.
Group sets/counts and no-promotion state were independently cross-counted; this
mechanical check is not independent semantic review or an accuracy experiment.

Original files and source-label derivatives are unchanged. No source compiler,
model, training, inference, download or installation ran. Required applicable
checks, both integrity validators, fresh isolated full paper build/visual checks,
backed-up input-matched PDF refresh, final/staged review and local Git/held-push
state are recorded in the run report. Manuscript scientific claims are unchanged.
