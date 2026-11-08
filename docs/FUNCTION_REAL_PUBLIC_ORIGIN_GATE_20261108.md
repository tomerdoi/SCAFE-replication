# Real-public-data origin gate in the source-training loader

## Implemented boundary — 8 November 2026, 10:00 UTC

The real-public-data-only instruction is now enforced at the existing
`masked_source_example(..., for_training=True)` path, which is used by
`collate_source_examples` and the hash-pinned `load_source_batch` loader.
An eligibility flag plus an empty blocker list is no longer enough: the source
target must also carry an accepted, identity-bound real-public origin decision.
Existing readiness, source/span/category, partial-mask, code-hash and code-only
input checks remain unchanged. Diagnostic batching remains explicitly non-training.
No model is loaded, trained or scored by this change.

`scripts/function_real_public_origin.py` checks the decision policy/status,
real-world origin and public-dataset assertion, record/source/revision/input-hash
binding, traceable decision/reviewer identifiers, a credential-free HTTPS dataset
reference and nonempty pinned origin-evidence references. Synthetic, unknown,
unreviewed, nonpublic, missing or mismatched decisions fail closed. A hosting URL
or category label alone is not accepted as origin evidence. No URL is fetched.
The accepted evidence-reference kinds concern upstream real-world documentation,
real-project source lineage or on-chain source provenance.

The decision is review metadata, not a proof of semantic truth. The code validates
its structure and binding; it cannot establish that a reviewer is qualified,
that an evidence hash supports the claim, that a public URL exists, or that origin
assertions are truthful. Obtain and verify the evidence under the declared review
protocol, pin the decision with the source release, and retain the original
artifacts. Passing this origin check grants neither eligibility nor independent
gold, correct labels, license permission or a valid split. All earlier gates apply.
Do not populate accepted decisions from a source name, role flag or this validator.

## Schema and integration scope

The target-side `real_public_origin_decision` holds policy
`real-public-function-origin-v1`, status `accepted`, origin `real_world`, boolean
`public_dataset`, record/source/revision/model-hash bindings, decision/reviewer
IDs, `public_dataset_url`, and evidence items with kind, artifact SHA-256 and a
nonempty locator. These fields remain outside the four-field code-only model input.
The current source release is immutable; no decision is attached or promoted here.
A future justified annotation release must be explicitly hash-pinned as usual.

The shared validator accepts explicit training/evaluation purposes, but only the
existing source-training path is integrated in this change. No evaluation runner
was added or retroactively certified; future/legacy research entrypoints must
apply the real-data rule and existing evaluation gates explicitly before use.
The low-level loss primitive is mathematical software, not a data-admission gate.

## Validation and result

Eight new metadata-only tests cover missing decisions, synthetic/unknown origin,
review/public-status failures, identity changes, absent/hosting-only evidence,
invalid hashes, absent review identifiers, credential-bearing/non-HTTPS URLs,
no mutation/promotion, and the same origin boundary for evaluation-purpose checks.
The existing batching integration test now verifies that an otherwise eligible
record is rejected without origin evidence. Its complete schema fixture tests
software behavior only; it is never research model training/evaluation data.
Historical tensor/unit fixtures remain software checks, not admitted dataset rows.

A bounded read-only check against all 110 native SCRUBD source targets demonstrates
that source hosting/claims without an accepted origin decision are rejected. This
measures gate behavior, not source authenticity or label quality. The original data,
33 RE proxies, prior derivatives, masks, groups, labels and eligibility/splits are
unchanged. No new real-origin decisions, qualified evaluation labels, corpus,
training or accuracy results were produced.

## Next critical step

Bind verified real-world/public-origin evidence to a selected source-defined cohort
in a separate reviewed release; do not infer eligibility from metadata shape. Pair
that with agreed target/source-quality and independently qualified grouped
evaluation, then matched baselines and the bounded CodeBERT pilot. Unknown-origin
examples remain excluded; synthetic augmentation remains prohibited. No new
acquisition/download permission or spending is implied. Historical synthetic
experiments remain clearly labeled and outside future model cohorts.
