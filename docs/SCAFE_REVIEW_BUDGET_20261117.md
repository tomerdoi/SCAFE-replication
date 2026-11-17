# Fixed-score review-budget comparison — 17 November 2026

## Outcome and interpretation

Completed the prespecified comparison on both unchanged already-inspected target caches. DIVE's full-versus-nuisance ordering changes across review budgets: full is higher at1%, lower at5%/10%, and close at20%. Its5%/10%expected precision is below the uniform-random expectation on these native labels. Empirical-Analysis has higher full-score point precision at all four budgets, with marginal conditional intervals excluding zero at5%/20%but including zero at1%/10%. All eight outcomes are retained; no budget, model or future threshold was selected from them.

This is a new conditional ranking diagnostic, not a new classifier, improved model, confirmed vulnerability/triage accuracy or fresh generalization evidence. High target positive prevalence matters: the DIVE1%full precision0.8858is only slightly above random expectation0.8810, despite its large contrast with the nuisance score0.6941. Native labels are source-record finding associations, not independent security gold or observed human-review success.

| Target | Budget | k | Full P | Nuisance P | Random P | Delta P [95% interval] | Full R | Nuisance R |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| DIVE | 1% | 219 | 0.8858 | 0.6941 | 0.8810 | +0.1918 [+0.1096, +0.2695] | 0.0100 | 0.0079 |
| DIVE | 5% | 1096 | 0.7920 | 0.8768 | 0.8810 | -0.0849 [-0.1159, -0.0556] | 0.0449 | 0.0497 |
| DIVE | 10% | 2193 | 0.8390 | 0.9115 | 0.8810 | -0.0725 [-0.0921, -0.0534] | 0.0952 | 0.1034 |
| DIVE | 20% | 4387 | 0.8972 | 0.8920 | 0.8810 | +0.0052 [-0.0059, +0.0146] | 0.2036 | 0.2025 |
| Empirical-Analysis | 1% | 21 | 0.9286 | 0.8095 | 0.7063 | +0.1190 [-0.0898, +0.3333] | 0.0127 | 0.0111 |
| Empirical-Analysis | 5% | 108 | 0.9444 | 0.7500 | 0.7063 | +0.1944 [+0.0926, +0.2870] | 0.0665 | 0.0528 |
| Empirical-Analysis | 10% | 217 | 0.8387 | 0.8018 | 0.7063 | +0.0369 [-0.0276, +0.0968] | 0.1186 | 0.1134 |
| Empirical-Analysis | 20% | 434 | 0.7951 | 0.7143 | 0.7063 | +0.0808 [+0.0317, +0.1392] | 0.2249 | 0.2021 |

P and R are expected observed-label precision and positive recall, respectively. Recall uses all19328DIVE or1534Empirical-Analysis observed positives in the denominator, not only reviewed/referred positives. Random P is native prevalence; expected random recall is k/N. Both views use exactly the same k; requested fractions are floored to integer record counts, with at least one record. No perfect-human-review assumption is made.

## Frozen protocol, exact ties and uncertainty

The protocol was saved before computing these budget outcomes:1%,5%,10%,20%; five-fit mean probabilities; same exact target identities/labels/seeds42--46 and ETidentifier; no favorable intersection;1,000paired iid-record resamples, seed2026111703for each target; eight marginal95%percentile intervals for precision differences. Protocol/source hashes and pre-outcome time are in analysis_freeze.json. Prior AUC/AP results were known, so this is retrospective descriptive/regression evidence, not preregistered external confirmation or independent hypothesis discovery. DAppSCAN and other non-target rows are skipped before outcome interpretation through the unchanged strict existing loader.

Exact equal mean scores define ties. At a boundary, expected positives equal all higher-score positives plus remaining slots times the positive fraction within the boundary tie. No record-ID or outcome-favorable tie ordering is used. The same rule is applied after reranking each paired bootstrap sample. Fractional expected counts are intentional: the21-record EA budget has19.5expected positives. At the434-record EA budget,75slots lie in a239-record score tie. These expectations are not realized fractional reviews; the intervals concern expected tie-handled precision under iid-record sampling, not random tie-realization variance.

A post hoc interpretive check retains possible boundary-tie subset extrema for all16view/budget cells in posthoc_tie_bounds.json. Those finite-cohort bounds are not confidence intervals, achievable label-blind policies or replacement estimators, and they were not used to select a result. The original frozen primary outcomes remain unchanged. In particular, the EA434full possible precision spans0.7903--0.7972around expectation0.7951. The analysis uses the native CSV/NumPy mean convention retained by the prior paired comparison, with exact equality and no rounding/jitter/tolerance search.

The intervals are marginal, not multiplicity-adjusted or a joint eight-contrast test. They condition on fixed cached means and observed labels; project/clone/corpus dependence, annotation correctness, model/source selection and training variability are not covered. No project IDs are invented, no probability calibration/threshold fitting occurs, and no operational policy is recommended. Future deployment claims still need independently verified outcomes, prospective cohorts and observed review effort.

## Numerical and software validation

Eighteen software controls cover known perfect/reversed/tied rankings, exhaustive tie permutations, weighted versus expanded bootstrap samples, missing bootstrap score groups, row-order/strict-monotone invariance, all-record review, recall denominator, exact zero identical-score intervals, score-swap sign reversal, invalid inputs, CPU cap and output-overwrite refusal. Numeric arrays are software fixtures, not model training examples or synthetic function records. The real-data identical-score100drawcontrol also yields exactly zero for every budget.

An independent expanded-index and direct Boolean score-threshold implementation replayed all16precision cells and all eight1,000drawintervals: maximum absolute error0for both. It reuses the strict original cohort loader and is not independent source-label verification. The primary weighted-score-group calculation took about1.20CPU seconds with about262MiB own-child peakRSS; reference replay about1.41CPU seconds. No model weights or Solidity code were loaded or executed; no model was fitted. Initial tiny numeric controls omitted the shared lease; they were rerun under the lease before any real-data calculation, and final mandatory checks were serialized. No retrospective non-overlap claim is made for that initial0.023secondfixture invocation; no foreign process was probed, stopped or modified.

## Exact lineage and scope

- Full cache: data/derived/sanitized_v1/ml_results/primary_lodo/scafe_repeated_lodo_predictions.csv, SHA2569af4efe3e4ca17e907b364c09dff0b547fcc4d8fef2ca9627ea80aabd5a9eca4.
- Nuisance cache: data/derived/sanitized_v1/ml_results/nuisance_baseline/predictions.csv, SHA25639efe0be2742b71b7c6495d4d960c1ac45710230424f9db75f68334f48424a62.
- Original scored input:9bd0bfeb51b46df80a602eb334e7ebdcaf14d5777ce69ccce76a223e5283cfcb. Both caches retain the same seeds/input; full244features and fixed six-feature nuisance protocol verified.
- Aggregate result SHA256:7e3c3c45b1991f0a1461ac31752687c7beb4e8ffbad520372ff7c12fa705012d. Inputs remain unchanged after calculation. Private task evidence: scafe_iteration_20261117_1300UTC.

Run scripts/review_budget.py with --root, --output(new authorized local path), and --protocol pointing to the retained frozen protocol. No per-record predictions, labels, IDs, raw source or findings are exported. Existing source/access/C4/reserved/publication holds remain. This local derived-cache use is not a new source-rights or distribution grant. Function-data records, held context views, labels, masks, eligibility and all original unknowns remain unchanged; no held MetaSwap/Nexus/Alchemy screening or substitute was attempted. Target values do not guide another paper's adaptive selection.

## Rolling outcome milestones

1. This session: fixed-score result, algorithmic replay, manuscript methods/table/limits and mandatory validation/PDF/Git checkpoint. No update to the older dated aggregate review package is claimed.
2. Conditional one to two focused sessions with required evidence/review access: substantive contribution/claim and distribution decision against the current results. Avoid more post hoc metric variants, target-selected models or repeated source screens as a substitute for that decision.
3. Final manuscript/evidence freeze after those gates, conditionally one to two further sessions for justified revisions. External review/rights elapsed time is unknown; no calendar-week, Q1-ready or acceptance promise. The function extension remains gated on compatible independent real groups, source/context/quality, partial-label/grouped baseline and evaluation-role requirements.
