# Remaining Reviewer Action Items

This checklist tracks the unresolved items from `reviewers-bcra.md` and
`reviewers-bcra-2.md`. Mark an item complete only when the implementation,
experiment artifact, manuscript update, and relevant validation have all been
completed.

## Priority 0 — Required before treating SCAFE headline results as submission evidence

- [x] **Remove non-source metadata from the primary SCAFE feature matrix.**
  `has_verified_source` and `is_solc_pre_08` can use availability/compiler
  metadata. Exclude them or derive them solely from Solidity source; add
  regression tests; rerun affected structural experiments; regenerate dependent
  figures and tables. Completed with 255 source-only features, corrected
  five-seed fixed-threshold and train-only-calibrated LODO artifacts, and
  regenerated primary tables.

- [x] **Run and report parser/extractor reliability evaluation against Slither.**
  Use a representative compiler-compatible subset. Report parser/extractor
  success and failure rates, results by Solidity-version strata, behavior on
  proxies/`delegatecall`, interfaces, and inline assembly, and agreement or
  coverage for features with meaningful Slither counterparts. Completed
  artifacts: `data/derived/slither_results/slither_summary.json`,
  `data/derived/slither_reliability/`, and
  `data/derived/slither_correlation/correlation_results.csv`; the manuscript
  now reports the audit in Table~\ref{tab:parser_reliability}.

- [x] **Remove the unsupported parser-reliability claim until the evaluation exists.**
  Superseded: the Slither audit now exists and the manuscript reports its
  scope, failure rate, construct strata, and correlation limits.

- [x] **Validate every current manuscript value against its source artifact.**
  Added `scripts/validate_current_paper_artifacts.py`; it verifies the corrected
  255-feature manifest, five-seed fixed/calibrated artifacts, source-only
  exclusions, corrected headline values, PDF, and primary figures.

## Priority 1 — DL baseline scope decision

- [x] **Define the DL repetition scope.**
  Seeds 43--46 are not run. The manuscript reports seed 42 as a controlled,
  single-seed exploratory baseline and makes no five-seed DL uncertainty or
  superiority claim. Repeating the DL seeds remains optional strengthening,
  not an explicit reviewer requirement.

- [x] **Qualify the matched-comparison claim.**
  The paper reports the available seed-42 contract-level results with
  prevalence, ranking metrics, operating-point metrics, and chunk statistics;
  it does not present them as a matched five-seed comparison.

- [x] **Resolve Transformer fine-tuning comparability.**
  The manuscript states that complete-input chunking is used but partial
  fine-tuning remains; the Transformer results are explicitly exploratory and
  not treated as definitive architecture comparisons.

- [x] **Keep legacy DL results out of headline conclusions.**
  Only the leakage-safe `fair_baseline_eval.py` artifacts may support the new
  comparison. Do not use legacy test-threshold-tuned or truncated-input outputs
  as primary evidence. The manuscript confines them to the alternate-protocol
  appendix and labels the seed-42 controlled run separately.

## Priority 2 — Strongly recommended scientific evidence

- [x] **Provide local interpretability evidence or narrow the contribution.**
  Generated fold-consistent local SHAP artifacts for one XGBoost head, seed 42,
  and TP/FP/FN cases from each primary held-out dataset. The manuscript limits
  interpretation to score contributions, not causal mechanisms or line-level
  localization.

- [x] **Justify the high-dimensional feature representation.**
  Report repeated strict-LODO feature-group ablations, correlation/redundancy
  analysis, and extraction time/memory cost per feature group. Do not claim
  individual features are necessary without corresponding evidence. The current
  package reports ablations, 255-feature correlation summaries, and stratified
  extraction timings.

- [x] **Strengthen provenance-confounding controls or narrow generalization.**
  Add a temporal, developer/project, audit-firm, or comparable cohort-disjoint
  evaluation with both classes if feasible. Otherwise state that LODO reduces
  exact-source leakage but cannot remove provenance/style confounding and limit
  claims to the named corpora. The manuscript now makes this limitation
  explicit.

## Priority 3 — Positioning and manuscript hardening

- [x] **Decide the LLM comparison scope.**
  Either run a reproducible, source-deduplicated matched LLM evaluation with
  saved predictions and coverage/truncation data, or retain contextual-only LLM
  discussion and make no matched empirical claim. The current manuscript uses
  the latter scope explicitly; no matched LLM superiority claim is made.

- [x] **Correct and verify the bibliography.**
  Correct TRACE and PropertyGPT venue/year metadata and independently verify all
  previously questioned references.

- [x] **Perform the final concision and wording pass.**
  Remove repeated compilation-free/lightweight claims, preserve concrete
  limitations, state dataset and label policy in each table caption, and use
  “observed corpus coverage” rather than unverified parser “support.”
