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

- [ ] **Validate every current manuscript value against its source artifact.**
  Replace or update the obsolete paper-number validator. Check all feature
  counts, dataset/label policies, source-only claims, tables, and figures after
  regeneration.

## Priority 1 — Required for a final fair DL baseline comparison

- [ ] **Run fair DL-baseline seeds 43–46.**
  Complete TextCNN, BiLSTM, CodeBERT, and UniXcoder under the existing
  duplicate-safe, full-source, contract-level LODO protocol. Seed 42 is
  complete but is single-seed evidence only.

- [ ] **Report matched five-seed SCAFE and DL baseline results.**
  Use the same canonical LODO rows. Report per-dataset mean and uncertainty for
  ROC-AUC, PR-AUC, precision, recall, F1, prevalence, and coverage/chunk
  statistics.

- [ ] **Resolve Transformer fine-tuning comparability.**
  Add a full-fine-tuning sensitivity for CodeBERT and UniXcoder, or explain and
  qualify the current eight-layer freezing configuration.

- [ ] **Keep legacy DL results out of headline conclusions.**
  Only the leakage-safe `fair_baseline_eval.py` artifacts may support the new
  comparison. Do not use legacy test-threshold-tuned or truncated-input outputs
  as primary evidence.

## Priority 2 — Strongly recommended scientific evidence

- [ ] **Provide local interpretability evidence or narrow the contribution.**
  Generate fold-consistent per-contract SHAP artifacts with a few auditable case
  studies. Otherwise remove per-contract attribution as an achieved result and
  retain only feature-group/model-level transparency.

- [ ] **Justify the high-dimensional feature representation.**
  Report repeated strict-LODO feature-group ablations, correlation/redundancy
  analysis, and extraction time/memory cost per feature group. Do not claim
  individual features are necessary without corresponding evidence.

- [ ] **Strengthen provenance-confounding controls or narrow generalization.**
  Add a temporal, developer/project, audit-firm, or comparable cohort-disjoint
  evaluation with both classes if feasible. Otherwise state that LODO reduces
  exact-source leakage but cannot remove provenance/style confounding and limit
  claims to the named corpora.

## Priority 3 — Positioning and manuscript hardening

- [ ] **Decide the LLM comparison scope.**
  Either run a reproducible, source-deduplicated matched LLM evaluation with
  saved predictions and coverage/truncation data, or retain contextual-only LLM
  discussion and make no matched empirical claim.

- [ ] **Correct and verify the bibliography.**
  Correct TRACE and PropertyGPT venue/year metadata and independently verify all
  previously questioned references.

- [ ] **Perform the final concision and wording pass.**
  Remove repeated compilation-free/lightweight claims, preserve concrete
  limitations, state dataset and label policy in each table caption, and use
  “observed corpus coverage” rather than unverified parser “support.”
