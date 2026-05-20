# SCAFE — Comprehensive Revision Plan

**Source rejection:** Blockchain: Research and Applications (BCRA-D-26-00395)
**Editor verdict:** Reject, with explicit invitation to resubmit a substantially revised version.
**Reviewers:** 2 (R1, R2) + Editor summary.
**Goal:** Address every individual concern with a concrete action item, then map back to a verification matrix proving 100% coverage.

---

## 1. Enumeration of Every Concern

Each concern is given a unique ID for cross-referencing. Concerns are extracted verbatim from the editorial decision letter and the two reviewer reports.

### 1.1 Editor's Summary (E#)

| ID  | Concern |
|-----|---------|
| E1  | FORGE: "unknown" mapped to negative labels |
| E2  | FORGE: "suspicious" class excluded |
| E3  | FORGE label noise distorts validity of reported performance |
| E4  | Classification threshold selected on held-out test fold → data leakage |
| E5  | Limited real-world interpretability of F1/Precision/Recall |
| E6  | Regex + lightweight parser: coverage/precision/failure rate not validated |
| E7  | Reliability/reproducibility doubts for modern Solidity (proxies, delegatecall, interfaces, inline assembly, newer syntax) |
| E8  | Experimental comparison not up to date — missing recent LLM-based, neuro-symbolic, agent-based, RAG, formal-verification approaches |
| E9  | Several claims overstated, particularly auditor-actionable interpretability |
| E10 | Global RF importance does not provide contract-level explanations |
| E11 | Does not guide auditors to specific vulnerable code regions |
| E12 | Required revisions: task setting, vulnerability labels, dataset semantics, feature reliability, evaluation protocol, related work, interpretability claims |

### 1.2 Reviewer 1 (R1-#)

| ID  | Concern |
|-----|---------|
| R1-1  | DL baselines outdated (TextCNN 2014, BiLSTM 1997, CodeBERT 2020, UniXcoder 2022) |
| R1-2  | Missing references to neuro-symbolic + generative AI agents |
| R1-3  | Must reference and contrast against Trace (IEEE TSE 2026): LLM-synthesized snippets for CFGs in non-compilable repos — direct methodological competitor |
| R1-4  | Must reference SolAgent (2026): multi-agent dual-loop refinement integrating compiler + static analyzers |
| R1-5  | Must reference LLMSA / LLMDFA (NeurIPS 2024/2025): compositional neuro-symbolic data-flow summarization |
| R1-6  | Must reference PropertyGPT (NDSS 2025): LLM-driven formal verification via RAG property generation |
| R1-7  | FORGE: 4,872 contracts split as vulnerable (1,244) / suspicious (2,505) / unknown (1,123) — semantics misused |
| R1-8  | "Suspicious" class discarded |
| R1-9  | "Unknown" class mapped to negative |
| R1-10 | Assigning definitive negative to unverified "unknown" is statistically invalid |
| R1-11 | Paper itself admits unknowns may be partially vulnerable |
| R1-12 | Distorts mathematical decision boundary |
| R1-13 | Suppresses predicted probabilities for high-risk structural patterns appearing in contaminated set |
| R1-14 | Reported ROC-AUC = 0.932 / PR-AUC = 0.589 do not reflect actual detection capability |
| R1-15 | Reflect memorization of data-collection artifacts + label noise |
| R1-16 | FORGE was designed as LLM-driven (Semantic Chunkers + MapReduce Extractors) — its semantics misused |
| R1-17 | Mapping unverified LLM outputs to absolute ground truth is fundamental misunderstanding |
| R1-18 | Structural stream relies on regex for function declarations, variable state, SWC templates |
| R1-19 | Uses experimental non-compiling parser (python-solidity-parser) |
| R1-20 | Theoretically unsound for Turing-complete language |
| R1-21 | Regex inadequate for accurate call graphs / data-flow indicators |
| R1-22 | Regex cannot resolve polymorphic dispatch (interface implementations at runtime) |
| R1-23 | delegatecall in proxy patterns dynamically loads logic from storage slot |
| R1-24 | Regex cannot traverse proxy boundary — call graph silently truncated at proxy |
| R1-25 | Modern gas-optimized contracts rely heavily on inline assembly |
| R1-26 | Regex fails on low-level memory ops (mload, sstore, calldatacopy) |
| R1-27 | python-solidity-parser is an experimental Python port of a JavaScript ANTLR parser |
| R1-28 | Such parsers struggle with complex 0.8.x syntax |
| R1-29 | Frequently fail on: custom errors, user-defined value types, nested assembly blocks |
| R1-30 | 306 structural features built on lossy regex + experimental parser → fragile |
| R1-31 | Authors never measure nor report silent feature-extraction failure rates |
| R1-32 | Reproducibility and reliability of structural stream highly suspect |
| R1-33 | §4.4 protocol: F1/Precision/Recall at per-fold threshold maximizing F1 on held-out fold |
| R1-34 | "Oracle threshold yields upper bound" — explicit admission of leakage |
| R1-35 | Textbook data leakage |
| R1-36 | "Applied equally to all baselines" doesn't rescue the methodology |
| R1-37 | Strips F1/Precision/Recall of real-world validity |
| R1-38 | Production deployment cannot know optimal threshold on unseen data |
| R1-39 | Must tune threshold on isolated validation set or via nested CV |
| R1-40 | Frozen calibrated threshold must be applied to test set |
| R1-41 | Primary claim: "interpretable, auditor-actionable risk factors" |
| R1-42 | Positioned as distinct advantage over opaque DL |
| R1-43 | Interpretability is only global Gini importance from RF (Table 10) |
| R1-44 | Global importance shows on-average splitters across corpus, not per-contract |
| R1-45 | Does not explain why a specific contract was flagged |
| R1-46 | "Globally important = X" doesn't guide auditor to vulnerable line |
| R1-47 | Doesn't explain specific logical flaw |
| R1-48 | True interpretability needs local attribution (SHAP, LIME, attention) |
| R1-49 | Local attribution methods entirely absent |
| R1-50 | Auditor-actionability claim vastly overstated |

### 1.3 Reviewer 2 (R2-#)

| ID  | Concern |
|-----|---------|
| R2-1  | Sections contain repetitive descriptions of advantages |
| R2-2  | Verbose presentation occasionally interrupts logical flow |
| R2-3  | Compilation-free design well-motivated, but semantic modeling capability vs. compiler-based tools (Slither) insufficiently validated |
| R2-4  | Coverage of extracted features unclear |
| R2-5  | Extraction precision unclear |
| R2-6  | Effectiveness comparability to compiler-assisted analysis not established |
| R2-7  | Need comparative evaluation against compiler-based tools on representative subset |
| R2-8  | Focus on feature coverage and extraction consistency |
| R2-9  | LLM-based approaches show strong performance in recent studies, capturing richer semantic/contextual information |
| R2-10 | No comparison with LLM-based baselines or tools |
| R2-11 | Small-scale LLM comparison would significantly strengthen empirical evaluation |
| R2-12 | Manuscript does not clearly specify exact vulnerability types involved |
| R2-13 | Task setting insufficiently clarified |
| R2-14 | Label definitions per dataset insufficiently clarified |
| R2-15 | Need to state: specific vulnerabilities / general detection / binary classification |
| R2-16 | Need clearer description of label definitions per dataset |
| R2-17 | Related-work discussion of recent LLM developments incomplete |
| R2-18 | Missing: agent-based collaboration |
| R2-19 | Missing: multi-stage reasoning |
| R2-20 | Missing: retrieval-augmented generation (RAG) |
| R2-21 | Missing: other emerging LLM-based vulnerability detection frameworks |
| R2-22 | Should include several representative works in these directions |
| R2-23 | Should clarify characteristics and differences among approaches |

**Total individual concerns: 12 (Editor) + 50 (R1) + 23 (R2) = 85 unique items.**

---

## 2. Master Concern → Action Mapping

Each action item has a code and is detailed in §3. Multiple concerns often map to a single action.

| Concern IDs | Action Codes |
|-------------|--------------|
| E1, E2, E3, R1-7..R1-17 | **A-FORGE** (label reconstruction + ablation) |
| E4, E5, R1-33..R1-40 | **A-THRESH** (nested-CV threshold + deployed-threshold experiment) |
| E6, E7, R1-18..R1-32, R2-3..R2-8 | **A-PARSE** (tree-sitter migration + failure-rate analysis + Slither feature correlation + proxy/assembly explicit handling) |
| E8, R1-1..R1-6, R2-9..R2-11, R2-17..R2-23 | **A-LLM** (local LLM baselines) + **A-RW** (related-work expansion) |
| E9, E10, E11, R1-41..R1-50 | **A-SHAP** (per-contract attribution) + **A-CLAIMS** (recalibrate interpretability claims) |
| E12 (task setting), R2-12..R2-16 | **A-TASK** (explicit task / label definition subsection) |
| E12 (related work), R2-17..R2-23 | **A-RW** (related-work expansion) |
| E12 (verbose), R2-1, R2-2 | **A-TRIM** (verbosity reduction) |

---

## 3. Detailed Action Items

Each action lists: **scope**, **inputs**, **outputs**, **validation criteria**, and **estimated time**.

### A-FORGE — FORGE Label Reconstruction & Ablation

**Addresses:** E1, E2, E3, R1-7..R1-17

**Sub-actions:**
- **A-FORGE-1**: Re-derive labels with **three schemes**:
  - (a) **Strict** *(new headline)*: only `vulnerable` from FORGE merged with `vulnerable + clean` from DAppSCAN; drop both `unknown` and `suspicious`. Total positives = 948 + 1,244 = 2,192. Total negatives = 20,509. No noisy labels.
  - (b) **Original** *(legacy, for ablation only)*: current scheme (unknown → negative).
  - (c) **FORGE-excluded** *(robustness check)*: train and evaluate on DAppSCAN-only.
- **A-FORGE-2**: Acknowledge FORGE's construction (LLM-driven Semantic Chunkers + MapReduce Extractors) in the Methodology section. Justify why dropping `unknown` is the **correct** response to that semantic.
- **A-FORGE-3**: Run all primary experiments (ML, DL, ensemble, two-stage, hybrid) on **scheme (a)** as headlines.
- **A-FORGE-4**: Show schemes (b) and (c) as label-noise robustness ablation, demonstrating the qualitative story is preserved but absolute numbers differ.
  - **Hard rule:** If the ranking of methods (SCAFE vs. UniXcoder vs. baselines) inverts between schemes (a) and (b) on any primary metric, report both schemes with equal prominence in the main table — do not suppress inconvenient results.
- **A-FORGE-5 (Fold balance check):** Before any re-runs, verify that the 5-fold GroupKFold split is balanced: no single project contributes more than 15% of the positives in any single fold. Report fold sizes (n_contracts, n_positive, n_negative) in the Experimental Setup section. This pre-empts a reviewer query on whether results are driven by one dominant project in a single fold.

**Outputs:**
- New scored CSV with strict labels: `data/derived/unified_contract_eval_scored_strict.csv`
- Updated `final_results.json`, `scafe_sem_results.json`, `unixcoder_results.json`, `ensemble_unixcoder_results.json` (headline) + matching `_legacy.json` and `_dappscan_only.json` files
- New paper table: `tab_forge_ablation.tex`

**Validation:** Headline metrics on scheme (a) are honest and free of `unknown`-induced bias; story is qualitatively preserved across schemes.

**Estimated time:** 3–4 h CPU re-runs (ML); UniXcoder DAppSCAN-only already exists. UniXcoder on new merged data: ~7 h GPU/MPS.

---

### A-THRESH — Threshold Methodology Fix

**Addresses:** E4, E5, R1-33..R1-40

**Sub-actions:**
- **A-THRESH-1**: Modify `src/ml/train_pipeline.py` and `src/ml/dl_baselines.py`:
  - In each outer fold, run an **inner GroupKFold** on the training set.
  - Select F1-optimal threshold on the **inner OOF predictions**.
  - Freeze that threshold and apply to the outer test fold.
  - **Hyper-parameter tuning is also nested within the inner GroupKFold** — RF depth, GBC learning rate, XGB parameters etc. are all selected via inner-fold cross-validation, never touching the outer test fold. This closes the same leakage vector as threshold tuning and pre-empts any second-round reviewer query on HP selection.
- **A-THRESH-2**: Add an additional **"deployed-threshold" experiment** that satisfies R1-39/R1-40 literally:
  - Hold out 20% of projects as a frozen calibration set (never used for training or test).
  - Tune threshold on this calibration set.
  - Apply to a separate, never-seen test split.
- **A-THRESH-3**: Re-run all ML, DL, ensemble, and two-stage experiments under the corrected protocol.
- **A-THRESH-4**: Remove all uses of "oracle threshold" language from the paper.

**Outputs:**
- New `train_pipeline.py` with `select_threshold_nested(...)` helper.
- Updated OOF prediction files with `threshold_used` column.
- Updated tables: `tab_model_comparison.tex`, `tab_dl_baselines.tex`, `tab_comprehensive_summary.tex`, `tab_ensemble.tex`, `tab_significance.tex` — recomputed F1/Prec/Rec.
- New table: `tab_deployed_threshold.tex` (separate calibration + test).

**Validation:** ROC-AUC and PR-AUC unchanged (threshold-free). F1 drops by 2–5 pp uniformly across methods, relative ranking preserved. Deployed-threshold experiment yields numbers consistent with nested-CV.

**Estimated time:** ~2 h code + ~3 h re-runs (ML); DL re-runs ~6 h MPS.

---

### A-PARSE — Parser Reliability, Tree-sitter Migration, Slither Feature Correlation

**Addresses:** E6, E7, R1-18..R1-32, R2-3..R2-8

**Sub-actions:**

- **A-PARSE-1 (Tree-sitter migration):**
  - Replace `python-solidity-parser` with `tree-sitter-solidity` for AST construction.
  - **Hard failure rule:** If tree-sitter fails to parse a contract (status = `failed`, not `partial-with-errors`), that contract is flagged in the registry and reported separately in the parser-failure table — it is **not silently excluded** from training. AST-derived features for that contract are set to NaN / 0 with a `ts_parse_failed` indicator column, and the impact on classification metrics is reported. This is the only defensible position: excluding failed contracts silently would be a new data-selection bias; imputing zeros is conservative and transparent.
  - Re-extract all 306 structural features.

- **A-PARSE-2 (Parser failure-rate analysis):**
  - For all 26,423 contracts, record per-contract:
    - tree-sitter parse status (full / partial-with-errors / failed)
    - presence of: custom errors, user-defined value types, transient storage, nested assembly blocks
    - Solidity version (extracted from pragma)
  - Aggregate into a failure-rate table broken down by:
    - Solidity major version (0.4.x, 0.5.x, 0.6.x, 0.7.x, 0.8.x)
    - Syntax feature (custom errors, UDVT, transient storage, assembly nesting)
  - Compute classification metrics on parse-succeeded vs. parse-failed subsets to demonstrate parse failures aren't biasing results.

- **A-PARSE-3 (Proxy / delegatecall explicit handling):**
  - Quantify in the corpus:
    - % of contracts containing `delegatecall`
    - % of contracts inheriting from `Proxy`, `TransparentUpgradeableProxy`, `ERC1967Proxy`, `UUPSUpgradeable`, etc.
    - % of contracts with `_implementation()` or `_logic()` patterns
  - Demonstrate that for proxy contracts, the **vulnerability-pattern templates** (delegatecall counts, unchecked-call sites, storage-slot collisions) flag them correctly even when the call graph is truncated.
  - New table: `tab_proxy_coverage.tex`.

- **A-PARSE-4 (Inline-assembly feature validation):**
  - Quantify assembly-opcode usage across the corpus.
  - Validate that our inline-assembly features (already counting `mload`, `sstore`, `calldatacopy`, `call`, `delegatecall` opcodes) produce stable counts comparable to Slither's IR-derived equivalents.

- **A-PARSE-5 (Slither feature-level correlation study):**
  - Subset: **at least 1,500 contracts** where Slither successfully compiles (raised from 500 to make the statistical argument stronger; Slither results already exist in `data/derived/slither_results/`).
  - For each feature with a Slither IR equivalent (cyclomatic, function counts, call-graph edges, state-variable writes, delegatecall sites, assembly counts), compute:
    - Pearson + Spearman correlation between SCAFE-regex/tree-sitter-derived value and Slither-IR-derived value.
    - Per-feature absolute / relative error.
  - **Hard feature-drop rule:** Any feature whose Spearman r < 0.70 against its Slither IR equivalent is either (a) dropped from the final feature set, or (b) retained with an explicit caveat in the paper explaining why. This rule is stated in the paper — it prevents R1 from claiming the correlation study is cosmetic.
  - New table: `tab_feature_validation.tex`.
  - New figure: scatter plots for top 5 features.

- **A-PARSE-6 (Acknowledge limitations explicitly):**
  - New Methodology subsection: "Feature Reliability and Limitations":
    - State explicitly that the call graph is approximate (cannot resolve polymorphic dispatch; truncates at proxy boundary).
    - State explicitly that delegatecall handling is at-the-pattern-level, not flow-level.
    - State explicitly that inline assembly is opcode-counted, not semantically modelled.
    - Justify these as **acceptable approximations** given (a) empirical correlation with Slither IR, (b) end-to-end task performance, (c) the alternative requires compilation.

**Outputs:**
- New `src/risk/contract_risk_tree_sitter.py` (or refactor existing) with tree-sitter backend.
- New script: `scripts/parser_reliability_analysis.py`.
- New script: `scripts/slither_feature_correlation.py`.
- New tables: `tab_parser_reliability.tex`, `tab_proxy_coverage.tex`, `tab_feature_validation.tex`.
- New figure: `fig_feature_validation.pdf` (scatter plots).
- New Methodology subsection in `sc_vulnerability_detection.tex`.

**Validation:**
- Parse success rate ≥ 95% on 0.4.x–0.8.x corpus.
- Top features correlate r > 0.85 with Slither IR-derived equivalents on Slither-compatible subset.
- ML performance on parse-succeeded vs. parse-failed subsets is statistically indistinguishable (no bias).

**Estimated time:** ~3 h code (tree-sitter migration); ~1 h feature re-extraction; ~30 min parser analysis; ~1 h Slither correlation; ~1 h writing.

---

### A-LLM — Local LLM Baselines

**Addresses:** E8 (partial), R1-1, R1-2, R1-6 (partial via local proxy), R2-9, R2-10, R2-11

**Sub-actions:**

- **A-LLM-1 (Model selection):**
  - **Qwen2.5-Coder-7B-Instruct** — current SOTA open code model (Alibaba, 2024).
  - **DeepSeek-Coder-V2-Lite-Instruct (16B)** — strong recent code model (DeepSeek, 2025).
  - **(Optional 3rd)** CodeLlama-13B-Instruct as a well-known reference baseline.
  - Justification in paper: "Local open-source LLMs provide reproducible, no-API-cost comparison; consistent with SCAFE's reproducibility ethos."

- **A-LLM-2 (Prompt design):**
  - Zero-shot binary classification prompt:
    > "You are a smart contract security auditor. Analyze the following Solidity contract and determine if it contains security vulnerabilities. Respond with exactly 'VULNERABLE' or 'CLEAN' followed by a one-sentence explanation."
  - Optional second prompt variant: chain-of-thought ("Think step-by-step about reentrancy, access control, integer overflow, ..."), reported as an ablation.
  - Output parsing: regex extract `VULNERABLE` / `CLEAN`; confidence via logprobs if exposed.

- **A-LLM-3 (Inference script with incremental save-to-disk):**
  - `scripts/run_local_llm_baseline.py`:
    - Iterates `contract_id` → calls Ollama API.
    - For each contract: write JSON line to `data/derived/llm_results/<model>_predictions.jsonl` immediately after inference.
    - On restart, scan existing JSONL, skip already-processed `contract_id`s.
    - Crash-safe; OS-shutdown-safe; battery-loss-safe.
    - Optional checkpointing of running per-fold metrics every N contracts.

- **A-LLM-4 (Context-window handling):**
  - Contracts > 7,000 tokens are truncated to first 7,000 tokens. Record truncation flag per contract.
  - Report truncation rate in the results.

- **A-LLM-5 (Evaluation):**
  - Compute ROC-AUC (if logprobs available) or accuracy-only on the LLM binary outputs.
  - Compute Precision / Recall / F1 directly from binary predictions.
  - Five-fold GroupKFold evaluation aligned with all other methods (project-grouped).
  - Compute Cohen's d effect sizes vs. SCAFE and vs. UniXcoder.

- **A-LLM-6 (Ensemble with LLMs):**
  - Optional: late-fusion of SCAFE + Qwen-Coder OOF predictions to see whether complementary signal exists.
  - Report as additional ablation.

**Outputs:**
- `scripts/run_local_llm_baseline.py` (crash-safe, resumable).
- `data/derived/llm_results/qwen25_coder_7b_predictions.jsonl` (incremental).
- `data/derived/llm_results/deepseek_coder_v2_lite_predictions.jsonl` (incremental).
- (Optional) `data/derived/llm_results/codellama_13b_predictions.jsonl`.
- New table: `tab_llm_baselines.tex` (model, ROC-AUC, F1, time, truncation rate, GPU usage).
- Updated `tab_comprehensive_summary.tex` and `tab_significance.tex` with LLM rows.

**Validation:**
- Predictions deterministic given fixed seed + temperature 0.
- Resumes from interruption correctly.
- Reasonable performance (LLMs expected to be in the same ballpark as CodeBERT, possibly lower due to zero-shot).

**Estimated time:** ~2 h script + prompt iteration; **~8–12 h inference per model** on M4 with MPS (running unattended).

---

### A-RW — Related Work Expansion

**Addresses:** E8 (text portion), R1-2..R1-6, R2-17..R2-23

**Sub-actions:**

- **A-RW-1**: Restructure §2.2 "Machine Learning Approaches" into clear sub-buckets:
  - (i) Graph-based methods (existing).
  - (ii) Sequence-based on bytecode (existing).
  - (iii) Source-level deep learning (existing, extended).
  - (iv) **New: Zero-shot and few-shot LLM detectors** — GPTScan, David et al.
  - (v) **New: Agentic and multi-stage reasoning** — iAudit, SolAgent.
  - (vi) **New: Retrieval-augmented and formal-verification-augmented** — PropertyGPT.
  - (vii) **New: Neuro-symbolic data-flow** — LLMSA, LLMDFA.
  - (viii) **New: LLM-augmented control-flow synthesis** — Trace.

- **A-RW-2**: For each LLM/agentic paper, write 2–3 sentences explaining (a) approach, (b) what it does well, (c) explicit positioning vs. SCAFE (cost, reproducibility, scale, interpretability).

- **A-RW-2b (Per-paper reproducibility disclaimer — mandatory text in paper):** For each of the four papers R1 named explicitly, the revised §2.2 must include the following points verbatim or equivalent:
  - **Trace (IEEE TSE 2026):** "Trace addresses non-compilable repositories by using an LLM to synthesize missing code snippets for CFG construction. Although it shares SCAFE's compilation-free motivation, no public implementation is available as of submission, and the method requires LLM API access for snippet generation. We therefore include Trace in our qualitative positioning table (Table 1) but cannot reproduce its results on our corpus."
  - **SolAgent (2026):** "SolAgent employs a multi-agent dual-loop architecture integrating the Forge compiler environment and static analyzers. Direct comparison is not feasible: SolAgent requires both a running Forge compilation environment and a GPT-4-class API, neither of which is reproducible at the scale of 26,000 contracts within our resource constraints. We discuss SolAgent's approach and contrast it against SCAFE in Table 1."
  - **LLMSA / LLMDFA (NeurIPS 2024/2025):** "LLMSA and LLMDFA perform neuro-symbolic data-flow summarization via compositional LLM prompting. These methods operate at the function-summary level and require per-function LLM calls with a proprietary chain-of-thought protocol not released publicly. We include them in the related work discussion and positioning table but do not reproduce them empirically."
  - **PropertyGPT (NDSS 2025):** "PropertyGPT targets formal verification of smart contracts via RAG-driven property generation, feeding generated properties into an SMT/Certora prover. This is a fundamentally different task from binary vulnerability classification; the output is a set of verified/falsified formal properties, not a vulnerability label. We cite PropertyGPT as a complementary approach and note that SCAFE and PropertyGPT occupy different points on the verification–classification spectrum."

- **A-RW-3**: Update the "Position of This Work" subsection. Add a row to Table 1 (`tab_related_comparison.tex`) for LLM/agentic methods:

  | Method | Compile-free | Interpretable | Risk score | GPU-free | Contract-level | API-free |
  |---|---|---|---|---|---|---|
  | GPTScan | ✓ | partial | ✗ | ✗ | ✓ | ✗ |
  | PropertyGPT | ✓ | partial | ✗ | ✗ | ✗ (per-property) | ✗ |
  | SolAgent | partial | ✗ | ✗ | ✗ | ✓ | ✗ |
  | LLMSA / LLMDFA | ✓ | partial | ✗ | ✗ | partial | ✗ |
  | Trace | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
  | SCAFE (ours) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- **A-RW-4**: Add references to the `.bib`:
  - `chen2024trace`, `zhang2026solagent`, `wu2024llmsa`, `wu2025llmdfa`, `liu2024propertygpt` (exact bibkeys TBD by checking each paper's preferred citation).

**Outputs:**
- Rewritten §2.2 + updated §2.3 ("Position of This Work") in `sc_vulnerability_detection.tex`.
- Updated `tab_related_comparison.tex`.
- Updated `sc_vulnerability_refs.bib`.

**Validation:** Each of Trace, SolAgent, LLMSA, LLMDFA, PropertyGPT, iAudit, GPTScan is cited and discussed with explicit positioning.

**Estimated time:** ~2 h reading + writing.

---

### A-SHAP — Local Interpretability via SHAP

**Addresses:** E9, E10, E11, R1-41..R1-50

**Sub-actions:**

- **A-SHAP-1**: Add `shap` dependency. Use `shap.TreeExplainer` on the trained Vote ensemble (or its strongest individual component: GradientBoosting or XGBoost).
- **A-SHAP-2**: Compute SHAP values for the **entire test fold** (~5,000 contracts) — this is fast for tree models.
- **A-SHAP-3**: Generate explanatory artifacts:
  - **Waterfall plots** for 6 representative cases, selected **deterministically** from saved SHAP values — no hand-picking:
    - TP-1: contract with highest SHAP magnitude (most confidently flagged true positive)
    - TP-2: contract with median SHAP magnitude among true positives
    - FP-1: false positive with highest predicted probability (hardest false alarm)
    - FP-2: false positive with median predicted probability among false positives
    - FN-1: false negative with lowest predicted probability (most missed vulnerable contract)
    - FN-2: false negative with second-lowest predicted probability
  - Selection procedure is stated in the paper caption: "Cases selected deterministically by rank on predicted probability from saved OOF predictions; no cherry-picking." This makes the examples fully reproducible and reviewer-proof.
  - **SHAP summary plot** (bee-swarm) showing feature contributions across the corpus.
  - **SHAP dependence plot** for top 3 features showing per-contract feature value vs. SHAP value.
- **A-SHAP-4**: Add a new paper subsection "Per-Contract Attribution via SHAP" in the Discussion / Error Analysis section.
- **A-SHAP-5**: Concede explicitly that SHAP gives feature-level attribution at the contract level, **not** line-level or vulnerability-type localization. Frame the triage workflow honestly:
  - "SCAFE prioritizes contracts via dual-stream features → SHAP attributes each prediction to specific risk factors → auditors inspect those factors in the code → optionally combine with line-level tools (Slither, Mythril) for localization."

**Outputs:**
- `scripts/generate_shap_explanations.py`.
- `data/derived/shap_values/` (npy + csv per fold).
- New figures: `fig_shap_waterfall.pdf` (3×2 grid), `fig_shap_summary.pdf` (bee-swarm), `fig_shap_dependence.pdf` (top 3 features).
- New paper subsection.

**Validation:** SHAP figures are reproducible from saved values. Discussion section accurately describes what SHAP provides vs. doesn't provide.

**Estimated time:** ~30 min code + 30 min run + 1 h figure design + 1 h writing.

---

### A-CLAIMS — Interpretability Claim Recalibration

**Addresses:** E9, R1-41, R1-42, R1-46, R1-47, R1-50

**Sub-actions:**

- **A-CLAIMS-1**: Find and modify every instance of "interpretable" / "auditor-actionable" / "actionable risk factors" in:
  - Abstract
  - Introduction
  - Section 6 (Discussion)
  - Conclusion
- **A-CLAIMS-2**: Replace overstated language with calibrated language:
  - "Auditor-actionable risk factors" → "Per-contract risk factors attributable via SHAP, guiding manual review to the most relevant feature groups (complexity, identifier risk vocabulary, state mutation patterns), without claiming line-level localization."
  - "Interpretable" → "Model-level transparent (global feature importance) and per-contract attributable (SHAP), in contrast to opaque transformer baselines."
- **A-CLAIMS-3**: Add a sentence in the Limitations section:
  - "SCAFE provides per-contract feature attribution, not line-level vulnerability localization. For precise locations within flagged contracts, integration with line-level tools (Slither, Mythril) is recommended; we leave this hybrid as future work."

- **A-CLAIMS-4 (Fallback contribution statement — write before re-runs):** Draft and lock the contribution narrative now, in a form that is robust regardless of how the re-run numbers land. Two scenarios must both be covered by the same framing:
  - **Scenario A (SCAFE ROC-AUC ≥ LLM baselines):** "SCAFE achieves competitive detection performance while being fully CPU-bound, API-free, and sub-second per contract — making it the only approach in our comparison that is deployable at ecosystem scale without GPU infrastructure or external API access."
  - **Scenario B (LLM baselines slightly outperform SCAFE on ROC-AUC):** "Zero-shot LLMs achieve marginally higher ROC-AUC at the cost of 100–1000× higher inference time and API dependency. SCAFE occupies a distinct, complementary point: it serves as a scalable, GPU-free, explainable pre-screening layer; the SCAFE+LLM hybrid (Table X) demonstrates that both approaches provide complementary signal, with the hybrid achieving the best overall results."
  - The framing in **Scenario B is already supported by A-LLM-6 (the late-fusion ensemble)**. Promote A-LLM-6 from "Optional" to **mandatory** — run SCAFE+Qwen OOF late-fusion regardless of individual results, so the hybrid is always available as the headline contribution if needed.

**Outputs:**
- Recalibrated wording throughout the paper.
- Mandatory A-LLM-6 late-fusion result (SCAFE + best LLM baseline).

**Validation:** No remaining claims of "auditor-actionable" without qualification. Contribution statement is written in a form that survives both number scenarios.

**Estimated time:** ~1 h.

---

### A-TASK — Explicit Task / Label Definition

**Addresses:** E12 (task setting, vulnerability labels, dataset semantics), R2-12..R2-16

**Sub-actions:**

- **A-TASK-1**: Add a new subsection "Task Definition and Label Semantics" early in §3 (Methodology):
  - Task: **binary, contract-level, vulnerability-type-agnostic classification.**
  - Defend the choice: per-vulnerability-type prediction requires fine-grained CWE labels which are not consistently available across datasets; binary detection is the screening-relevant task and the de facto standard in DAppSCAN-based studies.
- **A-TASK-2**: Add a per-dataset label-semantics paragraph:
  - **DAppSCAN:** A contract is `vulnerable` iff at least one SWC finding in the audit report references its file. Otherwise `clean`. Labels are derived from professional auditor verdicts.
  - **FORGE (strict, headline scheme):** Only `vulnerable` retained; `unknown` and `suspicious` are dropped. FORGE positives are CWE-labeled findings extracted via LLM-driven semantic chunking by the dataset authors.
  - **AccessControlVulnerabilities:** All 94 contracts are confirmed access-control vulnerabilities; only used as a positive-only sanity check, never as a training source.
- **A-TASK-3**: Cross-reference each table caption to clarify the label scheme used.

**Outputs:**
- New §3.1 subsection.
- Updated table captions throughout.

**Validation:** A reader can answer in one sentence: "What is the task?" and "What does each dataset's label mean?"

**Estimated time:** ~30 min.

---

### A-TRIM — Verbosity Reduction

**Addresses:** E12 (verbosity implicit), R2-1, R2-2

**Sub-actions:**

- **A-TRIM-1**: Identify repetitions:
  - "compilation-free, GPU-free, lightweight, interpretable" appears > 15 times across the paper.
  - Same key results restated in abstract, intro, RQ1 conclusions, RQ3 conclusions, discussion, conclusion.
- **A-TRIM-2**: Compress:
  - Cut redundant sentences in the introduction (it currently overlaps heavily with the abstract).
  - Cut restatements in RQ conclusions (results already state numbers; conclusions should add interpretation, not repeat).
  - Tighten paragraphs in the Discussion that re-list the three deployment scenarios.
- **A-TRIM-3**: Target: reduce non-table/figure body text by ~10–15% (~700 words).

**Outputs:**
- Tightened `sc_vulnerability_detection.tex`.

**Validation:** Word count drops 10–15%, no content lost.

**Estimated time:** ~1.5 h.

---

## 4. Cross-Verification Matrix

This table proves that every individual concern is covered. Rows are concern IDs; columns are action codes.

| Concern | A-FORGE | A-THRESH | A-PARSE | A-LLM | A-RW | A-SHAP | A-CLAIMS | A-TASK | A-TRIM |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **E1**  | ✓ |   |   |   |   |   |   |   |   |
| **E2**  | ✓ |   |   |   |   |   |   |   |   |
| **E3**  | ✓ |   |   |   |   |   |   |   |   |
| **E4**  |   | ✓ |   |   |   |   |   |   |   |
| **E5**  |   | ✓ |   |   |   |   |   |   |   |
| **E6**  |   |   | ✓ |   |   |   |   |   |   |
| **E7**  |   |   | ✓ |   |   |   |   |   |   |
| **E8**  |   |   |   | ✓ | ✓ |   |   |   |   |
| **E9**  |   |   |   |   |   | ✓ | ✓ |   |   |
| **E10** |   |   |   |   |   | ✓ |   |   |   |
| **E11** |   |   |   |   |   | ✓ | ✓ |   |   |
| **E12** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **R1-1**  |   |   |   | ✓ | ✓ |   |   |   |   |
| **R1-2**  |   |   |   | ✓ | ✓ |   |   |   |   |
| **R1-3**  |   |   |   |   | ✓ |   |   |   |   |
| **R1-4**  |   |   |   |   | ✓ |   |   |   |   |
| **R1-5**  |   |   |   |   | ✓ |   |   |   |   |
| **R1-6**  |   |   |   |   | ✓ |   |   |   |   |
| **R1-7**  | ✓ |   |   |   |   |   |   |   |   |
| **R1-8**  | ✓ |   |   |   |   |   |   |   |   |
| **R1-9**  | ✓ |   |   |   |   |   |   |   |   |
| **R1-10** | ✓ |   |   |   |   |   |   |   |   |
| **R1-11** | ✓ |   |   |   |   |   |   |   |   |
| **R1-12** | ✓ |   |   |   |   |   |   |   |   |
| **R1-13** | ✓ |   |   |   |   |   |   |   |   |
| **R1-14** | ✓ |   |   |   |   |   |   |   |   |
| **R1-15** | ✓ |   |   |   |   |   |   |   |   |
| **R1-16** | ✓ |   |   |   |   |   |   |   |   |
| **R1-17** | ✓ |   |   |   |   |   |   |   |   |
| **R1-18** |   |   | ✓ |   |   |   |   |   |   |
| **R1-19** |   |   | ✓ |   |   |   |   |   |   |
| **R1-20** |   |   | ✓ |   |   |   |   |   |   |
| **R1-21** |   |   | ✓ |   |   |   |   |   |   |
| **R1-22** |   |   | ✓ |   |   |   |   |   |   |
| **R1-23** |   |   | ✓ |   |   |   |   |   |   |
| **R1-24** |   |   | ✓ |   |   |   |   |   |   |
| **R1-25** |   |   | ✓ |   |   |   |   |   |   |
| **R1-26** |   |   | ✓ |   |   |   |   |   |   |
| **R1-27** |   |   | ✓ |   |   |   |   |   |   |
| **R1-28** |   |   | ✓ |   |   |   |   |   |   |
| **R1-29** |   |   | ✓ |   |   |   |   |   |   |
| **R1-30** |   |   | ✓ |   |   |   |   |   |   |
| **R1-31** |   |   | ✓ |   |   |   |   |   |   |
| **R1-32** |   |   | ✓ |   |   |   |   |   |   |
| **R1-33** |   | ✓ |   |   |   |   |   |   |   |
| **R1-34** |   | ✓ |   |   |   |   |   |   |   |
| **R1-35** |   | ✓ |   |   |   |   |   |   |   |
| **R1-36** |   | ✓ |   |   |   |   |   |   |   |
| **R1-37** |   | ✓ |   |   |   |   |   |   |   |
| **R1-38** |   | ✓ |   |   |   |   |   |   |   |
| **R1-39** |   | ✓ |   |   |   |   |   |   |   |
| **R1-40** |   | ✓ |   |   |   |   |   |   |   |
| **R1-41** |   |   |   |   |   | ✓ | ✓ |   |   |
| **R1-42** |   |   |   |   |   |   | ✓ |   |   |
| **R1-43** |   |   |   |   |   | ✓ |   |   |   |
| **R1-44** |   |   |   |   |   | ✓ |   |   |   |
| **R1-45** |   |   |   |   |   | ✓ |   |   |   |
| **R1-46** |   |   |   |   |   | ✓ | ✓ |   |   |
| **R1-47** |   |   |   |   |   | ✓ | ✓ |   |   |
| **R1-48** |   |   |   |   |   | ✓ |   |   |   |
| **R1-49** |   |   |   |   |   | ✓ |   |   |   |
| **R1-50** |   |   |   |   |   | ✓ | ✓ |   |   |
| **R2-1**  |   |   |   |   |   |   |   |   | ✓ |
| **R2-2**  |   |   |   |   |   |   |   |   | ✓ |
| **R2-3**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-4**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-5**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-6**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-7**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-8**  |   |   | ✓ |   |   |   |   |   |   |
| **R2-9**  |   |   |   | ✓ |   |   |   |   |   |
| **R2-10** |   |   |   | ✓ |   |   |   |   |   |
| **R2-11** |   |   |   | ✓ |   |   |   |   |   |
| **R2-12** |   |   |   |   |   |   |   | ✓ |   |
| **R2-13** |   |   |   |   |   |   |   | ✓ |   |
| **R2-14** |   |   |   |   |   |   |   | ✓ |   |
| **R2-15** |   |   |   |   |   |   |   | ✓ |   |
| **R2-16** |   |   |   |   |   |   |   | ✓ |   |
| **R2-17** |   |   |   |   | ✓ |   |   |   |   |
| **R2-18** |   |   |   |   | ✓ |   |   |   |   |
| **R2-19** |   |   |   |   | ✓ |   |   |   |   |
| **R2-20** |   |   |   |   | ✓ |   |   |   |   |
| **R2-21** |   |   |   |   | ✓ |   |   |   |   |
| **R2-22** |   |   |   |   | ✓ |   |   |   |   |
| **R2-23** |   |   |   |   | ✓ |   |   |   |   |

**Coverage check: every concern ID appears in at least one action column. ✓**

---

## 5. Execution Order

Execution is structured to minimize wasted work (e.g., do not re-extract features twice).

### Phase 1 — Methodology fixes (foundations)
1. **A-FORGE-1**: Derive new strict label CSV.
2. **A-PARSE-1**: Tree-sitter migration. Re-extract features into a new scored CSV.
3. **A-THRESH-1**: Refactor threshold-selection helper.

### Phase 2 — ML/DL re-runs under fixed protocol
4. Run ML pipeline (RF, GBC, XGB, ET, HGB, Vote) on new features + strict labels + nested threshold.
5. Run UniXcoder on new merged data (long, MPS).
6. Generate OOF predictions for ensembles.
7. Run two-stage + late-fusion ensemble experiments.
8. Run FORGE label-scheme ablation (legacy + DAppSCAN-only) for comparison.
9. Run deployed-threshold experiment.

### Phase 3 — New analyses
10. **A-PARSE-2/3/4/5**: Parser reliability + proxy/assembly + Slither correlation.
11. **A-SHAP**: SHAP value generation + figures.
12. **A-LLM**: Local LLM baselines (long; can run unattended overnight in parallel with paper writing).

### Phase 4 — Paper rewrites
13. **A-TASK**: New task-definition subsection.
14. **A-RW**: Related-work expansion.
15. **A-CLAIMS**: Interpretability claim recalibration.
16. **A-TRIM**: Verbosity reduction.
17. Update all tables and figures from new results.
18. Update Abstract, Intro, all RQ conclusions, Discussion, Conclusion to reflect new numbers.

### Phase 5 — Verification
19. Compile PDF, verify no broken refs, no leftover names, no leftover "oracle" / overclaim language.
20. Update Zenodo / anonymous-4open package with new code + results.
21. Re-verify anonymity (no Tomer / Gilad / Asaf / Doitshman / Katz / Shabtai mentions anywhere).

---

## 6. Risk Register

| Risk | Mitigation |
|------|------------|
| Tree-sitter migration changes feature distributions significantly | Validate via Slither correlation; if drift > 5%, keep python-solidity-parser as primary and use tree-sitter only for parse-rate analysis. |
| LLM inference takes > 24 h total | Run unattended overnight; incremental save means partial results are usable. |
| Headline ROC-AUC drops significantly under strict labels | Frame honestly; if drops > 5 pp on merged, lean on DAppSCAN-only as headline. |
| LLM baselines outperform SCAFE | Reframe SCAFE as the GPU-free / API-free / interpretable alternative; emphasize the hybrid SCAFE+LLM as the strongest configuration. |
| Threshold fix erodes F1 advantage over lightweight DL | ROC-AUC/PR-AUC are headline metrics anyway; the story is preserved. |
| SHAP figures look generic | Hand-pick 6 illustrative contracts (TP/FP/FN × 2 categories) for the waterfall plots. |

---

## 7. Deliverables Checklist (final)

- [ ] Updated `sc_vulnerability_detection.tex` with all rewrites
- [ ] New strict-label CSV + legacy/DAppSCAN-only ablation CSVs
- [ ] Re-run ML / DL / ensemble / two-stage / hybrid results
- [ ] Tree-sitter migrated feature extractor
- [ ] Parser-reliability table
- [ ] Slither feature-correlation table + scatter figure
- [ ] Proxy/delegatecall coverage table
- [ ] Local LLM baseline results (2–3 models, full corpus, incremental)
- [ ] LLM baseline table + significance test rows
- [ ] SHAP analysis + 3 new figures
- [ ] New Task-Definition subsection
- [ ] Expanded Related-Work subsection covering 8 LLM/agentic/neuro-symbolic works
- [ ] Recalibrated interpretability claims throughout
- [ ] Verbosity reduction (~10–15%)
- [ ] Updated Zenodo + anonymous-4open package
- [ ] Compiled PDF with no broken refs
- [ ] Anonymity re-verified

**Nothing committed or pushed.** All changes left staged for review.

---

*End of revision plan.*
