# SCAFE — Submission Readiness Gap Analysis
## Cynical Reviewer Perspective

**Date:** May 2026 (v2 — expanded after deeper audit)
**Context:** Paper revised per BCRA-D-26-00395 rejection. All original reviewer concerns (85 items across Editor + R1 + R2) were addressed at the methodology level. This document audits the *current state of the paper* for any remaining gaps that a **second-round reviewer** would flag.

The original revision plan is in `REVISION_PLAN.md`. The reviewer letter is in `reviewers-bcra.md`.

---

## Executive Summary

After a deep audit of the paper, the data files, and the LaTeX tables, **the paper has a SYSTEMIC protocol inconsistency** that runs much deeper than just the DL baselines:

**Most of the paper's headline claims are on STRICT labels (22,701 contracts), but almost every ablation, sensitivity, robustness, efficiency, complementarity, significance, and cross-dataset experiment uses LEGACY labels (23,824 contracts).**

A cynical reviewer reading the paper top-to-bottom will see headline ROC-AUC of 0.957 and then hit ablations claiming 0.925, 0.928, 0.932 — and will demand to know why the ablations are on a different protocol.

The single most important action is the overnight DL strict re-run (resolves Gaps 1, 2, 3, 5, 8, 10, 13). However there are now **18 distinct gaps** identified, requiring careful sequencing.

---

## Part 1 — Original Revision Plan: Verification Audit

### ✅ Fully Addressed at the Headline Level

| Action | What was done |
|--------|---------------|
| **A-FORGE** | Strict label scheme implemented in `unified_contract_eval_scored_strict.csv`; 3-way ablation table in paper; FORGE semantics explained; fold balance verified ("no project >7.4%"). |
| **A-THRESH** | Nested inner-CV threshold selection implemented; "oracle threshold" language removed from main text; deployed-threshold experiment in §5.10. |
| **A-LLM** | Three local LLM baselines evaluated; `tab_llm_baselines.tex` reports F1 0.494/0.377/0.170; OR-fusion hybrid achieves recall ≈0.87. |
| **A-RW** | Trace, SolAgent, LLMSA/LLMDFA, PropertyGPT, GPTScan, iAudit added in §2.2 with reproducibility disclaimers; `tab_related_comparison` updated. |
| **A-SHAP** | SHAP TreeExplainer applied to XGBoost; bee-swarm + waterfall plots generated from strict CSV; 3-step triage workflow in §6.7. |
| **A-CLAIMS** | "Auditor-actionable" replaced with "feature-level attribution"; SHAP-not-line-level stated everywhere. |
| **A-TASK** | Task Definition subsection §3.1 with explicit per-dataset label semantics. |
| **A-TRIM** | Introduction restructured into compact numbered list, verbosity reduced. |

### ⚠️ Partially Addressed

| Action | Gap |
|--------|-----|
| **A-FORGE-3** (re-run ALL experiments on strict) | ✅ ML headlines on strict. ❌ DL baselines on legacy. ❌ Ablations on legacy. ❌ Cross-dataset on legacy. ❌ Significance test on legacy. ❌ Efficiency table on legacy. ❌ Two-stage and ensemble on legacy. |
| **A-PARSE-1** (Tree-sitter migration) | ❌ Paper still uses `python-solidity-parser`. Mitigated by zero-NaN audit + Slither correlation, but the original migration was not done. |
| **A-PARSE-5** (Slither correlation at 1,500+ contracts) | ⚠️ Only 293 contracts used (statistically valid but below plan target). |
| **A-PARSE-2** (per-version parser failure breakdown) | ⚠️ Zero NaN reported, but the per-Solidity-version breakdown table from the plan is not included. |

---

## Part 2 — Newly Discovered Gaps (Reviewer Perspective)

### 🔴 GAP-1 — DL Baselines on Legacy Labels, SCAFE on Strict Labels

**Severity: CRITICAL.**

- `tab_model_comparison` (SCAFE headline): strict, 22,701 contracts.
- `tab_dl_baselines` Merged section: legacy, 23,824 contracts.

A reviewer will say: *"The PR-AUC gap (SCAFE 0.822 vs UniXcoder 0.645 = 17.7 pp) is almost certainly a labeling artifact, not a model-quality difference."*

**Fix:** Run `scripts/run_dl_strict.py --models unixcoder,codebert` (~12–13h).

---

### 🔴 GAP-2 — PR-AUC Gap May Be Labeling Artifact

**Severity: CRITICAL (same root cause as GAP-1).**

The 17.7 pp PR-AUC advantage is the most prominent number in the paper. Under strict labels, ~2,400 noisy negatives are removed; SCAFE's own PR-AUC improved by +23.5 pp going legacy→strict. We have no idea how much UniXcoder gains.

**Fix:** Same as GAP-1.

---

### 🔴 GAP-3 — Hybrid Ensemble and Two-Stage Pipeline Use Legacy OOF

**Severity: IMPORTANT.**

- `tab_ensemble`: legacy SCAFE OOF + legacy UniXcoder OOF. *Internally consistent* (both legacy) but DL inputs are now stale.
- `tab_two_stage`: claims "97% F1 retention" using legacy UniXcoder as reference. Strict baseline may shift this number.

**Fix:** After GAP-1 strict DL runs are done, re-run `scripts/ensemble_analysis.py` and `scripts/two_stage_pipeline_analysis.py` using strict OOF.

---

### 🔴 GAP-4 — Feature Count Inconsistency: 203 in Methodology vs 292 in Experiments

**Severity: IMPORTANT.**

- §3.4 + Algorithm 1: claim 80 selected structural + 123 semantic = **203 features**.
- Headline experiments (`ml_results_strict_all`): `n_features = 292` (no MI+RF selection applied).
- SHAP `shap_values.npz`: shape (22701, **292**).
- §5.5 ablation: "All features (292 after selection)" — contradicts §3.4.

A reviewer will say: *"Algorithm 1 describes a 203-feature pipeline, but Table 5 reports n=292. This is unexplained."*

**Fix:** Edit §3.4 and Algorithm 1 to honestly describe the actual pipeline. Headline experiments use ALL available 292 source-text features (no MI+RF top-80 cut). The 80-feature ablation in §5.5 is a separate evaluation. Takes 30 min.

---

### 🔴 GAP-5 — Error Analysis on Legacy Data

**Severity: IMPORTANT.**

§6.4 error analysis (FP/FN characteristics) and size-stratified complementarity use legacy OOF predictions. Numbers like "FP mean complexity 1.82, FN mean LOC 285" may shift under strict labels.

**Fix:** After GAP-1, re-run error analysis with strict OOF. ~2h.

---

### 🔴 GAP-6 — Cross-Dataset Table is LEGACY, Paper Text Says STRICT (Inconsistency)

**Severity: CRITICAL — explicit numerical contradiction.**

- Paper §5.2 line 567: *"Combining both datasets under the strict-label scheme (FORGE-vulnerable only) yields the best results. Table~\ref{tab:cross_dataset} reports GradientBoosting with the structural stream alone at ROC-AUC=0.921, F1=0.597"*.
- But `tab_cross_dataset.tex` shows the merged row at N=**23,824** (legacy count) and ROC=0.921 (legacy GBC structural-only).
- Under STRICT structural-only merged, GBC achieves ROC=**0.952** (verified from `ml_results_strict/model_comparison.csv`), not 0.921.

**The table is literally legacy data while the surrounding text claims it's strict.**

**Fix:** Either (a) re-run cross-dataset experiments under strict and update the table, or (b) change the text and table caption to honestly state legacy was used for the cross-dataset transfer narrative. Option (a) preferred. ~1h.

---

### 🔴 GAP-7 — risk_z Baseline ROC-AUC = 0.448 (BELOW Random!) is Unexplained

**Severity: CRITICAL — a reviewer WILL flag this.**

- `tab_model_comparison` and §5.1: *"Baseline (risk\_z): ROC-AUC=0.448"*.
- ROC-AUC < 0.5 means the score is **anti-predictive** (predicts opposite of vulnerability).
- On DAppSCAN-only legacy, risk_z=0.815 (reasonable). Adding FORGE-vulnerable flips it to anti-predictive because FORGE positives have different structural characteristics than the hand-tuned DAppSCAN-derived `risk_z` rule weights.
- The paper just states 0.448 without comment.

A reviewer will say: *"Your rule-based baseline performs worse than random. This is either a bug, a sign the dataset is contaminated, or evidence that the hand-crafted score is not generalizing. Explain."*

**Fix:** Add 1-2 sentences in §5.1 explaining that risk_z weights were calibrated on DAppSCAN patterns and become anti-predictive on FORGE because FORGE positives use distinct patterns. Frame this as **motivation** for learned features, not a weakness. ~10 min.

---

### 🔴 GAP-8 — All §5.5 Ablations Use LEGACY Data (Whole Section Inconsistent with Headlines)

**Severity: CRITICAL — entire ablation section is on a different protocol than the headlines.**

Every ablation in §5.5 uses LEGACY OOF data (numbers in 0.92x range), but the headline §5.1 result is STRICT (0.957):

- **Trigram contribution (line 685):** *"Random Forest drops from ROC-AUC=0.924 to 0.921"* ← LEGACY
- **Sensitivity to k (line 687):** *"k=20: 0.891 → k=306: 0.907"* ← LEGACY
- **Semantic decomposition Table 8 (lines 700-703):** *"Structural only: 0.925, +Both: 0.932"* ← LEGACY
- **TF-IDF + LogReg control (line 708):** *"reaches ROC-AUC=0.910 vs full SCAFE 0.932"* ← LEGACY (compares against legacy SCAFE)
- **Identifier obfuscation (line 710):** *"degrades from 0.928 to 0.917"* ← LEGACY
- **RQ5 conclusions (line 712):** *"9-feature summary reaches 0.856"* ← LEGACY

A reviewer will see Table 4 saying 0.957 and Table 8 saying 0.932 and ask: *"Why does your semantic decomposition show a different baseline than your headline?"* The answer ("the ablations are on the legacy corpus for historical comparability") needs to either be stated explicitly OR the ablations need to be re-run.

**Two fix options:**

- **Option A (preferred, ~6h CPU):** Re-run `scripts/run_scafe_sem_experiments.py` and `scripts/scafe_sem_analysis.py` on strict labels to update Table 8 and the surrounding ablations. Also re-run trigram and obfuscation ablations.
- **Option B (~30 min):** Add a single explicit caveat sentence at the start of §5.5: *"For consistency with prior submissions and to enable direct comparison against the original DL baseline results, the ablations in this section use the legacy merged corpus (23,824 contracts) unless otherwise noted. The strict-label headline (Section 5.1) is reproduced at the legacy protocol as the 'Both' configuration of Table 8."*

Option A is much stronger and is the only one that fully closes this gap.

---

### 🟠 GAP-9 — Significance Table Compares LEGACY Vote vs LEGACY DL, but Paper Headlines STRICT

**Severity: IMPORTANT.**

`tab_significance` (Table 7) tests "Ours (Vote ensemble)" vs TextCNN/BiLSTM/CodeBERT/UniXcoder. Vote PR-AUC=0.589 indicates LEGACY (strict Vote PR-AUC ~0.82). So the significance test is between LEGACY Vote and LEGACY DL — internally fair, but the paper headlines STRICT Vote (0.822).

A reviewer will ask: *"Why is the statistical significance assessment on the legacy protocol when the headline results are strict?"*

**Fix:** Same approach as GAP-8 — re-run significance tests after strict DL is available. ~30 min after GAP-1.

---

### 🟠 GAP-10 — Efficiency Table Uses Legacy N=23,824

**Severity: MODERATE.**

`tab_efficiency` caption: *"5-fold GroupKFold CV, merged data, **23,824 contracts**"* — legacy size. Vote ROC=0.932 (legacy).

This is a minor but visible inconsistency: the paper headlines 22,701 contracts but the efficiency table uses 23,824.

**Fix:** Re-time the strict-corpus pipeline (~25 min recompute) and update the table; OR update caption to clarify. ~15 min.

---

### 🟠 GAP-11 — Comprehensive Summary Table EXPLICITLY Mixes Protocols

**Severity: MODERATE — but already footnoted.**

`tab_comprehensive_summary` has SCAFE rows labeled "Merged (strict)" and DL rows labeled "Merged (legacy)" with footnote $^{\ddagger}$: *"Results from original submission (legacy label scheme)."*

The footnote helps, but the visual side-by-side comparison still looks like a fair head-to-head. A reviewer might call it "intentionally misleading."

**Fix:** After GAP-1, update DL rows to strict. Also add a horizontal line separator between the strict and legacy sections to make the visual division clear. ~10 min after GAP-1.

---

### 🟠 GAP-12 — Cross-Dataset Uses STRUCTURAL ONLY (Weakened Pipeline)

**Severity: MODERATE.**

`tab_cross_dataset` caption explicitly says *"using SCAFE's structural-stream features"* — but the paper's headline pipeline is the **full dual-stream**. A reviewer will ask: *"Why is cross-dataset transfer evaluated on a weakened structural-only model rather than the full pipeline?"*

The likely reason: it was easier to run cross-dataset on a simpler config. But it's not adequately defended.

**Fix:** Either (a) re-run cross-dataset transfer with the full dual-stream pipeline (best), or (b) add 1 sentence explaining the design choice. ~1h for (a), 5 min for (b).

---

### 🟠 GAP-13 — SHAP Analysis: Which Protocol and Model?

**Severity: MODERATE.**

§5.7 SHAP analysis says: *"we apply SHAP TreeExplainer to the trained soft-voting ensemble, specifically to its XGBoost constituent."*

- Verified: `run_shap_analysis.py` uses `unified_contract_eval_scored_strict.csv`. ✅ STRICT.
- Model: XGBoost on strict (so this is consistent with headlines).

**But:** the SHAP figure captions and §5.7 text do not state explicitly that strict labels are used. A reviewer will ask.

**Fix:** Add 1 sentence in §5.7: *"SHAP values are computed for the XGBoost classifier trained on the strict-label corpus (22,701 contracts), matching the headline configuration in Table 4."* ~5 min.

---

### 🟡 GAP-14 — "24× Cost Advantage" Based on Apple MPS Only

**Severity: LOW — already partially caveated.**

The "24× lower cost" claim is repeated throughout the paper. The paper does say *"the ratio would be lower on NVIDIA CUDA hardware"* — but never says *how* much lower. A reviewer with NVIDIA hardware will challenge this.

**Fix:** Quote a published UniXcoder fine-tuning runtime on NVIDIA V100/A100 if available, and state the worst-case range: *"On NVIDIA A100, published fine-tuning runtimes for UniXcoder-base on similar corpora are 2–3h, suggesting a 10–15× cost ratio relative to SCAFE on a single CPU."* ~15 min.

---

### 🟡 GAP-15 — Deployed-Threshold Experiment Is Simulated, Not True Held-Out

**Severity: LOW — but R1 specifically asked for a "frozen calibrated threshold applied to test set" (R1-39, R1-40).**

§5.10 says: *"we simulated this protocol using the 5-fold OOF predicted probabilities of XGBoost on the strict-label corpus"*. This is a simulation using OOF, not a true 20% held-out calibration split.

R1's exact words: *"Tune the threshold on an isolated validation set... and apply that frozen, calibrated threshold to the test set."*

A picky reviewer will say: *"You promised a frozen calibration set in the revision plan. The simulation is OK but does not literally satisfy what was asked."*

**Fix:** Either (a) run an actual 80/20 project-grouped split, calibrate threshold on 20%, apply to 80% test, report the F1 — ~30 min CPU work, OR (b) defend the simulation more rigorously in the text. ~30 min.

---

### 🟡 GAP-16 — Hybrid α=0.08 Means SCAFE Adds Only 8% Weight; Reviewer Could Reframe

**Severity: LOW — narrative risk.**

§5.6 RQ6: *"Merged dataset. The weighted average (α ≈ 0.08, giving dominant weight to UniXcoder) reaches ROC-AUC=0.959"*.

A cynical reviewer can flip this: *"With α=0.08, SCAFE contributes only 8% of the prediction weight. The hybrid ensemble result is effectively UniXcoder with a tiny SCAFE perturbation."*

**Fix:** Frame it correctly: *"Even an 8% SCAFE weighting yields +0.6 pp ROC-AUC and +1.7 pp PR-AUC, showing that SCAFE's signal is sufficiently complementary to UniXcoder that even a small weight produces measurable gain."* ~5 min.

---

### 🟡 GAP-17 — AccessControlVuln Sanity Check Trivially Succeeds (F1=1.0)

**Severity: LOW — already disclosed in caption.**

`tab_cross_dataset` shows F1=1.0 on AccessControlVuln with footnote: *"ROC/PR-AUC undefined; F1 trivially 1.0 since FP=0"*. A picky reviewer might say *"a positive-only sanity check is not a real evaluation."*

**Fix:** No code change needed; just verify the text in §5.2 acknowledges this clearly. ~5 min check.

---

### 🟡 GAP-18 — Repo Anonymity Not Verified

**Severity: LOW — important to check before final submission.**

The LaTeX file is clean (no author names). But:
- The local repo has commit history with author names. The 4open.science anonymous repo should be regenerated to strip git history.
- File headers or comments in `src/` may contain author info.

**Fix:** Verify by visiting `https://anonymous.4open.science/r/scafepapercode/` and checking for: real names, emails, institution URLs, or non-anonymized commit history. Re-upload if needed. ~30 min.

---

## Part 3 — End-to-End Execution Plan

This plan is **designed to be fully automatable**. There is one upfront preparation step (Phase A0) that I (the assistant) must complete before the user says "go." After Phase A0, a single command triggers everything else.

### 🟣 Phase A0 — Preparation (one-time, ~2h, no user input needed)

**Why this is needed:** Most existing experiment scripts (`run_final_experiments.py`, `run_scafe_sem_experiments.py`, `run_gilad_experiments.py`, `ensemble_analysis.py`, `two_stage_pipeline_analysis.py`, `error_analysis_vs_codebert.py`) are hardcoded to the legacy `unified_contract_eval_scored_v2.csv` and include FORGE-unknown contracts. To run all §5.5 ablations, ensembles, error analysis, and cross-dataset experiments on the strict label scheme, I must first create strict-aware variants OR patch the existing scripts to accept a `--protocol strict|legacy` flag.

**Concrete prep work I'll do upfront:**

| # | File | Action |
|---|------|--------|
| P1 | `scripts/run_final_experiments_strict.py` | Strict variant: reads strict CSV, drops FORGE-unknown, writes OOF as `ml_{m}_strict_merged.csv` + trigram ablation as JSON |
| P2 | `scripts/run_scafe_sem_strict.py` | Strict variant of `run_scafe_sem_experiments.py` for semantic decomposition Table 8 (structural-only / +Id / +LSA / +Both) |
| P3 | `scripts/run_gilad_strict.py` | Strict variants of TF-IDF+LR control and identifier obfuscation experiments |
| P4 | Patch `scripts/ensemble_analysis.py` | Add `--protocol strict\|legacy` flag; strict path uses `*_strict_merged.csv` OOF files |
| P5 | Patch `scripts/two_stage_pipeline_analysis.py` | Same `--protocol` flag |
| P6 | Patch `scripts/error_analysis_vs_codebert.py` | Same `--protocol` flag |
| P7 | `scripts/run_cross_dataset_strict.py` | Cross-dataset transfer on strict labels with full dual-stream pipeline (GAP-12) |
| P8 | `scripts/run_deployed_threshold_holdout.py` | True 80/20 project-grouped held-out calibration (GAP-15) |
| P9 | `scripts/run_significance_strict.py` | Pairwise Welch's t-tests on strict OOF (GAP-9) |
| P10 | `scripts/finalize_strict_submission.sh` | **Master orchestrator** that runs everything in order. This is what "go" triggers. |
| P11 | `scripts/update_paper_tables_strict.py` | Parses all strict JSONs and rewrites every affected `.tex` table |
| P12 | `scripts/check_anonymity.py` | Scans repo + paper for author info and verifies 4open.science upload structure (GAP-18 prep) |

After Phase A0, all 18 gaps are addressable by a single `bash scripts/finalize_strict_submission.sh` invocation that runs sequentially: DL strict → all ablation re-runs → all table updates → recompile PDF.

### 🔴 Phase A: Critical fixes BEFORE submission (~14h wall-clock, mostly overnight)

**Phase A1 — Start tonight (overnight, fully resumable):**

```bash
cd /Users/tomer.doitshman/PycharmProjects/sc_review
source venv/bin/activate
nohup python3 -u scripts/run_dl_strict.py --models unixcoder,codebert \
    > /tmp/dl_strict.log 2>&1 &
```
**Resolves GAP-1, GAP-2** + provides data for GAP-3, GAP-5, GAP-8 (DL portions), GAP-9, GAP-11, GAP-13 verification.

**Phase A2 — Independent fixes while DL runs (~1.5h, can do tonight):**

| ID | Task | Effort |
|----|------|--------|
| GAP-4 | Update §3.4 + Algorithm 1: 203 → 292; describe pipeline accurately | 30 min |
| GAP-6 | Fix cross-dataset table/text mismatch (either re-run or relabel) | 1h |
| GAP-7 | Add 1-2 sentence explanation for risk_z=0.448 anti-predictive | 10 min |
| GAP-13 | Add 1 sentence to §5.7 stating SHAP is on strict | 5 min |
| GAP-16 | Reframe α=0.08 hybrid weighting positively | 5 min |
| GAP-17 | Verify AccessControlVuln framing in §5.2 | 5 min |

### 🟠 Phase B: After overnight DL is done (~6h work)

| ID | Task | Effort |
|----|------|--------|
| GAP-1, GAP-11 | Update `tab_dl_baselines` and `tab_comprehensive_summary` with strict UniXcoder + CodeBERT | 30 min |
| GAP-3 | Re-run `ensemble_analysis.py` + `two_stage_pipeline_analysis.py` on strict OOF | 1h |
| GAP-8 | Re-run all §5.5 ablations on strict (semantic decomposition, trigram, k-sensitivity, TF-IDF control, obfuscation) | 4h CPU |
| GAP-9 | Re-run significance tests | 30 min |
| GAP-5 | Re-run error analysis and size-stratified complementarity | 2h |
| GAP-12 | Re-run cross-dataset with full dual-stream OR caveat | 1h or 5 min |
| GAP-10 | Re-time efficiency table or update caption | 15 min |
| Update all narrative text in §5.3, §6.1, §7 with strict numbers | 1h |
| Recompile PDF, verify no broken refs, no leftover legacy numbers | 30 min |

### 🟡 Phase C: Polish (~1h)

| ID | Task | Effort |
|----|------|--------|
| GAP-14 | Quote NVIDIA CUDA runtimes for "24×" claim | 15 min |
| GAP-15 | Defend simulated deployed-threshold OR run true held-out | 30 min |
| GAP-18 | Verify 4open.science anonymity | 30 min |
| Final anonymity check across paper + repo | 15 min |

---

## Part 4 — Updated Reviewer Mindset Checklist

A cynical R1 second-round reading the revision will check:

- [ ] **GAP-1, GAP-2:** "Did they re-run DL baselines on the same strict corpus they used for SCAFE?" → Must be YES after Phase A.
- [ ] **GAP-4:** "Is the feature count in the methodology consistent with what was actually run?" → Must be YES after A2.
- [ ] **GAP-6:** "Does the cross-dataset table use the protocol claimed by the surrounding text?" → Must be YES after A2.
- [ ] **GAP-7:** "Why is the rule-based baseline below random?" → Must be explained.
- [ ] **GAP-8:** "Why are the ablations on a different protocol than the headlines?" → Must be addressed (re-run or caveat).
- [ ] **GAP-9:** "Why is statistical significance on the legacy protocol?" → Must be re-run.
- [ ] **GAP-3, GAP-5:** "Is the ensemble fair? Is error analysis on the right protocol?" → Must be YES after Phase B.
- [ ] **GAP-13:** "Is SHAP on the same data as the headlines?" → Must say YES explicitly.
- [ ] "Are all 85 original concerns addressed?" → YES per REVISION_PLAN.md §4 verification.
- [ ] "Is the paper anonymized (incl. repo)?" → Must be YES after GAP-18.

---

## Part 5 — Summary Table (All 18 Gaps)

| Gap | Severity | Resolved by | Effort | Blocks submission? |
|-----|----------|-------------|--------|---------------------|
| GAP-1: DL on legacy labels | 🔴 Critical | Overnight DL run | 13h GPU | **YES** |
| GAP-2: PR-AUC may be artifact | 🔴 Critical | Resolved by GAP-1 | — | **YES** |
| GAP-3: Ensemble/two-stage on legacy OOF | 🔴 Important | Re-run analyses after GAP-1 | 1h | **YES** |
| GAP-4: 203 vs 292 feature count | 🔴 Important | Edit §3.4 + Algorithm 1 | 30 min | **YES** |
| GAP-5: Error analysis on legacy | 🔴 Important | Re-run after GAP-1 | 2h | **YES** |
| GAP-6: Cross-dataset table N=23,824 vs text "strict" | 🔴 Critical | Re-run or relabel | 1h | **YES** |
| GAP-7: risk_z = 0.448 below random unexplained | 🔴 Critical | 1-2 sentence explanation | 10 min | **YES** |
| GAP-8: All §5.5 ablations on legacy | 🔴 Critical | Re-run on strict OR explicit caveat | 4h or 30 min | **YES** |
| GAP-9: Significance test on legacy | 🟠 Important | Re-run after GAP-1 | 30 min | Yes |
| GAP-10: Efficiency table N=23,824 | 🟠 Moderate | Re-time or caveat | 15 min | No |
| GAP-11: Comprehensive table mixes protocols | 🟠 Moderate | Update after GAP-1 | 10 min | No |
| GAP-12: Cross-dataset structural-only | 🟠 Moderate | Re-run dual-stream or caveat | 1h or 5 min | Borderline |
| GAP-13: SHAP protocol not stated | 🟠 Moderate | 1 sentence | 5 min | No |
| GAP-14: 24× cost based on Apple MPS only | 🟡 Low | NVIDIA reference quote | 15 min | No |
| GAP-15: Deployed threshold is simulated | 🟡 Low | Defend or true held-out | 30 min | No |
| GAP-16: α=0.08 hybrid framing | 🟡 Low | Reframe sentence | 5 min | No |
| GAP-17: AccessControlVuln F1=1.0 trivial | 🟡 Low | Verify framing | 5 min | No |
| GAP-18: Repo anonymity | 🟡 Low | Verify 4open.science | 30 min | Cosmetic |

**Total effort if all gaps addressed:** ~14h overnight + ~9h human work (much can run while DL trains).

---

## Part 6 — End-to-End Automation

After Phase A0 is complete, the entire workflow becomes:

```bash
bash scripts/finalize_strict_submission.sh
```

This single invocation does **all of the following sequentially, with no human input**:

1. **Sanity checks** — strict CSV exists, all scripts present, Python venv active, GPU/MPS available.
2. **Phase A1 (overnight, ~13h):** Run `run_dl_strict.py --models unixcoder,codebert` to produce strict DL OOF.
3. **Phase A2 text edits (~5 min):** Apply GAP-4, GAP-6, GAP-7, GAP-13, GAP-16, GAP-17 edits to `.tex` via `update_paper_tables_strict.py` text-patch routine.
4. **Phase B re-runs (~6h CPU):**
   - `run_final_experiments_strict.py` → strict ML OOF + trigram ablation
   - `run_scafe_sem_strict.py` → strict semantic decomposition Table 8
   - `run_gilad_strict.py` → strict TF-IDF+LR + strict obfuscation experiment
   - `ensemble_analysis.py --protocol strict` → strict ensemble Table 9
   - `two_stage_pipeline_analysis.py --protocol strict` → strict two-stage Table 10
   - `error_analysis_vs_codebert.py --protocol strict` → strict error analysis
   - `run_cross_dataset_strict.py` → strict full-dual-stream cross-dataset Table 6
   - `run_significance_strict.py` → strict significance Table 7
   - `run_deployed_threshold_holdout.py` → true 80/20 deployed threshold (GAP-15)
5. **Phase B table updates (~5 min):** `update_paper_tables_strict.py` parses all new JSONs and overwrites:
   - `tab_dl_baselines.tex`, `tab_comprehensive_summary.tex`, `tab_ensemble.tex`, `tab_two_stage.tex` (if exists)
   - `tab_semantic_ablation.tex`, `tab_significance.tex`, `tab_cross_dataset.tex`, `tab_efficiency.tex`
   - Inline numbers in §5.5 ablation paragraphs (trigram, k-sensitivity, TF-IDF, obfuscation)
6. **Phase C polish (~10 min):**
   - `check_anonymity.py` — scans paper + repo for author info
   - Compile PDF via `pdflatex` + `bibtex` + `pdflatex` ×2
   - Verify zero undefined refs, zero "Warning" log lines
7. **Final report:** dump to `plans/EXECUTION_REPORT.md` summarizing every change, every metric delta, every table updated.

**The user's role after saying "go":** zero. The script logs everything to `/tmp/finalize_strict.log` and notifies on completion.

**Single failure-mode to be aware of:** if UniXcoder MPS training crashes mid-fold (rare but possible), the orchestrator pauses with a clear error and instructions to resume. No silent corruption.

---

## Part 7 — Items That Will Still Require User Input (Honest Disclosure)

Even with full end-to-end automation, three items remain that I cannot fully automate without user involvement:

1. **GAP-14 (NVIDIA cost reference):** I can web-search for published UniXcoder fine-tuning runtimes on A100/V100 and propose a citation, but the user should verify the citation is appropriate. ~5 min user review.

2. **GAP-18 (4open.science anonymity):** I can produce a clean export of the repo with stripped git history into `/tmp/scafe-anon-export/`, and write a script to verify no author info is present. But the actual upload to anonymous.4open.science requires the user's browser session. ~10 min user upload.

3. **Final PDF visual review:** After all auto-updates, the user should skim the PDF once for any layout issue the auto-update couldn't anticipate (e.g., a long table running off-page). ~15 min review.

Everything else — every number, every table, every experiment, every text edit — runs automatically.

---

*End of v2 audit. Updated after deep review of paper + data + ablation tables + cross-dataset table + significance table + efficiency table.*
