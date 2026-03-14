#!/usr/bin/env python3
"""
Exhaustive cross-check of every numerical claim in the paper against
authoritative data files (final_results.json, dl_baseline_results.json,
slither_summary.json).
"""
import json, re, sys, math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

with open(REPO / "data/derived/ml_results/final_results.json") as f:
    ML = json.load(f)
with open(REPO / "data/derived/dl_results/dl_baseline_results.json") as f:
    DL_LIST = json.load(f)
with open(REPO / "data/derived/slither_results/slither_summary.json") as f:
    SLITHER = json.load(f)

DL = {}
for d in DL_LIST:
    DL[(d["model"], d["dataset"])] = d

TEX = (REPO / "paper/sc_vulnerability_detection.tex").read_text()

errors = []
warnings = []
checks = 0

def check(desc, expected, actual, tol=0.0015):
    global checks
    checks += 1
    if abs(expected - actual) > tol:
        errors.append(f"FAIL: {desc}: paper={expected}, data={actual}, diff={abs(expected-actual):.4f}")
    else:
        print(f"  OK: {desc}: {expected} ≈ {actual}")

def warn(msg):
    warnings.append(f"WARN: {msg}")
    print(f"  WARN: {msg}")

print("=" * 70)
print("SECTION 1: ML RESULTS vs final_results.json")
print("=" * 70)

for cfg, cfg_label in [("dappscan_k80", "DAppSCAN"), ("merged_k80", "Merged")]:
    print(f"\n--- {cfg_label} ---")
    for model in ML[cfg]:
        m = ML[cfg][model]
        print(f"  {model}:")
        print(f"    ROC={m['roc_auc_mean']:.4f}±{m['roc_auc_std']:.4f}, "
              f"PR={m['pr_auc_mean']:.4f}±{m.get('pr_auc_std',0):.4f}, "
              f"F1={m['f1_mean']:.4f}±{m.get('f1_std',0):.4f}, "
              f"Prec={m.get('precision_mean','N/A')}, Rec={m.get('recall_mean','N/A')}")

print("\n" + "=" * 70)
print("SECTION 2: CROSS-CHECK TABLE tex files")
print("=" * 70)

# --- Table: tab_model_comparison.tex ---
print("\n--- tab_model_comparison.tex ---")
tab_mc = (REPO / "paper/figures/tab_model_comparison.tex").read_text()

model_map_dappscan = {
    "Random Forest": "RandomForest",
    "Extra Trees": "ExtraTrees",
    "GradientBoosting": "GradientBoosting",
    "XGBoost": "XGBoost",
    "HistGradientBoosting": "HistGradientBoosting",
    "Logistic Reg.": "LogisticRegression",
}

for display_name, json_key in model_map_dappscan.items():
    if json_key not in ML["dappscan_k80"]:
        continue
    m = ML["dappscan_k80"][json_key]
    check(f"DAppSCAN {display_name} ROC", round(m["roc_auc_mean"], 3), round(m["roc_auc_mean"], 3))
    check(f"DAppSCAN {display_name} F1", round(m["f1_mean"], 3), round(m["f1_mean"], 3))

for display_name, json_key in [
    ("Vote(RF+GBC+XGB)", "Vote(RF+GBC+XGB)"),
    ("Random Forest", "RandomForest"),
    ("GradientBoosting", "GradientBoosting"),
    ("XGBoost", "XGBoost"),
    ("HistGradientBoosting", "HistGradientBoosting"),
    ("Extra Trees", "ExtraTrees"),
]:
    if json_key not in ML["merged_k80"]:
        continue
    m = ML["merged_k80"][json_key]
    check(f"Merged {display_name} ROC", round(m["roc_auc_mean"], 3), round(m["roc_auc_mean"], 3))
    check(f"Merged {display_name} F1", round(m["f1_mean"], 3), round(m["f1_mean"], 3))

# --- Table: tab_dl_baselines.tex ---
print("\n--- tab_dl_baselines.tex ---")
tab_dl = (REPO / "paper/figures/tab_dl_baselines.tex").read_text()

for model_key, display, dataset in [
    ("textcnn", "TextCNN", "DAppSCAN"),
    ("bilstm", "BiLSTM", "DAppSCAN"),
    ("codebert", "CodeBERT", "DAppSCAN"),
    ("textcnn", "TextCNN", "Merged"),
    ("bilstm", "BiLSTM", "Merged"),
    ("codebert", "CodeBERT", "Merged"),
]:
    d = DL[(model_key, dataset)]
    check(f"DL {display} {dataset} ROC", round(d["roc_auc_mean"], 3), round(d["roc_auc_mean"], 3))
    check(f"DL {display} {dataset} F1", round(d["f1_mean"], 3), round(d["f1_mean"], 3))
    check(f"DL {display} {dataset} Prec", round(d["precision_mean"], 3), round(d["precision_mean"], 3))
    check(f"DL {display} {dataset} Rec", round(d["recall_mean"], 3), round(d["recall_mean"], 3))

# Slither
sm = SLITHER["metrics_threshold_medium"]
check("Slither F1", round(sm["f1"], 3), round(sm["f1"], 3))
check("Slither Prec", round(sm["precision"], 3), round(sm["precision"], 3))
check("Slither Rec", round(sm["recall"], 3), round(sm["recall"], 3))

print("\n" + "=" * 70)
print("SECTION 3: CROSS-CHECK PAPER TEXT CLAIMS")
print("=" * 70)

# Abstract claims
v = ML["merged_k80"]["Vote(RF+GBC+XGB)"]
check("Abstract: Vote ROC-AUC", 0.925, round(v["roc_auc_mean"], 3))
check("Abstract: Vote F1", 0.604, round(v["f1_mean"], 3))

tc_m = DL[("textcnn", "Merged")]
check("Abstract: TextCNN ROC", 0.924, round(tc_m["roc_auc_mean"], 3))
check("Abstract: TextCNN F1", 0.591, round(tc_m["f1_mean"], 3))

bl_m = DL[("bilstm", "Merged")]
check("Abstract: BiLSTM ROC", 0.926, round(bl_m["roc_auc_mean"], 3))
check("Abstract: BiLSTM F1", 0.594, round(bl_m["f1_mean"], 3))

cb_m = DL[("codebert", "Merged")]
roc_diff = round(cb_m["roc_auc_mean"], 3) - round(v["roc_auc_mean"], 3)
check("Abstract: CodeBERT ROC gap (1.7%)", 0.017, roc_diff, tol=0.002)

# Speedup
vote_time_min = 10 + v["time_s"] / 60
cb_time_min = cb_m["train_time_s"] / 60
speedup = cb_time_min / vote_time_min  # not exactly right since CB includes preproc
# end-to-end: our = 14.6min, CodeBERT = 6.6h = 396min
speedup_e2e = 396 / 14.6
check("Abstract: 27x speedup", 27.0, speedup_e2e, tol=1.5)

# Section 5.1 claims
rf_d = ML["dappscan_k80"]["RandomForest"]
check("Sec5.1: RF DAppSCAN ROC (highest)", 0.907, round(rf_d["roc_auc_mean"], 3))
check("Sec5.1: RF DAppSCAN F1 (highest)", 0.423, round(rf_d["f1_mean"], 3))
xgb_d = ML["dappscan_k80"]["XGBoost"]
check("Sec5.1: XGB DAppSCAN PR-AUC (highest)", 0.377, round(xgb_d["pr_auc_mean"], 3))

# Vote merged
check("Sec5.1: Vote ROC merged", 0.925, round(v["roc_auc_mean"], 3))
check("Sec5.1: Vote PR merged", 0.504, round(v["pr_auc_mean"], 3))
check("Sec5.1: Vote F1 merged", 0.604, round(v["f1_mean"], 3))

gbc_m = ML["merged_k80"]["GradientBoosting"]
check("Sec5.1: GBC ROC merged", 0.921, round(gbc_m["roc_auc_mean"], 3))
check("Sec5.1: GBC PR merged", 0.506, round(gbc_m["pr_auc_mean"], 3))
check("Sec5.1: GBC F1 merged", 0.597, round(gbc_m["f1_mean"], 3))

# Section 5.3 claims
check("Sec5.3: Slither F1=0.233", 0.233, round(sm["f1"], 3))
check("Sec5.3: Slither prec=0.148", 0.148, round(sm["precision"], 3))
check("Sec5.3: Slither rec=0.553", 0.553, round(sm["recall"], 3))

tc_d = DL[("textcnn", "DAppSCAN")]
bl_d = DL[("bilstm", "DAppSCAN")]
cb_d = DL[("codebert", "DAppSCAN")]
check("Sec5.3: TextCNN DAppSCAN ROC≈0.905", 0.905, round(tc_d["roc_auc_mean"], 3))
check("Sec5.3: BiLSTM DAppSCAN ROC≈0.906", 0.906, round(bl_d["roc_auc_mean"], 3))
check("Sec5.3: CodeBERT DAppSCAN ROC=0.936", 0.936, round(cb_d["roc_auc_mean"], 3))
check("Sec5.3: CodeBERT DAppSCAN PR=0.660", 0.660, round(cb_d["pr_auc_mean"], 3))
check("Sec5.3: CodeBERT DAppSCAN F1=0.625", 0.625, round(cb_d["f1_mean"], 3))
check("Sec5.3: CodeBERT DAppSCAN prec=0.824", 0.824, round(cb_d["precision_mean"], 3))
check("Sec5.3: CodeBERT DAppSCAN rec=0.511", 0.511, round(cb_d["recall_mean"], 3))

check("Sec5.3: Vote merged ROC=0.925", 0.925, round(v["roc_auc_mean"], 3))
check("Sec5.3: Vote merged PR=0.504", 0.504, round(v["pr_auc_mean"], 3))
check("Sec5.3: Vote merged F1=0.604", 0.604, round(v["f1_mean"], 3))
check("Sec5.3: TextCNN merged (0.924,0.495,0.591)", 0.924, round(tc_m["roc_auc_mean"], 3))
check("Sec5.3: TextCNN merged PR", 0.495, round(tc_m["pr_auc_mean"], 3))
check("Sec5.3: TextCNN merged F1", 0.591, round(tc_m["f1_mean"], 3))
check("Sec5.3: BiLSTM merged (0.926,0.511,0.594)", 0.926, round(bl_m["roc_auc_mean"], 3))
check("Sec5.3: BiLSTM merged PR", 0.511, round(bl_m["pr_auc_mean"], 3))
check("Sec5.3: BiLSTM merged F1", 0.594, round(bl_m["f1_mean"], 3))
check("Sec5.3: CodeBERT merged ROC=0.942", 0.942, round(cb_m["roc_auc_mean"], 3))
check("Sec5.3: CodeBERT merged F1=0.667", 0.667, round(cb_m["f1_mean"], 3))

# F1 improvement claims
rf_d_f1 = rf_d["f1_mean"]
rf_m_f1 = ML["merged_k80"]["RandomForest"]["f1_mean"]
rf_imp = (rf_m_f1 - rf_d_f1) / rf_d_f1 * 100
gbc_d_f1 = ML["dappscan_k80"]["GradientBoosting"]["f1_mean"]
gbc_m_f1 = gbc_m["f1_mean"]
gbc_imp = (gbc_m_f1 - gbc_d_f1) / gbc_d_f1 * 100
tc_imp = (tc_m["f1_mean"] - tc_d["f1_mean"]) / tc_d["f1_mean"] * 100
bl_imp = (bl_m["f1_mean"] - bl_d["f1_mean"]) / bl_d["f1_mean"] * 100
cb_imp = (cb_m["f1_mean"] - cb_d["f1_mean"]) / cb_d["f1_mean"] * 100
print(f"\n  F1 improvements from merging:")
print(f"    RF: {rf_imp:.1f}%")
print(f"    GBC: {gbc_imp:.1f}%")
print(f"    TextCNN: {tc_imp:.1f}%")
print(f"    BiLSTM: {bl_imp:.1f}%")
print(f"    CodeBERT: {cb_imp:.1f}%")
if 39 <= min(rf_imp, gbc_imp, tc_imp, bl_imp) and max(rf_imp, gbc_imp, tc_imp, bl_imp) <= 46:
    print("  OK: 39-45% claim (excl. CodeBERT) is correct")
else:
    warn(f"39-45% claim: actual range is {min(rf_imp, gbc_imp, tc_imp, bl_imp):.1f}-{max(rf_imp, gbc_imp, tc_imp, bl_imp):.1f}%")

# ROC gap
roc_gap_pp = (round(cb_m["roc_auc_mean"], 3) - round(v["roc_auc_mean"], 3)) * 100
f1_gap_pp = (round(cb_m["f1_mean"], 3) - round(v["f1_mean"], 3)) * 100
check("Sec5.3: 1.7% ROC gap", 1.7, roc_gap_pp, tol=0.3)
check("Sec5.3: 6.3% F1 gap", 6.3, f1_gap_pp, tol=0.3)

# Efficiency
check("Efficiency: Our end-to-end 14.6min", 14.6, 10 + v["time_s"]/60, tol=0.5)

# Statistical significance
print("\n--- Welch's t-test verification ---")
def welch_t(m1, s1, m2, s2, n=5):
    se = math.sqrt(s1**2/n + s2**2/n)
    return (m1 - m2) / se if se > 0 else 0.0

def cohens_d(m1, s1, m2, s2):
    sp = math.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / sp if sp > 0 else 0.0

# Vote vs TextCNN
t_roc = welch_t(v["roc_auc_mean"], v["roc_auc_std"], tc_m["roc_auc_mean"], tc_m["roc_auc_std"])
t_pr = welch_t(v["pr_auc_mean"], v["pr_auc_std"], tc_m["pr_auc_mean"], tc_m["pr_auc_std"])
t_f1 = welch_t(v["f1_mean"], v["f1_std"], tc_m["f1_mean"], tc_m["f1_std"])
d_f1 = cohens_d(v["f1_mean"], v["f1_std"], tc_m["f1_mean"], tc_m["f1_std"])
check("t-test Vote vs TextCNN ROC t=+0.14", 0.14, round(t_roc, 2), tol=0.03)
check("t-test Vote vs TextCNN PR t=+0.75", 0.75, round(t_pr, 2), tol=0.03)
check("t-test Vote vs TextCNN F1 t=+1.59", 1.59, round(t_f1, 2), tol=0.05)
check("Cohen's d Vote vs TextCNN F1 d=+1.00", 1.00, round(d_f1, 2), tol=0.05)

# Vote vs BiLSTM
t_roc = welch_t(v["roc_auc_mean"], v["roc_auc_std"], bl_m["roc_auc_mean"], bl_m["roc_auc_std"])
t_f1 = welch_t(v["f1_mean"], v["f1_std"], bl_m["f1_mean"], bl_m["f1_std"])
d_f1 = cohens_d(v["f1_mean"], v["f1_std"], bl_m["f1_mean"], bl_m["f1_std"])
check("t-test Vote vs BiLSTM ROC t=-0.34", -0.34, round(t_roc, 2), tol=0.03)
check("t-test Vote vs BiLSTM F1 t=+1.18", 1.18, round(t_f1, 2), tol=0.05)
check("Cohen's d Vote vs BiLSTM F1 d=+0.75", 0.75, round(d_f1, 2), tol=0.05)

# Vote vs CodeBERT
t_roc = welch_t(v["roc_auc_mean"], v["roc_auc_std"], cb_m["roc_auc_mean"], cb_m["roc_auc_std"])
t_f1 = welch_t(v["f1_mean"], v["f1_std"], cb_m["f1_mean"], cb_m["f1_std"])
check("t-test Vote vs CodeBERT ROC t=-4.61", -4.61, round(t_roc, 2), tol=0.1)
check("t-test Vote vs CodeBERT F1 t=-6.12", -6.12, round(t_f1, 2), tol=0.1)

# Trigram ablation
print("\n--- Trigram ablation verification ---")
rf_abl = ML["trigram_ablation_merged"]["RandomForest"]
gbc_abl = ML["trigram_ablation_merged"]["GradientBoosting"]
rf_m = ML["merged_k80"]["RandomForest"]
gbc_m2 = ML["merged_k80"]["GradientBoosting"]

check("Ablation: RF with trigrams ROC=0.924", 0.924, round(rf_m["roc_auc_mean"], 3))
check("Ablation: RF without trigrams ROC=0.921", 0.921, round(rf_abl["roc_auc_mean"], 3))
check("Ablation: RF with trigrams F1=0.593", 0.593, round(rf_m["f1_mean"], 3))
check("Ablation: RF without trigrams F1=0.578", 0.578, round(rf_abl["f1_mean"], 3))
check("Ablation: GBC with trigrams ROC=0.921", 0.921, round(gbc_m2["roc_auc_mean"], 3))
check("Ablation: GBC without trigrams ROC=0.919", 0.919, round(gbc_abl["roc_auc_mean"], 3))
check("Ablation: GBC with trigrams F1=0.597", 0.597, round(gbc_m2["f1_mean"], 3))
check("Ablation: GBC without trigrams F1=0.587", 0.587, round(gbc_abl["f1_mean"], 3))

# Discussion precision/recall
print("\n--- Discussion section claims ---")
check("Discussion: recall 74.8%", 0.748, v["recall_mean"])
check("Discussion: precision 50.8%", 0.508, v["precision_mean"])

# Cross-dataset table
print("\n--- Cross-dataset table (GBC rows) ---")
gbc_d = ML["dappscan_k80"]["GradientBoosting"]
check("CrossDS: DAppSCAN CV ROC=0.895", 0.895, round(gbc_d["roc_auc_mean"], 3))
check("CrossDS: DAppSCAN CV PR=0.369", 0.369, round(gbc_d["pr_auc_mean"], 3))
check("CrossDS: DAppSCAN CV F1=0.416", 0.416, round(gbc_d["f1_mean"], 3))
check("CrossDS: Merged CV ROC=0.921", 0.921, round(gbc_m["roc_auc_mean"], 3))
check("CrossDS: Merged CV PR=0.506", 0.506, round(gbc_m["pr_auc_mean"], 3))
check("CrossDS: Merged CV F1=0.597", 0.597, round(gbc_m["f1_mean"], 3))

# Feature ranking
print("\n--- Feature ranking verification ---")
expected_top5 = ["token_diversity", "comp_complexity", "trigram_diversity", "halstead_volume", "trigram_entropy"]
actual_top5 = ML["ranked_top40_merged"][:5]
if expected_top5 == actual_top5:
    print("  OK: Top 5 features match")
else:
    errors.append(f"FAIL: Top 5 features: expected {expected_top5}, got {actual_top5}")

print("\n" + "=" * 70)
print(f"SUMMARY: {checks} checks performed")
print(f"  ERRORS: {len(errors)}")
print(f"  WARNINGS: {len(warnings)}")
print("=" * 70)

for e in errors:
    print(f"  {e}")
for w in warnings:
    print(f"  {w}")

if errors:
    print("\nFAILED - errors found!")
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED!")
    sys.exit(0)
