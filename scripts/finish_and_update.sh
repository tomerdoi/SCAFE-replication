#!/usr/bin/env bash
# Waits for all 3 LLM models to finish, runs model-1 retry, updates tables, recompiles PDF.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="${PAPER_DIR:-$REPO/../sc_risk_score_overleaf}"
VENV="$REPO/venv/bin/python3"
LOG="$REPO/logs/finish_and_update.log"

echo "=== finish_and_update.sh started: $(date) ===" | tee -a $LOG

# ── Step 1: Wait for qwen2.5 (model 3) to produce its metrics file ──────────
echo "[1/4] Waiting for qwen2.5 metrics..." | tee -a $LOG
while [ ! -f "$REPO/data/derived/llm_results/qwen2_5_7b_zero_shot_metrics.json" ]; do
    sleep 120
done
echo "      qwen2.5 done: $(date)" | tee -a $LOG

# ── Step 2: Wait for the shell pipeline (78475 or its child) to fully exit ───
sleep 30
while pgrep -f "run_all_llm_baselines.sh" > /dev/null 2>&1; do
    sleep 30
done
echo "      Pipeline exited: $(date)" | tee -a $LOG

# ── Step 3: Model-1 retry (fill in None predictions) ────────────────────────
echo "[2/4] Retrying qwen2.5-coder None predictions..." | tee -a $LOG
$VENV -u $REPO/scripts/run_local_llm_baseline.py \
    --model qwen2.5-coder:7b-instruct --variant zero_shot \
    --max-pos 2200 --max-neg 2200 \
    >> $REPO/logs/llm_qwen_coder_retry.log 2>&1
echo "      Retry done: $(date)" | tee -a $LOG

# ── Step 4: Generate updated LaTeX tables ───────────────────────────────────
echo "[3/4] Updating paper tables..." | tee -a $LOG
$VENV $REPO/scripts/update_paper_tables.py >> $LOG 2>&1
echo "      Tables updated: $(date)" | tee -a $LOG

# ── Step 5: Recompile PDF ────────────────────────────────────────────────────
echo "[4/4] Recompiling PDF..." | tee -a $LOG
cd $PAPER
pdflatex -interaction=nonstopmode sc_vulnerability_detection.tex >> $LOG 2>&1
bibtex sc_vulnerability_detection >> $LOG 2>&1
pdflatex -interaction=nonstopmode sc_vulnerability_detection.tex >> $LOG 2>&1
pdflatex -interaction=nonstopmode sc_vulnerability_detection.tex >> $LOG 2>&1
echo "      PDF compiled: $(date)" | tee -a $LOG

echo "=== ALL DONE: $(date) ===" | tee -a $LOG
