#!/usr/bin/env bash
# Run all 3 LLM baselines sequentially (qwen-coder → codellama → qwen-general).
# Each model resumes from checkpoint if interrupted.
# Runs are serialized because Ollama processes one request at a time.
#
# Usage: bash scripts/run_all_llm_baselines.sh

set -e
VENV=venv/bin/python3
SCRIPT=scripts/run_local_llm_baseline.py
LOG_DIR=logs

echo "=== LLM Baseline Pipeline ===" | tee -a $LOG_DIR/llm_all.log
echo "Started: $(date)" | tee -a $LOG_DIR/llm_all.log

echo "[1/3] qwen2.5-coder:7b-instruct (code-specialized)" | tee -a $LOG_DIR/llm_all.log
$VENV -u $SCRIPT --model qwen2.5-coder:7b-instruct --variant zero_shot \
    --max-pos 2200 --max-neg 2200 \
    >> $LOG_DIR/llm_qwen_coder.log 2>&1
echo "[1/3] DONE: $(date)" | tee -a $LOG_DIR/llm_all.log

echo "[2/3] codellama:latest (code LLM reference)" | tee -a $LOG_DIR/llm_all.log
$VENV -u $SCRIPT --model codellama:latest --variant zero_shot \
    --max-pos 2200 --max-neg 2200 \
    >> $LOG_DIR/llm_codellama.log 2>&1
echo "[2/3] DONE: $(date)" | tee -a $LOG_DIR/llm_all.log

echo "[3/3] qwen2.5:7b (general — code-specialization ablation)" | tee -a $LOG_DIR/llm_all.log
$VENV -u $SCRIPT --model qwen2.5:7b --variant zero_shot \
    --max-pos 2200 --max-neg 2200 \
    >> $LOG_DIR/llm_qwen_general.log 2>&1
echo "[3/3] DONE: $(date)" | tee -a $LOG_DIR/llm_all.log

echo "=== All LLM baselines complete: $(date) ===" | tee -a $LOG_DIR/llm_all.log
