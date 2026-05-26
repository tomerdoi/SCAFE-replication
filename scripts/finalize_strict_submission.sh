#!/usr/bin/env bash
# ============================================================================
# Master orchestrator: end-to-end STRICT submission pipeline.
#
# Runs all stages of the SUBMISSION_READINESS_PLAN.md in dependency order,
# with stage-level checkpointing.  Each stage is a Python script that
# writes a marker file under data/derived/.checkpoints/ on success.
# Re-running the orchestrator skips stages that already have markers,
# so you can interrupt with Ctrl-C and resume by re-running this script.
#
# Usage:
#   ./scripts/finalize_strict_submission.sh             # full pipeline
#   ./scripts/finalize_strict_submission.sh --force     # ignore checkpoints
#   ./scripts/finalize_strict_submission.sh --stages a,b # run only these
#   ./scripts/finalize_strict_submission.sh --dry-run   # show what would run
#
# Stages:
#   A  Strict SCAFE structural ML (run_final_experiments_strict)
#   B  Strict SCAFE-Sem decomposition (run_scafe_sem_strict)
#   C  Strict Gilad experiments (run_gilad_strict)
#   D  Strict DL baselines (UniXcoder, CodeBERT, TextCNN, BiLSTM)
#   E  Strict ensemble analysis
#   F  Strict two-stage pipeline analysis
#   G  Strict error analysis (SCAFE-Sem vs CodeBERT)
#   H  Strict cross-dataset evaluation
#   I  Deployed-threshold held-out (true 80/20)
#   J  Significance tests
#   K  Update LaTeX tables
#   L  Anonymity check
#
# Notes:
#   * Long stages (D = DL, ~12 h on M-series MPS) re-use their own internal
#     fold-level checkpoints, so even within a stage you can resume safely.
#   * Each stage prints its own progress; this script captures stdout/stderr
#     to logs/strict_<stage>.log.
# ============================================================================
set -euo pipefail
shopt -s expand_aliases

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${VENV}" ]]; then
    # Fall back to system python3
    VENV="$(command -v python3)"
    echo "[orchestrator] venv not found at ${REPO_ROOT}/venv/bin/python; using ${VENV}" >&2
fi

CKPT_DIR="${REPO_ROOT}/data/derived/.checkpoints"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# ── Argument parsing ─────────────────────────────────────────────────────────
FORCE=0
DRY_RUN=0
STAGE_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    FORCE=1;       shift ;;
        --dry-run)  DRY_RUN=1;     shift ;;
        --stages)   STAGE_FILTER="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^# Stages/p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# ── Stage runner ─────────────────────────────────────────────────────────────
# stage_id, description, command(s)
run_stage() {
    local id="$1"
    local desc="$2"
    shift 2

    if [[ -n "${STAGE_FILTER}" ]] && ! [[ ",${STAGE_FILTER}," == *",${id},"* ]]; then
        echo "[skip ${id}] not in --stages filter"
        return 0
    fi

    local marker="${CKPT_DIR}/stage_${id}.done"
    local log="${LOG_DIR}/strict_${id}.log"

    echo
    echo "==============================================================================="
    printf "[stage %s] %s\n" "${id}" "${desc}"
    echo "==============================================================================="
    if [[ -f "${marker}" && ${FORCE} -eq 0 ]]; then
        echo "  Already done (marker: ${marker}).  Use --force to re-run."
        return 0
    fi

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] would execute:"
        for cmd in "$@"; do printf "    %s\n" "${cmd}"; done
        return 0
    fi

    local start=$(date +%s)
    : > "${log}"
    for cmd in "$@"; do
        echo "  > ${cmd}" | tee -a "${log}"
        # shellcheck disable=SC2086
        bash -c "${cmd}" 2>&1 | tee -a "${log}"
        local rc=${PIPESTATUS[0]}
        if [[ ${rc} -ne 0 ]]; then
            echo "  ✗ Stage ${id} failed (rc=${rc}).  Log: ${log}"
            exit "${rc}"
        fi
    done
    local elapsed=$(($(date +%s) - start))
    date > "${marker}"
    echo "  ✓ Stage ${id} done in ${elapsed}s.  Log: ${log}"
}

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "==============================================================================="
echo "STRICT SUBMISSION PIPELINE"
echo "==============================================================================="
echo "  Repo:        ${REPO_ROOT}"
echo "  Python:      ${VENV}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Force:       ${FORCE}"
echo "  Dry-run:     ${DRY_RUN}"
[[ -n "${STAGE_FILTER}" ]] && echo "  Stage filter: ${STAGE_FILTER}"

if [[ ! -f "${REPO_ROOT}/data/derived/unified_contract_eval_scored_strict.csv" ]]; then
    echo
    echo "  ✗ Strict CSV not found: data/derived/unified_contract_eval_scored_strict.csv"
    echo "    Generate it first with the labelling pipeline."
    exit 3
fi

# ── Stages ───────────────────────────────────────────────────────────────────
# A: Strict ML (~30-60 min)
run_stage A "Strict SCAFE structural ML + trigram ablation + k-sweep" \
  "${VENV} -u scripts/run_final_experiments_strict.py"

# B: Strict SCAFE-Sem decomposition (~1-2 h)
run_stage B "Strict SCAFE-Sem decomposition (Table 8)" \
  "${VENV} -u scripts/run_scafe_sem_strict.py"

# C: Strict Gilad experiments (~30-60 min)
run_stage C "Strict TF-IDF+LR baseline + identifier obfuscation" \
  "${VENV} -u scripts/run_gilad_strict.py"

# D: Strict DL baselines (~12 h)
run_stage D "Strict DL baselines (UniXcoder, CodeBERT, TextCNN, BiLSTM)" \
  "${VENV} -u scripts/run_dl_strict.py --models unixcoder,codebert,textcnn,bilstm"

# E: Strict ensemble analysis
run_stage E "Strict ensemble (weighted-avg + stacking)" \
  "${VENV} -u scripts/ensemble_analysis.py --protocol strict --datasets merged dappscan"

# F: Strict two-stage pipeline
run_stage F "Strict two-stage (SCAFE pre-filter + CodeBERT)" \
  "${VENV} -u scripts/two_stage_pipeline_analysis.py --protocol strict --datasets merged dappscan"

# G: Strict error analysis
run_stage G "Strict error analysis (SCAFE-Sem vs CodeBERT)" \
  "${VENV} -u scripts/error_analysis_vs_codebert.py --protocol strict"

# H: Strict cross-dataset
run_stage H "Strict cross-dataset evaluation (full dual-stream)" \
  "${VENV} -u scripts/run_cross_dataset_strict.py"

# I: Deployed-threshold held-out
run_stage I "Deployed-threshold true 80/20 project-grouped held-out" \
  "${VENV} -u scripts/run_deployed_threshold_holdout.py"

# J: Significance tests
run_stage J "Pairwise Welch's t-tests on strict per-fold scores" \
  "${VENV} -u scripts/run_significance_strict.py"

# K: Update LaTeX tables
run_stage K "Auto-update LaTeX tables in Overleaf repo (companion files only)" \
  "${VENV} -u scripts/update_paper_tables_strict.py"

# L: Anonymity / submission-readiness scan
run_stage L "Anonymity scan (will exit non-zero if findings; non-fatal)" \
  "${VENV} -u scripts/check_anonymity.py --quiet || true"

# ── Final summary ────────────────────────────────────────────────────────────
echo
echo "==============================================================================="
echo "PIPELINE COMPLETE"
echo "==============================================================================="
echo "  Companion tables:   sc_risk_score_overleaf/figures/*_strict_generated.tex"
echo "  Results JSONs:      data/derived/ml_results/*strict*.json"
echo "  OOF predictions:    data/derived/oof_predictions/*strict*.csv"
echo "  Logs:               ${LOG_DIR}/strict_*.log"
echo
echo "Next steps (manual):"
echo "  1. Review companion *_strict_generated.tex files."
echo "  2. Run with --apply if you want the orchestrator to overwrite the"
echo "     canonical figures/*.tex files:"
echo "       ${VENV} scripts/update_paper_tables_strict.py --apply"
echo "  3. Re-compile the paper, eyeball numbers."
echo "  4. Address any anonymity-scan findings."
echo
