"""
Generate updated LaTeX tables from ML experiment results.

Reads:
  data/derived/ml_results_strict_all/model_comparison.csv  (all feature set)
  data/derived/ml_results_strict/model_comparison.csv     (static_only, components_only)
  data/derived/llm_results/*_metrics.json                 (LLM results)

Writes:
  figures/tab_model_comparison.tex
  figures/tab_comprehensive_summary.tex (updated rows only)
  figures/tab_llm_baselines.tex

Usage:
    python scripts/update_paper_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_FIG = _REPO.parent / "sc_risk_score_overleaf" / "figures"
_ML_ALL = _REPO / "data" / "derived" / "ml_results_strict_all" / "model_comparison.csv"
_ML_ABL = _REPO / "data" / "derived" / "ml_results_strict" / "model_comparison.csv"
_LLM_DIR = _REPO / "data" / "derived" / "llm_results"


def fmt(mean: float, std: float | None = None) -> str:
    if std is not None:
        return f"${mean:.4f}\\pm{std:.4f}$"
    return f"${mean:.4f}$"


def bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def generate_model_comparison_table(df_all: pd.DataFrame) -> str:
    """Generate tab_model_comparison.tex from all-feature results."""
    # Show top models sorted by ROC-AUC
    df = df_all[df_all["feature_set"] == "all"].copy()
    df = df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)

    # Mark best per column
    best_roc = df["roc_auc_mean"].max()
    best_pr = df["pr_auc_mean"].max()
    best_f1 = df["f1_mean"].max()
    best_prec = df["precision_mean"].max()
    best_rec = df["recall_mean"].max()

    rows = []
    for _, r in df.iterrows():
        if r["family"] == "baseline":
            continue  # baseline shown separately

        roc = f"${r['roc_auc_mean']:.4f}\\pm{r['roc_auc_std']:.4f}$"
        pr = f"${r['pr_auc_mean']:.4f}\\pm{r['pr_auc_std']:.4f}$"
        f1 = f"${r['f1_mean']:.4f}\\pm{r['f1_std']:.4f}$"
        prec = f"${r['precision_mean']:.4f}$"
        rec = f"${r['recall_mean']:.4f}$"

        if abs(r['roc_auc_mean'] - best_roc) < 1e-6:
            roc = f"\\textbf{{{r['roc_auc_mean']:.4f}}}$\\pm{r['roc_auc_std']:.4f}$"
            roc = f"$\\mathbf{{{r['roc_auc_mean']:.4f}}}\\pm{r['roc_auc_std']:.4f}$"
        if abs(r['pr_auc_mean'] - best_pr) < 1e-6:
            pr = f"$\\mathbf{{{r['pr_auc_mean']:.4f}}}\\pm{r['pr_auc_std']:.4f}$"
        if abs(r['f1_mean'] - best_f1) < 1e-6:
            f1 = f"$\\mathbf{{{r['f1_mean']:.4f}}}\\pm{r['f1_std']:.4f}$"

        name = r["display_name"]
        rows.append(f"Merged (strict) & {name} & {roc} & {pr} & {f1} & {prec} & {rec} \\\\")

    # Add baseline row
    bl = df_all[df_all["family"] == "baseline"].iloc[0]
    rows_with_bl = [
        f"Merged (strict) & Baseline (risk\\_z) & ${bl['roc_auc_mean']:.4f}\\pm{bl['roc_auc_std']:.4f}$ "
        f"& ${bl['pr_auc_mean']:.4f}\\pm{bl['pr_auc_std']:.4f}$ "
        f"& ${bl['f1_mean']:.4f}\\pm{bl['f1_std']:.4f}$ "
        f"& ${bl['precision_mean']:.4f}$ & ${bl['recall_mean']:.4f}$ \\\\"
    ] + rows

    table = r"""\begin{table}[t]
\centering
\caption{Classification performance on merged strict-label dataset (DAppSCAN + FORGE-vulnerable, 22{,}701 contracts; 5-fold GroupKFold CV, mean $\pm$ std). Threshold selected via nested inner GroupKFold on training fold only. Best values in \textbf{bold}.}
\label{tab:model_comparison}
\setlength{\tabcolsep}{3pt}
\small
\begin{tabular}{ll ccccc}
\toprule
\textbf{Training} & \textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} \\
\midrule
""" + "\n".join(rows_with_bl) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_llm_table(llm_results: list[dict]) -> str:
    """Generate tab_llm_baselines.tex from LLM metrics."""
    if not llm_results:
        return None

    model_display = {
        "qwen2.5-coder:7b-instruct": ("Qwen2.5-Coder-7B-Instruct", r"\checkmark"),
        "codellama:latest": ("CodeLlama-7B", r"\checkmark"),
        "qwen2.5:7b": ("Qwen2.5-7B (general)", r"\ding{55}"),
    }

    rows = []
    for r in sorted(llm_results, key=lambda x: -x.get("f1_mean", 0)):
        model = r["model"]
        disp, code_mark = model_display.get(model, (model, "--"))
        f1 = f"${r['f1_mean']:.4f}\\pm{r['f1_std']:.4f}$"
        prec = f"${r['precision_mean']:.4f}$"
        rec = f"${r['recall_mean']:.4f}$"
        trunc = f"{r.get('truncation_rate', 0)*100:.1f}\\%"
        n = r.get("n_contracts", "?")
        rows.append(f"{disp} & {code_mark} & {f1} & {prec} & {rec} & {n} & {trunc} \\\\")

    table = r"""\begin{table}[t]
\centering
\caption{Zero-shot LLM baselines on strict-label stratified evaluation set (all positives + equal negatives; 5-fold GroupKFold CV). Models served locally via Ollama (no API). ``Code-spec.'' = code-specialized pre-training. $N$ = number of evaluated contracts. Trunc.\ = fraction of contracts truncated at """ + f"{24000}" + r""" characters.}
\label{tab:llm_baselines}
\setlength{\tabcolsep}{3pt}
\small
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Code-} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & $N$ & \textbf{Trunc.} \\
 & \textbf{spec.} & & & & & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return table


def main():
    print("Updating paper tables...")

    # Load results
    llm_results = []
    for f in sorted(_LLM_DIR.glob("*_metrics.json")):
        with f.open() as fp:
            llm_results.append(json.load(fp))
        print(f"  Loaded LLM: {f.name}")

    if _ML_ALL.exists():
        df_all = pd.read_csv(_ML_ALL)
        print(f"  Loaded ML (all): {len(df_all)} rows")

        # Generate model comparison table
        table = generate_model_comparison_table(df_all)
        out_path = _FIG / "tab_model_comparison.tex"
        out_path.write_text(table)
        print(f"  Written: {out_path}")

    if llm_results:
        table = generate_llm_table(llm_results)
        if table:
            out_path = _FIG / "tab_llm_baselines.tex"
            out_path.write_text(table)
            print(f"  Written: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
