"""
Docker-based Slither batch analysis on DAppSCAN contracts.

Runs Slither inside a linux/amd64 Docker container to avoid ARM64 solc
compatibility issues. Processes ALL no-import contracts (all solc versions).

Output:
  - data/derived/slither_results/slither_per_contract.csv
  - data/derived/slither_results/slither_vs_ground_truth.csv
  - data/derived/slither_results/slither_summary.json

Usage:
    python -m src.baselines.slither_docker_batch [--max-contracts N] [--timeout SECS]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONTRACTS_DIR = DATA_DIR / "raw" / "DAppSCAN-main" / "DAppSCAN-source" / "contracts"
OUT_DIR = DATA_DIR / "derived" / "slither_results"

DOCKER_IMAGE = "slither-batch"
DOCKER_DATA_MOUNT = "/data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEVERITY_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Informational": 0, "Optimization": 0}

IGNORE_CHECKS = {
    "solc-version", "pragma", "naming-convention", "too-many-digits",
    "assembly", "low-level-calls", "redundant-statements", "dead-code",
    "constable-states", "immutable-states", "external-function",
    "similar-names", "unused-state", "unused-return",
}

import re

def _extract_pragma_info(sol_path: Path) -> Optional[Tuple[str, bool]]:
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"pragma\s+solidity\s+([^;]+);", text)
    if not m:
        return None
    pragma = m.group(1).strip()
    exact = re.match(r"^=?\s*(0\.\d+\.\d+)\s*$", pragma)
    if exact:
        return (exact.group(1), True)
    caret = re.match(r"^\^?\s*(0\.\d+\.\d+)", pragma)
    if caret:
        return (caret.group(1), False)
    gte = re.match(r">=?\s*(0\.\d+\.\d+)", pragma)
    if gte:
        return (gte.group(1), False)
    return None


_SOLC_MAP = {
    "0.4": "0.4.26", "0.5": "0.5.17", "0.6": "0.6.12",
    "0.7": "0.7.6", "0.8": "0.8.10",
}


def _resolve_solc_version(pragma_ver: str, is_exact: bool) -> str:
    if is_exact:
        return pragma_ver
    parts = pragma_ver.split(".")
    if len(parts) >= 3:
        minor_key = f"{parts[0]}.{parts[1]}"
        return _SOLC_MAP.get(minor_key, pragma_ver)
    return pragma_ver


def _has_any_import(sol_path: Path) -> bool:
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    return bool(re.search(r"^\s*import\s", text, re.MULTILINE))


def discover_contracts(contracts_dir: Path) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str]] = []
    all_sol = sorted(contracts_dir.rglob("*.sol"))
    log.info("Total .sol files found: %d", len(all_sol))

    skipped_import = 0
    skipped_pragma = 0

    for sol_path in all_sol:
        if _has_any_import(sol_path):
            skipped_import += 1
            continue
        info = _extract_pragma_info(sol_path)
        if not info:
            skipped_pragma += 1
            continue
        pragma_ver, is_exact = info
        solc_ver = _resolve_solc_version(pragma_ver, is_exact)
        candidates.append((sol_path, solc_ver))

    log.info("Candidates: %d  (skipped: %d imports, %d no pragma)", len(candidates), skipped_import, skipped_pragma)
    return candidates


def _install_solc_in_docker(versions: List[str]) -> None:
    """Pre-install all needed solc versions in a persistent Docker volume."""
    unique = sorted(set(versions))
    log.info("Pre-installing %d unique solc versions in Docker...", len(unique))

    versions_str = " ".join(unique)
    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", "slither-solc-cache:/root/.solc-select",
        DOCKER_IMAGE,
        "bash", "-c", f"solc-select install {versions_str} 2>&1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.warning("Some solc installs may have failed: %s", result.stderr[-300:])
    else:
        log.info("All solc versions installed successfully.")


def _run_slither_docker(
    sol_path: Path,
    solc_ver: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Run Slither on a single file via Docker."""
    docker_sol_path = str(sol_path).replace(str(DATA_DIR), DOCKER_DATA_MOUNT)

    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{DATA_DIR}:{DOCKER_DATA_MOUNT}:ro",
        "-v", "slither-solc-cache:/root/.solc-select",
        DOCKER_IMAGE,
        "bash", "-c",
        f"solc-select use {solc_ver} >/dev/null 2>&1 && slither '{docker_sol_path}' --json - 2>/dev/null",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "detectors": [], "n_findings": 0, "severities": {}}
    except Exception as e:
        return {"success": False, "error": str(e), "detectors": [], "n_findings": 0, "severities": {}}

    stdout = result.stdout.strip()
    if not stdout:
        stderr_msg = result.stderr.strip()[-300:] if result.stderr else "no output"
        return {"success": False, "error": stderr_msg, "detectors": [], "n_findings": 0, "severities": {}}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False, "error": "invalid json", "detectors": [], "n_findings": 0, "severities": {}}

    if not data.get("success", False):
        return {"success": False, "error": str(data.get("error", "unknown"))[:200], "detectors": [], "n_findings": 0, "severities": {}}

    detectors = data.get("results", {}).get("detectors", [])
    meaningful = [d for d in detectors if d.get("check", "") not in IGNORE_CHECKS]

    severity_counts: Dict[str, int] = {}
    checks_found: List[str] = []
    for d in meaningful:
        sev = d.get("impact", "Informational")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        checks_found.append(d.get("check", "unknown"))

    return {
        "success": True,
        "error": None,
        "detectors": checks_found,
        "n_findings": len(meaningful),
        "severities": severity_counts,
    }


def _contract_to_label(severities: Dict[str, int], threshold: str = "Medium") -> int:
    if threshold == "High":
        return 1 if severities.get("High", 0) > 0 else 0
    return 1 if (severities.get("High", 0) + severities.get("Medium", 0)) > 0 else 0


def run_batch(
    candidates: List[Tuple[Path, str]],
    max_contracts: int = 0,
    timeout: int = 120,
) -> pd.DataFrame:
    if max_contracts > 0:
        candidates = candidates[:max_contracts]

    rows: List[Dict[str, Any]] = []
    total = len(candidates)
    t0 = time.time()

    for i, (sol_path, solc_ver) in enumerate(candidates):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            n_success = sum(1 for r in rows if r["slither_success"])
            log.info(
                "[%d/%d] %.1f c/min  ETA %.0fm  success=%d (%.0f%%)  |  %s",
                i + 1, total, rate * 60, eta / 60,
                n_success, 100 * n_success / max(1, len(rows)),
                sol_path.name,
            )

        result = _run_slither_docker(sol_path, solc_ver, timeout=timeout)

        rows.append({
            "sol_path": str(sol_path),
            "solc_version": solc_ver,
            "slither_success": result["success"],
            "slither_error": result["error"] or "",
            "n_findings": result["n_findings"],
            "n_high": result["severities"].get("High", 0),
            "n_medium": result["severities"].get("Medium", 0),
            "n_low": result["severities"].get("Low", 0),
            "n_informational": result["severities"].get("Informational", 0),
            "checks_found": ";".join(result["detectors"]),
            "slither_label_medium": _contract_to_label(result["severities"], "Medium"),
            "slither_label_high": _contract_to_label(result["severities"], "High"),
        })

        # Save intermediate results every 200 contracts
        if (i + 1) % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / "slither_per_contract_partial.csv", index=False)

    return pd.DataFrame(rows)


def match_with_unified(slither_df: pd.DataFrame, unified_csv: Path) -> pd.DataFrame:
    unified = pd.read_csv(unified_csv, low_memory=False)
    unified_dappscan = unified[unified["source_dataset"] == "DAppSCAN"].copy()
    slither_df["sol_path_norm"] = slither_df["sol_path"].str.strip()
    unified_dappscan["local_source_path_norm"] = unified_dappscan["local_source_path"].str.strip()
    merged = slither_df.merge(
        unified_dappscan[["local_source_path_norm", "contract_id", "vuln_present", "label_status",
                          "source_subdataset", "loc"]],
        left_on="sol_path_norm",
        right_on="local_source_path_norm",
        how="inner",
    )
    log.info("Matched %d of %d Slither results with unified dataset", len(merged), len(slither_df))
    return merged


def compute_metrics(merged_df: pd.DataFrame, pred_col: str, truth_col: str = "vuln_present") -> Dict[str, Any]:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    successful = merged_df[merged_df["slither_success"]].copy()
    if len(successful) == 0:
        return {"error": "no successful analyses"}
    y_true = successful[truth_col].astype(int).values
    y_pred = successful[pred_col].astype(int).values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n_contracts": len(successful),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Docker-based Slither batch analysis")
    parser.add_argument("--max-contracts", type=int, default=0, help="Max contracts (0=all)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-contract timeout (seconds)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = discover_contracts(CONTRACTS_DIR)
    if not candidates:
        log.error("No candidates found"); sys.exit(1)

    version_dist = {}
    for _, v in candidates:
        version_dist[v] = version_dist.get(v, 0) + 1
    log.info("Solc version distribution: %s", dict(sorted(version_dist.items())))

    all_versions = list(set(v for _, v in candidates))
    _install_solc_in_docker(all_versions)

    total = len(candidates) if args.max_contracts == 0 else min(args.max_contracts, len(candidates))
    log.info("Starting Docker-based Slither batch on %d contracts...", total)
    slither_df = run_batch(candidates, max_contracts=args.max_contracts, timeout=args.timeout)

    out_csv = OUT_DIR / "slither_per_contract.csv"
    slither_df.to_csv(out_csv, index=False)
    log.info("Saved per-contract results to %s", out_csv)

    n_success = slither_df["slither_success"].sum()
    n_fail = (~slither_df["slither_success"]).sum()
    log.info("Slither success: %d / %d (%.1f%%)", n_success, len(slither_df), 100 * n_success / max(1, len(slither_df)))

    unified_csv = DATA_DIR / "derived" / "unified_contract_eval_scored.csv"
    if unified_csv.exists():
        merged = match_with_unified(slither_df, unified_csv)
        merged.to_csv(OUT_DIR / "slither_vs_ground_truth.csv", index=False)

        metrics_medium = compute_metrics(merged, "slither_label_medium")
        metrics_high = compute_metrics(merged, "slither_label_high")

        summary = {
            "total_sol_files": 21457,
            "total_candidates_no_import": len(candidates),
            "total_analyzed": len(slither_df),
            "n_success": int(n_success),
            "n_fail": int(n_fail),
            "success_rate": round(float(n_success / max(1, len(slither_df))), 4),
            "coverage_of_dataset": round(float(n_success / 21457), 4),
            "n_matched_with_unified": len(merged),
            "metrics_threshold_medium": metrics_medium,
            "metrics_threshold_high": metrics_high,
            "version_distribution": version_dist,
        }
        summary_path = OUT_DIR / "slither_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        log.info("\n=== SLITHER BASELINE RESULTS ===")
        log.info("Coverage: %d of 21,457 DAppSCAN contracts (%.1f%%)", n_success, 100 * n_success / 21457)
        log.info("\nThreshold = High+Medium:")
        for k, v in metrics_medium.items():
            log.info("  %-15s %s", k, v)
        log.info("\nThreshold = High only:")
        for k, v in metrics_high.items():
            log.info("  %-15s %s", k, v)

    log.info("Done.")


if __name__ == "__main__":
    main()
