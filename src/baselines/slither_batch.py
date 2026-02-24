"""
Batch Slither analysis on DAppSCAN contracts for baseline comparison.

Strategy:
  1. Identify contracts without external imports (compilable standalone).
  2. Detect the pragma solidity version for each contract.
  3. Run Slither with the matching solc version.
  4. Aggregate findings to contract-level: if any High/Medium finding → vulnerable.
  5. Save results for comparison with our ML model.

Output:
  - data/derived/slither_results/slither_per_contract.csv
  - data/derived/slither_results/slither_summary.json

Usage:
    python -m src.baselines.slither_batch [--max-contracts N] [--timeout SECS]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEVERITY_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Informational": 0, "Optimization": 0}

# Detectors that are purely informational / style and should NOT count
# as vulnerability findings for our comparison.
IGNORE_CHECKS = {
    "solc-version",
    "pragma",
    "naming-convention",
    "too-many-digits",
    "assembly",
    "low-level-calls",
    "redundant-statements",
    "dead-code",
    "constable-states",
    "immutable-states",
    "external-function",
    "similar-names",
    "unused-state",
    "unused-return",
}


def _extract_pragma_info(sol_path: Path) -> Optional[Tuple[str, bool]]:
    """
    Extract solc version and whether it's an exact pin from pragma.
    Returns (version, is_exact) or None.

    'pragma solidity 0.6.11;'        → ('0.6.11', True)
    'pragma solidity ^0.6.0;'        → ('0.6.0',  False)
    'pragma solidity >=0.6.0 <0.8.0' → ('0.6.0',  False)
    """
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    m = re.search(r"pragma\s+solidity\s+([^;]+);", text)
    if not m:
        return None

    pragma = m.group(1).strip()

    # Exact pin: no ^, no >=, no <, just a version
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


# Map minor versions to recommended solc releases.
_SOLC_MAP = {
    "0.4": "0.4.26",
    "0.5": "0.5.17",
    "0.6": "0.6.12",
    "0.7": "0.7.6",
    "0.8": "0.8.10",
}


def _resolve_solc_version(pragma_ver: str, is_exact: bool) -> str:
    """
    Pick a concrete installable solc version for a given pragma.
    For exact pins (e.g. 'pragma solidity 0.6.11'), use that exact version.
    For ranges/carets, use the latest patch for that minor.
    """
    if is_exact:
        return pragma_ver

    parts = pragma_ver.split(".")
    if len(parts) >= 3:
        minor_key = f"{parts[0]}.{parts[1]}"
        return _SOLC_MAP.get(minor_key, pragma_ver)
    return pragma_ver


def _has_external_imports(sol_path: Path) -> bool:
    """Check if the file has imports that require npm packages."""
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    return bool(re.search(
        r'''import\s+["']@|import\s+["']hardhat|import\s+["']forge-std''',
        text,
    ))


def _has_any_import(sol_path: Path) -> bool:
    """Check if the file has any import statement."""
    try:
        text = sol_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    return bool(re.search(r"^\s*import\s", text, re.MULTILINE))


def _ensure_solc_installed(version: str) -> bool:
    """Install solc version if not already available."""
    try:
        check = subprocess.run(
            ["solc-select", "versions"],
            capture_output=True, text=True, timeout=10,
        )
        if version in check.stdout:
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["solc-select", "install", version],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0 or "already installed" in (result.stdout + result.stderr).lower()
    except Exception as e:
        log.warning("Failed to install solc %s: %s", version, e)
        return False


def _set_solc_version(version: str) -> bool:
    """Set the active solc version."""
    try:
        result = subprocess.run(
            ["solc-select", "use", version],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_slither(sol_path: Path, timeout: int = 120) -> Dict[str, Any]:
    """
    Run slither on a single .sol file and return parsed JSON results.
    Returns dict with keys: success, error, detectors, n_findings, severities.
    """
    try:
        result = subprocess.run(
            ["slither", str(sol_path), "--json", "-"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "detectors": [], "n_findings": 0, "severities": {}}
    except Exception as e:
        return {"success": False, "error": str(e), "detectors": [], "n_findings": 0, "severities": {}}

    # Slither sends JSON to stdout and logs to stderr.
    # On compilation failure, stdout may be empty or contain partial JSON.
    stdout = result.stdout.strip()
    if not stdout:
        # Try stderr for error info
        stderr_msg = result.stderr.strip()[-500:] if result.stderr else "no output"
        # Check if stderr contains compilation error hints
        if "Error" in (result.stderr or ""):
            return {"success": False, "error": f"compilation: {stderr_msg}", "detectors": [], "n_findings": 0, "severities": {}}
        return {"success": False, "error": stderr_msg, "detectors": [], "n_findings": 0, "severities": {}}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False, "error": "invalid json", "detectors": [], "n_findings": 0, "severities": {}}

    if not data.get("success", False):
        return {"success": False, "error": data.get("error", "unknown"), "detectors": [], "n_findings": 0, "severities": {}}

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
    """
    Convert Slither severity counts to binary label.
    threshold="Medium" → vulnerable if any High or Medium finding.
    threshold="High" → vulnerable only if High finding.
    """
    if threshold == "High":
        return 1 if severities.get("High", 0) > 0 else 0
    else:
        return 1 if (severities.get("High", 0) + severities.get("Medium", 0)) > 0 else 0


def discover_contracts(
    contracts_dir: Path,
    require_no_imports: bool = True,
    min_solc_minor: int = 7,
) -> List[Tuple[Path, str]]:
    """
    Find .sol files suitable for standalone Slither analysis.
    Returns list of (path, solc_version) tuples.

    min_solc_minor: minimum minor version to include (default 7 = 0.7.x+).
        On ARM64 Macs, solc versions < 0.7.0 are x86_64-only and may be
        blocked by solc-select. Set to 0 to include all versions.
    """
    candidates: List[Tuple[Path, str]] = []
    all_sol = sorted(contracts_dir.rglob("*.sol"))
    log.info("Total .sol files found: %d", len(all_sol))

    skipped_import = 0
    skipped_pragma = 0
    skipped_old_solc = 0

    for sol_path in all_sol:
        if require_no_imports and _has_any_import(sol_path):
            skipped_import += 1
            continue

        info = _extract_pragma_info(sol_path)
        if not info:
            skipped_pragma += 1
            continue

        pragma_ver, is_exact = info

        # Skip old solc versions that won't run on ARM64
        parts = pragma_ver.split(".")
        if len(parts) >= 2:
            try:
                minor = int(parts[1])
                if minor < min_solc_minor:
                    skipped_old_solc += 1
                    continue
            except ValueError:
                pass

        solc_ver = _resolve_solc_version(pragma_ver, is_exact)
        candidates.append((sol_path, solc_ver))

    log.info(
        "Candidates: %d  (skipped: %d imports, %d no pragma, %d solc<0.%d.0)",
        len(candidates), skipped_import, skipped_pragma, skipped_old_solc, min_solc_minor,
    )
    return candidates


def run_batch(
    candidates: List[Tuple[Path, str]],
    max_contracts: int = 0,
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Run Slither on each candidate contract and collect results.
    """
    if max_contracts > 0:
        candidates = candidates[:max_contracts]

    installed_versions: set = set()
    current_version: Optional[str] = None

    rows: List[Dict[str, Any]] = []
    total = len(candidates)
    t0 = time.time()

    for i, (sol_path, solc_ver) in enumerate(candidates):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %.1f contracts/min  ETA %.0f min  |  %s",
                i + 1, total, rate * 60, eta / 60, sol_path.name,
            )

        if solc_ver not in installed_versions:
            if _ensure_solc_installed(solc_ver):
                installed_versions.add(solc_ver)
            else:
                rows.append({
                    "sol_path": str(sol_path),
                    "solc_version": solc_ver,
                    "slither_success": False,
                    "slither_error": f"solc {solc_ver} install failed",
                    "n_findings": 0,
                    "n_high": 0,
                    "n_medium": 0,
                    "n_low": 0,
                    "n_informational": 0,
                    "checks_found": "",
                    "slither_label_medium": 0,
                    "slither_label_high": 0,
                })
                continue

        if solc_ver != current_version:
            if _set_solc_version(solc_ver):
                current_version = solc_ver
            else:
                rows.append({
                    "sol_path": str(sol_path),
                    "solc_version": solc_ver,
                    "slither_success": False,
                    "slither_error": f"solc-select use {solc_ver} failed",
                    "n_findings": 0,
                    "n_high": 0,
                    "n_medium": 0,
                    "n_low": 0,
                    "n_informational": 0,
                    "checks_found": "",
                    "slither_label_medium": 0,
                    "slither_label_high": 0,
                })
                continue

        result = _run_slither(sol_path, timeout=timeout)

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

    return pd.DataFrame(rows)


def match_with_unified(
    slither_df: pd.DataFrame,
    unified_csv: Path,
) -> pd.DataFrame:
    """
    Join Slither results with the unified dataset to get ground-truth labels
    and ML model predictions on the same contracts.
    """
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


def compute_metrics(
    merged_df: pd.DataFrame,
    pred_col: str = "slither_label_medium",
    truth_col: str = "vuln_present",
) -> Dict[str, Any]:
    """Compute classification metrics for Slither's predictions."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )

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
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Slither analysis on DAppSCAN")
    parser.add_argument("--max-contracts", type=int, default=0,
                        help="Max contracts to analyze (0=all)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-contract Slither timeout in seconds")
    parser.add_argument("--include-relative-imports", action="store_true",
                        help="Also include contracts with relative imports")
    parser.add_argument("--min-solc-minor", type=int, default=7,
                        help="Minimum solc minor version (default 7 = 0.7.x+, set 0 for all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    require_no_imports = not args.include_relative_imports
    candidates = discover_contracts(
        CONTRACTS_DIR,
        require_no_imports=require_no_imports,
        min_solc_minor=args.min_solc_minor,
    )

    if not candidates:
        log.error("No candidate contracts found in %s", CONTRACTS_DIR)
        sys.exit(1)

    version_dist = {}
    for _, v in candidates:
        version_dist[v] = version_dist.get(v, 0) + 1
    log.info("Solc version distribution: %s", dict(sorted(version_dist.items())))

    log.info("Starting Slither batch analysis on %d contracts...", len(candidates) if args.max_contracts == 0 else args.max_contracts)
    slither_df = run_batch(candidates, max_contracts=args.max_contracts, timeout=args.timeout)

    out_csv = OUT_DIR / "slither_per_contract.csv"
    slither_df.to_csv(out_csv, index=False)
    log.info("Saved per-contract results to %s", out_csv)

    n_success = slither_df["slither_success"].sum()
    n_fail = (~slither_df["slither_success"]).sum()
    log.info("Slither success: %d / %d  (%.1f%%)", n_success, len(slither_df), 100 * n_success / max(1, len(slither_df)))

    unified_csv = DATA_DIR / "derived" / "unified_contract_eval_scored.csv"
    if unified_csv.exists():
        merged = match_with_unified(slither_df, unified_csv)
        out_merged = OUT_DIR / "slither_vs_ground_truth.csv"
        merged.to_csv(out_merged, index=False)
        log.info("Saved merged results to %s", out_merged)

        metrics_medium = compute_metrics(merged, "slither_label_medium")
        metrics_high = compute_metrics(merged, "slither_label_high")

        summary = {
            "total_candidates": len(candidates),
            "total_analyzed": len(slither_df),
            "n_success": int(n_success),
            "n_fail": int(n_fail),
            "success_rate": float(n_success / max(1, len(slither_df))),
            "n_matched_with_unified": len(merged),
            "metrics_threshold_medium": metrics_medium,
            "metrics_threshold_high": metrics_high,
            "version_distribution": version_dist,
        }

        summary_path = OUT_DIR / "slither_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Saved summary to %s", summary_path)

        log.info("\n=== SLITHER BASELINE RESULTS ===")
        log.info("Coverage: %d of 21,457 DAppSCAN contracts (%.1f%%)",
                 n_success, 100 * n_success / 21457)
        log.info("\nThreshold = High+Medium:")
        for k, v in metrics_medium.items():
            log.info("  %-15s %s", k, v)
        log.info("\nThreshold = High only:")
        for k, v in metrics_high.items():
            log.info("  %-15s %s", k, v)
    else:
        log.warning("Unified CSV not found at %s; skipping comparison", unified_csv)

    log.info("Done.")


if __name__ == "__main__":
    main()
