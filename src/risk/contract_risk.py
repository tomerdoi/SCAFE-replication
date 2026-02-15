"""
Contract-level technical risk scoring from verified Solidity sources.

This is the pivoted project's core: one row per contract address with:
- extracted feature flags (upgrade/admin surface, dangerous patterns, complexity proxies)
- raw risk score and normalized z-score

Input:
- data/contract_registry.csv            (address universe)
- data/etherscan_sources/<addr>.json    (cached Etherscan payloads from etherscan_source_util)

Output:
- data/derived/contract_risk.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# src/risk/contract_risk.py -> repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
ETHERSCAN_SOURCES_DIR = DATA_DIR / "etherscan_sources"
ETHERSCAN_ACTIVITY_DIR = DATA_DIR / "etherscan_activity"


def _is_addr(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("0x") and len(s) == 42


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now_ts() -> int:
    # avoid importing datetime everywhere in hot paths
    import time

    return int(time.time())


def _clamp01(x: float) -> float:
    try:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)
    except Exception:
        return 0.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return int(default)


def _eth_from_wei(wei: Any) -> float:
    try:
        return float(_safe_int(wei, 0)) / 1e18
    except Exception:
        return 0.0

def _hex_to_bytes(hex_str: str) -> bytes:
    s = (hex_str or "").strip()
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        return b""
    try:
        return bytes.fromhex(s)
    except Exception:
        return b""


def _disassemble_opcode_counts(bytecode_hex: str) -> Dict[str, int]:
    """
    Very small EVM disassembler: iterates opcodes and skips PUSH immediate data.
    Returns counts of selected opcodes (by mnemonic).
    """
    b = _hex_to_bytes(bytecode_hex)
    i = 0
    counts: Dict[str, int] = {}

    def inc(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    while i < len(b):
        op = b[i]
        i += 1

        # PUSH1..PUSH32 => skip immediate bytes
        if 0x60 <= op <= 0x7F:
            i += (op - 0x5F)
            continue

        # Selected opcodes
        if op == 0xF1:
            inc("CALL")
        elif op == 0xF2:
            inc("CALLCODE")
        elif op == 0xF4:
            inc("DELEGATECALL")
        elif op == 0xFA:
            inc("STATICCALL")
        elif op == 0xFF:
            inc("SELFDESTRUCT")
        elif op == 0xF0:
            inc("CREATE")
        elif op == 0xF5:
            inc("CREATE2")
        elif op == 0x55:
            inc("SSTORE")
        elif op == 0x54:
            inc("SLOAD")
    return counts


def _detect_solc_pre_08(compiler_version: str, source_text: str) -> int:
    """
    Detect if a contract uses Solidity < 0.8 (which lacks built-in overflow checks).
    Checks compiler_version metadata first, then falls back to pragma in source.
    """
    # Try compiler_version metadata (e.g. "v0.6.12+commit.xyz")
    m = re.search(r"v?0\.(\d+)\.\d+", compiler_version)
    if m:
        minor = int(m.group(1))
        return 1 if minor < 8 else 0
    # Fallback: look at pragma in source text
    pragma_matches = re.findall(r"pragma\s+solidity\s+[\^>=<]*\s*0\.(\d+)", source_text)
    if pragma_matches:
        # Use the most common pragma minor version
        minors = [int(x) for x in pragma_matches]
        # If any pragma is < 0.8, mark it
        if min(minors) < 8:
            return 1
        return 0
    return 0  # unknown


def _extract_v4_features(
    base: Dict[str, Any],
    t: str,
    func_matches: list,
) -> None:
    """
    v4 feature extraction: cyclomatic complexity, Halstead metrics, token entropy,
    function-size distributions, state mutation analysis, internal coupling,
    data-structure complexity, code duplication, and inheritance depth.

    These features were discovered via systematic exploration and AUC testing
    on DAppSCAN (300 vuln vs 300 clean), yielding AUC 0.66–0.81.

    Many of these are SIZE-INDEPENDENT (ratios, per-function, entropy) which
    helps capture vulnerability signal beyond the "bigger code = more bugs" confound.
    """
    if not t:
        return

    loc = max(1, base.get("loc", 0) or 0)
    n_funcs_raw = base.get("n_functions", 0) or 0
    n_funcs = max(1, n_funcs_raw)

    # ------------------------------------------------------------------
    # L. Cyclomatic complexity (AUC 0.77–0.81)
    # ------------------------------------------------------------------
    if func_matches:
        ccs = []
        for _, _, body in func_matches:
            cc = (
                len(re.findall(r'\bif\s*\(', body))
                + len(re.findall(r'\bfor\s*\(', body))
                + len(re.findall(r'\bwhile\s*\(', body))
                + len(re.findall(r'\brequire\s*\(', body))
                + len(re.findall(r'&&|\|\|', body))
                + 1  # base path
            )
            ccs.append(cc)
        cc_arr = np.array(ccs, dtype=float)
        base["max_cyclomatic"] = int(cc_arr.max())
        base["avg_cyclomatic"] = float(cc_arr.mean())
        base["std_cyclomatic"] = float(cc_arr.std()) if len(cc_arr) > 1 else 0.0
        base["sum_cyclomatic"] = int(cc_arr.sum())
        base["cyclomatic_per_loc"] = float(cc_arr.sum()) / loc
        base["cyclomatic_variance_ratio"] = (
            float(cc_arr.std() / cc_arr.mean()) if cc_arr.mean() > 0 else 0.0
        )
        base["n_high_complexity_fns"] = int(np.sum(cc_arr > 10))

    # ------------------------------------------------------------------
    # M. Halstead complexity metrics (AUC 0.72–0.78)
    # ------------------------------------------------------------------
    operators = set(re.findall(r'[+\-*/=<>!&|^~%]+', t))
    operands = set(re.findall(r'\b[a-zA-Z_]\w*\b', t)) | set(re.findall(r'\b\d+\b', t))
    N1 = len(re.findall(r'[+\-*/=<>!&|^~%]+', t))
    N2 = len(re.findall(r'\b[a-zA-Z_]\w*\b|\b\d+\b', t))
    n1 = len(operators)
    n2 = max(1, len(operands))
    n_vocab = max(2, n1 + n2)
    N_total = N1 + N2

    volume = float(N_total * np.log2(n_vocab))
    difficulty = float((n1 / 2.0) * (N2 / n2))
    base["halstead_volume"] = volume
    base["halstead_difficulty"] = difficulty
    base["halstead_effort"] = float(volume * difficulty)
    base["halstead_volume_per_loc"] = volume / loc

    # ------------------------------------------------------------------
    # N. Token entropy & diversity (AUC 0.77–0.79)
    # ------------------------------------------------------------------
    tokens = re.findall(r'[a-zA-Z_]\w*', t)
    unique_tokens = len(set(tokens))
    total_tokens = max(1, len(tokens))
    base["n_unique_tokens"] = int(unique_tokens)
    base["n_total_tokens"] = int(total_tokens)
    base["token_diversity"] = float(unique_tokens) / total_tokens

    if tokens:
        from collections import Counter as _Counter
        tc = _Counter(tokens)
        probs = np.array(list(tc.values()), dtype=float) / total_tokens
        base["token_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        base["token_entropy"] = 0.0

    # ------------------------------------------------------------------
    # O. Function-size distribution metrics (AUC 0.71–0.77)
    # ------------------------------------------------------------------
    if func_matches:
        fl = np.array([len(body.splitlines()) for _, _, body in func_matches], dtype=float)
        n_fl = len(fl)

        base["fn_loc_p75"] = float(np.percentile(fl, 75))
        base["fn_loc_p90"] = float(np.percentile(fl, 90))
        base["fn_loc_iqr"] = float(np.percentile(fl, 75) - np.percentile(fl, 25))
        base["fn_loc_skew"] = float(pd.Series(fl).skew()) if n_fl > 2 else 0.0
        base["fn_loc_cv"] = float(np.std(fl) / np.mean(fl)) if np.mean(fl) > 0 else 0.0

        # Gini coefficient (inequality of function sizes)
        sorted_fl = np.sort(fl)
        idx = np.arange(1, n_fl + 1)
        total = np.sum(sorted_fl)
        if total > 0:
            base["fn_loc_gini"] = float(
                (2 * np.sum(idx * sorted_fl) / (n_fl * total) - (n_fl + 1) / n_fl)
            )
        else:
            base["fn_loc_gini"] = 0.0

        # % of LOC in complex functions (> 30 lines)
        base["complex_fn_loc_ratio"] = float(sum(l for l in fl if l > 30)) / loc

        # Max nesting density (normalized by log of function size)
        max_nd = 0.0
        for _, _, body in func_matches:
            body_lines = max(1, len(body.splitlines()))
            depth = 0
            max_d = 0
            for ch in body:
                if ch == '{':
                    depth += 1
                    if depth > max_d:
                        max_d = depth
                elif ch == '}':
                    depth -= 1
            nd = max_d / np.log1p(body_lines)
            if nd > max_nd:
                max_nd = nd
        base["max_nesting_density"] = float(max_nd)

    # ------------------------------------------------------------------
    # P. State mutation analysis (AUC 0.74–0.77)
    # ------------------------------------------------------------------
    sv_decls = set(re.findall(
        r'\b(?:uint|int|bool|address|bytes|string|mapping)\w*\s+'
        r'(?:public|private|internal)?\s*(\w+)\s*[;=]', t
    ))
    written = sum(1 for sv in sv_decls if re.search(rf'\b{re.escape(sv)}\s*=', t))
    base["n_written_state_vars"] = int(written)
    base["state_var_write_ratio"] = float(written) / max(1, len(sv_decls))
    base["state_writes_per_fn"] = float(written) / n_funcs

    if func_matches:
        fn_writes = [len(re.findall(r'\w+\s*=\s*[^=]', body)) for _, _, body in func_matches]
        base["max_fn_state_writes"] = int(max(fn_writes))
        base["mean_fn_state_writes"] = float(np.mean(fn_writes))

    # ------------------------------------------------------------------
    # Q. Internal coupling (AUC 0.73)
    # ------------------------------------------------------------------
    func_names = set(re.findall(r'\bfunction\s+(\w+)\s*\(', t))
    internal_calls = 0
    for fn_name in func_names:
        # -1 for the declaration itself
        n_refs = len(re.findall(rf'\b{re.escape(fn_name)}\s*\(', t))
        internal_calls += max(0, n_refs - 1)
    base["n_internal_calls"] = int(internal_calls)
    base["internal_call_density"] = float(internal_calls) / max(1, len(func_names))

    # ------------------------------------------------------------------
    # R. Data structure complexity (AUC 0.71)
    # ------------------------------------------------------------------
    base["n_mapping_access"] = int(len(re.findall(r'\w+\[\w+\]', t)))
    base["n_nested_access"] = int(len(re.findall(r'\w+\[\w+\]\[\w+\]', t)))
    base["mapping_access_per_fn"] = float(base["n_mapping_access"]) / n_funcs
    base["nested_access_per_fn"] = float(base["n_nested_access"]) / n_funcs

    # ------------------------------------------------------------------
    # S. Code duplication (AUC 0.69)
    # ------------------------------------------------------------------
    lines = t.splitlines()
    meaningful = [ln.strip() for ln in lines if len(ln.strip()) > 20]
    if meaningful:
        from collections import Counter as _Counter2
        lc = _Counter2(meaningful)
        n_dup = sum(c - 1 for c in lc.values() if c > 1)
        base["n_duplicated_lines"] = int(n_dup)
        base["duplication_ratio"] = float(n_dup) / max(1, len(meaningful))
    else:
        base["n_duplicated_lines"] = 0
        base["duplication_ratio"] = 0.0

    # ------------------------------------------------------------------
    # T. Inheritance depth analysis (AUC 0.66–0.68)
    # ------------------------------------------------------------------
    contracts_with_is = re.findall(r'contract\s+(\w+)\s+is\s+([\w,\s]+)\{', t)
    inh_graph: Dict[str, list] = {}
    for cname, parents in contracts_with_is:
        parent_list = [p.strip() for p in parents.split(",") if p.strip()]
        inh_graph[cname] = parent_list

    def _inh_depth(name: str, seen: set = None) -> int:
        if seen is None:
            seen = set()
        if name in seen or name not in inh_graph:
            return 0
        seen.add(name)
        return 1 + max((_inh_depth(p, seen) for p in inh_graph[name]), default=0)

    depths = [_inh_depth(c) for c in inh_graph]
    base["max_inheritance_depth"] = int(max(depths)) if depths else 0
    base["total_inheritance_edges"] = int(sum(len(v) for v in inh_graph.values()))

    # ------------------------------------------------------------------
    # U. Additional statement/pattern counts
    # ------------------------------------------------------------------
    base["n_return_stmts"] = int(len(re.findall(r'\breturn\b', t)))
    base["n_delete_stmts"] = int(len(re.findall(r'\bdelete\b', t)))
    base["n_new_stmts"] = int(len(re.findall(r'\bnew\s+\w+', t)))
    base["return_per_function"] = float(base["n_return_stmts"]) / n_funcs
    base["n_override_fns"] = int(len(re.findall(r'\boverride\b', t)))
    base["n_virtual_fns"] = int(len(re.findall(r'\bvirtual\b', t)))
    base["n_ternary_ops"] = int(len(re.findall(r'\?[^:]+:', t)))
    base["n_type_conversions"] = int(len(re.findall(r'\b(?:uint|int|bytes)\d*\s*\(', t)))

    # ------------------------------------------------------------------
    # V. Size-normalized ratio features
    # ------------------------------------------------------------------
    n_assign = max(1, base.get("n_assignments", 0) or 0)
    base["require_to_assign_ratio"] = float(base.get("n_require_assert", 0) or 0) / n_assign
    base["conditional_to_assign_ratio"] = float(base.get("n_conditionals", 0) or 0) / n_assign

    # Guard coverage: fraction of public/external functions with access modifiers
    pub_ext_sigs_v4 = re.findall(
        r'function\s+\w+\s*\([^)]*\)\s*((?:public|external)[^{;]*)', t
    )
    n_pe_v4 = len(pub_ext_sigs_v4)
    n_guarded = sum(
        1 for sig in pub_ext_sigs_v4
        if re.search(r'\b(onlyOwner|onlyRole|whenNotPaused|nonReentrant|only\w+)\b', sig)
    )
    base["guard_coverage"] = float(n_guarded) / max(1, n_pe_v4)

    # Comment-to-code ratio
    n_code_lines = sum(
        1 for ln in lines
        if ln.strip() and not ln.strip().startswith("//") and not ln.strip().startswith("*")
    )
    n_comment_lines = sum(
        1 for ln in lines
        if ln.strip().startswith("//") or ln.strip().startswith("*")
    )
    base["comment_to_code_ratio"] = float(n_comment_lines) / max(1, n_code_lines)

    # ------------------------------------------------------------------
    # W. Import analysis
    # ------------------------------------------------------------------
    import_paths = re.findall(r'import\s+["\']([^"\']+)', t)
    base["n_unique_imports"] = int(len(set(import_paths)))
    base["n_openzeppelin_imports"] = int(
        sum(1 for p in import_paths if "openzeppelin" in p.lower())
    )


def _extract_v3_features(base: Dict[str, Any], t: str) -> None:
    """
    v3 feature extraction: function-level granularity, semantic patterns,
    vulnerability-specific detectors, DeFi/token features, code quality signals.

    Populates ``base`` dict in-place.  Designed to be called from
    ``_extract_source_features`` after the v2 features are already set.

    Feature groups and their empirical AUC on DAppSCAN (200 vuln vs 200 clean):
      A. Function-level granularity  (AUC 0.77–0.82)
      B. Code pattern counts         (AUC 0.71–0.82)
      C. Solidity special variables   (AUC 0.57–0.77)
      D. Data location / storage      (AUC 0.57–0.64)
      E. Structural complexity        (AUC 0.53–0.72)
      F. Encoding & crypto            (AUC 0.52–0.58)
      G. Token / DeFi semantic        (AUC 0.53–0.68)
      H. Vulnerability detectors      (AUC 0.52–0.63)
      I. Code quality                 (AUC 0.52–0.59)
      J. Density / ratio features     (AUC 0.60–0.74)
      K. Cross-contract interaction   (AUC 0.52–0.66)
    """
    if not t:
        return

    loc = max(1, base.get("loc", 0) or 0)
    n_funcs_raw = base.get("n_functions", 0) or 0
    n_funcs = max(1, n_funcs_raw)

    # ------------------------------------------------------------------
    # A. Function-level granularity (highest-impact new features)
    # ------------------------------------------------------------------
    # Extract function signatures and their bodies using a regex that handles
    # nested braces up to 3 levels deep (covers >99% of real Solidity).
    _FUNC_BODY_RE = re.compile(
        r'function\s+(\w+)\s*\(([^)]*)\)[^{;]*'
        r'\{((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)\}',
        re.DOTALL,
    )
    func_matches = _FUNC_BODY_RE.findall(t)

    if func_matches:
        func_locs = [len(body.splitlines()) for _, _, body in func_matches]
        param_counts = [
            len([p.strip() for p in params.split(",") if p.strip()])
            for _, params, _ in func_matches
        ]

        base["max_function_loc"] = int(max(func_locs))
        base["std_function_loc"] = float(np.std(func_locs)) if len(func_locs) > 1 else 0.0
        base["n_complex_functions"] = int(sum(1 for fl in func_locs if fl > 30))
        base["max_params"] = int(max(param_counts)) if param_counts else 0
        base["avg_params"] = float(np.mean(param_counts)) if param_counts else 0.0
        base["total_params"] = int(sum(param_counts))

        # Max nesting depth: count nested {} within each function body
        max_nest = 0
        for _, _, body in func_matches:
            depth = 0
            max_d = 0
            for ch in body:
                if ch == '{':
                    depth += 1
                    if depth > max_d:
                        max_d = depth
                elif ch == '}':
                    depth -= 1
            if max_d > max_nest:
                max_nest = max_d
        base["max_nesting_depth"] = int(max_nest)

    # ------------------------------------------------------------------
    # B. Code pattern counts
    # ------------------------------------------------------------------
    base["n_arithmetic_ops"] = int(len(re.findall(r'[\w\]]\s*[+\-*/]\s*[\w\[]', t)))
    n_if = len(re.findall(r'\bif\s*\(', t))
    n_else = len(re.findall(r'\belse\b', t))
    base["n_conditionals"] = int(n_if + n_else)
    base["n_assignments"] = int(len(re.findall(r'\w+\s*=\s*[^=]', t)))
    base["n_emit"] = int(len(re.findall(r'\bemit\s+\w+', t)))
    base["n_type_casts"] = int(len(re.findall(r'[A-Z]\w+\s*\(\s*\w+\s*\)', t)))
    base["n_address_casts"] = int(len(re.findall(r'\baddress\s*\(', t)))
    base["n_bitwise_ops"] = int(len(re.findall(r'[&|^~](?!=)', t)))
    base["n_exp_ops"] = int(len(re.findall(r'\*\*', t)))

    # ------------------------------------------------------------------
    # C. Solidity special variable usage
    # ------------------------------------------------------------------
    base["n_msg_sender"] = int(len(re.findall(r'\bmsg\.sender\b', t)))
    base["n_msg_value"] = int(len(re.findall(r'\bmsg\.value\b', t)))
    base["n_block_timestamp"] = int(len(re.findall(r'\bblock\.timestamp\b', t)))
    base["n_block_number"] = int(len(re.findall(r'\bblock\.number\b', t)))

    # ------------------------------------------------------------------
    # D. Data location & storage patterns
    # ------------------------------------------------------------------
    base["n_storage_refs"] = int(len(re.findall(r'\bstorage\b', t)))
    base["n_memory_refs"] = int(len(re.findall(r'\bmemory\b', t)))
    base["n_calldata_refs"] = int(len(re.findall(r'\bcalldata\b', t)))
    n_all_mappings = len(re.findall(r'\bmapping\s*\(', t))
    n_nested = len(re.findall(r'mapping\s*\([^)]*mapping', t))
    base["n_nested_mappings"] = int(n_nested)

    # ------------------------------------------------------------------
    # E. Structural complexity
    # ------------------------------------------------------------------
    base["n_imports"] = int(len(re.findall(r'\bimport\b', t)))
    base["n_interfaces"] = int(len(re.findall(r'\binterface\s+\w+', t)))
    base["n_contracts"] = int(len(re.findall(r'\bcontract\s+\w+', t)))
    base["n_abstract_contracts"] = int(len(re.findall(r'\babstract\s+contract\b', t)))
    # Max inheritance: e.g. "contract Foo is A, B, C {"
    inh_matches = re.findall(r'\bis\s+([\w,\s]+)\{', t)
    base["max_inheritance"] = int(max(
        [len(m.split(",")) for m in inh_matches] or [0]
    ))
    base["n_libraries"] = int(len(re.findall(r'\blibrary\s+\w+', t)))
    base["n_using_for"] = int(len(re.findall(r'\busing\s+\w+\s+for\b', t)))
    base["n_structs"] = int(len(re.findall(r'\bstruct\s+\w+', t)))
    base["n_enums"] = int(len(re.findall(r'\benum\s+\w+', t)))

    # ------------------------------------------------------------------
    # F. Encoding & crypto
    # ------------------------------------------------------------------
    base["n_abi_encode"] = int(len(re.findall(r'\babi\.encode\w*\(', t)))
    base["n_keccak256"] = int(len(re.findall(r'\bkeccak256\s*\(', t)))
    base["uses_ecrecover"] = 1 if "ecrecover" in t else 0

    # ------------------------------------------------------------------
    # G. Token standard & DeFi semantic features
    # ------------------------------------------------------------------
    base["is_erc20"] = 1 if any(
        p in t for p in ("IERC20", "ERC20", "totalSupply() public", "balanceOf(address")
    ) else 0
    base["is_erc721"] = 1 if any(
        p in t for p in ("IERC721", "ERC721", "ownerOf(", "safeTransferFrom(")
    ) else 0
    base["has_mint"] = 1 if re.search(r'\b_?mint\s*\(', t) else 0
    base["has_burn"] = 1 if re.search(r'\b_?burn\s*\(', t) else 0
    base["has_swap"] = 1 if re.search(r'\bswap\b', t, re.I) else 0
    base["has_deposit_withdraw"] = 1 if ("deposit(" in t and "withdraw(" in t) else 0
    base["has_flash_loan"] = 1 if re.search(r'\bflash(Loan|Borrow|Mint)\b', t) else 0
    base["has_staking"] = 1 if re.search(r'\bstak(e|ing)\b', t, re.I) else 0
    base["has_reward"] = 1 if re.search(r'\breward', t, re.I) else 0

    # ------------------------------------------------------------------
    # H. Vulnerability-specific detectors
    # ------------------------------------------------------------------
    # SWC-104: Unchecked low-level call return value
    n_calls_raw = len(re.findall(r'\.call[\s{(]', t))
    n_checked = len(re.findall(r'\(\s*bool\s+\w+\s*(?:,\s*)?\)\s*=\s*\w+\.call', t))
    base["n_unchecked_calls"] = int(max(0, n_calls_raw - n_checked))
    base["has_unchecked_call"] = 1 if base["n_unchecked_calls"] > 0 else 0

    # SWC-107: Reentrancy pattern (state change after external call)
    base["has_call_then_state"] = 1 if re.search(
        r'\.call[\s{(].*?;\s*\n\s*\w+\s*[^=!<>]=\s*[^=]', t, re.DOTALL
    ) else 0

    # Access control gaps: public/external functions without modifier keywords
    pub_ext_sigs = re.findall(
        r'function\s+\w+\s*\([^)]*\)\s*((?:public|external)[^{;]*)', t
    )
    n_pe = len(pub_ext_sigs)
    n_with_mod = sum(
        1 for sig in pub_ext_sigs
        if re.search(r'\b(onlyOwner|onlyRole|whenNotPaused|nonReentrant|only\w+)\b', sig)
    )
    base["n_unprotected_public"] = int(max(0, n_pe - n_with_mod))
    base["unprotected_ratio"] = float(base["n_unprotected_public"]) / max(1, n_pe)

    # ------------------------------------------------------------------
    # I. Code quality
    # ------------------------------------------------------------------
    lines = t.splitlines()
    n_lines = len(lines)
    n_comment = sum(
        1 for ln in lines
        if ln.strip().startswith("//") or ln.strip().startswith("*") or ln.strip().startswith("/*")
    )
    base["comment_ratio"] = float(n_comment) / max(1, n_lines)
    base["has_natspec"] = 1 if ("///" in t or "@notice" in t or "@dev" in t) else 0
    base["n_try_catch"] = int(len(re.findall(r'\btry\s+', t)))
    base["has_custom_errors"] = 1 if re.search(r'\berror\s+\w+\s*\(', t) else 0
    base["n_revert_custom"] = int(len(re.findall(r'\brevert\s+\w+\s*\(', t)))

    # ------------------------------------------------------------------
    # J. New density / ratio features
    # ------------------------------------------------------------------
    base["conditional_density"] = float(base["n_conditionals"]) / loc
    base["emit_to_assign_ratio"] = float(base["n_emit"]) / max(1, base["n_assignments"])
    base["state_var_per_fn"] = float(base.get("n_state_variables", 0) or 0) / n_funcs
    base["arithmetic_density"] = float(base["n_arithmetic_ops"]) / loc
    n_pub_ext = base.get("n_public_external_functions", 0) or 0
    base["modifier_coverage"] = float(base.get("n_modifiers", 0) or 0) / max(1, n_pub_ext)

    # Visibility distribution (requires parsing function visibility keywords)
    n_pub = len(re.findall(r'\bfunction\s+\w+[^{;]*\bpublic\b', t))
    n_ext = len(re.findall(r'\bfunction\s+\w+[^{;]*\bexternal\b', t))
    n_int = len(re.findall(r'\bfunction\s+\w+[^{;]*\binternal\b', t))
    n_priv = len(re.findall(r'\bfunction\s+\w+[^{;]*\bprivate\b', t))
    n_vis_total = max(1, n_pub + n_ext + n_int + n_priv)
    base["vis_public_ratio"] = float(n_pub) / n_vis_total
    base["vis_external_ratio"] = float(n_ext) / n_vis_total
    base["vis_internal_ratio"] = float(n_int) / n_vis_total
    base["vis_private_ratio"] = float(n_priv) / n_vis_total

    # ------------------------------------------------------------------
    # K. Cross-contract interaction
    # ------------------------------------------------------------------
    base["n_interface_calls"] = int(len(re.findall(r'I[A-Z]\w+\(', t)))
    base["n_super_calls"] = int(len(re.findall(r'\bsuper\.\w+', t)))
    base["uses_proxy_pattern"] = 1 if re.search(
        r'\b(implementation|_implementation|proxy|Proxy|upgrade)\b', t
    ) else 0

    # ======================================================================
    # v4 features — cyclomatic complexity, Halstead metrics, token entropy,
    # function distributions, state mutation, coupling, duplication
    # ======================================================================
    _extract_v4_features(base, t, func_matches if func_matches else [])


@dataclass(frozen=True, slots=True)
class ContractRiskConfig:
    """
    Scoring weights for a component-based, explainable contract technical risk score.

    v2 — redesigned based on empirical feature-discrimination analysis (MI, AUC,
    Cohen's d) on 26K+ contracts across DAppSCAN, FORGE, AccessControlVulnerabilities,
    and Empirical-Analysis datasets.

    Key changes from v1:
    - Removed comp_opacity (confound: correlates with dataset provenance, not vuln).
    - Fixed comp_complexity (was inverted because raw LoC is larger for clean projects;
      now uses per-function density metrics).
    - Added comp_code_smell (new component capturing dangerous-pattern density,
      interaction surface density, and low-safety-pattern signals).
    - Re-weighted based on empirical signal strength.
    """

    # --- component weights (how much each component contributes to risk_raw) ---
    # Tuned to maximize DAppSCAN ROC-AUC (our largest and best-labeled dataset).
    # DAppSCAN component AUCs:
    #   comp_complexity 0.821, comp_safety 0.753*, comp_code_smell 0.702,
    #   comp_interaction 0.689, comp_privilege 0.648, comp_dangerous 0.440*
    #   (* = inverted or confounded — weight reduced accordingly)
    w_upgrade: float = 0.2          # AUC ~0.50: no signal on DAppSCAN
    w_privilege: float = 1.0        # AUC 0.648 on DAppSCAN
    w_interaction: float = 1.5      # AUC 0.689 on DAppSCAN; genuine signal
    w_dangerous: float = 0.3        # AUC 0.440 on DAppSCAN (inverted!); kept low
    w_complexity: float = 3.0       # AUC 0.821 on DAppSCAN — strongest component
    w_code_smell: float = 1.5       # AUC 0.702 on DAppSCAN
    w_safety: float = 0.15          # subtracted; but vuln>clean on DAppSCAN (AUC 0.753) → low weight
    w_activity: float = 0.2         # AUC 0.502 on DAppSCAN; no within-dataset signal
    # NOTE: comp_opacity is computed but NOT added to risk_raw (it's a confound)

    # --- feature-level weights inside components ---
    w_selfdestruct: float = 8.0
    w_delegatecall: float = 3.0
    w_tx_origin: float = 4.0
    w_assembly: float = 1.5
    w_unchecked: float = 2.0        # ↑ from 1.0: strong discriminator (AUC 0.516, only in vuln)

    w_external_call: float = 1.5    # ↑ from 1.0: n_external_calls is top MI feature
    w_transfer_send: float = 1.0

    w_upgrade_patterns: float = 2.0
    w_proxy_flag: float = 1.0

    w_only_owner: float = 1.0
    w_access_control: float = 1.0
    w_role_admin_ops: float = 0.5
    w_initializer: float = 0.5

    # Safety credits (subtract from risk)
    w_reentrancy_guard: float = 1.5
    w_pausable: float = 0.5
    w_timelock: float = 1.0
    w_safe_math: float = 0.25
    w_two_step_ownable: float = 0.5

    # Live activity weights (non-code signals)
    w_tx_activity: float = 0.35
    w_unique_senders: float = 0.50
    w_failed_rate: float = 1.25
    w_token_activity: float = 0.25
    w_value_flow_eth: float = 0.20
    w_admin_event_churn: float = 0.65


class ContractRiskUtil:
    def __init__(self, cfg: Optional[ContractRiskConfig] = None) -> None:
        logger = logging.getLogger("contract_risk_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.cfg = cfg or ContractRiskConfig()

    def ensure_dirs(self) -> None:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    def _etherscan_payload_path(self, address: str) -> Path:
        return ETHERSCAN_SOURCES_DIR / f"{address.lower()}.json"

    def _etherscan_activity_path(self, address: str) -> Path:
        return ETHERSCAN_ACTIVITY_DIR / f"{address.lower()}.json"

    def load_etherscan_source_payload(self, address: str) -> Optional[Dict[str, Any]]:
        p = self._etherscan_payload_path(address)
        if not p.exists():
            return None
        obj = _read_json(p)
        return obj if isinstance(obj, dict) else None

    def load_etherscan_activity_payload(self, address: str) -> Optional[Dict[str, Any]]:
        p = self._etherscan_activity_path(address)
        if not p.exists():
            return None
        obj = _read_json(p)
        return obj if isinstance(obj, dict) else None

    def _extract_bytecode_hex(self, payload: Dict[str, Any]) -> str:
        """
        Extract bytecode hex from the cached etherscan payload.
        etherscan_source_util stores this under payload["bytecode"]["eth_getCode"]["result"].
        """
        bc = payload.get("bytecode") if isinstance(payload, dict) else None
        if not isinstance(bc, dict):
            return ""
        egc = bc.get("eth_getCode")
        if isinstance(egc, dict) and isinstance(egc.get("result"), str):
            return str(egc.get("result") or "")
        # tolerate alternate storage
        if isinstance(egc, str):
            return egc
        return ""

    # ------------------------------------------------------------------
    # Shared helpers for feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_feature_base(identifier: str) -> Dict[str, Any]:
        """Return a zeroed-out feature dict.  Shared by all extraction paths."""
        return {
            "address": str(identifier).lower(),
            "has_payload": 0,
            "has_verified_source": 0,
            "n_files": 0,
            "loc": 0,
            "compiler_version": "",
            "is_proxy": 0,
            "implementation_address": "",
            "has_upgrade_patterns": 0,
            "uses_ownable": 0,
            "uses_two_step_ownable": 0,
            "uses_accesscontrol": 0,
            "uses_pausable": 0,
            "uses_timelock": 0,
            "uses_reentrancy_guard": 0,
            "uses_call": 0,
            "uses_staticcall": 0,
            "uses_delegatecall": 0,
            "uses_selfdestruct": 0,
            "uses_transfer_send": 0,
            "uses_tx_origin": 0,
            "uses_assembly": 0,
            "uses_unchecked": 0,
            "n_external_calls": 0,
            "n_delegatecall": 0,
            "n_functions": 0,
            "n_public_external_functions": 0,
            "n_payable_functions": 0,
            "n_modifiers": 0,
            "uses_initializer": 0,
            "n_role_admin_ops": 0,
            "uses_safe_math": 0,
            "uses_chainlink_oracle": 0,
            "uses_uniswap_oracle": 0,
            "has_fallback_or_receive": 0,
            # --- derived density / ratio features (v2) ---
            "external_call_density": 0.0,       # n_external_calls / loc
            "delegatecall_density": 0.0,         # n_delegatecall / loc
            "function_density": 0.0,             # n_functions / loc
            "modifier_density": 0.0,             # n_modifiers / loc
            "payable_density": 0.0,              # n_payable_functions / loc
            "public_ratio": 0.0,                 # n_public_external / n_functions
            "payable_ratio": 0.0,                # n_payable / n_functions
            "avg_loc_per_function": 0.0,         # loc / n_functions
            "modifier_per_function": 0.0,        # n_modifiers / n_functions
            "dangerous_pattern_count": 0,        # sum of dangerous binary flags
            "safety_pattern_count": 0,           # sum of safety binary flags
            "dangerous_to_safety_ratio": 0.0,    # dangerous / max(1, safety)
            "interaction_surface": 0,            # n_ext_calls + n_delegatecall + n_payable + has_fallback
            "interaction_surface_density": 0.0,  # interaction_surface / loc
            "uses_any_oracle": 0,                # max(chainlink, uniswap)
            "is_solc_pre_08": 0,                 # compiler < 0.8 (no built-in overflow checks)
            "n_state_variables": 0,              # count of state variable declarations
            "n_events": 0,                       # count of event declarations
            "n_mappings": 0,                     # count of mapping declarations
            "n_require_assert": 0,               # count of require() + assert()
            "require_per_function": 0.0,         # n_require_assert / n_functions
            "uses_abi_encode_packed": 0,         # abi.encodePacked (hash collision risk)
            "n_loops": 0,                        # for/while/do loops
            "loop_density": 0.0,                 # n_loops / n_functions
            # --- v3 features: function-level granularity (AUC 0.77-0.82) ---
            "max_function_loc": 0,               # LOC of the longest function body
            "std_function_loc": 0.0,             # std deviation of function LOCs
            "max_nesting_depth": 0,              # deepest { } nesting in any function
            "n_complex_functions": 0,            # functions with > 30 LOC
            "max_params": 0,                     # max parameters in any function
            "avg_params": 0.0,                   # average parameters per function
            "total_params": 0,                   # total parameters across all functions
            # --- v3 features: code pattern counts (AUC 0.71-0.82) ---
            "n_arithmetic_ops": 0,               # +, -, *, / on variables (not comments)
            "n_conditionals": 0,                 # if + else count
            "n_assignments": 0,                  # state assignment operations
            "n_emit": 0,                         # emit statement count
            "n_type_casts": 0,                   # Type(variable) casts
            "n_address_casts": 0,                # address() casts
            "n_bitwise_ops": 0,                  # &, |, ^, ~ operations
            "n_exp_ops": 0,                      # ** exponentiation
            # --- v3 features: Solidity special variable usage ---
            "n_msg_sender": 0,                   # msg.sender count
            "n_msg_value": 0,                    # msg.value count
            "n_block_timestamp": 0,              # block.timestamp count
            "n_block_number": 0,                 # block.number count
            # --- v3 features: data location & storage patterns ---
            "n_storage_refs": 0,                 # 'storage' keyword
            "n_memory_refs": 0,                  # 'memory' keyword
            "n_calldata_refs": 0,                # 'calldata' keyword
            "n_nested_mappings": 0,              # mapping(... mapping(...))
            # --- v3 features: structural complexity ---
            "n_imports": 0,                      # import statement count
            "n_interfaces": 0,                   # interface declarations
            "n_contracts": 0,                    # contract declarations
            "n_abstract_contracts": 0,           # abstract contract declarations
            "max_inheritance": 0,                # max "is A, B, C" count
            "n_libraries": 0,                    # library declarations
            "n_using_for": 0,                    # using ... for ... statements
            "n_structs": 0,                      # struct declarations
            "n_enums": 0,                        # enum declarations
            # --- v3 features: encoding & crypto ---
            "n_abi_encode": 0,                   # abi.encode* calls
            "n_keccak256": 0,                    # keccak256 calls
            "uses_ecrecover": 0,                 # ecrecover (signature verification)
            # --- v3 features: token standard & DeFi semantic ---
            "is_erc20": 0,                       # ERC20 token contract
            "is_erc721": 0,                      # ERC721 NFT contract
            "has_mint": 0,                       # mint function present
            "has_burn": 0,                       # burn function present
            "has_swap": 0,                       # swap/DEX pattern
            "has_deposit_withdraw": 0,           # deposit+withdraw pattern
            "has_flash_loan": 0,                 # flash loan pattern
            "has_staking": 0,                    # staking pattern
            "has_reward": 0,                     # reward distribution
            # --- v3 features: vulnerability-specific detectors ---
            "n_unchecked_calls": 0,              # .call() without bool success check
            "has_unchecked_call": 0,             # binary: any unchecked call
            "has_call_then_state": 0,            # reentrancy pattern: state change after call
            "n_unprotected_public": 0,           # public/external fns without access modifiers
            "unprotected_ratio": 0.0,            # unprotected / total public+external
            # --- v3 features: code quality ---
            "comment_ratio": 0.0,                # comment lines / total lines
            "has_natspec": 0,                    # NatSpec documentation present
            "n_try_catch": 0,                    # try/catch blocks
            "has_custom_errors": 0,              # custom error declarations
            "n_revert_custom": 0,                # revert with custom error name
            # --- v3 features: new density/ratio features ---
            "conditional_density": 0.0,          # n_conditionals / loc
            "emit_to_assign_ratio": 0.0,         # n_emit / n_assignments
            "state_var_per_fn": 0.0,             # n_state_variables / n_functions
            "arithmetic_density": 0.0,           # n_arithmetic_ops / loc
            "modifier_coverage": 0.0,            # n_modifiers / n_public_external
            "vis_public_ratio": 0.0,             # public functions / all functions
            "vis_external_ratio": 0.0,           # external functions / all functions
            "vis_internal_ratio": 0.0,           # internal functions / all functions
            "vis_private_ratio": 0.0,            # private functions / all functions
            # --- v3 features: cross-contract interaction ---
            "n_interface_calls": 0,              # I<Name>() interface-based calls
            "n_super_calls": 0,                  # super.method() calls
            "uses_proxy_pattern": 0,             # proxy/implementation pattern
            # --- v4 features: cyclomatic complexity (AUC 0.77-0.81) ---
            "max_cyclomatic": 0,                 # max cyclomatic complexity of any function
            "avg_cyclomatic": 0.0,               # mean cyclomatic complexity per function
            "std_cyclomatic": 0.0,               # std dev of cyclomatic complexity
            "sum_cyclomatic": 0,                 # total cyclomatic complexity
            "cyclomatic_per_loc": 0.0,           # sum_cyclomatic / loc
            "cyclomatic_variance_ratio": 0.0,    # std / mean of cyclomatic (heterogeneity)
            "n_high_complexity_fns": 0,          # functions with cyclomatic > 10
            # --- v4 features: Halstead metrics (AUC 0.72-0.78) ---
            "halstead_volume": 0.0,              # N * log2(n) — information content
            "halstead_difficulty": 0.0,          # (n1/2) * (N2/n2) — coding difficulty
            "halstead_effort": 0.0,              # volume * difficulty
            "halstead_volume_per_loc": 0.0,      # volume normalized by LOC
            # --- v4 features: token entropy (AUC 0.77-0.79) ---
            "n_unique_tokens": 0,                # vocabulary size
            "n_total_tokens": 0,                 # total tokens
            "token_diversity": 0.0,              # unique / total (inverse AUC: lower = vuln)
            "token_entropy": 0.0,                # Shannon entropy of token distribution
            # --- v4 features: function size distribution (AUC 0.71-0.77) ---
            "fn_loc_gini": 0.0,                  # Gini coefficient of function LOCs
            "fn_loc_cv": 0.0,                    # coefficient of variation of fn LOCs
            "fn_loc_skew": 0.0,                  # skewness of function LOCs
            "fn_loc_p75": 0.0,                   # 75th percentile function LOC
            "fn_loc_p90": 0.0,                   # 90th percentile function LOC
            "fn_loc_iqr": 0.0,                   # IQR of function LOCs
            "complex_fn_loc_ratio": 0.0,         # LOC in complex fns / total LOC
            # --- v4 features: state mutation analysis (AUC 0.74-0.77) ---
            "n_written_state_vars": 0,           # state vars actually written
            "state_var_write_ratio": 0.0,        # written / declared state vars
            "max_fn_state_writes": 0,            # max assignments in any function
            "mean_fn_state_writes": 0.0,         # mean assignments per function
            "state_writes_per_fn": 0.0,          # n_written_state_vars / n_functions
            # --- v4 features: internal coupling (AUC 0.73) ---
            "n_internal_calls": 0,               # internal function-to-function calls
            "internal_call_density": 0.0,        # internal calls per function
            # --- v4 features: data structure complexity (AUC 0.71) ---
            "n_mapping_access": 0,               # array/mapping bracket access
            "n_nested_access": 0,                # nested mapping access (x[y][z])
            "mapping_access_per_fn": 0.0,        # mapping access per function
            # --- v4 features: code duplication (AUC 0.69) ---
            "n_duplicated_lines": 0,             # duplicated non-trivial lines
            "duplication_ratio": 0.0,            # duplicated / total meaningful lines
            # --- v4 features: inheritance analysis (AUC 0.66-0.68) ---
            "max_inheritance_depth": 0,          # deepest inheritance chain
            "total_inheritance_edges": 0,        # total parent references
            # --- v4 features: additional statement/pattern counts ---
            "n_return_stmts": 0,                 # return statement count
            "n_delete_stmts": 0,                 # delete statement count
            "n_new_stmts": 0,                    # new ContractName() count
            "return_per_function": 0.0,          # returns per function
            "n_override_fns": 0,                 # override keyword count
            "n_virtual_fns": 0,                  # virtual keyword count
            "n_ternary_ops": 0,                  # ternary operator count
            "n_type_conversions": 0,             # uint/int/bytes type conversions
            # --- v4 features: size-normalized ratio features ---
            "max_nesting_density": 0.0,          # max nesting / log(function LOC)
            "require_to_assign_ratio": 0.0,      # require / assignments
            "conditional_to_assign_ratio": 0.0,  # if / assignments
            "guard_coverage": 0.0,               # guarded / total public+external fns
            "nested_access_per_fn": 0.0,         # nested access per function
            "comment_to_code_ratio": 0.0,        # comment lines / code lines
            # --- v4 features: import analysis ---
            "n_unique_imports": 0,               # unique import paths
            "n_openzeppelin_imports": 0,         # OpenZeppelin imports
            # Bytecode-backed features
            "has_bytecode": 0,
            "bytecode_len": 0,
            "bc_has_call": 0,
            "bc_has_delegatecall": 0,
            "bc_has_staticcall": 0,
            "bc_has_selfdestruct": 0,
            "bc_n_call": 0,
            "bc_n_delegatecall": 0,
            "bc_n_selfdestruct": 0,
            "bc_proxy_like": 0,
            # --- live (non-code) activity features ---
            "has_activity_payload": 0,
            "age_days": 0.0,
            "last_tx_age_days": 0.0,
            "tx_count_7d": 0,
            "tx_count_30d": 0,
            "tx_count_90d": 0,
            "tx_in_30d": 0,
            "tx_out_30d": 0,
            "unique_senders_30d": 0,
            "failed_tx_rate_90d": 0.0,
            "internal_tx_count_30d": 0,
            "erc20_tx_count_30d": 0,
            "erc20_unique_tokens_30d": 0,
            "erc20_unique_senders_30d": 0,
            "erc20_unique_receivers_30d": 0,
            "value_in_eth_30d": 0.0,
            "value_out_eth_30d": 0.0,
            # admin/governance event churn (last 365d)
            "n_evt_upgraded_365d": 0,
            "n_evt_admin_changed_365d": 0,
            "n_evt_ownership_transferred_365d": 0,
            "n_evt_role_granted_365d": 0,
            "n_evt_role_revoked_365d": 0,
            "n_evt_paused_365d": 0,
            "n_evt_unpaused_365d": 0,
            "admin_event_churn_365d": 0.0,
        }

    @staticmethod
    def _extract_source_features(base: Dict[str, Any], text: str, n_files: int) -> Dict[str, Any]:
        """
        Populate source-code-derived features into *base* from concatenated Solidity *text*.
        This is the shared regex/pattern logic used by both Etherscan-payload and
        local-file extraction paths.  Modifies *base* in place and returns it.
        """
        t = text
        base["loc"] = int(sum(1 for _ in t.splitlines())) if t else 0
        base["n_files"] = int(n_files)
        base["has_verified_source"] = 1 if n_files > 0 else 0

        base["has_upgrade_patterns"] = 1 if any(
            p in t for p in ("upgradeTo(", "upgradeToAndCall(", "UUPSUpgradeable", "TransparentUpgradeableProxy")
        ) else 0
        base["uses_ownable"] = 1 if ("onlyOwner" in t or re.search(r"\bOwnable\b", t)) else 0
        base["uses_two_step_ownable"] = 1 if ("Ownable2Step" in t) else 0
        base["uses_accesscontrol"] = 1 if (
            "onlyRole" in t or "DEFAULT_ADMIN_ROLE" in t or re.search(r"\bAccessControl\b", t)
        ) else 0
        base["uses_pausable"] = 1 if ("whenNotPaused" in t or re.search(r"\bPausable\b", t) or "pause(" in t) else 0
        base["uses_timelock"] = 1 if ("TimelockController" in t or re.search(r"\bTimelock\b", t)) else 0
        base["uses_reentrancy_guard"] = 1 if ("nonReentrant" in t or re.search(r"\bReentrancyGuard\b", t)) else 0
        base["uses_delegatecall"] = 1 if "delegatecall" in t else 0
        base["uses_call"] = 1 if (".call{" in t or ".call(" in t) else 0
        base["uses_staticcall"] = 1 if ("staticcall" in t) else 0
        base["uses_selfdestruct"] = 1 if "selfdestruct" in t else 0
        base["uses_transfer_send"] = 1 if (".transfer(" in t or ".send(" in t) else 0
        base["uses_tx_origin"] = 1 if ("tx.origin" in t) else 0
        base["uses_assembly"] = 1 if re.search(r"\bassembly\s*\{", t) else 0
        base["uses_unchecked"] = 1 if re.search(r"\bunchecked\s*\{", t) else 0

        base["n_external_calls"] = int(len(re.findall(r"\.call\s*(\{|\()", t)))
        base["n_delegatecall"] = int(len(re.findall(r"\bdelegatecall\b", t)))

        base["n_functions"] = int(len(re.findall(r"\bfunction\b", t)))
        base["n_public_external_functions"] = int(len(re.findall(r"\bfunction\b[^{;]*\b(public|external)\b", t)))
        base["n_payable_functions"] = int(len(re.findall(r"\bfunction\b[^{;]*\bpayable\b", t)))
        base["n_modifiers"] = int(len(re.findall(r"\bmodifier\b", t)))

        base["uses_initializer"] = 1 if ("initializer" in t or "reinitializer" in t) else 0
        base["n_role_admin_ops"] = int(
            len(re.findall(r"\b(grantRole|revokeRole|renounceRole|_grantRole|_revokeRole)\b", t))
        )

        base["uses_safe_math"] = 1 if ("SafeMath" in t) else 0
        base["uses_chainlink_oracle"] = 1 if ("AggregatorV3Interface" in t or "latestRoundData" in t or "chainlink" in t.lower()) else 0
        base["uses_uniswap_oracle"] = 1 if ("IUniswapV3Pool" in t or "observe(" in t or "getReserves" in t) else 0
        base["has_fallback_or_receive"] = 1 if (re.search(r"\bfallback\s*\(", t) or re.search(r"\breceive\s*\(", t)) else 0

        # --- New v2 features: additional pattern counts ---
        base["n_state_variables"] = int(len(re.findall(
            r"\b(uint|int|bool|address|bytes|string|mapping)\d{0,3}\s+(public|private|internal)?\s*\w+\s*[;=]", t
        )))
        base["n_events"] = int(len(re.findall(r"\bevent\s+\w+", t)))
        base["n_mappings"] = int(len(re.findall(r"\bmapping\s*\(", t)))
        base["n_require_assert"] = int(len(re.findall(r"\b(require|assert)\s*\(", t)))
        base["uses_abi_encode_packed"] = 1 if "abi.encodePacked" in t else 0
        base["n_loops"] = int(len(re.findall(r"\b(for|while|do)\s*[\({]", t)))

        # --- Derived density/ratio features (v2) ---
        loc = max(1, base["loc"])
        n_funcs = max(1, base["n_functions"])

        base["external_call_density"] = float(base["n_external_calls"]) / loc
        base["delegatecall_density"] = float(base["n_delegatecall"]) / loc
        base["function_density"] = float(base["n_functions"]) / loc
        base["modifier_density"] = float(base["n_modifiers"]) / loc
        base["payable_density"] = float(base["n_payable_functions"]) / loc
        base["public_ratio"] = float(base["n_public_external_functions"]) / n_funcs
        base["payable_ratio"] = float(base["n_payable_functions"]) / n_funcs
        base["avg_loc_per_function"] = float(base["loc"]) / n_funcs
        base["modifier_per_function"] = float(base["n_modifiers"]) / n_funcs
        base["require_per_function"] = float(base["n_require_assert"]) / n_funcs
        base["loop_density"] = float(base["n_loops"]) / n_funcs

        base["dangerous_pattern_count"] = int(
            base["uses_delegatecall"] + base["uses_selfdestruct"] + base["uses_tx_origin"]
            + base["uses_assembly"] + base["uses_unchecked"] + base["uses_abi_encode_packed"]
        )
        base["safety_pattern_count"] = int(
            base["uses_reentrancy_guard"] + base["uses_pausable"] + base["uses_timelock"]
            + base["uses_safe_math"] + base["uses_two_step_ownable"]
        )
        base["dangerous_to_safety_ratio"] = float(
            base["dangerous_pattern_count"] / max(1, base["safety_pattern_count"])
        )
        base["interaction_surface"] = int(
            base["n_external_calls"] + base["n_delegatecall"]
            + base["n_payable_functions"] + base["has_fallback_or_receive"]
        )
        base["interaction_surface_density"] = float(base["interaction_surface"]) / loc
        base["uses_any_oracle"] = 1 if (base["uses_chainlink_oracle"] or base["uses_uniswap_oracle"]) else 0

        # Compiler version parsing: detect pre-0.8 (no built-in overflow protection)
        cv = base.get("compiler_version", "")
        base["is_solc_pre_08"] = _detect_solc_pre_08(cv, t)

        # ======================================================================
        # v3 features — function-level granularity, semantic patterns, vuln detectors
        # ======================================================================
        _extract_v3_features(base, t)

        return base

    def _enrich_activity_features(self, base: Dict[str, Any], act: Dict[str, Any]) -> None:
        """Populate on-chain activity features into *base* from an activity payload dict."""
        base["has_activity_payload"] = 1
        now_ts = _utc_now_ts()

        def _parse_rows(key: str) -> list:
            obj = act.get(key)
            if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                return [r for r in obj["rows"] if isinstance(r, dict)]
            return []

        tx_rows = _parse_rows("txlist")
        internal_rows = _parse_rows("txlistinternal")
        token_rows = _parse_rows("tokentx")

        def within_days(ts: int, days: int) -> bool:
            return ts >= now_ts - int(days) * 86400

        seen_senders_30: set = set()
        tx_in_30 = tx_out_30 = 0
        v_in_30 = v_out_30 = 0.0
        n_fail_90 = n_total_90 = 0
        ts_min: Optional[int] = None
        ts_max: Optional[int] = None

        for r in tx_rows:
            ts = _safe_int(r.get("timeStamp"), 0)
            if ts <= 0:
                continue
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            n_total_90 += 1
            if str(r.get("isError") or "").strip() == "1":
                n_fail_90 += 1
            if within_days(ts, 7):
                base["tx_count_7d"] += 1
            if within_days(ts, 30):
                base["tx_count_30d"] += 1
                frm = str(r.get("from") or "").lower()
                if _is_addr(frm):
                    seen_senders_30.add(frm)
                to = str(r.get("to") or "").lower()
                if to == base["address"]:
                    tx_in_30 += 1
                    v_in_30 += _eth_from_wei(r.get("value"))
                if frm == base["address"]:
                    tx_out_30 += 1
                    v_out_30 += _eth_from_wei(r.get("value"))
            if within_days(ts, 90):
                base["tx_count_90d"] += 1

        base["unique_senders_30d"] = int(len(seen_senders_30))
        base["tx_in_30d"] = int(tx_in_30)
        base["tx_out_30d"] = int(tx_out_30)
        base["value_in_eth_30d"] = float(v_in_30)
        base["value_out_eth_30d"] = float(v_out_30)
        base["failed_tx_rate_90d"] = float(n_fail_90 / max(1, n_total_90))
        if ts_min is not None:
            base["age_days"] = float(max(0.0, (now_ts - int(ts_min)) / 86400.0))
        if ts_max is not None:
            base["last_tx_age_days"] = float(max(0.0, (now_ts - int(ts_max)) / 86400.0))

        n_int_30 = sum(1 for r in internal_rows if _safe_int(r.get("timeStamp"), 0) > 0 and within_days(_safe_int(r.get("timeStamp"), 0), 30))
        base["internal_tx_count_30d"] = int(n_int_30)

        n_tok_30 = 0
        tok_set: set = set()
        tok_senders: set = set()
        tok_receivers: set = set()
        for r in token_rows:
            ts = _safe_int(r.get("timeStamp"), 0)
            if ts <= 0 or not within_days(ts, 30):
                continue
            n_tok_30 += 1
            token_addr = str(r.get("contractAddress") or "").lower()
            if _is_addr(token_addr):
                tok_set.add(token_addr)
            frm = str(r.get("from") or "").lower()
            to = str(r.get("to") or "").lower()
            if _is_addr(frm):
                tok_senders.add(frm)
            if _is_addr(to):
                tok_receivers.add(to)
        base["erc20_tx_count_30d"] = int(n_tok_30)
        base["erc20_unique_tokens_30d"] = int(len(tok_set))
        base["erc20_unique_senders_30d"] = int(len(tok_senders))
        base["erc20_unique_receivers_30d"] = int(len(tok_receivers))

        logs = act.get("logs", {}) if isinstance(act.get("logs"), dict) else {}

        def _count_logs(key: str) -> int:
            entry = logs.get(key) if isinstance(logs, dict) else None
            resp = entry.get("response") if isinstance(entry, dict) else None
            res = resp.get("result") if isinstance(resp, dict) else None
            return int(len([x for x in res if isinstance(x, dict)])) if isinstance(res, list) else 0

        base["n_evt_upgraded_365d"] = _count_logs("upgraded")
        base["n_evt_admin_changed_365d"] = _count_logs("admin_changed")
        base["n_evt_ownership_transferred_365d"] = _count_logs("ownership_transferred")
        base["n_evt_role_granted_365d"] = _count_logs("role_granted")
        base["n_evt_role_revoked_365d"] = _count_logs("role_revoked")
        base["n_evt_paused_365d"] = _count_logs("paused")
        base["n_evt_unpaused_365d"] = _count_logs("unpaused")
        base["admin_event_churn_365d"] = float(sum(
            base[k] for k in (
                "n_evt_upgraded_365d", "n_evt_admin_changed_365d", "n_evt_ownership_transferred_365d",
                "n_evt_role_granted_365d", "n_evt_role_revoked_365d", "n_evt_paused_365d", "n_evt_unpaused_365d",
            )
        ))

    # ------------------------------------------------------------------
    # Public extraction API: from Etherscan payload (existing path)
    # ------------------------------------------------------------------

    def extract_contract_features(self, address: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Lightweight, rule-based feature extraction from Etherscan metadata + Solidity source text.
        """
        base = self._empty_feature_base(address)
        base["has_payload"] = 1 if payload else 0

        if not payload:
            return base

        etherscan = payload.get("etherscan", {}) if isinstance(payload, dict) else {}
        result = etherscan.get("result", []) if isinstance(etherscan, dict) else []
        row = result[0] if isinstance(result, list) and result and isinstance(result[0], dict) else {}

        base["compiler_version"] = str(row.get("CompilerVersion") or "").strip()
        base["is_proxy"] = 1 if str(row.get("Proxy", "")).strip() == "1" else 0
        impl = row.get("Implementation")
        if isinstance(impl, str) and impl.lower().startswith("0x") and len(impl) == 42:
            base["implementation_address"] = impl.lower()

        ps = payload.get("parsed_sources", {}) if isinstance(payload, dict) else {}
        contracts = ps.get("contracts", []) if isinstance(ps, dict) else []
        if isinstance(contracts, list):
            n_files = len([c for c in contracts if isinstance(c, str) and c.strip()])
            text = "\n".join([c for c in contracts if isinstance(c, str)])
        else:
            n_files = 0
            text = ""

        self._extract_source_features(base, text, n_files)

        # --- bytecode features (present even when source is unverified) ---
        bytecode_hex = self._extract_bytecode_hex(payload)
        bb = _hex_to_bytes(bytecode_hex)
        base["has_bytecode"] = 1 if len(bb) > 0 else 0
        base["bytecode_len"] = int(len(bb))
        if len(bb) > 0:
            op = _disassemble_opcode_counts(bytecode_hex)
            base["bc_n_call"] = int(op.get("CALL", 0))
            base["bc_n_delegatecall"] = int(op.get("DELEGATECALL", 0))
            base["bc_n_selfdestruct"] = int(op.get("SELFDESTRUCT", 0))
            base["bc_has_call"] = 1 if base["bc_n_call"] > 0 else 0
            base["bc_has_delegatecall"] = 1 if base["bc_n_delegatecall"] > 0 else 0
            base["bc_has_staticcall"] = 1 if int(op.get("STATICCALL", 0)) > 0 else 0
            base["bc_has_selfdestruct"] = 1 if base["bc_n_selfdestruct"] > 0 else 0
            base["bc_proxy_like"] = 1 if (base["bc_has_delegatecall"] == 1 and base["bytecode_len"] <= 600) else 0

        # --- live activity payload (optional) ---
        act = self.load_etherscan_activity_payload(address)
        if isinstance(act, dict):
            self._enrich_activity_features(base, act)

        return base

    # ------------------------------------------------------------------
    # Public extraction API: from raw Solidity source text (local files)
    # ------------------------------------------------------------------

    def extract_features_from_source_text(
        self,
        contract_id: str,
        source_text: str,
        *,
        n_files: int = 1,
        address: str = "",
    ) -> Dict[str, Any]:
        """
        Extract features from raw Solidity source text (e.g. a local .sol file).

        Only static source-code features are populated; bytecode and on-chain
        activity features remain at their defaults (0).  If an on-chain *address*
        is also available, cached Etherscan data will be used to enrich bytecode
        and activity features.
        """
        identifier = address if _is_addr(address) else contract_id
        base = self._empty_feature_base(identifier)
        base["has_payload"] = 1
        self._extract_source_features(base, source_text, n_files)

        if _is_addr(address):
            src_payload = self.load_etherscan_source_payload(address)
            if isinstance(src_payload, dict):
                bytecode_hex = self._extract_bytecode_hex(src_payload)
                bb = _hex_to_bytes(bytecode_hex)
                base["has_bytecode"] = 1 if len(bb) > 0 else 0
                base["bytecode_len"] = int(len(bb))
                if len(bb) > 0:
                    op = _disassemble_opcode_counts(bytecode_hex)
                    base["bc_n_call"] = int(op.get("CALL", 0))
                    base["bc_n_delegatecall"] = int(op.get("DELEGATECALL", 0))
                    base["bc_n_selfdestruct"] = int(op.get("SELFDESTRUCT", 0))
                    base["bc_has_call"] = 1 if base["bc_n_call"] > 0 else 0
                    base["bc_has_delegatecall"] = 1 if base["bc_n_delegatecall"] > 0 else 0
                    base["bc_has_staticcall"] = 1 if int(op.get("STATICCALL", 0)) > 0 else 0
                    base["bc_has_selfdestruct"] = 1 if base["bc_n_selfdestruct"] > 0 else 0
                    base["bc_proxy_like"] = 1 if (base["bc_has_delegatecall"] == 1 and base["bytecode_len"] <= 600) else 0
            act = self.load_etherscan_activity_payload(address)
            if isinstance(act, dict):
                self._enrich_activity_features(base, act)

        return base

    def extract_features_from_local_file(
        self,
        contract_id: str,
        sol_path: str,
        *,
        address: str = "",
    ) -> Dict[str, Any]:
        """
        Read a local ``.sol`` file (or directory of .sol files) and extract features.

        If *sol_path* is a directory, all ``*.sol`` files inside are concatenated.
        """
        p = Path(sol_path)
        if p.is_dir():
            sols = sorted(p.rglob("*.sol"))
            texts = []
            for s in sols:
                try:
                    texts.append(s.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
            text = "\n".join(texts)
            n_files = len(texts)
        elif p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            n_files = 1 if text else 0
        else:
            text = ""
            n_files = 0

        return self.extract_features_from_source_text(
            contract_id, text, n_files=n_files, address=address,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _complexity_term(self, feats: Dict[str, Any]) -> float:
        """
        v2.1 complexity: blends log-scaled size signals with structural density.

        Empirical finding: in DAppSCAN (21K contracts, best labeled), vulnerable
        contracts are LARGER and more complex (more state vars, functions, loops).
        The signal is: more code surface = more potential bugs (AUC > 0.75 for
        n_state_variables, loc, n_functions, n_mappings).

        We use log-scaling to avoid outlier domination, and mix in per-function
        structural density to capture concentrated complexity.
        """
        loc = float(feats.get("loc", 0) or 0)
        n_funcs = float(feats.get("n_functions", 0) or 0)
        n_state_vars = float(feats.get("n_state_variables", 0) or 0)
        n_mappings = float(feats.get("n_mappings", 0) or 0)
        n_events = float(feats.get("n_events", 0) or 0)
        n_loops = float(feats.get("n_loops", 0) or 0)
        req_pf = float(feats.get("require_per_function", 0) or 0)

        term = 0.0
        # Size surface (log-scaled): more code = more potential bugs
        term += 0.35 * math.log1p(max(0.0, loc) / 100.0)
        term += 0.25 * math.log1p(max(0.0, n_funcs) / 10.0)
        # State storage complexity (DAppSCAN AUC 0.81 for n_state_variables)
        term += 0.40 * math.log1p(max(0.0, n_state_vars) / 5.0)
        term += 0.30 * math.log1p(max(0.0, n_mappings) / 3.0)
        # Event complexity (DAppSCAN AUC 0.70)
        term += 0.15 * math.log1p(max(0.0, n_events) / 3.0)
        # Loop complexity (DAppSCAN AUC 0.66)
        term += 0.20 * math.log1p(max(0.0, n_loops) / 3.0)
        # Require density bonus: more requires per function = more complex logic
        # (DAppSCAN AUC 0.72 for require_per_function)
        term += 0.25 * min(3.0, req_pf)
        return float(term)

    def score_components(self, feats: Dict[str, Any]) -> Dict[str, float]:
        """
        Return explainable component scores. Total raw score is a weighted sum of these components.

        v2 changes:
        - comp_opacity is still computed (for analysis/transparency) but NOT added to risk_raw.
        - comp_complexity redesigned: uses density metrics, not raw LoC.
        - comp_code_smell added: density-based combined signal from dangerous patterns,
          interaction surface, and safety-gap indicators.
        """
        c = self.cfg

        # Upgradeability: prefer explicit source/metadata; fall back to bytecode proxy-like signal.
        is_upgradeable = 1 if (
            int(feats.get("is_proxy", 0)) == 1
            or int(feats.get("has_upgrade_patterns", 0)) == 1
            or int(feats.get("bc_proxy_like", 0)) == 1
        ) else 0
        admin_power = 1 if (int(feats.get("uses_ownable", 0)) == 1 or int(feats.get("uses_accesscontrol", 0)) == 1) else 0

        upgrade = (
            (c.w_proxy_flag * int(feats.get("is_proxy", 0)))
            + (c.w_upgrade_patterns * int(feats.get("has_upgrade_patterns", 0)))
            + (0.5 * int(feats.get("bc_proxy_like", 0)))
        )
        # privilege includes "can someone change things?" + "admin ops exist?" + upgradeable init patterns
        privilege = 0.0
        privilege += c.w_only_owner * int(feats.get("uses_ownable", 0))
        privilege += c.w_access_control * int(feats.get("uses_accesscontrol", 0))
        privilege += c.w_role_admin_ops * math.log1p(max(0.0, float(feats.get("n_role_admin_ops", 0) or 0)))
        privilege += c.w_initializer * int(feats.get("uses_initializer", 0)) * float(is_upgradeable)

        # interaction: more external calls / payable surface tends to increase attack surface
        interaction = 0.0
        interaction += c.w_external_call * math.log1p(max(0.0, float(feats.get("n_external_calls", 0) or 0)))
        interaction += 0.5 * int(feats.get("uses_call", 0))
        interaction += 0.3 * int(feats.get("uses_staticcall", 0))
        # Bytecode fallback signals (helpful for unverified sources)
        interaction += 0.4 * math.log1p(max(0.0, float(feats.get("bc_n_call", 0) or 0)))
        interaction += 0.2 * int(feats.get("bc_has_staticcall", 0))
        interaction += c.w_transfer_send * int(feats.get("uses_transfer_send", 0))
        interaction += 0.3 * math.log1p(max(0.0, float(feats.get("n_payable_functions", 0) or 0)))
        # oracle dependencies add complexity/integration risk
        interaction += 0.7 * int(feats.get("uses_any_oracle", 0))
        # fallback/receive = open entry point
        interaction += 0.4 * int(feats.get("has_fallback_or_receive", 0))

        # dangerous / footguns
        dangerous = 0.0
        dangerous += c.w_selfdestruct * int(feats.get("uses_selfdestruct", 0))
        dangerous += c.w_delegatecall * (int(feats.get("uses_delegatecall", 0)) + 0.3 * math.log1p(max(0.0, float(feats.get("n_delegatecall", 0) or 0))))
        # Bytecode fallback (unverified)
        dangerous += 0.75 * c.w_selfdestruct * int(feats.get("bc_has_selfdestruct", 0))
        dangerous += 0.75 * c.w_delegatecall * (int(feats.get("bc_has_delegatecall", 0)) + 0.3 * math.log1p(max(0.0, float(feats.get("bc_n_delegatecall", 0) or 0))))
        dangerous += c.w_tx_origin * int(feats.get("uses_tx_origin", 0))
        dangerous += c.w_assembly * int(feats.get("uses_assembly", 0))
        dangerous += c.w_unchecked * int(feats.get("uses_unchecked", 0))
        dangerous += 0.5 * int(feats.get("uses_abi_encode_packed", 0))  # hash collision risk
        # Pre-0.8 compiler without SafeMath = arithmetic risk
        dangerous += 1.5 * int(feats.get("is_solc_pre_08", 0)) * (1 - int(feats.get("uses_safe_math", 0)))

        complexity = self._complexity_term(feats)

        # opacity: computed for transparency but NOT included in risk_raw (it's a confound).
        has_src = float(int(feats.get("has_verified_source", 0)))
        has_bc = float(int(feats.get("has_bytecode", 0)))
        opacity = (1.0 - has_src) * (1.0 - 0.7 * has_bc)

        # --- NEW: comp_code_smell --- combined absolute + density signal
        # Blends absolute dangerous-pattern counts (which have genuine AUC signal)
        # with density features where they add value. Avoids pure density which
        # inverted the signal on DAppSCAN (large projects had low density).
        code_smell = 0.0
        # Absolute interaction surface (AUC 0.55 overall, much higher DAppSCAN)
        code_smell += 0.3 * math.log1p(max(0.0, float(feats.get("interaction_surface", 0) or 0)))
        # Absolute dangerous pattern count
        code_smell += 0.5 * float(feats.get("dangerous_pattern_count", 0) or 0)
        # Dangerous-to-safety imbalance (genuine discriminator)
        code_smell += 0.4 * float(feats.get("dangerous_to_safety_ratio", 0) or 0)
        # Payable ratio (genuinely discriminative: more payable entry = more ETH risk)
        code_smell += 0.8 * float(feats.get("payable_ratio", 0) or 0)
        # Modifier density (AUC 0.55 overall, 0.65 DAppSCAN)
        code_smell += 0.6 * float(feats.get("modifier_density", 0) or 0)
        # External call density (still valid, but dampened)
        code_smell += 1.0 * float(feats.get("external_call_density", 0) or 0)
        code_smell += 2.0 * float(feats.get("delegatecall_density", 0) or 0)
        # n_loops absolute (DAppSCAN AUC 0.66)
        code_smell += 0.15 * math.log1p(max(0.0, float(feats.get("n_loops", 0) or 0)))
        req_pf = float(feats.get("require_per_function", 0) or 0)

        # safety credits (subtract)
        safety = 0.0
        safety += c.w_reentrancy_guard * int(feats.get("uses_reentrancy_guard", 0))
        safety += c.w_pausable * int(feats.get("uses_pausable", 0))
        safety += c.w_timelock * int(feats.get("uses_timelock", 0))
        safety += c.w_safe_math * int(feats.get("uses_safe_math", 0))
        safety += c.w_two_step_ownable * int(feats.get("uses_two_step_ownable", 0))
        # Require/assert density contributes to safety (more checks = safer)
        safety += 0.3 * min(2.0, req_pf)  # cap at 2 requires/function for credit

        # Live activity / exposure: higher usage + higher failure rate + admin churn => higher operational risk.
        tx_30 = float(_safe_int(feats.get("tx_count_30d"), 0))
        uniq_30 = float(_safe_int(feats.get("unique_senders_30d"), 0))
        tok_30 = float(_safe_int(feats.get("erc20_tx_count_30d"), 0))
        v_flow = float(max(0.0, _safe_float(feats.get("value_in_eth_30d"), 0.0) + _safe_float(feats.get("value_out_eth_30d"), 0.0)))
        fail_rate = _clamp01(_safe_float(feats.get("failed_tx_rate_90d"), 0.0))
        churn_365 = float(max(0.0, _safe_float(feats.get("admin_event_churn_365d"), 0.0)))

        activity = 0.0
        activity += c.w_tx_activity * math.log1p(tx_30)
        activity += c.w_unique_senders * math.log1p(uniq_30)
        activity += c.w_token_activity * math.log1p(tok_30)
        activity += c.w_value_flow_eth * math.log1p(v_flow)
        activity += c.w_failed_rate * fail_rate
        activity += c.w_admin_event_churn * math.log1p(churn_365)

        return {
            "comp_upgrade": float(upgrade),
            "comp_privilege": float(privilege),
            "comp_interaction": float(interaction),
            "comp_dangerous": float(dangerous),
            "comp_complexity": float(complexity),
            "comp_opacity": float(opacity),        # computed but NOT added to risk_raw
            "comp_code_smell": float(code_smell),   # NEW v2 component
            "comp_safety": float(safety),
            "comp_activity": float(activity),
            "aux_is_upgradeable": float(is_upgradeable),
            "aux_admin_power": float(admin_power),
        }

    def score_contract(self, feats: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Convert feature dict into a single raw risk score (higher => riskier).

        v2: comp_opacity is excluded from the sum (it's a dataset confound).
            comp_code_smell is added (density-based combined signal).
        """
        c = self.cfg
        comps = self.score_components(feats)
        score = 0.0
        score += c.w_upgrade * comps["comp_upgrade"]
        score += c.w_privilege * comps["comp_privilege"]
        score += c.w_interaction * comps["comp_interaction"]
        score += c.w_dangerous * comps["comp_dangerous"]
        score += c.w_complexity * comps["comp_complexity"]
        # NOTE: comp_opacity intentionally NOT added (confound, see config docstring)
        score += c.w_code_smell * comps["comp_code_smell"]
        score += c.w_activity * comps["comp_activity"]
        score -= c.w_safety * comps["comp_safety"]
        return float(score), comps

    def run(self, *, registry_csv: Path, out_csv: Path, limit: int = 0) -> Path:
        self.ensure_dirs()
        reg = pd.read_csv(Path(registry_csv))
        if "address" not in reg.columns:
            raise ValueError(f"Registry CSV must include `address` column: {registry_csv}")
        addrs = [str(a).strip().lower() for a in reg["address"].tolist()]
        addrs = [a for a in addrs if _is_addr(a)]
        if limit and limit > 0:
            addrs = addrs[: int(limit)]

        rows = []
        missing = 0
        for i, addr in enumerate(addrs, start=1):
            payload = self.load_etherscan_source_payload(addr)
            if payload is None:
                missing += 1
            feats = self.extract_contract_features(addr, payload)
            if payload:
                raw, comps = self.score_contract(feats)
            else:
                raw, comps = 0.0, self.score_components(feats)
            # Uncertainty score: not a label, just a measurement-quality signal.
            has_src = float(int(feats.get("has_verified_source", 0)))
            has_bc = float(int(feats.get("has_bytecode", 0)))
            risk_uncertainty = (1.0 - has_src) * (0.7 - 0.4 * has_bc) + (1.0 - has_bc) * 0.3
            rows.append({**feats, **comps, "risk_raw": float(raw), "risk_uncertainty": float(max(0.0, risk_uncertainty))})
            if i == 1 or i % 200 == 0 or i == len(addrs):
                self.logger.info(f"[contract risk] {i}/{len(addrs)}")

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No rows produced for contract risk scoring.")

        x = pd.to_numeric(df["risk_raw"], errors="coerce").fillna(0.0)
        mu = float(x.mean())
        sigma = float(x.std())
        sigma = sigma if sigma and sigma > 0 else 1.0
        df["risk_z"] = (x - mu) / sigma

        # Convenience rank columns
        df["risk_rank"] = pd.to_numeric(df["risk_z"], errors="coerce").rank(ascending=False, method="min").astype(int)

        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        self.logger.info(f"Wrote contract risk CSV: {out_csv} (rows={len(df)})")
        self.logger.info(f"Missing etherscan payloads: {missing}/{len(addrs)} (risk_raw=0 for missing)")
        return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute contract-level technical risk scores.")
    parser.add_argument(
        "--registry-csv",
        type=str,
        default=str(DATA_DIR / "contract_registry.csv"),
        help="Contract registry CSV (must have `address`).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DERIVED_DIR / "contract_risk.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0 = no limit).")
    args = parser.parse_args()

    ContractRiskUtil().run(registry_csv=Path(args.registry_csv), out_csv=Path(args.out_csv), limit=int(args.limit))


if __name__ == "__main__":
    main()
