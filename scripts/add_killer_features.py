"""
Add new features to the existing scored CSV to close the gap with DL baselines.

Strategy: TextCNN learns from raw token n-grams. We add features that capture
similar patterns but remain interpretable:
1. Token trigram features (what TextCNN filter-size-3 learns)
2. Additional SWC vulnerability detectors (uncovered patterns)
3. More nuanced code structure features
4. Better state mutation and external call patterns
"""
import re
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.features import load_scored_csv

_SOL_TOKEN_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|"
    r'"(?:[^"\\]|\\.)*"|'
    r"'(?:[^'\\]|\\.)*'|"
    r"0x[0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?|"
    r"[a-zA-Z_$][a-zA-Z0-9_$]*|"
    r"[{}()\[\];,.]|"
    r"[+\-*/%&|^~<>=!]=?=?|>>>=?|<<=?|&&|\|\||=>|"
    r"\S",
    re.MULTILINE,
)

# --- Risky trigram patterns (what TextCNN filter-size-3 captures) ---
RISKY_TRIGRAMS = [
    ("msg", ".", "value"),
    ("msg", ".", "sender"),
    ("tx", ".", "origin"),
    ("block", ".", "timestamp"),
    ("block", ".", "number"),
    ("this", ".", "balance"),
    (".", "call", "{"),
    (".", "call", "("),
    (".", "delegatecall", "("),
    (".", "transfer", "("),
    (".", "send", "("),
    ("address", "(", "0"),
    ("selfdestruct", "(", ""),
    ("suicide", "(", ""),
]

SAFE_TRIGRAMS = [
    ("require", "(", "msg"),
    ("require", "(", "_"),
    ("onlyOwner", "(", ")"),
    ("nonReentrant", "(", ")"),
    ("whenNotPaused", "(", ")"),
    ("SafeMath", ".", "add"),
    ("SafeMath", ".", "sub"),
    ("SafeERC20", ".", "safeTransfer"),
]


def read_source(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def compute_new_features(text: str) -> Dict[str, float]:
    """Compute new killer features from Solidity source text."""
    feats: Dict[str, float] = {}
    if not text.strip():
        return _empty_new_features()

    tokens = _SOL_TOKEN_RE.findall(text)
    lines = text.split("\n")
    n_lines = max(len(lines), 1)
    n_tokens = max(len(tokens), 1)

    # ----- 1. TOKEN TRIGRAM FEATURES -----
    trigrams = []
    for i in range(len(tokens) - 2):
        trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))
    
    trigram_counter = Counter(trigrams)
    n_trigrams = max(len(trigrams), 1)
    n_unique_trigrams = len(trigram_counter)
    
    feats["trigram_diversity"] = n_unique_trigrams / n_trigrams
    
    # Trigram entropy
    if trigram_counter:
        probs = np.array(list(trigram_counter.values()), dtype=float)
        probs /= probs.sum()
        feats["trigram_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        feats["trigram_entropy"] = 0.0

    # Count risky trigrams (flexible matching)
    n_risky_tri = 0
    for rt in RISKY_TRIGRAMS:
        for tri in trigrams:
            if all(rt[j] == "" or rt[j] == tri[j] for j in range(3)):
                n_risky_tri += 1
                break  # count pattern once per type
    feats["n_risky_trigrams"] = n_risky_tri
    feats["risky_trigram_density"] = n_risky_tri / n_trigrams

    # Count safe trigrams
    n_safe_tri = 0
    for st in SAFE_TRIGRAMS:
        for tri in trigrams:
            if all(st[j] == "" or st[j] == tri[j] for j in range(3)):
                n_safe_tri += 1
                break
    feats["n_safe_trigrams"] = n_safe_tri
    feats["safe_to_risky_trigram_ratio"] = (
        n_safe_tri / max(n_risky_tri, 1)
    )

    # ----- 2. SWC VULNERABILITY DETECTORS (uncovered patterns) -----
    
    # SWC-100: Default visibility (functions without explicit visibility)
    func_defs = re.findall(
        r"function\s+\w+\s*\([^)]*\)\s*((?:public|external|internal|private|pure|view|payable|virtual|override|\s)*)\s*(?:returns\s*\([^)]*\))?\s*[{;]",
        text
    )
    n_no_visibility = sum(
        1 for fd in func_defs
        if not re.search(r"\b(public|external|internal|private)\b", fd)
    )
    feats["n_default_visibility_fns"] = n_no_visibility

    # SWC-102/103: Outdated or floating pragma
    pragma_match = re.search(r"pragma\s+solidity\s+([^;]+)", text)
    feats["has_floating_pragma"] = int(
        bool(pragma_match and "^" in pragma_match.group(1))
    )
    feats["has_locked_pragma"] = int(
        bool(pragma_match and "^" not in pragma_match.group(1) and 
             ("=" in pragma_match.group(1) or pragma_match.group(1).strip()[0].isdigit()))
    )

    # SWC-106: Unprotected selfdestruct
    selfdestruct_matches = list(re.finditer(
        r"\bselfdestruct\s*\(", text
    ))
    n_selfdestruct = len(selfdestruct_matches)
    feats["n_selfdestruct"] = n_selfdestruct

    # SWC-111: Deprecated functions
    deprecated = ["suicide", "throw", "sha3", "callcode", "msg.gas", "block.blockhash"]
    n_deprecated = sum(
        len(re.findall(rf"\b{re.escape(d)}\b", text)) for d in deprecated
    )
    feats["n_deprecated_patterns"] = n_deprecated

    # SWC-113: DoS with failed call (transfer in loop)
    feats["has_transfer_in_loop"] = int(bool(re.search(
        r"\b(for|while)\s*\([^)]*\)\s*\{[^}]*(\.transfer\s*\(|\.send\s*\(|\.call\s*[\({])",
        text, re.DOTALL
    )))

    # SWC-120: Weak randomness (block.difficulty/prevrandao, blockhash)
    n_weak_random = (
        len(re.findall(r"\bblock\.difficulty\b", text)) +
        len(re.findall(r"\bblock\.prevrandao\b", text)) +
        len(re.findall(r"\bblockhash\s*\(", text))
    )
    feats["n_weak_randomness"] = n_weak_random

    # SWC-128: DoS with block gas limit (unbounded loops)
    array_loop_patterns = re.findall(
        r"\bfor\s*\(\s*\w+\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\w+)\.length",
        text
    )
    feats["n_unbounded_loops"] = len(array_loop_patterns)

    # SWC-131: Unused variables (state vars never referenced after declaration)
    # Simplified: count state vars declared but never used in function bodies
    state_var_decls = re.findall(
        r"\b(?:uint|int|bool|address|bytes|string|mapping)\w*\s+(?:public\s+|private\s+|internal\s+)?(\w+)\s*[;=]",
        text
    )
    func_bodies = re.findall(
        r"function\s+\w+[^{]*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        text, re.DOTALL
    )
    all_func_text = " ".join(func_bodies)
    n_unused_state = sum(
        1 for sv in state_var_decls
        if sv not in all_func_text and len(sv) > 1
    )
    feats["n_possibly_unused_state_vars"] = n_unused_state

    # SWC-132: Unexpected ether balance (address(this).balance)
    feats["n_balance_check"] = len(re.findall(
        r"address\s*\(\s*this\s*\)\s*\.\s*balance", text
    ))

    # SWC-134: Hardcoded gas
    feats["n_hardcoded_gas"] = len(re.findall(
        r"\.call\s*\{\s*gas\s*:", text
    ))

    # ----- 3. ADDITIONAL CODE STRUCTURE FEATURES -----

    # Compound assignments (missed by current assignment counting)
    n_compound = (
        len(re.findall(r"\+=", text)) +
        len(re.findall(r"-=", text)) +
        len(re.findall(r"\*=", text)) +
        len(re.findall(r"/=", text)) +
        len(re.findall(r"\+\+", text)) +
        len(re.findall(r"--", text))
    )
    feats["n_compound_assignments"] = n_compound
    feats["compound_assignment_density"] = n_compound / n_lines

    # Ternary operator complexity
    feats["n_ternary"] = len(re.findall(r"\?[^:]+:", text))

    # CREATE/CREATE2 in source
    feats["n_create_new"] = len(re.findall(r"\bnew\s+\w+\s*\(", text))
    feats["has_create2"] = int(bool(re.search(
        r"\bassembly\b[^}]*\bcreate2\b", text, re.DOTALL
    )))

    # Inline assembly analysis
    asm_blocks = re.findall(r"\bassembly\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text, re.DOTALL)
    asm_text = " ".join(asm_blocks)
    feats["n_assembly_blocks"] = len(asm_blocks)
    feats["asm_has_sstore"] = int("sstore" in asm_text)
    feats["asm_has_sload"] = int("sload" in asm_text)
    feats["asm_has_mstore"] = int("mstore" in asm_text)
    feats["asm_has_call"] = int(bool(re.search(r"\bcall\b", asm_text)))
    feats["asm_has_delegatecall"] = int(bool(re.search(r"\bdelegatecall\b", asm_text)))
    feats["asm_has_returndatasize"] = int("returndatasize" in asm_text)
    feats["n_asm_opcodes"] = len(re.findall(r"\b\w+\b", asm_text))

    # ----- 4. EXTERNAL CALL PATTERN ANALYSIS -----
    
    # Call with value (sending ETH)
    feats["n_call_with_value"] = len(re.findall(
        r"\.call\s*\{\s*value\s*:", text
    ))
    
    # Low-level call without value (data-only call)
    feats["n_call_no_value"] = max(
        0,
        len(re.findall(r"\.call\s*\(", text)) - 
        len(re.findall(r"\.call\s*\{\s*value", text))
    )

    # External calls to specific addresses vs arbitrary
    feats["n_address_param_calls"] = len(re.findall(
        r"\w+\s*\.\s*call\s*[\({]", text
    ))

    # ----- 5. MODIFIER & ACCESS CONTROL DEPTH -----
    
    # Specific modifier counts
    feats["n_only_owner"] = len(re.findall(r"\bonlyOwner\b", text))
    feats["n_only_role"] = len(re.findall(r"\bonlyRole\b", text))
    feats["n_when_not_paused"] = len(re.findall(r"\bwhenNotPaused\b", text))
    feats["n_non_reentrant"] = len(re.findall(r"\bnonReentrant\b", text))
    feats["n_initializer"] = len(re.findall(r"\binitializer\b", text))
    
    # Require message ratio (require with string vs without)
    n_require_with_msg = len(re.findall(r'require\s*\([^)]*"[^"]*"', text))
    n_require_total = len(re.findall(r"require\s*\(", text))
    feats["require_msg_ratio"] = (
        n_require_with_msg / max(n_require_total, 1)
    )

    # ----- 6. DeFi-SPECIFIC PATTERNS -----
    
    # Flash loan patterns
    feats["has_flash_callback"] = int(bool(re.search(
        r"function\s+\w*(flashLoan|flash|executeOperation|uniswapV\d|pancakeCall)\w*\s*\(",
        text, re.IGNORECASE
    )))

    # Price oracle patterns
    feats["n_price_oracle"] = (
        len(re.findall(r"\bgetPrice\b|\blatestAnswer\b|\blatestRoundData\b", text)) +
        len(re.findall(r"\bgetReserves\b|\bgetAmountsOut\b", text))
    )

    # Token approval patterns
    feats["n_approve_calls"] = len(re.findall(r"\.approve\s*\(", text))
    feats["n_increase_allowance"] = len(re.findall(
        r"\b(increaseAllowance|safeIncreaseAllowance)\s*\(", text
    ))

    # Reentrancy guard usage relative to external calls
    n_ext_calls = len(re.findall(r"\.(call|delegatecall|transfer|send)\s*[\({]", text))
    n_reentrancy_guards = len(re.findall(r"\bnonReentrant\b", text))
    feats["ext_call_to_guard_ratio"] = n_ext_calls / max(n_reentrancy_guards, 1)

    # ----- 7. CODE QUALITY INDICATORS -----
    
    # Error handling sophistication
    feats["n_custom_errors"] = len(re.findall(r"\berror\s+\w+\s*\(", text))
    feats["n_revert_with_error"] = len(re.findall(r"\brevert\s+\w+\s*\(", text))
    feats["n_try_catch"] = len(re.findall(r"\btry\b", text))

    # Code maturity signals
    feats["has_license"] = int(bool(re.search(r"SPDX-License-Identifier", text)))
    feats["n_todo_fixme"] = len(re.findall(r"\b(TODO|FIXME|HACK|XXX)\b", text))

    return feats


def _empty_new_features() -> Dict[str, float]:
    """Return empty feature dict with all zeros."""
    return {k: 0.0 for k in [
        "trigram_diversity", "trigram_entropy", "n_risky_trigrams",
        "risky_trigram_density", "n_safe_trigrams", "safe_to_risky_trigram_ratio",
        "n_default_visibility_fns", "has_floating_pragma", "has_locked_pragma",
        "n_selfdestruct", "n_deprecated_patterns", "has_transfer_in_loop",
        "n_weak_randomness", "n_unbounded_loops", "n_possibly_unused_state_vars",
        "n_balance_check", "n_hardcoded_gas",
        "n_compound_assignments", "compound_assignment_density", "n_ternary",
        "n_create_new", "has_create2",
        "n_assembly_blocks", "asm_has_sstore", "asm_has_sload", "asm_has_mstore",
        "asm_has_call", "asm_has_delegatecall", "asm_has_returndatasize",
        "n_asm_opcodes",
        "n_call_with_value", "n_call_no_value", "n_address_param_calls",
        "n_only_owner", "n_only_role", "n_when_not_paused",
        "n_non_reentrant", "n_initializer", "require_msg_ratio",
        "has_flash_callback", "n_price_oracle", "n_approve_calls",
        "n_increase_allowance", "ext_call_to_guard_ratio",
        "n_custom_errors", "n_revert_with_error", "n_try_catch",
        "has_license", "n_todo_fixme",
    ]}


def main():
    print("Loading scored CSV...")
    df = load_scored_csv()
    print(f"  {len(df)} contracts")

    paths = df["local_source_path"].tolist()
    contract_ids = df["contract_id"].tolist()

    print(f"Computing new features for {len(df)} contracts...")
    t0 = time.time()
    
    new_feat_rows = []
    skipped = 0
    for i, (cid, p) in enumerate(zip(contract_ids, paths)):
        text = read_source(str(p))
        if text.strip():
            feats = compute_new_features(text)
        else:
            feats = _empty_new_features()
            skipped += 1
        feats["contract_id"] = cid
        new_feat_rows.append(feats)
        
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(df)} ({elapsed:.1f}s, "
                  f"{skipped} skipped)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({skipped} contracts skipped)")

    new_df = pd.DataFrame(new_feat_rows)
    n_new = len([c for c in new_df.columns if c != "contract_id"])
    print(f"  {n_new} new features computed")

    # Merge with existing data
    merged = df.merge(new_df, on="contract_id", how="left", suffixes=("", "_new"))
    
    # Save
    out_path = Path("data/derived/unified_contract_eval_scored_v2.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved to {out_path} ({len(merged)} rows, "
          f"{len(merged.columns)} total columns)")

    # Quick discrimination check
    print("\n--- Quick MI check on new features (DAppSCAN) ---")
    from sklearn.feature_selection import mutual_info_classif
    
    dappscan = merged[
        (merged["source_dataset"] == "DAppSCAN") &
        (merged["label_status"].isin(["clean", "vulnerable"]))
    ].copy()
    y = (dappscan["label_status"] == "vulnerable").astype(int).values
    
    new_cols = [c for c in new_df.columns if c != "contract_id"]
    X_new = dappscan[new_cols].fillna(0).values
    
    mi = mutual_info_classif(X_new, y, random_state=42)
    mi_df = pd.DataFrame({"feature": new_cols, "mi": mi}).sort_values("mi", ascending=False)
    print(mi_df.head(20).to_string(index=False))
    
    # Show how many have non-zero MI
    n_informative = (mi > 0).sum()
    print(f"\n{n_informative}/{n_new} new features have non-zero MI")


if __name__ == "__main__":
    main()
