#!/usr/bin/env python3 -u
"""
Analyze whether new features can bridge the gap on small contracts.

Strategy:
1. Extract MANY candidate features from source code (security patterns, ratios, density metrics)
2. Test discriminative power on the CB_wins vs Clean subset
3. Test discriminative power on the full dataset via quick CV
"""
import sys
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"

from src.ml.semantic_features import read_source, _split_identifier, _extract_identifiers, tokenize_solidity


# ── Candidate new features targeting small-contract discrimination ──────

def extract_security_pattern_features(source: str) -> dict:
    """Extract fine-grained security-relevant patterns, especially useful for small contracts."""
    if not source:
        return {}
    
    lines = source.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]
    loc = len(code_lines)
    
    feats = {}
    
    # ── 1. ERC standard compliance signals ──────────────────────
    is_erc20 = bool(re.search(r'\bIERC20|ERC20\b', source))
    is_erc721 = bool(re.search(r'\bIERC721|ERC721\b', source))
    is_erc1155 = bool(re.search(r'\bIERC1155|ERC1155\b', source))
    feats['is_erc20'] = int(is_erc20)
    feats['is_erc721'] = int(is_erc721)
    feats['is_erc1155'] = int(is_erc1155)
    feats['is_any_token'] = int(is_erc20 or is_erc721 or is_erc1155)
    
    # ERC20 safety checks
    if is_erc20:
        feats['erc20_has_approve'] = int('approve(' in source)
        feats['erc20_has_transferFrom'] = int('transferFrom(' in source)
        feats['erc20_has_increaseAllowance'] = int('increaseAllowance' in source or 'safeIncreaseAllowance' in source)
        feats['erc20_has_decreaseAllowance'] = int('decreaseAllowance' in source or 'safeDecreaseAllowance' in source)
        feats['erc20_has_mint'] = int(bool(re.search(r'\b_?mint\b', source)))
        feats['erc20_has_burn'] = int(bool(re.search(r'\b_?burn\b', source)))
        feats['erc20_zero_check'] = int('address(0)' in source or 'address(0x0)' in source)
        feats['erc20_has_totalSupply'] = int('totalSupply' in source)
    else:
        for k in ['erc20_has_approve', 'erc20_has_transferFrom', 'erc20_has_increaseAllowance',
                   'erc20_has_decreaseAllowance', 'erc20_has_mint', 'erc20_has_burn',
                   'erc20_zero_check', 'erc20_has_totalSupply']:
            feats[k] = 0
    
    # ── 2. Missing safety pattern detection ─────────────────────
    n_fn = len(re.findall(r'\bfunction\s+\w+', source))
    n_external = len(re.findall(r'\bfunction\s+\w+[^{]*\b(?:external|public)\b', source))
    n_require = source.count('require(')
    n_assert = source.count('assert(')
    n_revert = source.count('revert(') + source.count('revert ')
    n_modifiers_used = len(re.findall(r'\bfunction\s+\w+[^{]*\b(?:onlyOwner|nonReentrant|whenNotPaused|onlyRole|onlyAdmin)\b', source))
    
    feats['n_external_public_fns'] = n_external
    feats['n_checks'] = n_require + n_assert + n_revert
    feats['checks_per_fn'] = (n_require + n_assert + n_revert) / max(n_fn, 1)
    feats['checks_per_external'] = (n_require + n_assert + n_revert) / max(n_external, 1)
    feats['has_any_guard'] = int(n_modifiers_used > 0)
    feats['guard_coverage'] = n_modifiers_used / max(n_external, 1)
    
    # Missing safety: external functions without any require/modifier
    fns = re.finditer(r'\bfunction\s+(\w+)\s*\([^)]*\)\s*([^{]*)\{', source)
    unprotected_external = 0
    total_external = 0
    for m in fns:
        vis_block = m.group(2)
        if 'external' in vis_block or 'public' in vis_block:
            total_external += 1
            fn_start = m.end()
            # Find the matching closing brace (simplified)
            depth = 1
            fn_end = fn_start
            while fn_end < len(source) and depth > 0:
                if source[fn_end] == '{':
                    depth += 1
                elif source[fn_end] == '}':
                    depth -= 1
                fn_end += 1
            fn_body = source[fn_start:fn_end]
            has_check = 'require(' in fn_body or 'assert(' in fn_body or 'revert(' in fn_body
            has_modifier = bool(re.search(r'\bonlyOwner|nonReentrant|whenNotPaused|onlyRole|onlyAdmin|auth\b', vis_block))
            if not has_check and not has_modifier:
                unprotected_external += 1
    
    feats['n_unprotected_external'] = unprotected_external
    feats['ratio_unprotected'] = unprotected_external / max(total_external, 1)
    
    # ── 3. Dangerous pattern density ────────────────────────────
    dangerous_ops = {
        'transfer': source.count('.transfer('),
        'send': source.count('.send('),
        'call_value': len(re.findall(r'\.call\{?\s*value', source)) + source.count('.call.value('),
        'delegatecall': source.count('delegatecall('),
        'selfdestruct': source.count('selfdestruct('),
        'suicide': source.count('suicide('),
        'assembly_blocks': len(re.findall(r'\bassembly\s*\{', source)),
        'tx_origin': source.count('tx.origin'),
        'block_timestamp': source.count('block.timestamp'),
        'blockhash': source.count('blockhash('),
        'extcodesize': source.count('extcodesize'),
        'staticcall': source.count('staticcall('),
    }
    
    for op, count in dangerous_ops.items():
        feats[f'dangerous_{op}'] = count
    
    feats['total_dangerous_ops'] = sum(dangerous_ops.values())
    feats['dangerous_per_fn'] = feats['total_dangerous_ops'] / max(n_fn, 1)
    feats['dangerous_per_loc'] = feats['total_dangerous_ops'] / max(loc, 1)
    
    # ── 4. State mutation indicators ────────────────────────────
    n_state_vars = len(re.findall(r'\b(?:uint\d*|int\d*|address|bool|string|bytes\d*|mapping)\s+(?:public\s+|private\s+|internal\s+)?(?!memory|calldata|storage)\w+\s*[;=]', source))
    feats['n_state_vars'] = n_state_vars
    feats['state_vars_per_fn'] = n_state_vars / max(n_fn, 1)
    
    # Storage writes (simplified: assignments to state)
    n_sstore = len(re.findall(r'\bsstore\b', source))
    feats['has_sstore'] = int(n_sstore > 0)
    
    # ── 5. Reentrancy vulnerability signals ─────────────────────
    has_nonreentrant = 'nonReentrant' in source or 'ReentrancyGuard' in source
    has_external_call = any(v > 0 for k, v in dangerous_ops.items() if k in ['call_value', 'delegatecall', 'send', 'transfer'])
    feats['reentrancy_risk'] = int(has_external_call and not has_nonreentrant)
    
    # ── 6. Access control completeness ──────────────────────────
    has_ownable = 'Ownable' in source
    has_access_control = 'AccessControl' in source
    has_any_acl = has_ownable or has_access_control or 'onlyOwner' in source or 'onlyRole' in source
    feats['has_acl'] = int(has_any_acl)
    feats['acl_coverage'] = feats['guard_coverage']
    
    # ── 7. Complexity density metrics ───────────────────────────
    n_if = len(re.findall(r'\bif\s*\(', source))
    n_for = len(re.findall(r'\bfor\s*\(', source))
    n_while = len(re.findall(r'\bwhile\s*\(', source))
    n_ternary = source.count(' ? ') + source.count('? ')
    
    feats['n_branches'] = n_if + n_ternary
    feats['n_loops'] = n_for + n_while
    feats['branches_per_fn'] = feats['n_branches'] / max(n_fn, 1)
    feats['loops_per_fn'] = feats['n_loops'] / max(n_fn, 1)
    feats['branches_per_loc'] = feats['n_branches'] / max(loc, 1)
    
    # ── 8. DeFi-specific patterns ───────────────────────────────
    defi_patterns = {
        'has_swap': int(bool(re.search(r'\bswap\b', source, re.I))),
        'has_liquidity': int(bool(re.search(r'\bliquidity\b', source, re.I))),
        'has_pool': int(bool(re.search(r'\bpool\b', source, re.I))),
        'has_oracle': int(bool(re.search(r'\boracle|priceFeed|getPrice|latestAnswer\b', source, re.I))),
        'has_flash': int(bool(re.search(r'\bflash\b', source, re.I))),
        'has_stake': int(bool(re.search(r'\bstake|staking|unstake\b', source, re.I))),
        'has_vault': int(bool(re.search(r'\bvault\b', source, re.I))),
        'has_reward': int(bool(re.search(r'\breward\b', source, re.I))),
        'has_fee': int(bool(re.search(r'\bfee\b', source, re.I))),
        'has_slippage': int(bool(re.search(r'\bslippage|deadline\b', source, re.I))),
    }
    feats.update(defi_patterns)
    feats['defi_pattern_count'] = sum(defi_patterns.values())
    
    # ── 9. Inheritance complexity ───────────────────────────────
    inherits = re.findall(r'\bcontract\s+\w+\s+is\s+([^{]+)', source)
    n_parents = 0
    for inh in inherits:
        n_parents += len(inh.split(','))
    feats['n_parent_contracts'] = n_parents
    
    # ── 10. Import analysis ─────────────────────────────────────
    imports = re.findall(r'import\s+["\']([^"\']+)', source) + re.findall(r'import\s+\{[^}]+\}\s+from\s+["\']([^"\']+)', source)
    feats['n_imports'] = len(imports)
    feats['imports_openzeppelin'] = sum(1 for i in imports if 'openzeppelin' in str(i).lower())
    feats['imports_uniswap'] = sum(1 for i in imports if 'uniswap' in str(i).lower())
    feats['imports_ratio_oz'] = feats['imports_openzeppelin'] / max(feats['n_imports'], 1)
    
    # ── 11. Code structure ratios ───────────────────────────────
    feats['loc'] = loc
    feats['loc_per_fn'] = loc / max(n_fn, 1)
    feats['n_functions'] = n_fn
    
    n_events = len(re.findall(r'\bevent\s+\w+', source))
    n_emits = source.count('emit ')
    feats['n_events'] = n_events
    feats['n_emits'] = n_emits
    feats['emit_coverage'] = n_emits / max(n_fn, 1)
    
    # ── 12. Identifier richness for small contracts ─────────────
    idents = _extract_identifiers(source)
    words = []
    for ident in idents:
        words.extend(_split_identifier(ident))
    
    from src.ml.semantic_features import RISK_WORDS, SAFETY_WORDS, DEFI_WORDS
    n_words = len(words)
    risk_count = sum(1 for w in words if w in RISK_WORDS)
    safety_count = sum(1 for w in words if w in SAFETY_WORDS)
    defi_count = sum(1 for w in words if w in DEFI_WORDS)
    
    feats['ident_risk_ratio'] = risk_count / max(n_words, 1)
    feats['ident_safety_ratio'] = safety_count / max(n_words, 1)
    feats['ident_defi_ratio'] = defi_count / max(n_words, 1)
    feats['ident_risk_minus_safety'] = (risk_count - safety_count) / max(n_words, 1)
    
    # Specific high-signal words from prior analysis
    token_words = {'token', 'transfer', 'approve', 'allowance', 'mint', 'burn', 'balance', 'spender'}
    defi_action_words = {'swap', 'liquidity', 'pool', 'stake', 'deposit', 'withdraw', 'claim', 'harvest'}
    danger_words = {'call', 'delegatecall', 'selfdestruct', 'suicide', 'assembly', 'raw'}
    
    word_set = set(words)
    feats['has_token_vocab'] = int(len(word_set & token_words) >= 2)
    feats['token_vocab_count'] = len(word_set & token_words)
    feats['has_defi_vocab'] = int(len(word_set & defi_action_words) >= 2)
    feats['defi_vocab_count'] = len(word_set & defi_action_words)
    feats['has_danger_vocab'] = int(len(word_set & danger_words) >= 1)
    feats['danger_vocab_count'] = len(word_set & danger_words)
    
    return feats


def main():
    print("=" * 100)
    print("BRIDGE-THE-GAP ANALYSIS: Finding features to close small-contract performance gap")
    print("=" * 100)

    # ── Load data ───────────────────────────────────────────────
    scored = pd.read_csv(SCORED_V2, low_memory=False)
    scored['contract_id'] = scored['contract_id'].fillna('').astype(str)
    scored['local_source_path'] = scored['local_source_path'].fillna('').astype(str)
    scored['source_dataset'] = scored['source_dataset'].fillna('').astype(str)
    scored['label_status'] = scored['label_status'].fillna('').astype(str)

    sem = pd.read_csv(OOF_DIR / "sem_Vote_RF_GBC_XGB_merged.csv")
    cb = pd.read_csv(OOF_DIR / "dl_codebert_Merged.csv")

    # Build evaluation dataframe from sem predictions
    eval_df = sem[['contract_id', 'fold', 'y_true', 'y_prob']].rename(columns={'y_prob': 'prob_sem'})
    eval_df = eval_df.merge(cb[['contract_id', 'fold', 'y_prob']].rename(columns={'y_prob': 'prob_cb'}),
                            on=['contract_id', 'fold'], how='inner')
    eval_df = eval_df.merge(scored[['contract_id', 'local_source_path']], on='contract_id', how='left')
    eval_df = eval_df.drop_duplicates(subset=['contract_id', 'fold'])

    print(f"\nEvaluation set: {len(eval_df)} samples, {eval_df['y_true'].sum()} positives")
    print(f"Folds: {sorted(eval_df['fold'].unique())}")

    # ── Extract new features for ALL contracts ──────────────────
    print("\n--- Extracting candidate features for all contracts ---")
    unique_contracts = eval_df[['contract_id', 'local_source_path']].drop_duplicates('contract_id')
    
    feat_rows = []
    for i, (_, row) in enumerate(unique_contracts.iterrows()):
        src = read_source(row['local_source_path'])
        feats = extract_security_pattern_features(src)
        feats['contract_id'] = row['contract_id']
        feat_rows.append(feats)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(unique_contracts)} contracts processed")
    
    print(f"  {len(feat_rows)} contracts processed")
    new_feats = pd.DataFrame(feat_rows)
    
    # Merge with eval_df
    eval_df = eval_df.merge(new_feats, on='contract_id', how='left')
    
    feat_cols = [c for c in new_feats.columns if c != 'contract_id']
    eval_df[feat_cols] = eval_df[feat_cols].fillna(0)
    
    print(f"\n  {len(feat_cols)} candidate features extracted")

    # ── Mutual information: how much do these features predict vulnerability? ──
    print(f"\n{'='*100}")
    print("MUTUAL INFORMATION: New features vs vulnerability label")
    print("=" * 100)
    
    X_new = eval_df[feat_cols].values
    y = eval_df['y_true'].values.astype(int)
    
    mi = mutual_info_classif(X_new, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': feat_cols, 'mi': mi}).sort_values('mi', ascending=False)
    
    print(f"\n  Top 30 features by Mutual Information:")
    for _, row in mi_df.head(30).iterrows():
        print(f"    {row['feature']:40s}  MI={row['mi']:.4f}")

    # ── Quick 5-fold CV with just the new features ──────────────
    print(f"\n{'='*100}")
    print("QUICK CV: New features alone")
    print("=" * 100)
    
    top_feats = mi_df.head(40)['feature'].tolist()
    
    fold_results = []
    for fid in sorted(eval_df['fold'].unique()):
        train_mask = eval_df['fold'] != fid
        test_mask = eval_df['fold'] == fid
        
        X_train = eval_df.loc[train_mask, top_feats].values
        y_train = eval_df.loc[train_mask, 'y_true'].values.astype(int)
        X_test = eval_df.loc[test_mask, top_feats].values
        y_test = eval_df.loc[test_mask, 'y_true'].values.astype(int)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        gbc = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, 
                                          subsample=0.8, random_state=42)
        gbc.fit(X_train, y_train)
        probs = gbc.predict_proba(X_test)[:, 1]
        
        roc = roc_auc_score(y_test, probs)
        fold_results.append({'fold': fid, 'roc_new_only': roc})
        print(f"  Fold {fid}: ROC-AUC = {roc:.4f}")
    
    print(f"  Mean ROC-AUC (new features only): {np.mean([r['roc_new_only'] for r in fold_results]):.4f}")

    # ── Now: combine existing SCAFE-Sem probabilities + new features ────
    print(f"\n{'='*100}")
    print("STACKING: SCAFE-Sem probs + new features as input to a meta-learner")
    print("=" * 100)
    
    meta_features = top_feats + ['prob_sem']
    
    fold_results_meta = []
    for fid in sorted(eval_df['fold'].unique()):
        train_mask = eval_df['fold'] != fid
        test_mask = eval_df['fold'] == fid
        
        X_train = eval_df.loc[train_mask, meta_features].values
        y_train = eval_df.loc[train_mask, 'y_true'].values.astype(int)
        X_test = eval_df.loc[test_mask, meta_features].values
        y_test = eval_df.loc[test_mask, 'y_true'].values.astype(int)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        gbc = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, 
                                          subsample=0.8, random_state=42)
        gbc.fit(X_train, y_train)
        probs = gbc.predict_proba(X_test)[:, 1]
        
        roc = roc_auc_score(y_test, probs)
        
        # Also measure existing sem and cb for comparison
        roc_sem = roc_auc_score(y_test, eval_df.loc[test_mask, 'prob_sem'].values)
        roc_cb = roc_auc_score(y_test, eval_df.loc[test_mask, 'prob_cb'].values)
        
        fold_results_meta.append({'fold': fid, 'roc_meta': roc, 'roc_sem': roc_sem, 'roc_cb': roc_cb})
        print(f"  Fold {fid}: META={roc:.4f}  SEM={roc_sem:.4f}  CB={roc_cb:.4f}  "
              f"gap_to_cb={roc - roc_cb:+.4f}")
    
    mean_meta = np.mean([r['roc_meta'] for r in fold_results_meta])
    mean_sem = np.mean([r['roc_sem'] for r in fold_results_meta])
    mean_cb = np.mean([r['roc_cb'] for r in fold_results_meta])
    print(f"\n  MEAN: META={mean_meta:.4f}  SEM={mean_sem:.4f}  CB={mean_cb:.4f}")
    print(f"  Improvement over SEM: {mean_meta - mean_sem:+.4f}")
    print(f"  Gap to CB: {mean_meta - mean_cb:+.4f}")

    # ── Combine SCAFE-Sem probs + CodeBERT probs + new features ─────────
    print(f"\n{'='*100}")
    print("TRIPLE STACK: SCAFE-Sem + new features + CodeBERT (hybrid upper bound)")
    print("=" * 100)
    
    triple_features = top_feats + ['prob_sem', 'prob_cb']
    
    fold_results_triple = []
    for fid in sorted(eval_df['fold'].unique()):
        train_mask = eval_df['fold'] != fid
        test_mask = eval_df['fold'] == fid
        
        X_train = eval_df.loc[train_mask, triple_features].values
        y_train = eval_df.loc[train_mask, 'y_true'].values.astype(int)
        X_test = eval_df.loc[test_mask, triple_features].values
        y_test = eval_df.loc[test_mask, 'y_true'].values.astype(int)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        gbc = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, 
                                          subsample=0.8, random_state=42)
        gbc.fit(X_train, y_train)
        probs = gbc.predict_proba(X_test)[:, 1]
        
        roc = roc_auc_score(y_test, probs)
        fold_results_triple.append({'fold': fid, 'roc_triple': roc})
        print(f"  Fold {fid}: TRIPLE={roc:.4f}")
    
    mean_triple = np.mean([r['roc_triple'] for r in fold_results_triple])
    print(f"\n  MEAN TRIPLE: {mean_triple:.4f}")

    # ── Full pipeline: add new features to existing structural+semantic ─
    print(f"\n{'='*100}")
    print("FULL PIPELINE: Existing features + new security pattern features")
    print("=" * 100)
    
    # Load existing structural features
    struct_cols = [c for c in scored.columns 
                   if c not in ['contract_id', 'project_id', 'local_source_path',
                               'source_dataset', 'label_status', 'is_vulnerable',
                               'contract_name', 'file_name', 'labels']]
    
    # Get identifier features
    from src.ml.semantic_features import compute_identifier_features, CodeLSA
    
    full_df = eval_df.copy()
    
    # Also compute identifier features for contracts that don't have them yet
    ident_rows = []
    token_docs = []
    for _, row in unique_contracts.iterrows():
        src = read_source(row['local_source_path'])
        id_feats = compute_identifier_features(src)
        id_feats['contract_id'] = row['contract_id']
        ident_rows.append(id_feats)
        token_docs.append(tokenize_solidity(src))
    
    ident_df = pd.DataFrame(ident_rows)
    ident_cols = [c for c in ident_df.columns if c != 'contract_id']
    
    full_df = full_df.merge(ident_df, on='contract_id', how='left')
    
    # Run full pipeline with EXISTING structural + identifier + LSA + NEW security features
    all_feat_cols = top_feats + ident_cols  # new security + identifier features
    
    # Also load structural features from scored (numeric only)
    numeric_struct = scored.select_dtypes(include=[np.number]).columns.tolist()
    struct_cols_clean = [c for c in numeric_struct 
                         if c not in ['is_vulnerable', 'vuln_present', 'row_id']][:80]
    struct_df = scored[['contract_id'] + struct_cols_clean].drop_duplicates('contract_id')
    full_df = full_df.merge(struct_df, on='contract_id', how='left')
    
    avail_struct = [c for c in struct_cols_clean if c in full_df.columns][:80]
    
    combined_cols = list(set(all_feat_cols + avail_struct))
    combined_cols = [c for c in combined_cols if c in full_df.columns]
    
    full_df[combined_cols] = full_df[combined_cols].fillna(0)
    
    print(f"\n  Combined feature set: {len(combined_cols)} features")
    print(f"    - Structural (existing): {len(avail_struct)}")
    print(f"    - Identifier (existing): {len(ident_cols)}")
    print(f"    - Security pattern (NEW): {len(top_feats)}")
    
    # Also add LSA per-fold
    fold_results_full = []
    for fid in sorted(full_df['fold'].unique()):
        train_mask = full_df['fold'] != fid
        test_mask = full_df['fold'] == fid
        
        train_ids = full_df.loc[train_mask, 'contract_id'].values
        test_ids = full_df.loc[test_mask, 'contract_id'].values
        
        # Find indices in unique_contracts for LSA
        id_to_idx = {cid: i for i, cid in enumerate(unique_contracts['contract_id'].values)}
        train_doc_idx = [id_to_idx[c] for c in train_ids if c in id_to_idx]
        test_doc_idx = [id_to_idx[c] for c in test_ids if c in id_to_idx]
        
        train_docs = [token_docs[i] for i in train_doc_idx]
        test_docs = [token_docs[i] for i in test_doc_idx]
        
        lsa = CodeLSA(n_components=50)
        lsa_train = lsa.fit_transform(train_docs)
        lsa_test = lsa.transform(test_docs)
        
        lsa_train_df = pd.DataFrame(lsa_train, columns=lsa.feature_names)
        lsa_test_df = pd.DataFrame(lsa_test, columns=lsa.feature_names)
        
        # Build feature matrices
        all_cols_fold = combined_cols + lsa.feature_names
        
        X_train = np.hstack([
            full_df.loc[train_mask, combined_cols].values,
            lsa_train
        ])
        y_train = full_df.loc[train_mask, 'y_true'].values.astype(int)
        
        X_test = np.hstack([
            full_df.loc[test_mask, combined_cols].values,
            lsa_test
        ])
        y_test = full_df.loc[test_mask, 'y_true'].values.astype(int)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'GBC': GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
                                               subsample=0.8, random_state=42),
            'RF': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=3,
                                          random_state=42, n_jobs=-1),
        }
        
        fold_probs = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            fold_probs[name] = probs
            roc = roc_auc_score(y_test, probs)
            print(f"  Fold {fid} {name}: ROC={roc:.4f}")
        
        # Ensemble
        ens_probs = np.mean(list(fold_probs.values()), axis=0)
        roc_ens = roc_auc_score(y_test, ens_probs)
        
        roc_sem = roc_auc_score(y_test, full_df.loc[test_mask, 'prob_sem'].values)
        roc_cb = roc_auc_score(y_test, full_df.loc[test_mask, 'prob_cb'].values)
        
        fold_results_full.append({
            'fold': fid, 'roc_full': roc_ens, 'roc_sem': roc_sem, 'roc_cb': roc_cb,
            **{f'roc_{n}': roc_auc_score(y_test, p) for n, p in fold_probs.items()}
        })
        print(f"  Fold {fid} ENS: ROC={roc_ens:.4f}  (SEM={roc_sem:.4f}  CB={roc_cb:.4f})")
    
    print(f"\n  MEAN RESULTS:")
    for key in ['roc_full', 'roc_GBC', 'roc_RF', 'roc_sem', 'roc_cb']:
        vals = [r[key] for r in fold_results_full]
        print(f"    {key:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    mean_full = np.mean([r['roc_full'] for r in fold_results_full])
    print(f"\n  Gap to CodeBERT: {mean_full - mean_cb:+.4f}")
    if mean_full > mean_cb:
        print("  >>> WE BEAT CODEBERT! <<<")
    else:
        print(f"  Still behind by {mean_cb - mean_full:.4f}")


if __name__ == "__main__":
    main()
