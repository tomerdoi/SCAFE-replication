#!/usr/bin/env python3 -u
"""
Deep analysis of small contracts where CodeBERT wins and we lose.

Goal: understand what distinguishes the 366 small vulnerable contracts
that CodeBERT catches but SCAFE-Sem misses, from small clean contracts.
Then design targeted features or strategies to close the gap.
"""
import sys
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

OOF_DIR = _REPO / "data" / "derived" / "oof_predictions"
SCORED_V2 = _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv"

from src.ml.semantic_features import read_source, _split_identifier, _extract_identifiers, tokenize_solidity


def optimal_f1_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def analyze_source_patterns(source_text):
    """Extract detailed patterns from source code for analysis."""
    if not source_text:
        return {}
    
    lines = source_text.split('\n')
    loc = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
    
    feats = {
        'loc': loc,
        'has_import': int('import' in source_text),
        'n_imports': source_text.count('import '),
        'has_interface': int(bool(re.search(r'\binterface\s+\w+', source_text))),
        'has_abstract': int('abstract contract' in source_text),
        'has_library': int(bool(re.search(r'\blibrary\s+\w+', source_text))),
        'is_interface_only': int(bool(re.search(r'\binterface\s+\w+', source_text)) 
                                 and not bool(re.search(r'\bcontract\s+\w+', source_text))),
        'has_inheritance': int(' is ' in source_text and 'contract' in source_text),
        'n_inheritance': len(re.findall(r'\bcontract\s+\w+\s+is\s+', source_text)),
        'has_using_for': int('using ' in source_text and ' for ' in source_text),
        'n_using_for': len(re.findall(r'\busing\s+\w+\s+for\s+', source_text)),
        
        'has_transfer': int('.transfer(' in source_text),
        'has_send': int('.send(' in source_text),
        'has_call_value': int('.call{value' in source_text or '.call.value(' in source_text),
        'has_delegatecall': int('delegatecall' in source_text),
        'has_selfdestruct': int('selfdestruct' in source_text),
        'has_assembly': int('assembly' in source_text),
        'has_tx_origin': int('tx.origin' in source_text),
        'has_block_timestamp': int('block.timestamp' in source_text),
        
        'has_onlyowner': int('onlyOwner' in source_text or 'onlyowner' in source_text),
        'has_nonreentrant': int('nonReentrant' in source_text or 'nonreentrant' in source_text),
        'has_require': int('require(' in source_text),
        'n_require': source_text.count('require('),
        'has_modifier': int(bool(re.search(r'\bmodifier\s+\w+', source_text))),
        'n_modifiers': len(re.findall(r'\bmodifier\s+\w+', source_text)),
        
        'has_mapping': int('mapping(' in source_text),
        'n_mappings': source_text.count('mapping('),
        'has_array': int('[]' in source_text),
        'has_struct': int(bool(re.search(r'\bstruct\s+\w+', source_text))),
        'has_event': int(bool(re.search(r'\bevent\s+\w+', source_text))),
        'has_enum': int(bool(re.search(r'\benum\s+\w+', source_text))),
        
        'has_erc20': int(bool(re.search(r'\bIERC20|ERC20|IERC20Metadata', source_text))),
        'has_erc721': int(bool(re.search(r'\bIERC721|ERC721', source_text))),
        'has_ownable': int('Ownable' in source_text),
        'has_pausable': int('Pausable' in source_text),
        'has_reentrancy_guard': int('ReentrancyGuard' in source_text),
        'has_safemath': int('SafeMath' in source_text),
        'has_access_control': int('AccessControl' in source_text),
        'has_proxy': int(bool(re.search(r'\bProxy|Upgradeable|proxy|upgradeable', source_text))),
        
        'n_functions': len(re.findall(r'\bfunction\s+\w+', source_text)),
        'n_public_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\bpublic\b', source_text)),
        'n_external_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\bexternal\b', source_text)),
        'n_internal_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\binternal\b', source_text)),
        'n_private_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\bprivate\b', source_text)),
        'n_view_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\bview\b', source_text)),
        'n_payable_fns': len(re.findall(r'\bfunction\s+\w+[^{]*\bpayable\b', source_text)),
        
        'has_receive': int('receive()' in source_text),
        'has_fallback': int(bool(re.search(r'\bfallback\s*\(', source_text))),
        'has_constructor': int('constructor(' in source_text),
        
        'ratio_external': 0.0,
        'ratio_payable': 0.0,
        'ratio_view': 0.0,
        'require_per_fn': 0.0,
        'modifier_per_fn': 0.0,
    }
    
    n_fn = feats['n_functions'] if feats['n_functions'] > 0 else 1
    feats['ratio_external'] = feats['n_external_fns'] / n_fn
    feats['ratio_payable'] = feats['n_payable_fns'] / n_fn
    feats['ratio_view'] = feats['n_view_fns'] / n_fn
    feats['require_per_fn'] = feats['n_require'] / n_fn
    feats['modifier_per_fn'] = feats['n_modifiers'] / n_fn
    
    return feats


def main():
    print("=" * 100)
    print("SMALL CONTRACT ANALYSIS: Where CodeBERT wins and we lose")
    print("=" * 100)

    sem = pd.read_csv(OOF_DIR / "sem_Vote_RF_GBC_XGB_merged.csv")
    cb = pd.read_csv(OOF_DIR / "dl_codebert_Merged.csv")

    df = sem.merge(
        cb[['contract_id', 'fold', 'y_prob']],
        on=['contract_id', 'fold'],
        suffixes=('_sem', '_cb')
    )

    # Compute per-fold binary predictions
    for fid in sorted(df['fold'].unique()):
        m = df['fold'] == fid
        yt = df.loc[m, 'y_true'].values
        t_sem, _ = optimal_f1_threshold(yt, df.loc[m, 'y_prob_sem'].values)
        t_cb, _ = optimal_f1_threshold(yt, df.loc[m, 'y_prob_cb'].values)
        df.loc[m, 'pred_sem'] = (df.loc[m, 'y_prob_sem'].values >= t_sem).astype(int)
        df.loc[m, 'pred_cb'] = (df.loc[m, 'y_prob_cb'].values >= t_cb).astype(int)

    # Load source paths
    scored = pd.read_csv(SCORED_V2, low_memory=False)
    scored['source_dataset'] = scored['source_dataset'].fillna('').astype(str)
    scored['label_status'] = scored['label_status'].fillna('').astype(str)
    scored['contract_id'] = scored['contract_id'].fillna('').astype(str)
    scored['local_source_path'] = scored['local_source_path'].fillna('').astype(str)

    df = df.merge(scored[['contract_id', 'local_source_path', 'source_dataset']],
                  on='contract_id', how='left')

    # Define groups
    y = df['y_true'].values.astype(int)
    cb_wins = (df['pred_cb'] == 1) & (df['pred_sem'] == 0) & (y == 1)
    we_win = (df['pred_sem'] == 1) & (df['pred_cb'] == 0) & (y == 1)
    both_tp = (df['pred_sem'] == 1) & (df['pred_cb'] == 1) & (y == 1)
    both_fn = (df['pred_sem'] == 0) & (df['pred_cb'] == 0) & (y == 1)
    our_fp = (df['pred_sem'] == 1) & (y == 0)
    clean_tn = (df['pred_sem'] == 0) & (df['pred_cb'] == 0) & (y == 0)

    groups = {
        'CB_wins_vuln': df[cb_wins],
        'We_win_vuln': df[we_win],
        'Both_TP': df[both_tp],
        'Both_FN': df[both_fn],
        'Our_FP': df[our_fp],
        'Clean_TN_sample': df[clean_tn].sample(min(500, clean_tn.sum()), random_state=42),
    }

    print(f"\nGroup sizes:")
    for name, grp in groups.items():
        print(f"  {name:20s}: {len(grp)}")

    # ── Analyze source code patterns for each group ─────────────────────
    print("\n--- Extracting source code patterns ---")
    all_patterns = {}
    for name, grp in groups.items():
        rows = []
        for _, row in grp.iterrows():
            src = read_source(row.get('local_source_path', ''))
            rows.append(analyze_source_patterns(src))
        all_patterns[name] = pd.DataFrame(rows)
        print(f"  {name}: {len(rows)} contracts analyzed")

    # ── Compare CB_wins vs Clean (small contracts) ──────────────────────
    print("\n" + "=" * 100)
    print("KEY COMPARISON: CB_wins (vuln we miss) vs Clean TN (sample)")
    print("=" * 100)

    cb_pat = all_patterns['CB_wins_vuln']
    clean_pat = all_patterns['Clean_TN_sample']

    features_to_compare = sorted(cb_pat.columns)
    print(f"\n  {'Feature':35s} {'CB_wins':>10s} {'Clean_TN':>10s} {'Diff':>10s} {'Ratio':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    diffs = []
    for feat in features_to_compare:
        m1 = cb_pat[feat].mean()
        m2 = clean_pat[feat].mean()
        diff = m1 - m2
        ratio = m1 / max(m2, 0.001)
        diffs.append((feat, m1, m2, diff, ratio))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for feat, m1, m2, diff, ratio in diffs[:30]:
        print(f"  {feat:35s} {m1:10.3f} {m2:10.3f} {diff:+10.3f} {ratio:8.2f}x")

    # ── Compare CB_wins vs Both_TP ──────────────────────────────────────
    print(f"\n{'='*100}")
    print("COMPARISON: CB_wins (vuln we miss) vs Both_TP (vuln we both catch)")
    print("=" * 100)

    both_pat = all_patterns['Both_TP']
    print(f"\n  {'Feature':35s} {'CB_wins':>10s} {'Both_TP':>10s} {'Diff':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    diffs2 = []
    for feat in features_to_compare:
        m1 = cb_pat[feat].mean()
        m2 = both_pat[feat].mean()
        diff = m1 - m2
        diffs2.append((feat, m1, m2, diff))

    diffs2.sort(key=lambda x: abs(x[3]), reverse=True)
    for feat, m1, m2, diff in diffs2[:25]:
        print(f"  {feat:35s} {m1:10.3f} {m2:10.3f} {diff:+10.3f}")

    # ── Check: are CB_wins mostly interfaces/libraries/abstracts? ───────
    print(f"\n{'='*100}")
    print("CONTRACT TYPE ANALYSIS")
    print("=" * 100)

    for name in ['CB_wins_vuln', 'We_win_vuln', 'Both_TP', 'Both_FN', 'Our_FP']:
        pat = all_patterns[name]
        n = len(pat)
        if n == 0:
            continue
        iface = (pat['is_interface_only'] == 1).sum()
        lib = (pat['has_library'] == 1).sum()
        abst = (pat['has_abstract'] == 1).sum()
        tiny = (pat['loc'] < 30).sum()
        small = (pat['loc'] < 100).sum()
        no_fn = (pat['n_functions'] == 0).sum()
        print(f"  {name:20s} (n={n:4d}): interface={iface:3d}({iface/n*100:4.1f}%), "
              f"library={lib:3d}({lib/n*100:4.1f}%), abstract={abst:3d}({abst/n*100:4.1f}%), "
              f"tiny<30LOC={tiny:3d}({tiny/n*100:4.1f}%), <100LOC={small:3d}({small/n*100:4.1f}%), "
              f"0_fns={no_fn:3d}({no_fn/n*100:4.1f}%)")

    # ── Word-level analysis of identifiers in CB_wins ───────────────────
    print(f"\n{'='*100}")
    print("IDENTIFIER WORD ANALYSIS (CB_wins vs Clean)")
    print("=" * 100)

    cb_words = Counter()
    clean_words = Counter()

    for _, row in groups['CB_wins_vuln'].iterrows():
        src = read_source(row.get('local_source_path', ''))
        if src:
            idents = _extract_identifiers(src)
            for ident in idents:
                for w in _split_identifier(ident):
                    cb_words[w] += 1

    for _, row in groups['Clean_TN_sample'].iterrows():
        src = read_source(row.get('local_source_path', ''))
        if src:
            idents = _extract_identifiers(src)
            for ident in idents:
                for w in _split_identifier(ident):
                    clean_words[w] += 1

    # Normalize and compare
    cb_total = sum(cb_words.values()) or 1
    clean_total = sum(clean_words.values()) or 1

    all_words_set = set(cb_words.keys()) | set(clean_words.keys())
    word_diffs = []
    for w in all_words_set:
        freq_cb = cb_words[w] / cb_total
        freq_clean = clean_words[w] / clean_total
        if cb_words[w] >= 3 or clean_words[w] >= 3:
            word_diffs.append((w, freq_cb, freq_clean, freq_cb - freq_clean))

    print(f"\n  Top words MORE common in CB_wins (vuln we miss):")
    word_diffs.sort(key=lambda x: x[3], reverse=True)
    for w, fc, fcl, diff in word_diffs[:20]:
        print(f"    {w:20s}  cb_wins={fc:.4f}  clean={fcl:.4f}  diff={diff:+.4f}")

    print(f"\n  Top words MORE common in Clean TN:")
    word_diffs.sort(key=lambda x: x[3])
    for w, fc, fcl, diff in word_diffs[:15]:
        print(f"    {w:20s}  cb_wins={fc:.4f}  clean={fcl:.4f}  diff={diff:+.4f}")

    # ── Check: what if we use different thresholds for small contracts? ──
    print(f"\n{'='*100}")
    print("SIZE-CONDITIONED THRESHOLD EXPERIMENT")
    print("=" * 100)

    # Load structural features to get LOC
    feat_cols = ['loc_total', 'n_functions', 'comp_complexity']
    avail_cols = [c for c in feat_cols if c in scored.columns]
    if avail_cols:
        df2 = df.merge(scored[['contract_id'] + avail_cols], on='contract_id', how='left')
    else:
        df2 = df.copy()

    if 'n_functions' in df2.columns:
        # Small contracts: ≤5 functions
        small = df2['n_functions'] <= 5
        large = df2['n_functions'] > 5

        for label, mask in [("Small (≤5 fns)", small), ("Large (>5 fns)", large)]:
            sub = df2[mask]
            n = len(sub)
            if n < 100 or sub['y_true'].nunique() < 2:
                continue
            roc_sem = roc_auc_score(sub['y_true'], sub['y_prob_sem'])
            roc_cb = roc_auc_score(sub['y_true'], sub['y_prob_cb'])
            pos = sub['y_true'].sum()
            print(f"  {label:20s}: n={n:6d}, pos={pos:4d}, "
                  f"SEM={roc_sem:.4f}, CB={roc_cb:.4f}, gap={roc_cb-roc_sem:+.4f}")

        # What if we use CodeBERT-like threshold behavior for small contracts?
        # Simulate: for small contracts, lower our threshold to catch more
        print(f"\n  Size-adaptive threshold simulation:")
        for t_small_offset in [-0.15, -0.10, -0.05, 0.0, 0.05]:
            fold_f1s = []
            fold_rocs = []
            for fid in sorted(df2['fold'].unique()):
                fm = df2['fold'] == fid
                sub = df2[fm]
                yt = sub['y_true'].values
                yp = sub['y_prob_sem'].values

                base_t, _ = optimal_f1_threshold(yt, yp)
                
                is_small = sub['n_functions'].values <= 5
                adjusted_prob = yp.copy()
                adjusted_prob[is_small] = np.clip(yp[is_small] + t_small_offset, 0, 1)
                
                roc = roc_auc_score(yt, adjusted_prob)
                _, f1 = optimal_f1_threshold(yt, adjusted_prob)
                fold_rocs.append(roc)
                fold_f1s.append(f1)

            print(f"    small_offset={t_small_offset:+.2f}: "
                  f"ROC={np.mean(fold_rocs):.4f}±{np.std(fold_rocs):.4f}  "
                  f"F1={np.mean(fold_f1s):.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ACTIONABLE INSIGHTS")
    print("=" * 100)


if __name__ == "__main__":
    main()
