#!/usr/bin/env python3 -u
"""
Quick analysis: Are identifier names (function, variable, contract names)
discriminative for vulnerability detection?
"""
import re
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]

# Regex patterns for Solidity identifiers
RE_FUNCTION = re.compile(r'\bfunction\s+([a-zA-Z_]\w*)')
RE_CONTRACT = re.compile(r'\bcontract\s+([a-zA-Z_]\w*)')
RE_EVENT = re.compile(r'\bevent\s+([a-zA-Z_]\w*)')
RE_MODIFIER = re.compile(r'\bmodifier\s+([a-zA-Z_]\w*)')
RE_VARIABLE = re.compile(r'\b(?:uint\d*|int\d*|address|bool|string|bytes\d*|mapping)\s+(?:public\s+|private\s+|internal\s+|external\s+)?([a-zA-Z_]\w*)')
RE_STRUCT = re.compile(r'\bstruct\s+([a-zA-Z_]\w*)')
RE_ENUM = re.compile(r'\benum\s+([a-zA-Z_]\w*)')


def split_identifier(name):
    """Split camelCase/PascalCase/snake_case into words."""
    parts = name.split('_')
    words = []
    for part in parts:
        if not part:
            continue
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', part)
        sub = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', sub)
        words.extend(w.lower() for w in sub.split('_') if w)
    return words


def extract_identifiers(source_text):
    """Extract all identifiers from Solidity source."""
    identifiers = []
    for pattern in [RE_FUNCTION, RE_CONTRACT, RE_EVENT, RE_MODIFIER,
                    RE_VARIABLE, RE_STRUCT, RE_ENUM]:
        identifiers.extend(pattern.findall(source_text))
    return identifiers


def read_source(path_str):
    """Read source file or directory of .sol files."""
    p = Path(path_str)
    if not p.exists():
        return ""
    if p.is_file():
        try:
            return p.read_text(errors='ignore')
        except:
            return ""
    elif p.is_dir():
        texts = []
        for sol in sorted(p.rglob("*.sol")):
            try:
                texts.append(sol.read_text(errors='ignore'))
            except:
                pass
        return "\n".join(texts)
    return ""


# Risk-related and safety-related word lists
RISK_WORDS = {
    'withdraw', 'transfer', 'send', 'call', 'delegate', 'delegatecall',
    'execute', 'proxy', 'flash', 'swap', 'borrow', 'lend', 'mint', 'burn',
    'stake', 'unstake', 'claim', 'deposit', 'liquidate', 'leverage',
    'selfdestruct', 'destroy', 'kill', 'suicide', 'external', 'raw',
    'arbitrary', 'unchecked', 'unsafe', 'low', 'asm', 'assembly',
    'fallback', 'receive', 'payable', 'ether', 'wei', 'callback',
    'approve', 'allowance', 'permit', 'migrate', 'upgrade',
}

SAFETY_WORDS = {
    'safe', 'check', 'verify', 'require', 'assert', 'guard', 'only',
    'owner', 'admin', 'pause', 'unpause', 'lock', 'unlock', 'reentrancy',
    'nonreentrant', 'restrict', 'protect', 'validate', 'limit', 'cap',
    'whitelist', 'blacklist', 'authorize', 'authenticated', 'access',
    'control', 'permission', 'role', 'modifier', 'internal', 'private',
    'immutable', 'constant', 'frozen', 'emergency', 'recover',
}

DEFI_WORDS = {
    'pool', 'liquidity', 'amm', 'oracle', 'price', 'feed', 'vault',
    'strategy', 'yield', 'farm', 'reward', 'token', 'erc20', 'erc721',
    'nft', 'pair', 'router', 'factory', 'bridge', 'cross', 'chain',
    'governance', 'vote', 'proposal', 'timelock', 'multi', 'sig',
}


def compute_name_features(source_text):
    """Compute features from identifier names."""
    identifiers = extract_identifiers(source_text)
    if not identifiers:
        return {}

    all_words = []
    for ident in identifiers:
        all_words.extend(split_identifier(ident))

    word_counts = Counter(all_words)
    n_words = len(all_words)
    n_unique = len(word_counts)

    risk_count = sum(word_counts.get(w, 0) for w in RISK_WORDS)
    safety_count = sum(word_counts.get(w, 0) for w in SAFETY_WORDS)
    defi_count = sum(word_counts.get(w, 0) for w in DEFI_WORDS)

    risk_unique = sum(1 for w in RISK_WORDS if w in word_counts)
    safety_unique = sum(1 for w in SAFETY_WORDS if w in word_counts)
    defi_unique = sum(1 for w in DEFI_WORDS if w in word_counts)

    features = {
        'name_n_identifiers': len(identifiers),
        'name_n_words': n_words,
        'name_n_unique_words': n_unique,
        'name_word_diversity': n_unique / max(n_words, 1),
        'name_avg_ident_len': np.mean([len(i) for i in identifiers]),
        'name_avg_words_per_ident': n_words / max(len(identifiers), 1),
        'name_risk_count': risk_count,
        'name_safety_count': safety_count,
        'name_defi_count': defi_count,
        'name_risk_unique': risk_unique,
        'name_safety_unique': safety_unique,
        'name_defi_unique': defi_unique,
        'name_risk_ratio': risk_count / max(n_words, 1),
        'name_safety_ratio': safety_count / max(n_words, 1),
        'name_defi_ratio': defi_count / max(n_words, 1),
        'name_risk_to_safety': risk_count / max(safety_count, 1),
        'name_safety_to_total': safety_count / max(risk_count + safety_count, 1),
    }

    # Top specific word presence
    for w in ['transfer', 'withdraw', 'delegate', 'proxy', 'flash',
              'swap', 'safe', 'guard', 'only', 'owner', 'check',
              'execute', 'mint', 'burn', 'stake', 'approve', 'permit',
              'oracle', 'pool', 'vault', 'upgrade', 'migrate', 'callback',
              'fallback', 'payable', 'lock', 'pause', 'reentrancy']:
        features[f'name_has_{w}'] = int(w in word_counts)
        features[f'name_count_{w}'] = word_counts.get(w, 0)

    return features


def main():
    print("=" * 100)
    print("IDENTIFIER NAME ANALYSIS: Are naming patterns discriminative?")
    print("=" * 100)

    # Load dataset
    df = pd.read_csv(
        _REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv",
        usecols=['contract_id', 'local_source_path', 'label_status', 'source_dataset'],
        low_memory=False
    )
    df["label_status"] = df["label_status"].fillna("").astype(str)
    df["source_dataset"] = df["source_dataset"].fillna("").astype(str)

    # Merged dataset
    dappscan = df[df["source_dataset"] == "DAppSCAN"]
    dappscan = dappscan[dappscan["label_status"].isin(["clean", "vulnerable"])]
    forge = df[df["source_dataset"] == "FORGE"]
    forge = forge[forge["label_status"].isin(["vulnerable", "unknown"])]
    data = pd.concat([dappscan, forge], ignore_index=True)
    data["y"] = (data["label_status"] == "vulnerable").astype(int)

    print(f"\n  Total contracts: {len(data)} (pos={data['y'].sum()}, neg={(data['y']==0).sum()})")

    # Extract features for all contracts
    print("\n--- Extracting identifier name features ---")
    feature_rows = []
    n_missing = 0
    for idx, row in data.iterrows():
        src = read_source(str(row['local_source_path']))
        if not src:
            n_missing += 1
            feature_rows.append({})
            continue
        features = compute_name_features(src)
        feature_rows.append(features)
        if len(feature_rows) % 5000 == 0:
            print(f"  Processed {len(feature_rows)}/{len(data)}...", flush=True)

    print(f"  Done. {n_missing} contracts had no source file.")

    feat_df = pd.DataFrame(feature_rows).fillna(0)
    feat_cols = [c for c in feat_df.columns if c.startswith('name_')]
    print(f"  Extracted {len(feat_cols)} name-based features")

    X_names = feat_df[feat_cols].values.astype(np.float64)
    y = data['y'].values

    # =========================================================================
    # Analysis 1: Univariate discrimination
    # =========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 1: UNIVARIATE DISCRIMINATION (AUC per feature)")
    print("=" * 100)

    feat_aucs = []
    for i, col in enumerate(feat_cols):
        try:
            auc = roc_auc_score(y, X_names[:, i])
            feat_aucs.append((col, auc))
        except:
            feat_aucs.append((col, 0.5))

    feat_aucs.sort(key=lambda x: -abs(x[1] - 0.5))
    print(f"\n  Top 30 most discriminative name features:")
    print(f"  {'Feature':50s} {'AUC':>8} {'Direction':>10}")
    print("  " + "-" * 70)
    for col, auc in feat_aucs[:30]:
        direction = "vuln↑" if auc > 0.5 else "vuln↓"
        print(f"  {col:50s} {auc:8.4f} {direction:>10}")

    # =========================================================================
    # Analysis 2: Word frequency comparison
    # =========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 2: MOST DISCRIMINATIVE WORDS (vuln vs clean)")
    print("=" * 100)

    vuln_words = Counter()
    clean_words = Counter()
    vuln_count = 0
    clean_count = 0

    for idx, row in data.iterrows():
        src = read_source(str(row['local_source_path']))
        if not src:
            continue
        idents = extract_identifiers(src)
        words = []
        for ident in idents:
            words.extend(split_identifier(ident))

        if row['y'] == 1:
            vuln_words.update(words)
            vuln_count += 1
        else:
            clean_words.update(words)
            clean_count += 1

    all_words = set(vuln_words.keys()) | set(clean_words.keys())
    word_scores = []
    for w in all_words:
        v_rate = vuln_words[w] / max(vuln_count, 1)
        c_rate = clean_words[w] / max(clean_count, 1)
        if v_rate + c_rate < 0.1:
            continue
        ratio = v_rate / max(c_rate, 0.001)
        word_scores.append((w, v_rate, c_rate, ratio))

    print(f"\n  Words MORE common in vulnerable contracts (ratio = vuln_rate / clean_rate):")
    word_scores.sort(key=lambda x: -x[3])
    for w, vr, cr, ratio in word_scores[:25]:
        print(f"    {w:25s} vuln_rate={vr:6.2f}  clean_rate={cr:6.2f}  ratio={ratio:6.2f}")

    print(f"\n  Words MORE common in clean contracts:")
    word_scores.sort(key=lambda x: x[3])
    for w, vr, cr, ratio in word_scores[:25]:
        print(f"    {w:25s} vuln_rate={vr:6.2f}  clean_rate={cr:6.2f}  ratio={ratio:6.2f}")

    # =========================================================================
    # Analysis 3: Classification with name features alone
    # =========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 3: CLASSIFICATION WITH NAME FEATURES ALONE (5-fold GroupKFold)")
    print("=" * 100)

    contract_ids = data["contract_id"].fillna("").astype(str).values
    group_key = pd.Series(contract_ids)
    group_map = {g: i for i, g in enumerate(sorted(group_key.unique()))}
    groups = group_key.map(group_map).values.astype(int)

    gkf = GroupKFold(n_splits=5)
    fold_rocs_rf, fold_f1s_rf = [], []
    fold_rocs_gbc, fold_f1s_gbc = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_names, y, groups)):
        Xtr, Xte = X_names[train_idx], X_names[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        prob = rf.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, prob)
        best_f1 = max(f1_score(yte, (prob >= t).astype(int), zero_division=0)
                      for t in np.linspace(0.01, 0.99, 100))
        fold_rocs_rf.append(roc)
        fold_f1s_rf.append(best_f1)

        gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42)
        gbc.fit(Xtr, ytr)
        prob_g = gbc.predict_proba(Xte)[:, 1]
        roc_g = roc_auc_score(yte, prob_g)
        best_f1_g = max(f1_score(yte, (prob_g >= t).astype(int), zero_division=0)
                        for t in np.linspace(0.01, 0.99, 100))
        fold_rocs_gbc.append(roc_g)
        fold_f1s_gbc.append(best_f1_g)

    print(f"\n  RF  (name features only): ROC={np.mean(fold_rocs_rf):.4f}±{np.std(fold_rocs_rf):.4f}  "
          f"F1={np.mean(fold_f1s_rf):.4f}±{np.std(fold_f1s_rf):.4f}")
    print(f"  GBC (name features only): ROC={np.mean(fold_rocs_gbc):.4f}±{np.std(fold_rocs_gbc):.4f}  "
          f"F1={np.mean(fold_f1s_gbc):.4f}±{np.std(fold_f1s_gbc):.4f}")

    # =========================================================================
    # Analysis 4: Combined (original 80 + name features)
    # =========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS 4: ORIGINAL 80 FEATURES + NAME FEATURES (5-fold GroupKFold)")
    print("=" * 100)

    sys.path.insert(0, str(_REPO))
    from src.ml.features import _EXCLUDE, _ONCHAIN_FEATURES

    full_df = pd.read_csv(_REPO / "data" / "derived" / "unified_contract_eval_scored_v2.csv", low_memory=False)
    full_df["label_status"] = full_df["label_status"].fillna("").astype(str)
    full_df["source_dataset"] = full_df["source_dataset"].fillna("").astype(str)

    dappscan_f = full_df[full_df["source_dataset"] == "DAppSCAN"]
    dappscan_f = dappscan_f[dappscan_f["label_status"].isin(["clean", "vulnerable"])]
    forge_f = full_df[full_df["source_dataset"] == "FORGE"]
    forge_f = forge_f[forge_f["label_status"].isin(["vulnerable", "unknown"])]
    data_full = pd.concat([dappscan_f, forge_f], ignore_index=True)

    orig_feat_cols = sorted([
        c for c in data_full.columns
        if pd.api.types.is_numeric_dtype(data_full[c])
        and c not in _EXCLUDE and c not in _ONCHAIN_FEATURES
    ])
    X_orig = data_full[orig_feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X_orig = np.nan_to_num(X_orig, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X_orig, y, random_state=42, n_neighbors=5)
    mask = mi > 0.001
    X_mi = X_orig[:, mask]
    rf_rank = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    rf_rank.fit(X_mi, y)
    imp = rf_rank.feature_importances_
    ranked = np.argsort(imp)[::-1]
    mi_indices = [i for i, m in enumerate(mask) if m]
    selected_80 = sorted([mi_indices[ranked[i]] for i in range(80)])

    X_80 = X_orig[:, selected_80]
    X_combined = np.hstack([X_80, X_names])
    print(f"  Original features: {X_80.shape[1]}, Name features: {X_names.shape[1]}, Combined: {X_combined.shape[1]}")

    for label, X_use in [("Original 80 only", X_80), ("Name features only", X_names),
                          ("Original 80 + Names", X_combined)]:
        fold_rocs, fold_f1s = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_use, y, groups)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_use[train_idx])
            Xte = sc.transform(X_use[test_idx])
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                         class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            rf.fit(Xtr, y[train_idx])
            prob = rf.predict_proba(Xte)[:, 1]
            fold_rocs.append(roc_auc_score(y[test_idx], prob))
            best_f1 = max(f1_score(y[test_idx], (prob >= t).astype(int), zero_division=0)
                          for t in np.linspace(0.01, 0.99, 100))
            fold_f1s.append(best_f1)
        print(f"  {label:30s} ROC={np.mean(fold_rocs):.4f}±{np.std(fold_rocs):.4f}  "
              f"F1={np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}")

    print("\n  Reference DL baselines:")
    print(f"  {'TextCNN':30s} ROC=0.9275  F1=0.6009")
    print(f"  {'BiLSTM':30s} ROC=0.9283  F1=0.5986")
    print(f"  {'CodeBERT':30s} ROC=0.9438  F1=0.6644")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
