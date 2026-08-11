"""
Semantic feature extraction for SCAFE-Sem.

Three complementary semantic feature streams:
  1. Identifier Semantic Profiling — extracts function/variable/contract names,
     decomposes them into constituent words, and computes risk/safety/DeFi
     word profiles plus structural naming statistics.
  2. Context Profiling — counts call, guard, state, and ordering patterns
     associated with vulnerability classes.
  3. Code Token LSA — tokenizes Solidity source, builds TF-IDF, and applies
     Truncated SVD to discover latent code topics (fitted per fold to avoid leakage).

Both streams are compilation-free and run on CPU.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ── Regex patterns for Solidity identifiers ─────────────────────────────────
_RE_FUNCTION = re.compile(r'\bfunction\s+([a-zA-Z_]\w*)')
_RE_CONTRACT = re.compile(r'\bcontract\s+([a-zA-Z_]\w*)')
_RE_EVENT = re.compile(r'\bevent\s+([a-zA-Z_]\w*)')
_RE_MODIFIER = re.compile(r'\bmodifier\s+([a-zA-Z_]\w*)')
_RE_VARIABLE = re.compile(
    r'\b(?:uint\d*|int\d*|address|bool|string|bytes\d*|mapping)'
    r'\s+(?:public\s+|private\s+|internal\s+|external\s+)?'
    r'([a-zA-Z_]\w*)'
)
_RE_STRUCT = re.compile(r'\bstruct\s+([a-zA-Z_]\w*)')
_RE_ENUM = re.compile(r'\benum\s+([a-zA-Z_]\w*)')

_IDENT_PATTERNS = [
    _RE_FUNCTION, _RE_CONTRACT, _RE_EVENT, _RE_MODIFIER,
    _RE_VARIABLE, _RE_STRUCT, _RE_ENUM,
]

# ── Word categories for identifier profiling ────────────────────────────────
RISK_WORDS = frozenset({
    'withdraw', 'transfer', 'send', 'call', 'delegate', 'delegatecall',
    'execute', 'proxy', 'flash', 'swap', 'borrow', 'lend', 'mint', 'burn',
    'stake', 'unstake', 'claim', 'deposit', 'liquidate', 'leverage',
    'selfdestruct', 'destroy', 'kill', 'suicide', 'external', 'raw',
    'arbitrary', 'unchecked', 'unsafe', 'low', 'asm', 'assembly',
    'fallback', 'receive', 'payable', 'ether', 'wei', 'callback',
    'approve', 'allowance', 'permit', 'migrate', 'upgrade',
})

SAFETY_WORDS = frozenset({
    'safe', 'check', 'verify', 'require', 'assert', 'guard', 'only',
    'owner', 'admin', 'pause', 'unpause', 'lock', 'unlock', 'reentrancy',
    'nonreentrant', 'restrict', 'protect', 'validate', 'limit', 'cap',
    'whitelist', 'blacklist', 'authorize', 'authenticated', 'access',
    'control', 'permission', 'role', 'modifier', 'internal', 'private',
    'immutable', 'constant', 'frozen', 'emergency', 'recover',
})

DEFI_WORDS = frozenset({
    'pool', 'liquidity', 'amm', 'oracle', 'price', 'feed', 'vault',
    'strategy', 'yield', 'farm', 'reward', 'token', 'erc20', 'erc721',
    'nft', 'pair', 'router', 'factory', 'bridge', 'cross', 'chain',
    'governance', 'vote', 'proposal', 'timelock', 'multi', 'sig',
})

TRACKED_WORDS = [
    'transfer', 'withdraw', 'delegate', 'proxy', 'flash', 'swap',
    'safe', 'guard', 'only', 'owner', 'check', 'execute', 'mint',
    'burn', 'stake', 'approve', 'permit', 'oracle', 'pool', 'vault',
    'upgrade', 'migrate', 'callback', 'fallback', 'payable', 'lock',
    'pause', 'reentrancy',
]

# ── Solidity keywords/punctuation to strip from code token stream ───────────
_SOL_KEYWORDS = frozenset({
    'pragma', 'solidity', 'import', 'contract', 'interface', 'library',
    'abstract', 'is', 'using', 'for', 'struct', 'enum', 'event',
    'function', 'modifier', 'constructor', 'fallback', 'receive',
    'public', 'private', 'internal', 'external', 'view', 'pure',
    'payable', 'virtual', 'override', 'returns', 'return', 'if',
    'else', 'while', 'do', 'for', 'break', 'continue', 'emit',
    'require', 'assert', 'revert', 'try', 'catch', 'new', 'delete',
    'mapping', 'memory', 'storage', 'calldata', 'constant', 'immutable',
    'indexed', 'anonymous', 'assembly', 'let', 'true', 'false',
    'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'uint256',
    'int', 'int8', 'int16', 'int32', 'int64', 'int128', 'int256',
    'address', 'bool', 'string', 'bytes', 'bytes1', 'bytes4', 'bytes8',
    'bytes16', 'bytes20', 'bytes32',
})

_RE_COMMENT_LINE = re.compile(r'//.*$', re.MULTILINE)
_RE_COMMENT_BLOCK = re.compile(r'/\*.*?\*/', re.DOTALL)
_RE_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_RE_HEX_LITERAL = re.compile(r'0x[0-9a-fA-F]+')
_RE_NUMBER_LITERAL = re.compile(r'\b\d+\b')
_RE_WORD_TOKEN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')


def read_source(path_str: str) -> str:
    """Read Solidity source from a file path or directory of .sol files."""
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.exists():
        return ""
    if p.is_file():
        try:
            return p.read_text(errors='ignore')
        except Exception:
            return ""
    if p.is_dir():
        texts = []
        for sol in sorted(p.rglob("*.sol")):
            try:
                texts.append(sol.read_text(errors='ignore'))
            except Exception:
                pass
        return "\n".join(texts)
    return ""


def _split_identifier(name: str) -> List[str]:
    """Split camelCase / PascalCase / snake_case into constituent words."""
    parts = name.split('_')
    words = []
    for part in parts:
        if not part:
            continue
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', part)
        sub = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', sub)
        words.extend(w.lower() for w in sub.split('_') if w)
    return words


def _extract_identifiers(source: str) -> List[str]:
    """Extract all declared identifiers from Solidity source."""
    idents = []
    for pat in _IDENT_PATTERNS:
        idents.extend(pat.findall(source))
    return idents


def compute_identifier_features(source: str) -> Dict[str, float]:
    """Compute identifier-based semantic features for one contract.

    Returns a dict of ~73 features covering naming statistics,
    risk/safety/DeFi word profiles, and tracked word indicators.
    """
    identifiers = _extract_identifiers(source)
    if not identifiers:
        return {f"name_{k}": 0.0 for k in [
            'n_identifiers', 'n_words', 'n_unique_words', 'word_diversity',
            'avg_ident_len', 'avg_words_per_ident',
            'risk_count', 'safety_count', 'defi_count',
            'risk_unique', 'safety_unique', 'defi_unique',
            'risk_ratio', 'safety_ratio', 'defi_ratio',
            'risk_to_safety', 'safety_to_total',
        ] + [f'has_{w}' for w in TRACKED_WORDS]
          + [f'count_{w}' for w in TRACKED_WORDS]}

    all_words: List[str] = []
    for ident in identifiers:
        all_words.extend(_split_identifier(ident))

    wc = Counter(all_words)
    n_words = len(all_words)
    n_unique = len(wc)

    risk_c = sum(wc.get(w, 0) for w in RISK_WORDS)
    safety_c = sum(wc.get(w, 0) for w in SAFETY_WORDS)
    defi_c = sum(wc.get(w, 0) for w in DEFI_WORDS)

    feats: Dict[str, float] = {
        'name_n_identifiers': len(identifiers),
        'name_n_words': n_words,
        'name_n_unique_words': n_unique,
        'name_word_diversity': n_unique / max(n_words, 1),
        'name_avg_ident_len': float(np.mean([len(i) for i in identifiers])),
        'name_avg_words_per_ident': n_words / max(len(identifiers), 1),
        'name_risk_count': risk_c,
        'name_safety_count': safety_c,
        'name_defi_count': defi_c,
        'name_risk_unique': sum(1 for w in RISK_WORDS if w in wc),
        'name_safety_unique': sum(1 for w in SAFETY_WORDS if w in wc),
        'name_defi_unique': sum(1 for w in DEFI_WORDS if w in wc),
        'name_risk_ratio': risk_c / max(n_words, 1),
        'name_safety_ratio': safety_c / max(n_words, 1),
        'name_defi_ratio': defi_c / max(n_words, 1),
        'name_risk_to_safety': risk_c / max(safety_c, 1),
        'name_safety_to_total': safety_c / max(risk_c + safety_c, 1),
    }

    for w in TRACKED_WORDS:
        feats[f'name_has_{w}'] = float(w in wc)
        feats[f'name_count_{w}'] = float(wc.get(w, 0))

    return feats


def add_identifier_features_to_frame(
    frame: pd.DataFrame,
    *,
    source_column: str = "local_source_path",
) -> pd.DataFrame:
    """Add leakage-safe identifier features, caching repeated source paths."""

    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")
    cache: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, float]] = []
    for value in frame[source_column].fillna("").astype(str):
        if value not in cache:
            cache[value] = compute_identifier_features(read_source(value))
        rows.append(cache[value])
    semantic = pd.DataFrame(rows, index=frame.index).fillna(0.0)
    semantic = semantic.add_prefix("semantic_")
    return pd.concat([frame, semantic], axis=1)


def compute_context_features(source: str) -> Dict[str, float]:
    """Extract compilation-free call, guard, state, and ordering signals."""

    lines = source.splitlines()
    lower = source.lower()
    external_calls = len(re.findall(r"\.(?:call|delegatecall|send|transfer)\b", lower))
    state_write_candidates = len(
        re.findall(r"\b(?:balances|owner|admin|allowance|state|status)\b[^;\n]*=", lower)
    )
    guard_count = len(re.findall(r"\b(?:require|assert|revert)\s*\(", lower))
    access_guard_count = len(
        re.findall(r"\b(?:msg\.sender|tx\.origin)\b|only[a-z0-9_]*", lower)
    )
    unchecked_count = len(re.findall(r"\bunchecked\s*\{|\bassembly\s*\{", lower))
    arithmetic_count = len(re.findall(r"[+\-*/%]\s*=?", source))
    timestamp_count = len(re.findall(r"\b(?:block\.timestamp|now)\b", lower))
    randomness_count = len(
        re.findall(r"\b(?:block\.hash|blockhash|prevrandao|difficulty)\b", lower)
    )
    loop_count = len(re.findall(r"\b(?:for|while)\s*\(", lower))
    payable_count = len(re.findall(r"\bpayable\b", lower))
    external_function_count = len(
        re.findall(r"\bfunction\b[^{}\n]*\bexternal\b", lower)
    )

    call_lines = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\.(?:call|delegatecall|send|transfer)\b", line.lower())
    ]
    write_lines = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\b(?:balances|owner|admin|allowance|state|status)\b[^;\n]*=", line.lower())
    ]
    calls_before_writes = sum(
        any(write_line > call_line for write_line in write_lines)
        for call_line in call_lines
    )
    return {
        "context_external_call_count": float(external_calls),
        "context_state_write_count": float(state_write_candidates),
        "context_guard_count": float(guard_count),
        "context_access_guard_count": float(access_guard_count),
        "context_unchecked_assembly_count": float(unchecked_count),
        "context_arithmetic_count": float(arithmetic_count),
        "context_timestamp_count": float(timestamp_count),
        "context_randomness_count": float(randomness_count),
        "context_loop_count": float(loop_count),
        "context_payable_count": float(payable_count),
        "context_external_function_count": float(external_function_count),
        "context_call_before_state_write_count": float(calls_before_writes),
        "context_call_to_guard_ratio": external_calls / max(guard_count, 1),
        "context_state_write_to_call_ratio": state_write_candidates / max(external_calls, 1),
    }


def add_semantic_features_to_frame(
    frame: pd.DataFrame,
    *,
    source_column: str = "local_source_path",
) -> pd.DataFrame:
    """Add identifier and targeted context features with source-path caching."""

    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")
    cache: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, float]] = []
    for value in frame[source_column].fillna("").astype(str):
        if value not in cache:
            source = read_source(value)
            features = compute_identifier_features(source)
            features.update(compute_context_features(source))
            cache[value] = features
        rows.append(cache[value])
    semantic = pd.DataFrame(rows, index=frame.index).fillna(0.0)
    semantic = semantic.add_prefix("semantic_")
    return pd.concat([frame, semantic], axis=1)


def tokenize_solidity(source: str) -> str:
    """Tokenize Solidity source into a space-separated string of meaningful
    code tokens (for TF-IDF).  Strips comments, strings, hex/number literals,
    and Solidity keywords so only identifier-like tokens and camelCase
    sub-words remain."""
    text = _RE_COMMENT_BLOCK.sub(' ', source)
    text = _RE_COMMENT_LINE.sub(' ', text)
    text = _RE_STRING_LITERAL.sub(' ', text)
    text = _RE_HEX_LITERAL.sub(' ', text)
    text = _RE_NUMBER_LITERAL.sub(' ', text)

    raw_tokens = _RE_WORD_TOKEN.findall(text)

    words = []
    for tok in raw_tokens:
        if tok.lower() in _SOL_KEYWORDS:
            continue
        words.extend(_split_identifier(tok))

    return " ".join(words)


class CodeLSA:
    """Latent Semantic Analysis over Solidity code tokens.

    Wraps TF-IDF + TruncatedSVD. Must be fit on training data only,
    then used to transform both train and test sets.
    """

    def __init__(self, n_components: int = 50, max_features: int = 5000,
                 min_df: int = 5, max_df: float = 0.95):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            norm='l2',
        )
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        )

    def fit_transform(self, token_docs: List[str]) -> np.ndarray:
        """Fit TF-IDF + SVD on training documents, return topic matrix."""
        tfidf_matrix = self.tfidf.fit_transform(token_docs)
        return self.svd.fit_transform(tfidf_matrix)

    def transform(self, token_docs: List[str]) -> np.ndarray:
        """Transform new documents using the fitted pipeline."""
        tfidf_matrix = self.tfidf.transform(token_docs)
        return self.svd.transform(tfidf_matrix)

    @property
    def feature_names(self) -> List[str]:
        return [f"lsa_topic_{i}" for i in range(self.n_components)]
