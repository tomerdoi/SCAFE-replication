"""Experimental comment-control preprocessing; frozen feature recipes stay unchanged."""
from __future__ import annotations
from dataclasses import dataclass

MAX_SOURCE_BYTES = 2 * 1024**2
VERSION = 'literal-aware-comment-mask-v1'


@dataclass(frozen=True)
class CommentView:
    text: str
    spans: tuple[tuple[int, int, str], ...]


def mask_comments(source: str) -> CommentView:
    """Blank comments, retaining all strings, character positions and CR/LF breaks.

    Reject unfinished block comments or quotes rather than silently discard code.
    Character offsets are preserved; Unicode comment byte lengths may differ.
    """
    if not isinstance(source, str) or len(source.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise ValueError('Expected bounded UTF-8 source text')
    chars = list(source); spans = []; index = 0
    while index < len(source):
        char = source[index]
        if char in {'"', "'"}:
            quote = char; index += 1
            while index < len(source):
                if source[index] == '\\':
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            else:
                raise ValueError('Unterminated source string')
            continue
        if source.startswith('//', index):
            start = index; index += 2
            while index < len(source) and source[index] not in '\r\n': index += 1
            end = index; kind = 'line'
        elif source.startswith('/*', index):
            start = index; close = source.find('*/', index + 2)
            if close < 0: raise ValueError('Unterminated source block comment')
            end = close + 2; index = end; kind = 'block'
        else:
            index += 1
            continue
        for offset in range(start, end):
            if chars[offset] not in '\r\n': chars[offset] = ' '
        spans.append((start, end, kind))
    return CommentView(''.join(chars), tuple(spans))


def extract_comment_control_features(source: str, *, n_files: int = 1) -> dict:
    """Return a separately versioned, opt-in feature view; no model/cache mutation.

    All legacy features are recomputed from comment-masked text, so intentional
    comment/style channels also change. This is not a drop-in model replacement.
    Strings are preserved, and remaining regex/AST approximations are unchanged.
    Metadata is separate from the feature dictionary and must not enter a model.
    """
    import hashlib
    import os
    from pathlib import Path
    from src.risk import contract_risk
    if type(n_files) is not int or not 1 <= n_files <= 1000:
        raise ValueError('Expected a bounded positive source-file count')
    view = mask_comments(source)
    base = contract_risk.ContractRiskUtil._empty_feature_base('comment-control')
    features = contract_risk.ContractRiskUtil._extract_source_features(base, view.text, n_files)
    return {
        'metadata': {
            'version': VERSION + ':experimental-static-view-v1',
            'raw_source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
            'processed_source_sha256': hashlib.sha256(view.text.encode('utf-8')).hexdigest(),
            'legacy_extractor_sha256': hashlib.sha256(Path(contract_risk.__file__).read_bytes()).hexdigest(),
            'experimental_extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'comment_spans': len(view.spans),
            'n_files': n_files,
            'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'unspecified'),
            'numeric_replay_tolerance': {'relative': 1e-12, 'absolute': 1e-12},
            'model_training_or_scoring': False,
            'compatibility': 'experimental;recompute_retrain_and_evaluate_separately;not_legacy_cache_or_model_compatible',
        },
        'features': features,
    }
