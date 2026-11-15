"""Evidence for reserved-source exclusion; absence of a match never admits training."""
from __future__ import annotations
import hashlib
from map_function_scopes import scope_index
from group_function_candidates import signatures

VERSION = 'reserved-source-function-match-v1'


def named_function_fragments(text):
    """Yield lexical direct named implemented members; no compiler/source execution.

    Abstract/special forms, inherited resolution and malformed sources are not
    certified. A lexical failure is recorded by the caller, never a clean result.
    """
    index = scope_index(text)
    tokens, depths, pairs = index['tokens'], index['depth'], index['brace_pairs']
    for i, token in enumerate(tokens):
        if token[:2] != ('identifier', 'function') or depths[i] != 1:
            continue
        if i+2 >= len(tokens) or tokens[i+1][0] != 'identifier' or tokens[i+2][1] != '(':
            continue
        owners = [d for d in index['declarations'] if d['body_open'] < token[2] < d['body_close']]
        if len(owners) != 1:
            continue
        parens = brackets = 0
        for j in range(i+2, len(tokens)):
            kind, value, start, end = tokens[j]
            if kind == 'literal':
                continue
            if value == '(': parens += 1
            elif value == ')': parens -= 1
            elif value == '[': brackets += 1
            elif value == ']': brackets -= 1
            elif value == '{':
                if parens or brackets or j not in pairs:
                    break
                closing = tokens[pairs[j]][3]
                if closing <= owners[0]['body_close']:
                    yield {'owner': owners[0]['name'], 'name': tokens[i+1][1],
                           'start_character': token[2], 'end_character': closing,
                           'code': text[token[2]:closing]}
                break
            elif value in (';', '}'):
                break
            if parens < 0 or brackets < 0:
                break


def fingerprints(code):
    token, template, count = signatures(code)
    return {'exact': hashlib.sha256(code.encode()).hexdigest(),
            'tokens': token, 'template': template, 'token_count': count}


def exclusion_status(reasons):
    """Never convert nonmatch into a provenance/near-clone clearance."""
    reasons = set(reasons)
    if reasons & {'exact', 'tokens', 'context'}:
        return 'exclude_direct_reserved_content_match'
    if 'template' in reasons:
        return 'quarantine_reserved_template_match'
    if reasons & {'project_family', 'constraint_group'}:
        return 'quarantine_reserved_related_group'
    return 'unresolved_no_reserved_match_found'
