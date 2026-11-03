#!/usr/bin/env python3
"""Bind exact function spans to explicit source-qualified lexical contract scopes."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import group_function_candidates as grouping
import recover_function_context as recovery
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import safe_file, verify_artifacts
from prepare_function_review import CATEGORIES

VERSION = 'qualified-lexical-function-scope-v1'
IDENTIFIER = r'[A-Za-z_$][A-Za-z0-9_$]*'


def qualifier(row):
    separator = {'sujeetc/SCRUBD': r'\.', 'CoinFabrik/solidity-rnd': '::'}.get(row['source'])
    value = row.get('function_identifier')
    match = re.fullmatch('(' + IDENTIFIER + ')' + separator + '(' + IDENTIFIER + ')', value) if separator and isinstance(value, str) else None
    return {'contract': match[1], 'function': match[2], 'authority': 'producer_metadata_not_independently_verified'} if match else None


def scope_index(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 16 * 1024**2:
        raise ValueError('Expected bounded context text')
    masked, _, _ = grouping.masked_source_and_imports(text)
    tokens = [(m.lastgroup, m.group(), m.start(), m.end()) for m in grouping.TOKEN.finditer(masked)]
    depth = []; stack = []; pairs = {}
    for i, token in enumerate(tokens):
        depth.append(len(stack))
        if token[0] != 'literal' and token[1] == '{': stack.append(i)
        elif token[0] != 'literal' and token[1] == '}':
            if not stack: raise ValueError('Unmatched source closing brace')
            pairs[stack.pop()] = i
    if stack: raise ValueError('Unclosed source brace')
    declarations = []
    for i, token in enumerate(tokens):
        if token[0] != 'identifier' or token[1] not in ('contract', 'library', 'interface') or depth[i] != 0: continue
        if i + 1 >= len(tokens) or tokens[i+1][0] != 'identifier': raise ValueError('Unsupported top-level declaration name')
        parens = brackets = 0; opening = None
        for j in range(i+2, len(tokens)):
            kind, value, _, _ = tokens[j]
            if kind == 'literal': continue
            if value == '(': parens += 1
            elif value == ')': parens -= 1
            elif value == '[': brackets += 1
            elif value == ']': brackets -= 1
            elif value == '{':
                if parens or brackets: raise ValueError('Unsupported brace in declaration header')
                opening = j; break
            elif value in (';', '}'): raise ValueError('Unsupported incomplete declaration header')
            if parens < 0 or brackets < 0: raise ValueError('Unbalanced declaration header')
        if opening is None or depth[opening] != 0: raise ValueError('Missing top-level declaration body')
        closing = pairs[opening]
        declarations.append({'kind': token[1], 'name': tokens[i+1][1], 'declaration_start': token[2],
                             'header_end': tokens[opening][3], 'body_open': tokens[opening][2], 'body_close': tokens[closing][2],
                             'declaration_end': tokens[closing][3]})
    return {'tokens': tokens, 'depth': depth, 'brace_pairs': pairs, 'declarations': declarations,
            'token_starts': {t[2]: i for i,t in enumerate(tokens)}}


def span_scope(span, index, expected_function):
    start, end = span['start_character'], span['end_character']; tokens = index['tokens']; i = index['token_starts'].get(start)
    evidence = {'span': span, 'owner': None, 'direct_member': False, 'function_name_matches': False, 'body_span_matches': False}
    owners = [d for d in index['declarations'] if d['body_open'] < start < end <= d['body_close']]
    if len(owners) == 1: evidence['owner'] = owners[0]
    if i is None or tokens[i][:2] != ('identifier', 'function'): return evidence
    evidence['direct_member'] = index['depth'][i] == 1 and len(owners) == 1
    if i + 1 >= len(tokens) or tokens[i+1][:2] != ('identifier', expected_function): return evidence
    evidence['function_name_matches'] = True
    parens = brackets = 0
    for j in range(i+2, len(tokens)):
        kind, value, begin, _ = tokens[j]
        if begin >= end: break
        if kind == 'literal': continue
        if value == '(': parens += 1
        elif value == ')': parens -= 1
        elif value == '[': brackets += 1
        elif value == ']': brackets -= 1
        elif value == '{':
            if parens or brackets: return evidence
            closing = index['brace_pairs'].get(j)
            evidence['body_span_matches'] = closing is not None and tokens[closing][3] == end
            return evidence
        elif value in (';', '}'): return evidence
        if parens < 0 or brackets < 0: return evidence
    return evidence


def resolve(row, link, index=None, diagnostic=None):
    expected = qualifier(row)
    result = {'record_id': row['record_id'], 'context_artifact': link['context_file'], 'pinned_context_sha256': row.get('context_sha256'),
              'source_qualified_identifier': row.get('function_identifier'), 'qualifier_claim': expected,
              'prior_fragment_localization': link['fragment_localization'], 'original_candidate_spans': link['function_spans'],
              'candidate_scope_evidence': [], 'selected_span': None, 'selected_declaration': None,
              'scope_status': 'context_unavailable', 'scope_diagnostic': diagnostic,
              'semantic_context_sufficiency': 'not_reviewed', 'evaluation_status': link['evaluation_status'],
              'pattern_labels': row['pattern_labels'], 'label_observation_mask': dict.fromkeys(CATEGORIES, False),
              'human_review_status': 'unreviewed', 'training_eligible': False, 'split': None}
    if not link['context_file']: return result
    if expected is None: result['scope_status'] = 'unsupported_source_qualifier'; return result
    if index is None: result['scope_status'] = 'context_lexical_scope_unavailable'; return result
    if expected['function'] in ('constructor', 'receive', 'fallback'):
        result['scope_status'] = 'unsupported_special_function_form'; return result
    evidence = [span_scope(span, index, expected['function']) for span in link['function_spans']]
    result['candidate_scope_evidence'] = evidence
    declarations = [d for d in index['declarations'] if d['name'] == expected['contract']]
    if len(declarations) > 1: result['scope_status'] = 'repeated_contract_name_requires_review'; return result
    if not declarations: result['scope_status'] = 'qualified_contract_not_found'; return result
    compatible = [e for e in evidence if e['owner'] is not None and e['owner']['name'] == expected['contract'] and all(e[k] for k in ('direct_member','function_name_matches','body_span_matches'))]
    if len(compatible) == 1:
        result.update(scope_status='unique_qualified_scope_match', selected_span=compatible[0]['span'], selected_declaration=compatible[0]['owner'],
                      resolution_basis='source_qualified_owner_distinguishes_identical_fragments' if len(evidence)>1 else 'unique_fragment_and_source_qualified_owner_agree')
    elif len(compatible) > 1: result['scope_status'] = 'multiple_qualified_scope_matches'
    else: result['scope_status'] = 'qualified_scope_conflict_or_unsupported_span'
    return result


def prepare(pilot, contexts, output):
    pilot, contexts, output = [p.resolve() for p in (pilot, contexts, output)]
    if output == pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or output == contexts or output.is_relative_to(contexts):
        raise ValueError('Preserve originals and prior release; use new versioned output')
    if output.exists(): raise FileExistsError(output)
    delivered = verify_delivery(pilot); parent_hash = sha256(safe_file(contexts,'recovery_manifest.json'))
    parent = verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256')
    if parent['delivery_manifest_sha256'] != sha256(pilot/'DELIVERY_SHA256.json'): raise ValueError('Context release belongs to another delivery')
    rows = list(jsonl_rows(pilot/'public_output/function_candidates.jsonl')); links = list(jsonl_rows(contexts/'candidate_context_links.jsonl'))
    by_id = {r['record_id']:r for r in rows}
    if len(rows)>5000 or len(by_id)!=len(rows) or len(links)!=len(rows) or {r['record_id'] for r in links}!=set(by_id): raise ValueError('Candidate inventories differ or exceed budget')
    for row in [*rows,*links]:
        if set(row['pattern_labels']) != set(CATEGORIES) or any(v is not None for v in row['pattern_labels'].values()) or row['training_eligible'] is not False or row['split'] is not None or any(row.get('label_observation_mask',{}).values()) or row.get('human_review_status','unreviewed')!='unreviewed':
            raise ValueError('Annotated or assigned input requires explicit migration')
    documents = {}; mappings = []; summaries = []
    for link in sorted(links,key=lambda r:r['record_id']):
        row=by_id[link['record_id']];digest=row.get('context_sha256');index=None;diagnostic=None
        if link['context_file']:
            if digest!=link['pinned_context_sha256']: raise ValueError('Context link differs from original pin')
            path=safe_file(contexts,link['context_file'])
            if sha256(path)!=digest: raise ValueError('Context bytes differ from original pin')
            if digest not in documents:
                if len(documents)>=128: raise ValueError('Source-document budget exceeded')
                text=path.read_bytes().decode('utf-8')
                try: parsed=scope_index(text);error=None
                except ValueError as exc: parsed=None;error=str(exc)
                documents[digest]=(text,parsed,error)
                summaries.append({'context_sha256':digest,'context_artifact':link['context_file'],'scope_status':'computed_lexically' if parsed else 'unavailable',
                                  'diagnostic':error,'declarations':parsed['declarations'] if parsed else []})
            text,index,diagnostic=documents[digest]
            for span in link['function_spans']:
                start,end=span['start_character'],span['end_character']
                if not 0<=start<end<=len(text) or end-start!=row['characters'] or recovery.text_hash(text[start:end])!=row['raw_sha256'] or span['raw_fragment_sha256']!=row['raw_sha256']:
                    raise ValueError('Original fragment span is not exact evidence')
        mappings.append(resolve(row,link,index,diagnostic))
    if verify_delivery(pilot)!=delivered or sha256(contexts/'recovery_manifest.json')!=parent_hash: raise ValueError('Inputs changed during mapping')
    verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256')
    covered=[r for r in mappings if r['context_artifact']];resolved=[r for r in covered if r['selected_span']]
    summary={'protocol':VERSION,'candidate_count':len(rows),'context_documents':len(documents),'covered_candidates':len(covered),
             'all_candidate_scope_status_counts':dict(Counter(r['scope_status'] for r in mappings)),
             'covered_scope_status_counts':dict(Counter(r['scope_status'] for r in covered)),
             'qualified_lexical_scope_matches':len(resolved),
             'previously_ambiguous_scope_status_counts':dict(Counter(r['scope_status'] for r in covered if r['prior_fragment_localization']=='ambiguous_identical_fragments')),
             'prior_unique_scope_status_counts':dict(Counter(r['scope_status'] for r in covered if r['prior_fragment_localization']=='exact_unique_fragment')),
             'newly_disambiguated_records':[r['record_id'] for r in resolved if r['prior_fragment_localization']=='ambiguous_identical_fragments'],
             'independent_reviews_completed':0,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0,
             'limitations':['Producer contract qualifiers are source claims, not independently verified semantic identity. Lexical scope/brace evidence is not a Solidity AST, compiler, inheritance or overload resolver.',
                            'Only exact original spans at direct declaration-member depth with one compatible declared owner are selected. Unsupported/conflicting cases remain unresolved; inherited functions and malformed/complex headers can be conservatively rejected.',
                            'Raw-source contexts and original candidate spans remain immutable; model code, labels, corpus overlap constraints, project/license evidence and evaluation readiness do not change. No training, inference, accuracy or Q1-readiness gain.']}
    output.mkdir(parents=True,exist_ok=False)
    recovery.write_jsonl(output/'function_scope_links.jsonl',mappings);recovery.write_jsonl(output/'context_scope_index.jsonl',sorted(summaries,key=lambda r:r['context_sha256']))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'helper_sha256':{Path(m.__file__).name:sha256(Path(m.__file__)) for m in (grouping,recovery)},
              'comment_masker_sha256':sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)),
              'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,
              'context_release_manifest_sha256':parent_hash,'context_reference_rule':'context_artifact is relative to the immutable hash-bound input recovery release, not this derivative',
              'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','context-release-dir','output-dir'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.pilot_dir,args.context_release_dir,args.output_dir),indent=2))

if __name__=='__main__': main()
