#!/usr/bin/env python3
"""Extract traceable call-shaped source facts for review, never security labels."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import group_function_candidates as grouping
import recover_function_context as recovery
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import safe_file, verify_artifacts
from prepare_function_review import CATEGORIES

VERSION='function-call-result-evidence-v1'
MEMBERS={'call','delegatecall','staticcall','callcode','send'}
GUARDS={'require','assert','if'}


def structure(tokens):
    stack=[];pairs={};states=[];braces=parens=brackets=0
    for i,token in enumerate(tokens):
        states.append((braces,parens,brackets));kind,value,_,_=token
        if kind=='literal':continue
        if value in ('{','(','['):
            stack.append((value,i))
            if value=='{':braces+=1
            elif value=='(':parens+=1
            else:brackets+=1
        elif value in ('}',')',']'):
            if not stack or stack[-1][0]!={'}':'{',')':'(',']':'['}[value]:raise ValueError('Unbalanced lexical delimiter')
            opening,j=stack.pop();pairs[j]=i
            if opening=='{':braces-=1
            elif opening=='(':parens-=1
            else:brackets-=1
    if stack:raise ValueError('Unclosed lexical delimiter')
    return pairs,states


def invocation(i,tokens,pairs):
    values=[t[1] for t in tokens];j=i+1
    if j>=len(tokens):return None,'unsupported_or_incomplete'
    if values[j]=='(' and j in pairs:return pairs[j],'direct'
    if values[j]=='{' and j in pairs:
        j=pairs[j]+1
        if j<len(tokens) and values[j]=='(' and j in pairs:return pairs[j],'modern_options'
        return None,'unsupported_or_incomplete'
    options=0
    while j+2<len(tokens) and values[j]=='.' and values[j+1] in ('gas','value') and values[j+2]=='(' and j+2 in pairs:
        options+=1
        if options>4:return None,'unsupported_or_incomplete'
        j=pairs[j+2]+1
    if options and j<len(tokens) and values[j]=='(' and j in pairs:return pairs[j],'legacy_gas_value'
    return None,'unsupported_or_incomplete'


def simple_receiver(values):
    def path(v):return bool(v) and len(v)%2==1 and all((x=='.' if i%2 else x.replace('_','a').replace('$','a').isascii() and x.replace('_','a').replace('$','a').isalnum() and not x[0].isdigit()) for i,x in enumerate(v))
    if path(values):return True
    return len(values)>=4 and values[0] in ('address','payable') and values[1]=='(' and values[-1]==')' and path(values[2:-1])


def disposition(i,end,tokens,pairs,states):
    values=[t[1] for t in tokens];ancestors=[j for j,close in pairs.items() if values[j]=='(' and j<i<close]
    guards=sorted({values[j-1] for j in ancestors if j>0 and tokens[j-1][0]=='identifier' and values[j-1] in GUARDS})
    containers=[j for j,close in pairs.items() if values[j]=='{' and j<i<close]
    if not containers:return 'ambiguous_or_other',None,guards
    opening=max(containers);closing=pairs[opening];level=states[i][0];start=opening+1
    for j in range(opening+1,i):
        if values[j]==';' and states[j]==(level,0,0):start=j+1
    finish=next((j for j in range(i+1,closing) if values[j]==';' and states[j]==(level,0,0)),None)
    statement=(start,finish) if finish is not None and finish-start<=256 else None
    if guards:return 'lexical_guard_expression',statement,guards
    if statement is None or end is None or states[i][1:]!=(0,0) or end!=finish-1:return 'ambiguous_or_other',statement,guards
    prefix=values[start:i-1]  # Exclude the member-access dot.
    if simple_receiver(prefix):return 'bare_expression_statement',statement,guards
    if prefix and prefix[0]=='return' and simple_receiver(prefix[1:]):return 'return_expression',statement,guards
    assignments=[j for j,v in enumerate(prefix) if v=='=']
    if len(assignments)==1 and assignments[0]>0 and simple_receiver(prefix[assignments[0]+1:]):return 'assignment_context',statement,guards
    return 'ambiguous_or_other',statement,guards


def span_evidence(raw,masked,start,end,offset=0):
    return {'start_character':offset+start,'end_character':offset+end,'raw_span_sha256':recovery.text_hash(raw[start:end]),
            'masked_span_sha256':recovery.text_hash(masked[start:end]),'comment_masked_text':masked[start:end]}


def extract(raw,offset,record_id,context_sha):
    masked,_,_=grouping.masked_source_and_imports(raw)
    tokens=[(m.lastgroup,m.group(),m.start(),m.end()) for m in grouping.TOKEN.finditer(masked)]
    if len(tokens)>40_000:raise ValueError('Function token budget exceeded')
    pairs,states=structure(tokens);sites=[]
    for i,token in enumerate(tokens):
        if token[0]!='identifier' or token[1] not in MEMBERS or i==0 or tokens[i-1][1]!='.':continue
        end,form=invocation(i,tokens,pairs);use,statement,guards=disposition(i,end,tokens,pairs,states)
        if end is None:use='unsupported_invocation'
        member=span_evidence(raw,masked,token[2],token[3],offset)
        call=span_evidence(raw,masked,token[2],tokens[end][3],offset) if end is not None else None
        stmt=span_evidence(raw,masked,tokens[statement[0]][2],tokens[statement[1]][3],offset) if statement else None
        identity=recovery.text_hash(VERSION+':'+record_id+':'+str(member['start_character']))
        sites.append({'site_id':identity,'record_id':record_id,'context_sha256':context_sha,'member_spelling':token[1],
                      'member_span':member,'invocation_shape':form,'invocation_span':call,'statement_window_span':stmt,
                      'result_disposition_observation':use,'lexical_guard_spellings':guards,'receiver_type_status':'unresolved_not_typechecked',
                      'review_hint':'inspect_possible_ignored_result' if use=='bare_expression_statement' else 'inspect_result_use_and_context',
                      'target_review_category':'unchecked_external_call','category_label':None,'independently_reviewed':False,
                      'warning':'Member spelling/result shape is source evidence, not verified external-call semantics, unchecked-call truth or safety.'})
        if len(sites)>1000:raise ValueError('Function call-site budget exceeded')
    return sites


def prepare(pilot,contexts,scopes,output):
    pilot,contexts,scopes,output=[p.resolve() for p in (pilot,contexts,scopes,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or any(output==p or output.is_relative_to(p) for p in (contexts,scopes)):
        raise ValueError('Preserve original and parent releases; use new versioned output')
    if output.exists():raise FileExistsError(output)
    delivered=verify_delivery(pilot);context_hash=sha256(safe_file(contexts,'recovery_manifest.json'));scope_hash=sha256(safe_file(scopes,'manifest.json'))
    cm=verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256');sm=verify_artifacts(scopes,'manifest.json','artifact_sha256')
    if cm['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json') or sm['delivery_manifest_sha256']!=cm['delivery_manifest_sha256'] or sm['context_release_manifest_sha256']!=context_hash:
        raise ValueError('Parent lineage mismatch')
    rows=list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'));by_id={r['record_id']:r for r in rows};mappings=list(jsonl_rows(scopes/'function_scope_links.jsonl'))
    if len(rows)>5000 or len(by_id)!=len(rows) or len(mappings)!=len(rows) or {r['record_id'] for r in mappings}!=set(by_id):raise ValueError('Inventory differs or exceeds budget')
    for row in [*rows,*mappings]:
        if set(row['pattern_labels'])!=set(CATEGORIES) or any(v is not None for v in row['pattern_labels'].values()) or any(row.get('label_observation_mask',{}).values()) or row['training_eligible'] is not False or row['split'] is not None or row.get('human_review_status','unreviewed')!='unreviewed':raise ValueError('Reviewed/assigned input requires explicit migration')
    documents={};functions=[];sites=[];queue=[]
    for mapping in sorted(mappings,key=lambda r:r['record_id']):
        identity=mapping['record_id'];row=by_id[identity];items=[];diagnostic=None
        status='not_analyzed_scope_'+mapping['scope_status']
        if mapping['scope_status']=='unique_qualified_scope_match':
            digest=row['context_sha256'];path=safe_file(contexts,mapping['context_artifact'])
            if digest!=mapping['pinned_context_sha256'] or sha256(path)!=digest:raise ValueError('Context differs from original identity')
            if digest not in documents:
                if len(documents)>=128:raise ValueError('Context document budget exceeded')
                documents[digest]=path.read_bytes().decode('utf-8')
            text=documents[digest];span=mapping['selected_span'];start,end=span['start_character'],span['end_character'];raw=text[start:end]
            if span not in mapping['original_candidate_spans'] or end-start!=row['characters'] or recovery.text_hash(raw)!=row['raw_sha256']:raise ValueError('Selected function is not original exact evidence')
            try:items=extract(raw,start,identity,digest);status='analyzed_lexical_call_shapes'
            except ValueError as error:status='not_analyzed_lexical_structure';diagnostic=str(error)
        record={'record_id':identity,'analysis_status':status,'diagnostic':diagnostic,'qualified_scope_status':mapping['scope_status'],
                'context_artifact':mapping['context_artifact'],'selected_source_span':mapping['selected_span'],
                'observed_call_site_ids':[i['site_id'] for i in items],
                'observed_site_count':len(items) if status=='analyzed_lexical_call_shapes' else None,
                'bare_result_candidate_count':sum(i['result_disposition_observation']=='bare_expression_statement' for i in items) if status=='analyzed_lexical_call_shapes' else None,
                'pattern_labels':row['pattern_labels'],'label_observation_mask':dict.fromkeys(CATEGORIES,False),
                'human_review_status':'unreviewed','semantic_context_sufficiency':'unreviewed','training_eligible':False,'split':None}
        functions.append(record);sites.extend(items)
        if items:
            queue.append({'record_id':identity,'site_ids':record['observed_call_site_ids'],'review_reason':'possible_ignored_result_shape' if record['bare_result_candidate_count'] else 'other_result_use_comparator',
                          'selection':'all analyzed functions with target member spellings; descriptive convenience queue, not prevalence sample or verified hard negatives',
                          'context_artifact':mapping['context_artifact'],'source_span':mapping['selected_span'],
                          'needed_review':['receiver type and exact call semantics','full failure/result propagation and reachability','modifiers, callers/callees and dependency sufficiency','independent category adjudication'],
                          'pattern_labels':row['pattern_labels'],'label_observation_mask':dict.fromkeys(CATEGORIES,False),'human_review_status':'unreviewed','training_eligible':False,'split':None})
    if len(sites)>5000:raise ValueError('Cohort call-site budget exceeded')
    if verify_delivery(pilot)!=delivered or sha256(contexts/'recovery_manifest.json')!=context_hash or sha256(scopes/'manifest.json')!=scope_hash:raise ValueError('Inputs changed during extraction')
    verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256');verify_artifacts(scopes,'manifest.json','artifact_sha256')
    summary={'protocol':VERSION,'candidate_count':len(rows),'function_analysis_status_counts':dict(Counter(r['analysis_status'] for r in functions)),
             'observed_target_member_sites':len(sites),'functions_with_observed_sites':len(queue),
             'member_spelling_counts':dict(Counter(r['member_spelling'] for r in sites)),
             'invocation_shape_counts':dict(Counter(r['invocation_shape'] for r in sites)),
             'result_disposition_counts':dict(Counter(r['result_disposition_observation'] for r in sites)),
             'functions_with_bare_result_candidates':sum(bool(r['bare_result_candidate_count']) for r in functions),'context_documents_read':len(documents),
             'independent_reviews_completed':0,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0,
             'limitations':['Call/send member spellings are not typechecked external-call ground truth. transfer is excluded because address/token semantics cannot be inferred by spelling.',
                            'Bare expression shapes are review candidates, not verified unchecked-call labels. Assignments, guards and returned values do not establish checked failures, safety or absence of other suspicious patterns.',
                            'Only qualified named-function inputs are targeted; special functions, missing contexts and unsupported syntax are not negatives. No whole-program dataflow/CFG, dependency closure, label adjudication, model inference or training.',
                            'Comment-masked evidence is a coordinator aid, not guaranteed source-blinded annotation material or a replacement model-input release. Existing source/qualifier, license and corpus-overlap limitations persist.']}
    output.mkdir(parents=True,exist_ok=False)
    for name,data in [('function_call_evidence.jsonl',functions),('call_site_evidence.jsonl',sites),('review_queue.jsonl',sorted(queue,key=lambda r:(r['review_reason']!='possible_ignored_result_shape',r['record_id'])))]:recovery.write_jsonl(output/name,data)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'helper_sha256':{Path(m.__file__).name:sha256(Path(m.__file__)) for m in (grouping,recovery)},
              'comment_masker_sha256':sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)),
              'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,
              'context_release_manifest_sha256':context_hash,'scope_release_manifest_sha256':scope_hash,
              'target_member_spellings':sorted(MEMBERS),'context_reference_rule':'context_artifact is relative to the immutable hash-bound parent context release',
              'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','context-release-dir','scope-release-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(prepare(a.pilot_dir,a.context_release_dir,a.scope_release_dir,a.output_dir),indent=2))

if __name__=='__main__':main()
