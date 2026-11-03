#!/usr/bin/env python3
"""Create lineage-bound views masking explicitly anchored inline audit-issue prose."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_function_pilot_handoff import inspect_staging_rows, jsonl_rows, sha256, verify_delivery
import group_function_candidates as grouping
from prepare_function_review import text_hash, write_jsonl
from prepare_context_review import safe_file, verify_artifacts

VERSION='post-statement-inline-audit-tail-v1'
VERSION_IF_BLOCK='post-statement-and-if-block-inline-audit-tail-v2'
MARKER=re.compile(r'<-[ \t]*@?audit-issue\b',re.IGNORECASE)
MAX_CHARS=1_000_000


def next_line_end(text,start):
    return min((p for mark in ('\r','\n') if (p:=text.find(mark,start))>=0),default=len(text))



def complete_if_block_prefix(prefix):
    """Recognize only one complete, balanced, same-line if (...) { prefix."""
    if '\n' in prefix or '\r' in prefix:return False
    try:masked,_,_=grouping.masked_source_and_imports(prefix)
    except ValueError:return False
    tokens=[(match.lastgroup,match.group()) for match in grouping.TOKEN.finditer(masked)]
    if len(tokens)<5 or tokens[0]!=('identifier','if') or tokens[1][1]!='(' or tokens[-1][1]!='{':return False
    depth=0
    for position,(kind,value) in enumerate(tokens[1:-1],1):
        if kind in ('literal','identifier','number'):continue
        if value=='(':depth+=1
        elif value==')':
            depth-=1
            if depth<0:return False
            if depth==0:return position==len(tokens)-2
    return False


def filter_view(text, *, include_if_blocks=False):
    if not isinstance(text,str) or len(text)>MAX_CHARS:raise ValueError('Expected bounded source text')
    chars=list(text);events=[];unhandled=[];in_literals=0;in_comments=0;i=0;line_start=0;last_semicolon=None;diagnostic=None
    while i<len(text):
        current=text[i];following=text[i+1] if i+1<len(text) else ''
        if current in '\r\n':line_start=i+1;last_semicolon=None;i+=1;continue
        if current=='/' and following in ('/','*'):
            start=i
            if following=='/':end=next_line_end(text,i+2)
            else:
                close=text.find('*/',i+2)
                if close<0:diagnostic='unterminated_block_comment';break
                end=close+2
            in_comments+=len(MARKER.findall(text[start:end]))
            last_newline=max(text.rfind('\n',start,end),text.rfind('\r',start,end))
            if last_newline>=0:line_start=last_newline+1;last_semicolon=None
            i=end;continue
        if current in ('"',"'"):
            start=i;quote=current;i+=1
            while i<len(text):
                if text[i]=='\\':i+=2;continue
                if text[i]==quote:i+=1;break
                i+=1
            else:diagnostic='unterminated_literal_before_eligible_marker';break
            in_literals+=len(MARKER.findall(text[start:i]))
            last_newline=max(text.rfind('\n',start,i),text.rfind('\r',start,i))
            if last_newline>=0:line_start=last_newline+1;last_semicolon=None
            continue
        marker=MARKER.match(text,i)
        if marker:
            prefix=text[line_start:i].rstrip(' \t')
            semicolon_position=line_start+len(prefix)-1
            end=next_line_end(text,i);raw=text[i:end]
            anchor='statement_semicolon' if prefix.endswith(';') and last_semicolon==semicolon_position else None
            if anchor is None and include_if_blocks and prefix.endswith('{') and complete_if_block_prefix(prefix) and not any(char in raw for char in '{};'):
                anchor='complete_if_block_opener'
            if anchor is not None:
                event={'start':i,'end':end,'reason':'outside_literal_comment_post_statement_audit_issue_tail' if anchor=='statement_semicolon' else 'outside_literal_comment_complete_if_block_audit_issue_tail',
                       'parent_span_sha256':text_hash(raw),'marker_matches_masked':len(MARKER.findall(raw))}
                if include_if_blocks:event['anchor']=anchor
                events.append(event)
                for position in range(i,end):chars[position]=' '
                i=end;continue
            unhandled.append({'offset':i,'reason':'marker_not_immediately_after_same_line_statement_semicolon'})
            i=marker.end();continue
        if current==';':last_semicolon=i
        i+=1
    view=''.join(chars)
    return view,{'events':events,'unhandled_outside_markers':unhandled,'markers_in_literals_retained':in_literals,
                 'markers_in_comments_retained':in_comments,'scanner_diagnostic':diagnostic,
                 'parent_marker_matches':len(MARKER.findall(text)),'remaining_marker_matches':len(MARKER.findall(view))}


def signature_status(code):
    try:
        _,_,count=grouping.signatures(code);return {'status':'computed','token_count':count}
    except ValueError as error:return {'status':'lexical_error: '+str(error),'token_count':None}


def create_views(rows, *, include_if_blocks=False):
    views=[];audit=[];events=[]
    for row in sorted(rows,key=lambda r:r['record_id']):
        parent=row['model_code'];view,details=filter_view(parent,include_if_blocks=include_if_blocks)
        if len(view)!=len(parent):raise ValueError('Mask changed character offsets')
        ranges=details['events']
        for event in ranges:
            if text_hash(parent[event['start']:event['end']])!=event['parent_span_sha256']:raise ValueError('Span identity mismatch')
            events.append(dict(event,record_id=row['record_id']))
        changed={position for event in ranges for position in range(event['start'],event['end'])}
        if any(a!=b and position not in changed for position,(a,b) in enumerate(zip(parent,view))):raise ValueError('Unexpected out-of-span change')
        if [i for i,c in enumerate(parent) if c in '\r\n']!=[i for i,c in enumerate(view) if c in '\r\n']:raise ValueError('Line endings changed')
        before_status=signature_status(parent);after_status=signature_status(view)
        flags=[]
        if details['remaining_marker_matches']:flags.append('residual_declared_audit_marker_requires_review')
        if details['scanner_diagnostic']:flags.append('source_scanner_diagnostic')
        if after_status['status']!='computed':flags.append('lexical_signature_unavailable')
        record={key:row.get(key) for key in ('record_id','source','source_revision','source_identity','project_id','contract_address','source_path','context_sha256','license_status')}
        record.update(model_code=view,view_protocol=VERSION_IF_BLOCK if include_if_blocks else VERSION,parent_model_code_sha256=text_hash(parent),model_code_sha256=text_hash(view),
                      parent_candidate_raw_sha256=row.get('raw_sha256'),pattern_labels=row['pattern_labels'],representation_issue_flags=flags,representation_review_required=bool(flags),
                      label_observation_mask={category:False for category in row['pattern_labels']},human_review_status='unreviewed',training_eligible=False,split=None)
        views.append(record)
        audit.append({'record_id':row['record_id'],'changed':view!=parent,'event_count':len(ranges),
                      **{key:value for key,value in details.items() if key!='events'},
                      'before_signature':before_status,'after_signature':after_status,
                      'parent_model_code_sha256':text_hash(parent),'model_code_sha256':text_hash(view)})
    return views,audit,events


def prepare(pilot,baseline,recovery,output, *, include_if_blocks=False, prior_view=None):
    pilot,baseline,recovery,output=[p.resolve() for p in (pilot,baseline,recovery,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or any(output==root or output.is_relative_to(root) for root in (baseline,recovery)):
        raise ValueError('Preserve originals and input releases; use new versioned output')
    prior_view=prior_view.resolve() if prior_view is not None else None
    if include_if_blocks and prior_view is None:raise ValueError('V2 requires an immutable prior v1 view')
    if prior_view is not None and (output==prior_view or output.is_relative_to(prior_view)):raise ValueError('Preserve prior view release')
    delivered=verify_delivery(pilot);baseline_id=sha256(safe_file(baseline,'manifest.json'));recovery_id=sha256(safe_file(recovery,'recovery_manifest.json'))
    baseline_manifest=verify_artifacts(baseline,'manifest.json','artifact_sha256');recovery_manifest=verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    delivery_id=sha256(pilot/'DELIVERY_SHA256.json')
    if baseline_manifest['delivery_manifest_sha256']!=delivery_id or recovery_manifest['delivery_manifest_sha256']!=delivery_id or baseline_manifest['recovery_manifest_sha256']!=recovery_id:raise ValueError('Input releases use different ancestor identities')
    if baseline_manifest['generator_sha256']!=sha256(Path(grouping.__file__)) or baseline_manifest['comment_masker_sha256']!=sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)):raise ValueError('Frozen grouping implementation changed')
    rows=list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'));inspect_staging_rows(rows)
    if len(rows)>5000 or sum(len(row['model_code']) for row in rows)>20_000_000:raise ValueError('View inventory exceeds bounded budget')
    by_id={row['record_id']:row for row in rows};prior=list(jsonl_rows(baseline/'candidate_groups.jsonl'))
    if len(prior)!=len(rows) or {row['record_id'] for row in prior}!=set(by_id):raise ValueError('Baseline grouping membership changed')
    for row in prior:
        if row['exact_input_sha256']!=text_hash(by_id[row['record_id']]['model_code']):raise ValueError('Baseline groups do not bind current parent views')
    seeds={row['record_id'] for row in prior if row['direct_verified_context_corpus_overlap']}
    links=list(jsonl_rows(recovery/'candidate_context_links.jsonl'));recovered={row['record_id']:row for row in links if row.get('context_file')}
    if seeds!=set(recovered):raise ValueError('Baseline overlap seeds differ from recovered evidence')
    cache={}
    for identity,link in recovered.items():
        digest=by_id[identity].get('context_sha256');path=safe_file(recovery,link['context_file'])
        if path not in cache:cache[path]=sha256(path)
        if link['pinned_context_sha256']!=digest or cache[path]!=digest:raise ValueError('Overlap seed evidence mismatch')
    prior_identity=None;prior_comparison=None
    if prior_view is not None:
        prior_identity=sha256(safe_file(prior_view,'manifest.json'));prior_manifest=verify_artifacts(prior_view,'manifest.json','artifact_sha256')
        if prior_manifest['protocol']!=VERSION or prior_manifest['delivery_manifest_sha256']!=delivery_id or prior_manifest['baseline_group_manifest_sha256']!=baseline_id or prior_manifest['recovery_manifest_sha256']!=recovery_id:raise ValueError('Prior view ancestry/protocol mismatch')
        prior_rows=list(jsonl_rows(prior_view/'function_model_views.jsonl'));prior_by={row['record_id']:row for row in prior_rows}
        if len(prior_rows)!=len(rows) or set(prior_by)!=set(by_id):raise ValueError('Prior view inventory mismatch')
        expected_v1=create_views(rows)[0]
        for row in expected_v1:
            before=prior_by[row['record_id']]
            if before['model_code']!=row['model_code'] or before['parent_model_code_sha256']!=row['parent_model_code_sha256'] or before['view_protocol']!=VERSION:raise ValueError('Default v1 behavior or input binding changed')
            if any(value is not None for value in before['pattern_labels'].values()) or any(before['label_observation_mask'].values()) or before.get('human_review_status')!='unreviewed' or before['training_eligible'] is not False or before['split'] is not None:raise ValueError('Annotated prior views require explicit label migration')
    views,audit,events=create_views(rows,include_if_blocks=include_if_blocks)
    if prior_view is not None:
        changed_ids=[]
        for row,check in zip(views,audit):
            before=prior_by[row['record_id']];changed=row['model_code']!=before['model_code']
            if changed and 'residual_declared_audit_marker_requires_review' not in before['representation_issue_flags']:raise ValueError('V2 changed a non-residual prior view')
            row.update(prior_view_protocol=VERSION,prior_view_model_code_sha256=text_hash(before['model_code']))
            check.update(changed_vs_prior_view=changed,prior_view_model_code_sha256=text_hash(before['model_code']))
            if changed:changed_ids.append(row['record_id'])
        prior_comparison={'prior_protocol':VERSION,'changed_records':len(changed_ids),'changed_record_ids':changed_ids,'prior_remaining_marker_matches':sum(len(MARKER.findall(row['model_code'])) for row in prior_rows),'all_default_v1_views_reproduced':True}
        if sha256(prior_view/'manifest.json')!=prior_identity:raise ValueError('Prior view manifest changed during preparation')
        verify_artifacts(prior_view,'manifest.json','artifact_sha256')
    members,edges,components,group_summary=grouping.build_groups(views,seeds)
    if verify_delivery(pilot)!=delivered or sha256(baseline/'manifest.json')!=baseline_id or sha256(recovery/'recovery_manifest.json')!=recovery_id:raise ValueError('Input identities changed during processing')
    verify_artifacts(baseline,'manifest.json','artifact_sha256');verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    summary={'protocol':VERSION_IF_BLOCK if include_if_blocks else VERSION,'prior_view_comparison':prior_comparison,'records':len(rows),'changed_records':sum(row['changed'] for row in audit),'mask_events':len(events),
             'marker_matches_masked':sum(row['marker_matches_masked'] for row in events),
             'parent_marker_matches':sum(row['parent_marker_matches'] for row in audit),'remaining_marker_matches':sum(row['remaining_marker_matches'] for row in audit),
             'unhandled_outside_markers':sum(len(row['unhandled_outside_markers']) for row in audit),
             'retained_literal_markers':sum(row['markers_in_literals_retained'] for row in audit),'retained_comment_markers':sum(row['markers_in_comments_retained'] for row in audit),
             'representation_review_required_records':sum(row['representation_review_required'] for row in views),
             'scanner_diagnostic_counts':dict(Counter(row['scanner_diagnostic'] for row in audit if row['scanner_diagnostic'])),
             'before_signature_status_counts':dict(Counter(row['before_signature']['status'] for row in audit)),
             'after_signature_status_counts':dict(Counter(row['after_signature']['status'] for row in audit)),
             'independent_reviews_completed':0,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0,
             'baseline_group_summary_sha256':sha256(baseline/'summary.json'),'recomputed_group_summary':group_summary,
             'limitations':['Only the declared inline audit-issue form is filtered; other annotation cues may remain.',
                           'Filtered audit prose is source-claim evidence, not a verified vulnerability label. Removing it does not recover audited original contract bytes or certify compiler/semantic validity.',
                           'All source, context, project/license, grouping and independent-review limitations remain. All-false label observation masks mean unknown, not negative.',
                           'The new view and signatures cannot be silently mixed with old model inputs, producer token hashes, checkpoints or grouping artifacts. No model training or accuracy gain occurred.']}
    output.mkdir(parents=True,exist_ok=False);(output/'groups').mkdir()
    write_jsonl(output/'function_model_views.jsonl',views);write_jsonl(output/'view_audit.jsonl',audit);write_jsonl(output/'mask_events.jsonl',events)
    for name,value in [('candidate_groups.jsonl',members),('constraint_edges.jsonl',edges),('components.jsonl',components)]:write_jsonl(output/'groups'/name,value)
    (output/'groups/summary.json').write_text(json.dumps(group_summary,indent=2,sort_keys=True)+'\n');(output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION_IF_BLOCK if include_if_blocks else VERSION,'include_if_block_tails':include_if_blocks,'prior_view_manifest_sha256':prior_identity,'generator_sha256':sha256(Path(__file__)),'grouping_generator_sha256':sha256(Path(grouping.__file__)),'grouping_comment_masker_sha256':sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)),
              'delivery_manifest_sha256':delivery_id,'baseline_group_manifest_sha256':baseline_id,'recovery_manifest_sha256':recovery_id,
              'artifact_sha256':{str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','baseline-groups-dir','recovery-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--include-if-block-tails',action='store_true');parser.add_argument('--prior-view-dir',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.pilot_dir,args.baseline_groups_dir,args.recovery_dir,args.output_dir,include_if_blocks=args.include_if_block_tails,prior_view=args.prior_view_dir),indent=2))

if __name__=='__main__':main()
