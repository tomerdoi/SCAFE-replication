#!/usr/bin/env python3
"""Create lineage-bound views masking only post-statement inline audit-issue prose."""
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
MARKER=re.compile(r'<-[ \t]*@?audit-issue\b',re.IGNORECASE)
MAX_CHARS=1_000_000


def next_line_end(text,start):
    return min((p for mark in ('\r','\n') if (p:=text.find(mark,start))>=0),default=len(text))


def filter_view(text):
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
            if prefix.endswith(';') and last_semicolon==semicolon_position:
                end=next_line_end(text,i)
                raw=text[i:end]
                events.append({'start':i,'end':end,'reason':'outside_literal_comment_post_statement_audit_issue_tail',
                               'parent_span_sha256':text_hash(raw),'marker_matches_masked':len(MARKER.findall(raw))})
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


def create_views(rows):
    views=[];audit=[];events=[]
    for row in sorted(rows,key=lambda r:r['record_id']):
        parent=row['model_code'];view,details=filter_view(parent)
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
        record.update(model_code=view,view_protocol=VERSION,parent_model_code_sha256=text_hash(parent),model_code_sha256=text_hash(view),
                      parent_candidate_raw_sha256=row.get('raw_sha256'),pattern_labels=row['pattern_labels'],representation_issue_flags=flags,representation_review_required=bool(flags),
                      label_observation_mask={category:False for category in row['pattern_labels']},human_review_status='unreviewed',training_eligible=False,split=None)
        views.append(record)
        audit.append({'record_id':row['record_id'],'changed':view!=parent,'event_count':len(ranges),
                      **{key:value for key,value in details.items() if key!='events'},
                      'before_signature':before_status,'after_signature':after_status,
                      'parent_model_code_sha256':text_hash(parent),'model_code_sha256':text_hash(view)})
    return views,audit,events


def prepare(pilot,baseline,recovery,output):
    pilot,baseline,recovery,output=[p.resolve() for p in (pilot,baseline,recovery,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or any(output==root or output.is_relative_to(root) for root in (baseline,recovery)):
        raise ValueError('Preserve originals and input releases; use new versioned output')
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
    views,audit,events=create_views(rows);members,edges,components,group_summary=grouping.build_groups(views,seeds)
    if verify_delivery(pilot)!=delivered or sha256(baseline/'manifest.json')!=baseline_id or sha256(recovery/'recovery_manifest.json')!=recovery_id:raise ValueError('Input identities changed during processing')
    verify_artifacts(baseline,'manifest.json','artifact_sha256');verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    summary={'protocol':VERSION,'records':len(rows),'changed_records':sum(row['changed'] for row in audit),'mask_events':len(events),
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
                           'Post-statement audit prose is source-claim evidence, not a verified vulnerability label. Removing it does not recover audited original contract bytes or certify compiler/semantic validity.',
                           'All source, context, project/license, grouping and independent-review limitations remain. All-false label observation masks mean unknown, not negative.',
                           'The new view and signatures cannot be silently mixed with old model inputs, producer token hashes, checkpoints or grouping artifacts. No model training or accuracy gain occurred.']}
    output.mkdir(parents=True,exist_ok=False);(output/'groups').mkdir()
    write_jsonl(output/'function_model_views.jsonl',views);write_jsonl(output/'view_audit.jsonl',audit);write_jsonl(output/'mask_events.jsonl',events)
    for name,value in [('candidate_groups.jsonl',members),('constraint_edges.jsonl',edges),('components.jsonl',components)]:write_jsonl(output/'groups'/name,value)
    (output/'groups/summary.json').write_text(json.dumps(group_summary,indent=2,sort_keys=True)+'\n');(output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'grouping_generator_sha256':sha256(Path(grouping.__file__)),'grouping_comment_masker_sha256':sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)),
              'delivery_manifest_sha256':delivery_id,'baseline_group_manifest_sha256':baseline_id,'recovery_manifest_sha256':recovery_id,
              'artifact_sha256':{str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','baseline-groups-dir','recovery-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.pilot_dir,args.baseline_groups_dir,args.recovery_dir,args.output_dir),indent=2))

if __name__=='__main__':main()
