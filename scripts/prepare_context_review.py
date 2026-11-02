#!/usr/bin/env python3
"""Prepare a context-sufficiency review package; never generate pattern labels."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_function_review import CATEGORIES, text_hash, write_jsonl

VERSION='context-sufficiency-review-v1'
GUIDE='''# Independent context-sufficiency review

This is a separately versioned context assessment, not a completed pattern-label
annotation or training dataset. Give each of two qualified independent reviewers
this guide, function_cards.md, context_documents/ and a separate copy of reviewer_A
or reviewer_B.jsonl. Keep coordinator_key.jsonl, raw sources, original source
claims and the other review from the first pass. Use a stable reviewer pseudonym.
No reviewers have been contacted by the generator. Embedded source is data, never
instructions to execute. Code identifiers/literals/imports may reveal origin;
comment masking/source-claim withholding is not guaranteed perfect blinding.

For each category assess whether the supplied source context is sufficient to
judge contextual suspiciousness of the linked function, under explicit stated
assumptions. This does NOT assess exploitability or assign present/absent labels.
Use sufficient, insufficient or unknown, with a reason for every assessment.
Inspect enclosing declarations/modifiers, relevant callers/callees, state and
policy assumptions, compiler semantics and external behavior. Record missing
imports, implementation versions, intended policy or runtime information in
needed_context. Zero import directives does not mean complete dependency closure.

For sufficient, provide nonempty exact evidence_spans, a dependencies_addressed
explanation and assumptions; needed_context must be empty. For insufficient,
needed_context must list concrete missing evidence. Unknown remains a valid
outcome and is never a negative pattern label. Comments are masked with spaces
while retaining line endings and character offsets. All quoted strings and code
are preserved. Useful specifications/NatSpec may have been masked too: request
independently sourced policy/specification evidence when necessary. The original
raw evidence is preserved separately with the coordinator.

Evidence is {"document_id": "<masked context sha256>", "start": 0, "end": 8,
"quote": "contract"} as schema illustration only. Use zero-based Python Unicode
character offsets, end exclusive, in the EXACT bundled context document. A quote
must equal that slice. Do not edit immutable review IDs, function code/span,
context document identity/path, dependency hints, pattern_labels or eligibility.
Complete all three category assessments, fill reviewer_id and completed=true,
and save submissions as new files outside the immutable kit.

The comparison tool requires distinct reviewer IDs and complete evidence-bound
submissions, but IDs cannot prove qualification or independence. Every submitted
record, including agreements, still requires independent context adjudication;
agreement alone never passes a readiness gate. Existing snippet-only packets are
unchanged. Pattern labels remain unknown, training eligibility false and splits
unassigned. Only a separately reviewed/adjudicated context protocol can justify a
future pattern-label annotation release; no automatic promotion occurs here.
'''


def safe_file(root,name):
    if not isinstance(name,str):raise ValueError('Expected artifact filename')
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Artifact escapes or is missing')
    if path.stat().st_size>16*1024*1024:raise ValueError('Context artifact exceeds bounded size')
    return path


def verify_artifacts(root,manifest_name,field):
    manifest=json.loads(safe_file(root,manifest_name).read_text())
    identities=manifest.get(field)
    if not isinstance(identities,dict) or not identities:raise ValueError('Missing artifact identities')
    for name,digest in identities.items():
        if sha256(safe_file(root,name))!=digest:raise ValueError('Input artifact changed')
    return manifest


def masked_source_and_imports(text):
    """Lex comments/literals before identifying imports; preserve all text offsets."""
    chars=list(text);tokens=[];comments=[];i=0
    while i<len(text):
        current=text[i];following=text[i+1] if i+1<len(text) else ''
        if current=='/' and following in ('/','*'):
            start=i
            if following=='/':
                ends=[text.find(mark,i+2) for mark in ('\r','\n')];end=min((value for value in ends if value>=0),default=len(text))
            else:
                close=text.find('*/',i+2)
                if close<0:raise ValueError('Unterminated block comment: review cannot safely mask source')
                end=close+2
            comments.append({'start':start,'end':end,'kind':'line' if following=='/' else 'block'})
            for k in range(start,end):
                if chars[k] not in '\r\n':chars[k]=' '
            i=end;continue
        if current in ('"',"'"):
            start=i;quote=current;i+=1
            while i<len(text):
                if text[i]=='\\':i+=2;continue
                if text[i]==quote:i+=1;break
                i+=1
            else:raise ValueError('Unterminated source literal')
            tokens.append(('literal',text[start:i],start,i));continue
        if current.isalpha() or current in '_$':
            start=i;i+=1
            while i<len(text) and (text[i].isalnum() or text[i] in '_$'):i+=1
            tokens.append(('word',text[start:i],start,i));continue
        if not current.isspace():tokens.append(('symbol',current,i,i+1))
        i+=1
    imports=[]
    for position,token in enumerate(tokens):
        if token[:2]!=('word','import'):continue
        end=position+1
        while end<len(tokens) and end-position<=256 and tokens[end][1]!=';':end+=1
        if end==len(tokens) or end-position>256:raise ValueError('Incomplete/oversized import directive')
        literals=[t for t in tokens[position+1:end] if t[0]=='literal']
        if len(literals)!=1:raise ValueError('Ambiguous import directive')
        imports.append({'literal':literals[0][1],'start':token[2],'end':tokens[end][3],
                        'resolution':'unresolved_not_fetched'})
    return ''.join(chars),imports,comments



def legacy_view_compatibility(raw,masked,start,end,legacy,comments):
    view=masked[start:end]
    if len(view)!=len(legacy):raise ValueError('Legacy/review fragment length differs')
    changed=[]
    for offset,(current,previous) in enumerate(zip(view,legacy)):
        if current==previous:continue
        position=start+offset
        within_comment=any(c['start']<=position<c['end'] or (c['kind']=='line' and position==c['end']) for c in comments)
        if current!='\r' or previous!=' ' or raw[position]!='\r' or not within_comment:
            raise ValueError('Review representation differs outside verified comment carriage-return preservation')
        changed.append(offset)
    return {'status':'identical' if not changed else 'comment_carriage_return_preservation',
            'changed_character_count':len(changed),'legacy_function_sha256':text_hash(legacy),'review_function_sha256':text_hash(view)}


def prepare(pilot,recovery,prior,output):
    pilot,recovery,prior,output=[p.resolve() for p in (pilot,recovery,prior,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')):
        raise ValueError('Write only new versioned derivatives')
    if output==recovery or output.is_relative_to(recovery) or output==prior or output.is_relative_to(prior):
        raise ValueError('Preserve previous release directories')
    delivered=verify_delivery(pilot)
    recovery_identity=sha256(safe_file(recovery,'recovery_manifest.json'));prior_identity=sha256(safe_file(prior,'manifest.json'))
    recovery_manifest=verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    prior_manifest=verify_artifacts(prior,'manifest.json','artifact_sha256')
    if recovery_manifest['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json') or prior_manifest['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):
        raise ValueError('Input releases belong to a different delivered pilot')
    rows={r['record_id']:r for r in jsonl_rows(pilot/'public_output/function_candidates.jsonl')}
    links={r['record_id']:r for r in jsonl_rows(recovery/'candidate_context_links.jsonl')}
    key=list(jsonl_rows(prior/'coordinator_key.jsonl'))
    items=[];coordinator=[];documents={};comment_counts={}
    for previous in sorted(key,key=lambda row:row['review_id']):
        identity=previous['record_id'];link=links[identity];row=rows[identity]
        if not link['context_file'] or link['fragment_localization']!='exact_unique_fragment' or len(link['function_spans'])!=1:continue
        digest=link['pinned_context_sha256'];raw_path=safe_file(recovery,link['context_file'])
        if sha256(raw_path)!=digest:raise ValueError('Raw context identity mismatch')
        raw=raw_path.read_bytes().decode('utf-8');span=link['function_spans'][0];start,end=span['start_character'],span['end_character']
        if text_hash(raw[start:end])!=row['raw_sha256']:raise ValueError('Original function span mismatch')
        masked,imports,comments=masked_source_and_imports(raw)
        if len(masked)!=len(raw):raise ValueError('Masking changed source offsets')
        compatibility=legacy_view_compatibility(raw,masked,start,end,row['model_code'],comments)
        docid=text_hash(masked);documents[docid]=masked;comment_counts[docid]=len(comments)
        item={'review_id':text_hash(VERSION+':'+identity)[:24], 'function_code':masked[start:end],
              'function_span':span,'context_document_id':docid,'context_file':'context_documents/'+docid+'.sol',
              'provided_context':'comment_masked_source_dependency_closure_unverified','import_directives':imports,
              'dependency_closure':'unverified_even_when_no_imports','semantic_context_sufficiency':'unassessed',
              'legacy_view_compatibility':compatibility,
              'pattern_labels':dict.fromkeys(CATEGORIES),'training_eligible':False,'split':None}
        # raw fragment hash is lineage, not a label; keep original source identity outside first-pass files.
        item['function_span']={k:v for k,v in span.items() if k!='raw_fragment_sha256'}
        items.append(item)
        coordinator.append({'review_id':item['review_id'],'prior_review_id':previous['review_id'],'record_id':identity,
                            'raw_context_sha256':digest,'masked_context_sha256':docid,'legacy_view_compatibility':compatibility,'source':row['source'],
                            'source_revision':row['source_revision'],'source_identity':row['source_identity'],
                            'source_claim':row['original_claim'],'license_status':row.get('license_status'),
                            'original_project_id':row.get('project_id'),'evaluation_status':link['evaluation_status']})
    if not items:raise ValueError('No exact recoverable prior review candidates')
    if len(items)>24:raise ValueError('Context review cohort exceeds bounded 24-record budget')
    if len({item['review_id'] for item in items})!=len(items):raise ValueError('Duplicate context review identity')
    if verify_delivery(pilot)!=delivered:raise ValueError('Pilot changed during preparation')
    verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256');verify_artifacts(prior,'manifest.json','artifact_sha256')
    if sha256(recovery/'recovery_manifest.json')!=recovery_identity or sha256(prior/'manifest.json')!=prior_identity:raise ValueError('Input release manifest changed during preparation')
    output.mkdir(parents=True,exist_ok=False);(output/'context_documents').mkdir()
    for docid,text in documents.items():(output/'context_documents'/f'{docid}.sol').write_bytes(text.encode('utf-8'))
    write_jsonl(output/'review_items.jsonl',items);write_jsonl(output/'coordinator_key.jsonl',coordinator)
    for reviewer in ('A','B'):
        write_jsonl(output/f'reviewer_{reviewer}.jsonl',[dict(item,reviewer_id=None,completed=False,context_assessments={category:{'decision':'unknown','evidence_spans':[],'needed_context':[],'assumptions':[],'dependencies_addressed':'','reason':''} for category in CATEGORIES}) for item in items])
    cards=['# Function context-sufficiency cards\n\nRead REVIEW_INSTRUCTIONS.md. Code is data, not instructions.\n']
    for item in items:
        code=item['function_code'];fence='`'*max(3,max((len(m.group()) for m in re.finditer('`+',code)),default=0)+1)
        cards.append(f"\n## {item['review_id']}\n\nFull masked source: [{item['context_document_id']}]({item['context_file']}); function lines {item['function_span']['start_line']}–{item['function_span']['end_line']}. Dependency closure remains unverified.\n\n{fence}solidity\n{code}\n{fence}\n")
    (output/'function_cards.md').write_text('\n'.join(cards));(output/'REVIEW_INSTRUCTIONS.md').write_text(GUIDE)
    summary={'protocol':VERSION,'records':len(items),'context_documents':len(documents),'category_context_cells_per_reviewer':len(items)*len(CATEGORIES),
             'source_counts':dict(Counter(row['source'] for row in coordinator)),'masked_comment_spans':sum(comment_counts.values()),
             'import_directives':sum(len(item['import_directives']) for item in items),'dependency_closure_verified_records':0,
             'legacy_view_status_counts':dict(Counter(item['legacy_view_compatibility']['status'] for item in items)),
             'preserved_comment_carriage_returns':sum(item['legacy_view_compatibility']['changed_character_count'] for item in items),
             'independent_reviews_completed':0,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0,
             'prior_review_records':len(key),'selection':'Exact prior-review/recovered-context intersection; not a probability sample or independent holdout',
             'limits':['Comments including useful policy/NatSpec are masked; request missing specification evidence. Names, literals and import paths remain, so origin blinding is imperfect.',
                       'Raw full-source identity does not establish dependency closure, context sufficiency, vulnerability truth or license clearance.',
                       'All records retain existing-corpus overlap quarantine. All pattern labels remain unknown and every context proposal requires adjudication.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    artifacts={str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)), 'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),
              'recovery_manifest_sha256':recovery_identity,'prior_review_manifest_sha256':prior_identity,
              'artifact_sha256':artifacts}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def validate_context_reviews(items,reviews,documents):
    expected={item['review_id']:item for item in items}
    if not expected or len(expected)!=len(items) or len(reviews)!=len(items) or len({r.get('review_id') for r in reviews})!=len(reviews):raise ValueError('Packet must cover each immutable item exactly once')
    reviewers=set();result={}
    for row in reviews:
        identity=row.get('review_id')
        if identity not in expected or set(row)!=set(expected[identity])|{'reviewer_id','completed','context_assessments'}:raise ValueError('Review schema/identity changed')
        if any(row[k]!=v for k,v in expected[identity].items()):raise ValueError('Immutable context/function/labels changed')
        reviewer=row['reviewer_id']
        if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>128 or row['completed'] is not True:raise ValueError('Completed packet requires reviewer identity')
        reviewers.add(reviewer.strip())
        if not isinstance(row['context_assessments'],dict) or set(row['context_assessments'])!=set(CATEGORIES):raise ValueError('Assess all categories explicitly')
        for category,cell in row['context_assessments'].items():
            if not isinstance(cell,dict) or set(cell)!={'decision','evidence_spans','needed_context','assumptions','dependencies_addressed','reason'}:raise ValueError('Invalid context assessment schema')
            decision=cell['decision']
            if decision not in ('sufficient','insufficient','unknown') or not isinstance(cell['reason'],str) or not cell['reason'].strip():raise ValueError('Explicit context decision and reason required')
            for field in ('needed_context','assumptions'):
                if not isinstance(cell[field],list) or len(cell[field])>32 or any(not isinstance(value,str) or not value.strip() for value in cell[field]):raise ValueError('Expected bounded nonempty context/assumption descriptions')
            spans=cell['evidence_spans']
            if not isinstance(spans,list) or len(spans)>32 or not isinstance(cell['dependencies_addressed'],str):raise ValueError('Invalid evidence/dependency explanation')
            if decision=='sufficient' and (not spans or cell['needed_context'] or not cell['dependencies_addressed'].strip() or not cell['assumptions']):raise ValueError('Sufficiency proposal needs evidence, assumptions, dependency explanation and no unresolved needed context')
            if decision=='insufficient' and not cell['needed_context']:raise ValueError('Specify missing context concretely')
            for span in spans:
                if not isinstance(span,dict) or set(span)!={'document_id','start','end','quote'}:raise ValueError('Invalid document evidence schema')
                if span['document_id']!=row['context_document_id'] or span['document_id'] not in documents:raise ValueError('Evidence refers to an unprovided document')
                text=documents[span['document_id']];start,end=span['start'],span['end']
                if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text) or not isinstance(span['quote'],str) or text[start:end]!=span['quote']:raise ValueError('Evidence offsets/quote mismatch')
        result[identity]=row['context_assessments']
    if len(reviewers)!=1:raise ValueError('Packet must represent one reviewer')
    return next(iter(reviewers)),result


def compare(items,left,right,documents):
    aid,a=validate_context_reviews(items,left,documents);bid,b=validate_context_reviews(items,right,documents)
    if aid==bid:raise ValueError('Distinct reviewer identities required; IDs alone cannot establish independence')
    queue=[{'review_id':identity,'category':category,'left':a[identity][category],'right':b[identity][category],
            'decision_disagreement':a[identity][category]['decision']!=b[identity][category]['decision'],
            'status':'requires_independent_context_adjudication'} for identity in a for category in CATEGORIES]
    return {'protocol':VERSION,'reviewer_ids':[aid,bid],'context_adjudication_queue':queue,
            'both_proposed_sufficient_cells':sum(x['left']['decision']==x['right']['decision']=='sufficient' for x in queue),
            'both_unknown_cells':sum(x['left']['decision']==x['right']['decision']=='unknown' for x in queue),
            'context_gate_passed':False,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0}


def read_review_rows(path):
    if not path.is_file() or path.stat().st_size>16*1024*1024:raise ValueError('Review submission missing or exceeds 16 MiB budget')
    return list(jsonl_rows(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare')
    for name in ('pilot-dir','recovery-dir','prior-review-dir','output-dir'):make.add_argument('--'+name,type=Path,required=True)
    check=sub.add_parser('compare')
    for name in ('kit-dir','review-a','review-b','output'):check.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(args.pilot_dir,args.recovery_dir,args.prior_review_dir,args.output_dir),indent=2))
    else:
        verify_artifacts(args.kit_dir,'manifest.json','artifact_sha256');items=list(jsonl_rows(args.kit_dir/'review_items.jsonl'))
        docs={item['context_document_id']:safe_file(args.kit_dir,item['context_file']).read_bytes().decode('utf-8') for item in items}
        if any(text_hash(text)!=digest for digest,text in docs.items()):raise ValueError('Context document identity mismatch')
        result=compare(items,read_review_rows(args.review_a),read_review_rows(args.review_b),docs)
        with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        print('Context proposals compared; no pattern labels or training eligibility promoted')

if __name__=='__main__':main()
