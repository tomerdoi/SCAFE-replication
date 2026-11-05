#!/usr/bin/env python3
"""Measure retained role-pair input differences, never repair correctness or labels."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import re
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import verify_artifacts
from prepare_function_review import text_hash, write_jsonl
from adapt_coinfabrik_source_findings import SOURCE, source_finding_example

VERSION='coinfabrik-retained-role-pair-input-evidence-v1'
# Literal contents, identifiers and numbers are preserved. This is not an AST.
TOKEN=re.compile(r'''(?P<literal>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?P<number>0[xX][0-9a-fA-F_]+|(?:[0-9][0-9_]*(?:\.[0-9_]+)?|\.[0-9_]+)(?:[eE][+-]?[0-9][0-9_]*)?)|(?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)|(?P<operator>>>=|<<=|\*\*=|=>|:=|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|&&|\|\||<<|>>|\*\*)|(?P<other>\S)''',re.DOTALL)

def literal_tokens(code):
    if not isinstance(code,str) or not code or len(code)>1_000_000:raise ValueError('Invalid bounded input')
    tokens=[(m.lastgroup,m.group()) for m in TOKEN.finditer(code)]
    if not tokens or any(kind=='other' and value in {'"',"'"} for kind,value in tokens):raise ValueError('Unclosed literal or empty token input')
    return tokens

def token_hash(code):
    return text_hash(json.dumps(literal_tokens(code),ensure_ascii=False,separators=(',',':')))

def pair_key(row):
    parts=row['source_path'].split('/');role=row['version_role'];project=row['project_id'];symbol=row['function_identifier']
    if (len(parts)<3 or any(p in {'','.','..'} for p in parts) or parts[0]!=project
        or parts[1]!=role or role not in {'vulnerable','patched'} or not isinstance(symbol,str) or not symbol
        or PurePosixPath(row['source_path']).is_absolute()):raise ValueError('Invalid project/role/source path/symbol')
    return (project,'/'.join([parts[0],*parts[2:]]),symbol)

def pair_frame(rows):
    if not rows or len(rows)>5000 or len({r['record_id'] for r in rows})!=len(rows):raise ValueError('Expected bounded unique records')
    if any(r['source']!=SOURCE for r in rows) or len({r['source_revision'] for r in rows})!=1:raise ValueError('One pinned CoinFabrik revision required')
    buckets=defaultdict(lambda:defaultdict(list))
    for row in rows:buckets[pair_key(row)][row['version_role']].append(row['record_id'])
    pairs=[];unpaired=[]
    for key,roles in sorted(buckets.items()):
        v=sorted(roles['vulnerable']);p=sorted(roles['patched']);base={'project':key[0],'role_stripped_path':key[1],'qualified_identifier':key[2]}
        if len(v)==len(p)==1:pairs.append({**base,'vulnerable_record_id':v[0],'patched_record_id':p[0]})
        else:unpaired.append({**base,'vulnerable_record_ids':v,'patched_record_ids':p,'status':'ambiguous_endpoint' if len(v)>1 or len(p)>1 else 'missing_counterpart'})
    return pairs,unpaired

def unique_rows(path):
    rows=list(jsonl_rows(path))
    if len(rows)>5000 or len({r['record_id'] for r in rows})!=len(rows):raise ValueError('Duplicate or excessive identity frame')
    return {r['record_id']:r for r in rows}

def prepare(pilot,views,contexts,source_layer,output):
    pilot,views,contexts,source_layer,output=[Path(p).resolve() for p in (pilot,views,contexts,source_layer,output)]
    if output.exists():raise FileExistsError(output)
    for root in (pilot,views,contexts,source_layer):
        if root==output or root.is_relative_to(output):raise ValueError('Output cannot contain inputs')
    if output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives'):raise ValueError('Preserve delivered originals')
    if any(output.is_relative_to(root) for root in (views,contexts,source_layer)):raise ValueError('Preserve parent derivatives')
    delivered=verify_delivery(pilot);parents={}
    for key,root,name in [('views',views,'manifest.json'),('contexts',contexts,'recovery_manifest.json'),('source_layer',source_layer,'manifest.json')]:
        manifest=verify_artifacts(root,name,'artifact_sha256')
        if manifest['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Delivery lineage differs')
        parents[key]=(root,name,sha256(root/name),manifest)
    if parents['contexts'][3]['model_view_manifest_sha256']!=parents['views'][2] or parents['source_layer'][3]['parent_manifest_sha256']['views']!=parents['views'][2] or parents['source_layer'][3]['parent_manifest_sha256']['contexts']!=parents['contexts'][2]:raise ValueError('Current parent lineage differs')
    candidates=[r for r in jsonl_rows(pilot/'public_output/function_candidates.jsonl') if r['source']==SOURCE]
    models=unique_rows(views/'function_model_views.jsonl');groups=unique_rows(contexts/'groups/candidate_groups.jsonl');targets=unique_rows(source_layer/'source_proxy_targets.jsonl');source_models=unique_rows(source_layer/'model_inputs.jsonl')
    ids={r['record_id'] for r in candidates}
    if ids!=set(targets) or ids!=set(source_models):raise ValueError('Source layer frame differs')
    features={}
    if sum(len(models[i]['model_code']) for i in ids)>20_000_000:raise ValueError('20M-character budget exceeded')
    for row in candidates:
        i=row['record_id'];m=models[i];g=groups[i];t=targets[i]
        source_finding_example(source_models[i],t)
        if (text_hash(m['model_code'])!=m['model_code_sha256'] or m['model_code_sha256']!=source_models[i]['model_code_sha256'] or g['exact_input_sha256']!=m['model_code_sha256'] or text_hash(row['model_code'])!=m['parent_model_code_sha256'] or t['source_revision']!=row['source_revision'] or t['source_path']!=row['source_path'] or t['function_identifier']!=row['function_identifier'] or t['source_project']!=row['project_id'] or t['version_role']!=row['version_role']):raise ValueError('Candidate/current-view/group/source lineage differs')
        if any(x is not None for x in row['pattern_labels'].values()) or row['training_eligible'] is not False or row['split'] is not None:raise ValueError('Annotated parent needs reviewed migration')
        features[i]={'input_sha256':m['model_code_sha256'],'literal_token_sha256':token_hash(m['model_code'])}
    pairs,unpaired=pair_frame(candidates)
    for pair in pairs:
        a,b=pair['vulnerable_record_id'],pair['patched_record_id'];ga,gb=groups[a],groups[b]
        pair.update(vulnerable_input=features[a],patched_input=features[b],byte_identical=features[a]['input_sha256']==features[b]['input_sha256'],literal_token_identical=features[a]['literal_token_sha256']==features[b]['literal_token_sha256'],vulnerable_source_positive_categories=[c for c,m in targets[a]['source_proxy_observation_mask'].items() if m],source_finding_ids=sorted({f['finding_id'] for f in targets[a]['finding_support']}),evidence_component_same=ga['evidence_component']==gb['evidence_component'],strict_component_same=ga['strict_template_component']==gb['strict_template_component'],either_existing_corpus_overlap=ga['direct_verified_context_corpus_overlap'] or gb['direct_verified_context_corpus_overlap'],either_conservative_quarantine=ga['evidence_conservative_quarantine'] or gb['evidence_conservative_quarantine'],repair_verified=False,labels_assigned=False)
        if not pair['evidence_component_same'] or not pair['strict_component_same']:raise ValueError('Existing pair/project grouping insufficient; do not freeze splits')
    positive=[p for p in pairs if p['vulnerable_source_positive_categories']]
    summary={'protocol':VERSION,'source_revision':candidates[0]['source_revision'],'records':len(candidates),'role_counts':dict(Counter(r['version_role'] for r in candidates)),'unique_role_pairs':len(pairs),'paired_records':2*len(pairs),'unpaired_records':sum(len(r['vulnerable_record_ids'])+len(r['patched_record_ids']) for r in unpaired),'unpaired_buckets':len(unpaired),'unpaired_status_counts':dict(Counter(r['status'] for r in unpaired)),'byte_identical_pairs':sum(p['byte_identical'] for p in pairs),'literal_token_identical_pairs':sum(p['literal_token_identical'] for p in pairs),'source_positive_pairs':len(positive),'source_positive_byte_identical_pairs':sum(p['byte_identical'] for p in positive),'source_positive_literal_token_identical_pairs':sum(p['literal_token_identical'] for p in positive),'all_pairs_already_grouped':all(p['evidence_component_same'] and p['strict_component_same'] for p in pairs),'verified_repairs':0,'new_labels':0,'new_eligible_records':0,'assigned_splits':0,'training_or_inference':False}
    if verify_delivery(pilot)!=delivered:raise ValueError('Original delivery changed')
    for root,name,digest,manifest in parents.values():
        if sha256(root/name)!=digest:raise ValueError('Parent changed')
        verify_artifacts(root,name,'artifact_sha256')
    output.mkdir(parents=True,exist_ok=False)
    write_jsonl(output/'pair_input_evidence.jsonl',pairs);write_jsonl(output/'unpaired_candidates.jsonl',unpaired)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    card={'protocol':VERSION,'pairing':'Exactly one vulnerable and one patched candidate for identical project, role-stripped relative path and qualified identifier at one recorded pin; overloads/duplicates are ambiguous, not guessed.','token_measure':'Literal-preserving frozen heuristic regex sequence on current annotation-v2 views; whitespace outside tokens ignored, identifiers/literals/numbers/operators unchanged; not compiler parsing or semantic equivalence.','interpretation':'Patched-role membership and changed function text are not verified repair or category negatives. Identical retained inputs cannot distinguish role-derived opposing targets with a deterministic code-only classifier. Identical function inputs do not rule out contextual/dependency repairs.','grouping':'Both endpoints must remain grouped. Existing project/clone/overlap quarantine retained; no cohort or split assigned.','truth':'No labels, independent reviews, training/evaluation eligibility or model results added; source-positive claims keep their native separate target meanings.','payload':'Hashes and metadata only. Original code, finding text and prior derivatives unchanged.'}
    (output/'data_card.json').write_text(json.dumps(card,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'source_revision':summary['source_revision'],'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,'parent_manifest_sha256':{k:v[2] for k,v in parents.items()},'generator_sha256':sha256(Path(__file__)),'helper_sha256':{Path(f.__code__.co_filename).name:sha256(Path(f.__code__.co_filename)) for f in (source_finding_example,verify_delivery,verify_artifacts,write_jsonl)},'token_regex':TOKEN.pattern,'artifact_sha256':{f.name:sha256(f) for f in sorted(output.iterdir()) if f.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','view-dir','context-release-dir','source-layer-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.pilot_dir,a.view_dir,a.context_release_dir,a.source_layer_dir,a.output_dir),indent=2))

if __name__=='__main__':main()
