#!/usr/bin/env python3
"""Export provenance-bound native source supervision; never train or create gold labels."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
from audit_function_pilot_handoff import jsonl_rows,sha256,verify_delivery,inspect_staging_rows
from prepare_context_review import safe_file,verify_artifacts
from prepare_function_review import CATEGORIES,text_hash,write_jsonl
from function_real_public_origin import require_real_public_origin

VERSION='function-source-reentrancy-adapter-v1'
DEFINITION='scrubd-source-assessed-reentrancy-susceptibility-proxy-v1'
SOURCE='sujeetc/SCRUBD'
SEED=202611041000
GATES=['targeted_source_label_audit_pending','independent_evaluation_cohort_pending','frozen_grouped_split_pending','provenance_license_eligibility_pending','compatible_masked_training_protocol_pending']


def native_binary(value):
    if value is None or value=='':return None
    if type(value) is str and value in ('0','1'):return value=='1'
    if type(value) is int and value in (0,1):return bool(value)
    raise ValueError('Unsupported native binary assessment; preserve unknown/ambiguous input for review')


def masked_source_example(model,target, *, for_training=False):
    """Join only matching hashes/IDs and return code plus inseparable targets/masks."""
    if target['supervision_protocol']!=VERSION or target['target_definition_id']!=DEFINITION:raise ValueError('Explicit source-proxy schema required; not legacy eleven outputs or independent gold')
    if model['record_id']!=target['record_id'] or model['model_code_sha256']!=target['model_code_sha256'] or text_hash(model['model_code'])!=model['model_code_sha256']:raise ValueError('Input/target identity mismatch')
    if set(target['labels'])!=set(CATEGORIES) or set(target['label_observation_mask'])!=set(CATEGORIES):raise ValueError('Category schema mismatch')
    if target['independent_gold'] is not False or target['human_review_status']!='unreviewed':raise ValueError('Source-proxy adapter cannot certify independent gold')
    if for_training and (target['training_eligible'] is not True or target['training_readiness_blockers']):raise ValueError('Training readiness gates have not passed')
    if for_training:require_real_public_origin(model,target,purpose='training')
    if target['label_observation_mask']['reentrancy'] and (target['source']!=SOURCE or target['mapping_status']!='mapped_source_proxy' or not target['mapping_evidence'] or target['native_source_assessments']['RE'] is not target['labels']['reentrancy']):raise ValueError('Source target lacks matching native label/span evidence')
    values=[];masks=[]
    for category in CATEGORIES:
        label=target['labels'][category];mask=target['label_observation_mask'][category]
        if type(mask) is not bool or (mask and type(label) is not bool) or (not mask and label is not None):raise ValueError('Unknown must be null/unobserved; observed targets must be boolean')
        if category!='reentrancy' and mask:raise ValueError('No UX-to-unchecked-call or authorization mapping in v1')
        values.append(float(label) if mask else 0.0);masks.append(mask)
    # A zero placeholder with a false mask is not a negative; training requires masked loss.
    return model['model_code'],values,masks


def prepare(pilot,contexts,scopes,views,output):
    pilot,contexts,scopes,views,output=[p.resolve() for p in (pilot,contexts,scopes,views,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or any(output==p or output.is_relative_to(p) for p in (contexts,scopes,views)):raise ValueError('Preserve originals/ancestors; use new versioned output')
    if output.exists():raise FileExistsError(output)
    delivered=verify_delivery(pilot);parents={}
    helper_hashes={Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (verify_delivery,verify_artifacts,write_jsonl)}
    for name,root,filename in [('contexts',contexts,'recovery_manifest.json'),('scopes',scopes,'manifest.json'),('views',views,'manifest.json')]:
        identity=sha256(safe_file(root,filename));manifest=verify_artifacts(root,filename,'artifact_sha256')
        if manifest['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Parent belongs to another delivery')
        parents[name]=(root,filename,identity,manifest)
    if parents['scopes'][3]['context_release_manifest_sha256']!=parents['contexts'][2] or parents['contexts'][3]['model_view_manifest_sha256']!=parents['views'][2]:raise ValueError('Inconsistent context/scope/model-view ancestry')
    originals=list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'));inspect_staging_rows(originals);by_id={r['record_id']:r for r in originals}
    model_rows=list(jsonl_rows(views/'function_model_views.jsonl'));scope_rows=list(jsonl_rows(scopes/'function_scope_links.jsonl'))
    for rows in [model_rows,scope_rows]:
        if len(rows)!=len(originals) or {r['record_id'] for r in rows}!=set(by_id):raise ValueError('Parent inventory differs')
        for r in rows:
            if any(v is not None for v in r['pattern_labels'].values()) or any(r.get('label_observation_mask',{}).values()) or r['training_eligible'] is not False or r['split'] is not None or r.get('human_review_status','unreviewed')!='unreviewed':raise ValueError('Annotated/assigned parents require explicit migration')
    groups=list(jsonl_rows(contexts/'groups/candidate_groups.jsonl'))
    if len(groups)!=len(originals) or {g['record_id'] for g in groups}!=set(by_id):raise ValueError('Group inventory differs')
    group_by={g['record_id']:g for g in groups}
    model_by={r['record_id']:r for r in model_rows};scope_by={r['record_id']:r for r in scope_rows};revision=json.loads((pilot/'source_manifest.json').read_text())['sources'][SOURCE]['revision']
    native={};input_claims=defaultdict(set)
    for row in originals:
        model=model_by[row['record_id']]
        if group_by[row['record_id']]['exact_input_sha256']!=model['model_code_sha256']:raise ValueError('Group constraints refer to a different model input')
        if text_hash(model['model_code'])!=model['model_code_sha256'] or model['parent_model_code_sha256']!=text_hash(row['model_code']):raise ValueError('Model input is not current hash-bound source view')
        if row['source']!=SOURCE:continue
        if row['source_revision']!=revision:raise ValueError('Source revision differs from pinned manifest')
        metadata=row['source_metadata'];reentrancy=native_binary(metadata.get('RE'));ux=native_binary(metadata.get('UX'))
        for key,value in [('reentrancy',reentrancy),('unhandled_exceptions',ux)]:
            source_value=row['source_assessments'].get(key)
            if source_value is not None and type(source_value) is not int:raise ValueError('Unexpected native source-assessment type')
            if source_value!=(int(value) if value is not None else None):raise ValueError('Preserved source assessment contradicts original metadata')
        native[row['record_id']]={'RE':reentrancy,'UX':ux,'raw_RE':metadata.get('RE'),'raw_UX':metadata.get('UX'),'is_student':metadata.get('is_student')}
        if reentrancy is not None:input_claims[model['model_code_sha256']].add(reentrancy)
    conflicts={digest for digest,claims in input_claims.items() if len(claims)>1}
    targets=[];inputs=[];documents={}
    for identity,row in sorted(by_id.items()):
        model=model_by[identity];scope=scope_by[identity];labels=dict.fromkeys(CATEGORIES);mask=dict.fromkeys(CATEGORIES,False);evidence=None;status='not_mapped_other_source'
        claim=native.get(identity)
        if claim is not None:
            status='native_RE_unknown' if claim['RE'] is None else 'qualified_exact_source_span_pending'
            if claim['RE'] is not None and scope['scope_status']=='unique_qualified_scope_match':
                digest=row['context_sha256'];path=safe_file(contexts,scope['context_artifact'])
                if digest!=scope['pinned_context_sha256'] or sha256(path)!=digest:raise ValueError('Context differs from original pin')
                if digest not in documents:documents[digest]=path.read_bytes().decode('utf-8')
                text=documents[digest];span=scope['selected_span'];start,end=span['start_character'],span['end_character']
                if span not in scope['original_candidate_spans'] or not 0<=start<end<=len(text) or end-start!=row['characters'] or text_hash(text[start:end])!=row['raw_sha256']:raise ValueError('Function label does not bind exact original span')
                contract,separator,function=row['function_identifier'].rpartition('.')
                if not separator or scope['source_qualified_identifier']!=row['function_identifier'] or scope['selected_declaration']['name']!=contract:raise ValueError('Qualified label/owner mismatch')
                evidence={'context_artifact':scope['context_artifact'],'context_sha256':digest,'function_span':span,'qualified_identifier':row['function_identifier'],'scope_basis':'producer_qualifier_plus_exact_lexical_owner;not_independent_semantic_truth'}
                if model['model_code_sha256'] in conflicts:status='identical_input_RE_conflict_requires_adjudication'
                else:status='mapped_source_proxy';labels['reentrancy']=claim['RE'];mask['reentrancy']=True
        target={'record_id':identity,'supervision_protocol':VERSION,'target_definition_id':DEFINITION,'source':row['source'],'source_revision':row['source_revision'],
                'source_identity':row['source_identity'],'source_project_metadata':row.get('project_id'),'source_recorded_address':row.get('contract_address'),
                'group_constraints':{k:group_by[identity][k] for k in ('evidence_component','strict_template_component','token_sha256','template_sha256','direct_verified_context_corpus_overlap','evidence_conservative_quarantine','strict_template_conservative_quarantine')},'model_code_sha256':model['model_code_sha256'],'native_source_assessments':claim,
                'labels':labels,'label_observation_mask':mask,'mapping_status':status,'mapping_evidence':evidence,
                'supervision_tier':'upstream_source_proxy' if mask['reentrancy'] else 'unobserved_target','independent_gold':False,'human_review_status':'unreviewed',
                'mapping_available_for_source_supervision':mask['reentrancy'],'training_eligible':False,'training_readiness_blockers':GATES,'evaluation_eligible':False,'split':None}
        model_input={'record_id':identity,'model_code':model['model_code'],'model_code_sha256':model['model_code_sha256'],'input_protocol':model['view_protocol']}
        masked_source_example(model_input,target);targets.append(target);inputs.append(model_input)
    audit=[];buckets=defaultdict(list)
    for target in targets:
        if target['mapping_available_for_source_supervision']:buckets[(target['labels']['reentrancy'],str(target['native_source_assessments']['is_student']))].append(target)
    rng=random.Random(SEED)
    for stratum,rows in sorted(buckets.items()):
        for target in rng.sample(sorted(rows,key=lambda r:r['record_id']),min(2,len(rows))):
            audit.append({'record_id':target['record_id'],'stratum':{'source_RE':stratum[0],'is_student':stratum[1]},'stratum_population':len(rows),'stratum_selected':min(2,len(rows)),
                          'mapping_evidence':target['mapping_evidence'],'source_proxy_label':target['labels']['reentrancy'],'independent_decision':None,'review_status':'not_completed',
                          'review_questions':['Does producer function identity/span match?','Does native RE meaning support the declared proxy?','What does independent contextual review conclude, and what is missing?'],
                          'scope':'small mapping/semantic audit pilot;not_evaluation_gold_or_calibrated_noise_estimate'})
    if verify_delivery(pilot)!=delivered:raise ValueError('Original delivery changed')
    for fn in (verify_delivery,verify_artifacts,write_jsonl):
        path=Path(fn.__code__.co_filename)
        if sha256(path)!=helper_hashes[path.name]:raise ValueError('Helper changed during adaptation')
    for root,filename,identity,manifest in parents.values():
        if sha256(root/filename)!=identity:raise ValueError('Parent manifest changed')
        verify_artifacts(root,filename,'artifact_sha256')
    mapped=[r for r in targets if r['mapping_available_for_source_supervision']]
    summary={'protocol':VERSION,'target_definition_id':DEFINITION,'records':len(targets),'native_scrubd_records':len(native),
             'native_RE_counts':dict(Counter(str(r['RE']) for r in native.values())),'native_UX_counts':dict(Counter(str(r['UX']) for r in native.values())),
             'mapping_status_counts':dict(Counter(r['mapping_status'] for r in targets)),'mapped_source_proxy_rows':len(mapped),
             'observed_target_cells':sum(sum(r['label_observation_mask'].values()) for r in targets),'mapped_RE_counts':dict(Counter(str(r['labels']['reentrancy']) for r in mapped)),
             'exact_input_conflict_groups':len(conflicts),'mapped_exact_model_inputs':len({r['model_code_sha256'] for r in mapped}),'mapped_token_groups':len({r['group_constraints']['token_sha256'] for r in mapped}),'mapped_direct_overlap_rows':sum(r['group_constraints']['direct_verified_context_corpus_overlap'] for r in mapped),'pilot_audit_rows':len(audit),'pilot_audit_seed':SEED,
             'independent_reviews_completed':0,'training_eligible_records':0,'evaluation_eligible_records':0,'assigned_splits':0,
             'limits':['RE is a source-assessed susceptibility proxy, not a mapped syntactic pattern-presence label, independently reviewed contextual suspicion, proven exploitability or legacy eleven-output target.',
                       'Native RE/UX source claims are retained even when no target can be mapped. UX is not automatically unchecked-call; no authorization, blanket CLEAN, patched or no-known-vulnerability negative is inferred.',
                       'Missing context/span, source quality, licenses/provenance, code-only label conflicts, old-corpus overlap, grouped splits, independent evaluation and compatible masked learning remain gates. No training/inference or accuracy gain.']}
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('source_targets.jsonl',targets),('model_inputs.jsonl',inputs),('stratified_audit_queue.jsonl',audit)]:write_jsonl(output/name,rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    card={'protocol':VERSION,'target_definition_id':DEFINITION,'target_axis':'upstream-source-assessed reentrancy susceptibility proxy',
          'category_definitions':{'reentrancy':'Only explicit SCRUBD RE0/1 source assessments with qualified exact spans; not syntactic pattern presence, independent contextual suspicion or exploitability.','unchecked_external_call':'Unmapped; native UX is retained separately.','authorization':'Unmapped; no blanket negative inferred.'},
          'supervision':'upstream_source_proxy, not independent gold; human_review_status remains unreviewed',
          'loss_contract':'Unknown labels are null with false masks. A zero vector placeholder is disabled, not negative. A future learner must reduce supervised loss over observed cells only and skip unobserved cells/examples; the current legacy closed-set learner is not automatically compatible.',
          'audit_plan':{'frame':'mapped source-proxy rows only','strata':['native_RE','is_student'],'maximum_per_stratum':2,'seed':SEED,'status':'not_completed','scope':'pilot mapping/meaning audit, not calibrated label-noise estimate or independent evaluation'},
          'readiness':'All training/evaluation eligibility false; no assigned splits. Independent reviewed evaluation, source-quality audit, grouping/provenance/license and masked-learner gates remain.',
          'model_inputs':'Only current immutable code view fields; no audit explanation, source label or rationale concatenated to code.',
          'source_limits':summary['limits']}
    (output/'data_card.json').write_text(json.dumps(card,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'helper_sha256':helper_hashes,'generator_sha256':sha256(Path(__file__)),'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,
              'parent_manifest_sha256':{name:value[2] for name,value in parents.items()},'target_definition_id':DEFINITION,
              'model_input_fields':['record_id','model_code','model_code_sha256','input_protocol'],'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','context-release-dir','scope-release-dir','view-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(prepare(a.pilot_dir,a.context_release_dir,a.scope_release_dir,a.view_dir,a.output_dir),indent=2))

if __name__=='__main__':main()
