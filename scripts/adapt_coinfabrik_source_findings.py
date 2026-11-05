#!/usr/bin/env python3
"""Export localized CoinFabrik source-finding proxies; never train or create gold."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import safe_file, verify_artifacts, masked_source_and_imports
from prepare_function_review import CATEGORIES, text_hash, write_jsonl

VERSION = 'coinfabrik-localized-finding-proxy-v1'
SOURCE = 'CoinFabrik/solidity-rnd'
DEFINITIONS = {'reentrancy':'coinfabrik-source-reported-reentrancy-order-concern-v1',
               'authorization':'coinfabrik-source-reported-authority-policy-concern-v1'}
GROUP_KEYS = ('direct_verified_context_corpus_overlap','evidence_component','evidence_conservative_quarantine',
              'strict_template_component','strict_template_conservative_quarantine','template_sha256','token_sha256')
MODEL_FIELDS = {'record_id','model_code','model_code_sha256','input_protocol'}


def object_hash(value):
    return text_hash(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False))


def crosswalk(finding):
    cls=finding.get('vulnerability_class');cwe=finding.get('cwe_classification')
    if not isinstance(cls,str) or type(cwe) is not int:return None
    cls=cls.strip().lower()
    if cls=='reentrancy' and cwe in {367,696,691,280}:return 'reentrancy'
    if cls in {'authorization','access control'} and cwe in {284,285}:return 'authorization'
    return None


def anchored_pointer_pairs(finding):
    """Empty parentheses are a source symbol reference, not type/signature proof."""
    pointers=finding.get('vulnerable_functions');lines=finding.get('lines')
    if not isinstance(pointers,list) or not pointers or not isinstance(lines,str):return []
    anchors=[item.strip() for item in lines.split(',')]
    if len(anchors)!=len(pointers):return []
    pairs=[]
    for pointer,anchor in zip(pointers,anchors):
        if not isinstance(pointer,str):return []
        match=re.fullmatch(r'\s*([A-Za-z_$][\w$]*)(?:\.sol)?::\s*([A-Za-z_$][\w$]*)\(\)\s*',pointer)
        loc=re.fullmatch(r'([1-9][0-9]*)(?:-([1-9][0-9]*))?',anchor)
        if not match or not loc:return []
        start=int(loc[1]);end=int(loc[2] or loc[1])
        if end<start:return []
        pairs.append((match[1]+'::'+match[2],{'start_line':start,'end_line':end},pointer))
    return pairs


def localized_evidence(row,scope,raw):
    if row['version_role']!='vulnerable' or scope['scope_status']!='unique_qualified_scope_match':return None
    if scope['pinned_context_sha256']!=row['context_sha256'] or text_hash(raw)!=row['context_sha256']:raise ValueError('Context identity mismatch')
    if scope['source_qualified_identifier']!=row['function_identifier']:raise ValueError('Qualified function differs')
    span=scope['selected_span'];start,end=span['start_character'],span['end_character']
    if not 0<=start<end<=len(raw) or text_hash(raw[start:end])!=span['raw_fragment_sha256'] or span['raw_fragment_sha256']!=row['raw_sha256']:raise ValueError('Original fragment span differs')
    if row['source_identity'].rpartition(':')[2]!=str(start):raise ValueError('Producer source offset differs')
    if span['start_line']!=raw.count('\n',0,start)+1 or span['end_line']!=raw.count('\n',0,end-1)+1:raise ValueError('Function line bounds differ')
    owner=scope['selected_declaration'];contract,separator,function=row['function_identifier'].partition('::')
    if not separator or contract!=owner['name']:raise ValueError('Qualified owner differs')
    masked=masked_source_and_imports(raw)[0];body=masked[owner['body_open']+1:owner['body_close']]
    if len(re.findall(r'\bfunction\s+'+re.escape(function)+r'\s*\(',body))!=1:return None
    return {'context_artifact':scope['context_artifact'],'context_sha256':row['context_sha256'],
            'function_span':span,'qualified_identifier':row['function_identifier'],
            'basis':'exact_raw_span_unique_qualified_lexical_symbol_and_positional_source_line_anchor;not_compiler_signature_or_gold'}


def source_finding_example(model,target, *, for_training=False):
    if type(for_training) is not bool:raise ValueError('Purpose must be boolean')
    if for_training:raise ValueError('Quality/evaluation/group/license and multi-source learner migration remain pending')
    if target['supervision_protocol']!=VERSION or target['source']!=SOURCE:raise ValueError('Explicit CoinFabrik source-proxy schema required')
    if set(model)!=MODEL_FIELDS or model['record_id']!=target['record_id'] or text_hash(model['model_code'])!=model['model_code_sha256'] or model['model_code_sha256']!=target['model_code_sha256']:raise ValueError('Code-only input identity differs')
    if target['independent_gold'] is not False or target['human_review_status']!='unreviewed' or target['training_eligible'] is not False or target['evaluation_eligible'] is not False or target['split'] is not None:raise ValueError('Source layer cannot certify gold/eligibility/splits')
    labels=target['source_proxy_labels'];masks=target['source_proxy_observation_mask'];definitions=target['target_definition_ids']
    if any(set(x)!=set(CATEGORIES) for x in (labels,masks,definitions)):raise ValueError('Three-category schema required, not legacy11')
    values=[];observed=[]
    for category in CATEGORIES:
        mask=masks[category];label=labels[category]
        if type(mask) is not bool:raise ValueError('Masks must be boolean')
        if mask:
            if label is not True or category not in DEFINITIONS or definitions[category]!=DEFINITIONS[category] or target['version_role']!='vulnerable' or not target['mapping_evidence'] or not any(x['category']==category for x in target['finding_support']):raise ValueError('Only supported source-positive cells are available in v1')
        elif label is not None or definitions[category] is not None:raise ValueError('Unobserved is null, never negative')
        values.append(1. if mask else 0.);observed.append(mask)
    return model['model_code'],values,observed


def prepare(pilot,contexts,scopes,views,output):
    pilot,contexts,scopes,views,output=[Path(x).resolve() for x in (pilot,contexts,scopes,views,output)]
    if output.exists():raise FileExistsError(output)
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or any(output==p or output.is_relative_to(p) for p in (contexts,scopes,views)):raise ValueError('Preserve originals and ancestors')
    delivered=verify_delivery(pilot);parents={}
    for name,root,filename in [('contexts',contexts,'recovery_manifest.json'),('scopes',scopes,'manifest.json'),('views',views,'manifest.json')]:
        m=verify_artifacts(root,filename,'artifact_sha256');parents[name]=(root,filename,sha256(root/filename),m)
        if m['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Parent delivery differs')
    if parents['scopes'][3]['context_release_manifest_sha256']!=parents['contexts'][2] or parents['contexts'][3]['model_view_manifest_sha256']!=parents['views'][2]:raise ValueError('Parent lineage differs')
    source_manifest=json.loads((pilot/'source_manifest.json').read_text());revision=source_manifest['sources'][SOURCE]['revision']
    candidates=[r for r in jsonl_rows(pilot/'public_output/function_candidates.jsonl') if r['source']==SOURCE]
    model_by={r['record_id']:r for r in jsonl_rows(views/'function_model_views.jsonl')};scope_by={r['record_id']:r for r in jsonl_rows(scopes/'function_scope_links.jsonl')};group_by={r['record_id']:r for r in jsonl_rows(contexts/'groups/candidate_groups.jsonl')}
    original_findings=list(jsonl_rows(pilot/'public_output/coinfabrik_findings.jsonl'));finding_by={}
    for item in original_findings:
        if item['source']!=SOURCE or item['path']!=item['project']+'/vulnerable/findings.json':raise ValueError('Finding provenance shape differs')
        identity=item['path']+':'+str(item['index'])
        if identity in finding_by:raise ValueError('Duplicate finding identity')
        entries=[f for f in source_manifest['files'] if f['source']==SOURCE and f['path']==item['path']]
        if len(entries)!=1 or entries[0]['revision']!=revision:raise ValueError('Missing pinned finding-file attestation')
        finding_by[identity]=(item,entries[0])
    targets=[];inputs=[];documents={};audits=[]
    for row in sorted(candidates,key=lambda r:r['record_id']):
        identity=row['record_id'];model=model_by[identity];scope=scope_by[identity];group=group_by[identity]
        parts=row['source_path'].split('/')
        if (row['source_revision']!=revision or len(parts)<3 or parts[0]!=row['project_id']
            or parts[1]!=row['version_role'] or row['version_role'] not in {'vulnerable','patched'}
            or row['source_identity'].rpartition(':')[0]!=row['source_path']):raise ValueError('Candidate provenance/version role differs')
        for parent in (model,scope,group):
            if (any(v is not None for v in parent.get('pattern_labels',{}).values())
                or any(parent.get('label_observation_mask',{}).values()) or parent.get('training_eligible') is not False
                or parent.get('split') is not None or parent.get('human_review_status','unreviewed')!='unreviewed'):
                raise ValueError('Annotated/eligible parents require explicit migration')
        if any(v is not None for v in row['pattern_labels'].values()) or row['training_eligible'] is not False or row['split'] is not None or row['human_review_status']!='unreviewed':raise ValueError('Annotated parents require explicit migration')
        if text_hash(model['model_code'])!=model['model_code_sha256'] or model['parent_model_code_sha256']!=text_hash(row['model_code']) or group['exact_input_sha256']!=model['model_code_sha256']:raise ValueError('Current code view/group lineage differs')
        code={'record_id':identity,'model_code':model['model_code'],'model_code_sha256':model['model_code_sha256'],'input_protocol':model['view_protocol']}
        labels=dict.fromkeys(CATEGORIES);masks=dict.fromkeys(CATEGORIES,False);definitions=dict.fromkeys(CATEGORIES);evidence=None;support=[];reasons=[]
        if row['version_role']=='vulnerable' and scope['context_artifact'] and scope['scope_status']=='unique_qualified_scope_match':
            context_file=safe_file(contexts,scope['context_artifact'])
            if sha256(context_file)!=row['context_sha256']:raise ValueError('Pinned raw context differs')
            raw=documents.setdefault(row['context_sha256'],context_file.read_bytes().decode('utf-8'))
            evidence=localized_evidence(row,scope,raw)
        for finding_id in row['finding_links']:
            if finding_id not in finding_by:raise ValueError('Producer finding link unresolved')
            item,entry=finding_by[finding_id];finding=item['finding'];category=crosswalk(finding)
            if item['project']!=row['project_id']:raise ValueError('Cross-project finding link')
            reason=None
            if row['version_role']!='vulnerable':reason='patched_role_is_not_negative_or_positive'
            elif category is None:reason='unsupported_source_category_definition'
            elif evidence is None:reason='qualified_exact_context_span_unavailable'
            else:
                pairs=[(anchor,pointer) for qualified,anchor,pointer in anchored_pointer_pairs(finding) if qualified==row['function_identifier']]
                if len(pairs)!=1:reason='missing_or_ambiguous_symbol_anchor_correspondence'
                elif not evidence['function_span']['start_line']<=pairs[0][0]['start_line']<=pairs[0][0]['end_line']<=evidence['function_span']['end_line']:reason='finding_line_anchor_outside_function'
                else:
                    anchor,pointer=pairs[0];labels[category]=True;masks[category]=True;definitions[category]=DEFINITIONS[category]
                    support.append({'finding_id':finding_id,'category':category,'target_definition_id':DEFINITIONS[category],
                                    'retained_finding_object_sha256':object_hash(finding),'upstream_declared_file_sha256':entry['sha256'],
                                    'upstream_revision':revision,'upstream_path':item['path'],'source_pointer':pointer,'source_line_anchor':anchor,
                                    'source_title':finding.get('title'),'source_class':finding.get('vulnerability_class'),'source_subclass':finding.get('vulnerability_subclass'),'source_CWE':finding.get('cwe_classification'),
                                    'authority':'retained_upstream_dataset_finding_claim;not_fresh_blob_verification_or_independent_gold'})
            audits.append({'record_id':identity,'finding_id':finding_id,'category_candidate':category,'status':reason or 'mapped_source_positive_claim'})
            if reason:reasons.append(reason)
        if row['version_role']!='vulnerable':status='patched_unobserved'
        elif support:status='mapped_source_finding_proxy'
        elif not row['finding_links']:status='no_specific_finding_link_unobserved'
        else:status='linked_claims_not_mapped'
        target={'record_id':identity,'source':SOURCE,'source_revision':revision,'source_project':row['project_id'],'source_path':row['source_path'],'function_identifier':row['function_identifier'],'version_role':row['version_role'],
                'supervision_protocol':VERSION,'supervision_tier':'retained_upstream_finding_proxy','source_proxy_labels':labels,'source_proxy_observation_mask':masks,'target_definition_ids':definitions,
                'model_code_sha256':model['model_code_sha256'],'mapping_status':status,'mapping_evidence':evidence if support else None,'finding_support':support,'unmapped_reasons':sorted(set(reasons)),
                'group_constraints':{k:group[k] for k in GROUP_KEYS},'independent_gold':False,'human_review_status':'unreviewed','training_eligible':False,'evaluation_eligible':False,'split':None,
                'readiness_blockers':['target_alignment_and_targeted_source_quality_audit','independent_grouped_evaluation','frozen_group_splits','provenance_license_eligibility','reviewed_multisource_learner_protocol_migration']}
        source_finding_example(code,target);targets.append(target);inputs.append(code)
    if verify_delivery(pilot)!=delivered:raise ValueError('Original delivery changed')
    for root,name,digest,m in parents.values():
        if sha256(root/name)!=digest:raise ValueError('Parent changed')
        verify_artifacts(root,name,'artifact_sha256')
    mapped=[r for r in targets if any(r['source_proxy_observation_mask'].values())]
    summary={'protocol':VERSION,'records':len(targets),'retained_finding_objects':len(original_findings),'mapping_status_counts':dict(Counter(r['mapping_status'] for r in targets)),
             'observed_source_positive_cells':sum(sum(r['source_proxy_observation_mask'].values()) for r in targets),'observed_by_category':{c:sum(r['source_proxy_observation_mask'][c] for r in targets) for c in CATEGORIES},
             'mapped_functions':len(mapped),'mapped_projects':len({r['source_project'] for r in mapped}),'mapped_contexts':len({r['mapping_evidence']['context_sha256'] for r in mapped}),
             'mapped_finding_objects':len({f['finding_id'] for r in mapped for f in r['finding_support']}),'support_edges':sum(len(r['finding_support']) for r in mapped),'mapped_existing_corpus_overlap_rows':sum(r['group_constraints']['direct_verified_context_corpus_overlap'] for r in mapped),
             'observed_negative_cells':0,'independent_gold_reviews':0,'training_eligible':0,'evaluation_eligible':0,'assigned_splits':0,
             'exclusion_counts':dict(Counter(r['status'] for r in audits)),'source_proxy_namespaces_are_not_harmonized_gold':True,'model_training_inference_or_accuracy_gain':False}
    output.mkdir(parents=True,exist_ok=False)
    for name,items in [('source_proxy_targets.jsonl',targets),('model_inputs.jsonl',inputs),('finding_link_checks.jsonl',audits)]:write_jsonl(output/name,items)
    (output/'retained_finding_claims.jsonl').write_bytes((pilot/'public_output/coinfabrik_findings.jsonl').read_bytes())
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    card={'protocol':VERSION,'target_definition_ids':DEFINITIONS,'definitions':{'reentrancy':'Upstream dataset explicitly reports a reentrancy/order/mitigation concern at this localized function; not independently established contextual suspicion or exploitability.','authorization':'Upstream dataset explicitly reports authority/privilege-policy concern at this localized function; not proof of absent access checks, unauthorized execution or exploitability.','unchecked_external_call':'Unmapped in v1.'},
          'source_quality':'Explicit weak/source claims; category interpretation and linkage are not independent adjudication. Excessive-privilege/internal-helper and guarded RE cases need targeted quality review.',
          'input':'Byte-identical current annotation-v2 code-only view; no finding descriptions/rationales/labels concatenated.',
          'negative_policy':'All nonpositive cells unknown; patched/no-finding functions are not clean negatives.',
          'source_authentication':'Original delivered hashes and per-finding object fingerprints/pin attestations retained; no fresh upstream raw findings blob or independent audit report verification.',
          'readiness':'No eligible rows, splits or training; existing SCRUBD-only loader deliberately rejects this separate namespace. Reviewed multisource target/loss protocol, source quality, independent evaluation, grouping and license gates remain.',
          'scope':'Qualified lexical symbol and positional line anchors, not compiler type overload resolution, dependency closure, caller/guard correctness, calibrated confidence or confirmed vulnerability.'}
    (output/'data_card.json').write_text(json.dumps(card,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'source_revision':revision,'generator_sha256':sha256(Path(__file__)),'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,
              'parent_manifest_sha256':{name:data[2] for name,data in parents.items()},'helper_sha256':{Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (verify_delivery,verify_artifacts,write_jsonl)},
              'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','context-release-dir','scope-release-dir','view-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(prepare(a.pilot_dir,a.context_release_dir,a.scope_release_dir,a.view_dir,a.output_dir),indent=2))

if __name__=='__main__':main()
