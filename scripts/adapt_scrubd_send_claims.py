#!/usr/bin/env python3
"""Export narrow native-UX/typed-discarded-send weak claims, never gold or negatives."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
from audit_function_pilot_handoff import jsonl_rows,sha256,verify_delivery,inspect_staging_rows
from prepare_context_review import safe_file,verify_artifacts,masked_source_and_imports
from prepare_function_review import CATEGORIES,text_hash,write_jsonl
from adapt_function_source_labels import SOURCE,native_binary,VERSION as RE_VERSION,masked_source_example
from measure_coinfabrik_pairs import literal_tokens

VERSION='scrubd-native-UX-typed-discarded-send-proxy-v1'
DEFINITION='source-UX-supported-compiler-typed-discarded-address-send-result-v1'
COMPILER_SHA256='9dd7686975200be71ce731f153823e7ea0d68106f3354ea16fecd5975f5c98b7'

def walk(node):
    if isinstance(node,dict):
        yield node
        for value in node.values():yield from walk(value)
    elif isinstance(node,list):
        for value in node:yield from walk(value)

def bounds(node,source_id,total):
    fields=node['src'].split(':')
    if len(fields)!=3:raise ValueError('Malformed compiler span')
    start,length,identity=map(int,fields)
    if identity!=source_id or length<1 or start<0 or start+length>total:raise ValueError('Foreign/out-of-range compiler span')
    return start,start+length

def typed_send_evidence(source,ast,source_id,qualified,span,member):
    """Require exact owner/function byte span and an actual discarded address.send bool."""
    raw=source.encode('utf-8');lo=len(source[:span['start_character']].encode());hi=len(source[:span['end_character']].encode())
    if not 0<=span['start_character']<span['end_character']<=len(source) or text_hash(source[span['start_character']:span['end_character']])!=span['raw_fragment_sha256']:raise ValueError('Raw function identity differs')
    contract,sep,name=qualified.rpartition('.')
    if not sep:raise ValueError('Qualified source function required')
    owners=[n for n in walk(ast) if n.get('nodeType')=='ContractDefinition' and n.get('name')==contract]
    if len(owners)!=1:raise ValueError('Unique compiler owner required')
    functions=[n for n in owners[0].get('nodes',[]) if n.get('nodeType')=='FunctionDefinition' and n.get('name')==name and bounds(n,source_id,len(raw))==(lo,hi)]
    if len(functions)!=1:raise ValueError('Unique exact compiler function required')
    a=member['start_character'];b=member['end_character']
    if not span['start_character']<=a<b<=span['end_character'] or source[a:b]!='send':raise ValueError('Exact send member anchor required')
    target=(len(source[:a].encode()),len(source[:b].encode()));matches=[]
    for statement in walk(functions[0].get('body')):
        if statement.get('nodeType')!='ExpressionStatement':continue
        call=statement.get('expression',{});access=call.get('expression',{})
        if call.get('nodeType')!='FunctionCall' or call.get('kind')!='functionCall' or access.get('nodeType')!='MemberAccess' or access.get('memberName')!='send':continue
        receiver=access.get('expression',{}).get('typeDescriptions',{});typ=access.get('typeDescriptions',{});result=call.get('typeDescriptions',{})
        if receiver.get('typeIdentifier')!='t_address' or receiver.get('typeString')!='address' or not typ.get('typeIdentifier','').startswith('t_function_send_') or result.get('typeIdentifier')!='t_bool' or result.get('typeString')!='bool':continue
        ms,me=bounds(access,source_id,len(raw));cs,ce=bounds(call,source_id,len(raw));ss,se=bounds(statement,source_id,len(raw))
        if (me-4,me)!=target or not lo<=ss<=cs<=ms<me<=ce<=se<=hi or raw[me-4:me]!=b'send':continue
        # Compiler proves the local boolean is discarded; not reachability or vulnerability.
        matches.append({'statement_byte_span':[ss,se],'call_byte_span':[cs,ce],'member_byte_span':list(target),'call_sha256':text_hash(raw[cs:ce].decode('utf-8')),'function_byte_span':[lo,hi],'compiler_function_id':functions[0]['id'],'receiver_type':receiver,'member_type':typ,'result_type':result,'basis':'native_UX_positive_plus_exact_compiler_typed_bare_address_send_bool;not_independent_gold'})
    if len(matches)!=1:raise ValueError('Unique typed discarded native-send site required')
    return matches[0]

def retains_call(model_code,call_text):
    model=literal_tokens(model_code);call=literal_tokens(call_text)
    return any(model[i:i+len(call)]==call for i in range(len(model)-len(call)+1))

def send_proxy_example(model,target,*,for_training=False):
    if for_training:raise ValueError('Source quality/evaluation/group/license/multisource training gates pending')
    if target['protocol']!=VERSION or target['target_definition_id']!=DEFINITION or target['source']!=SOURCE:raise ValueError('Separate weak-proxy namespace required')
    if set(model)!={'record_id','model_code','model_code_sha256','input_protocol'} or model['record_id']!=target['record_id'] or text_hash(model['model_code'])!=model['model_code_sha256'] or model['model_code_sha256']!=target['model_code_sha256']:raise ValueError('Code-only identity differs')
    if target['independent_gold'] is not False or target['training_eligible'] is not False or target['evaluation_eligible'] is not False or target['split'] is not None or target['human_review_status']!='unreviewed':raise ValueError('No gold/eligibility/split promotion')
    labels=target['source_proxy_labels'];masks=target['source_proxy_observation_mask']
    if set(labels)!=set(CATEGORIES) or set(masks)!=set(CATEGORIES):raise ValueError('Partial three-category proxy schema required')
    for c in CATEGORIES:
        if type(masks[c]) is not bool:raise ValueError('Boolean observation mask required')
        if masks[c]:
            if c!='unchecked_external_call' or labels[c] is not True or target['native_UX'] is not True or not target['typed_site_evidence']:raise ValueError('Only supported native-positive discarded-send proxy allowed')
        elif labels[c] is not None:raise ValueError('Unknown is not negative')
    return model['model_code'],[float(masks[c]) for c in CATEGORIES],[masks[c] for c in CATEGORIES]

def prepare(pilot,source_layer,contexts,calls,binary,output):
    pilot,source_layer,contexts,calls,binary,output=[Path(p).resolve() for p in (pilot,source_layer,contexts,calls,binary,output)]
    if output.exists():raise FileExistsError(output)
    if any(root==output or root.is_relative_to(output) for root in (pilot,source_layer,contexts,calls,binary.parent)) or any(output.is_relative_to(root) for root in (source_layer,contexts,calls,binary.parent)) or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')):raise ValueError('Preserve original/parent/compiler paths')
    delivered=verify_delivery(pilot);parents={}
    for key,root,name in [('source',source_layer,'manifest.json'),('contexts',contexts,'recovery_manifest.json'),('calls',calls,'manifest.json')]:
        m=verify_artifacts(root,name,'artifact_sha256')
        if m['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Delivery lineage differs')
        parents[key]=(root,name,sha256(root/name),m)
    if parents['source'][3]['parent_manifest_sha256']['contexts']!=parents['contexts'][2] or parents['calls'][3]['context_release_manifest_sha256']!=parents['contexts'][2] or parents['source'][3]['parent_manifest_sha256']['scopes']!=parents['calls'][3]['scope_release_manifest_sha256']:raise ValueError('Source/call/context ancestry differs')
    if sha256(binary)!=COMPILER_SHA256:raise ValueError('Installed compiler checksum differs')
    all_originals=list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'));inspect_staging_rows(all_originals)
    if parents['source'][3]['protocol']!=RE_VERSION or parents['calls'][3]['protocol']!='function-call-result-evidence-v1':raise ValueError('Explicit parent protocols required')
    originals={r['record_id']:r for r in all_originals if r['source']==SOURCE};targets={r['record_id']:r for r in jsonl_rows(source_layer/'source_targets.jsonl') if r['source']==SOURCE};inputs={r['record_id']:r for r in jsonl_rows(source_layer/'model_inputs.jsonl') if r['record_id'] in originals}
    if not originals or len(originals)>5000 or set(originals)!=set(targets) or set(inputs)!=set(originals):raise ValueError('Unique bounded native-source frame required')
    frame=[x for x in jsonl_rows(calls/'call_site_evidence.jsonl') if x['record_id'] in targets and targets[x['record_id']]['mapping_available_for_source_supervision'] and targets[x['record_id']]['native_source_assessments']['UX'] is True and x['member_spelling']=='send' and x['invocation_shape']=='direct' and x['result_disposition_observation']=='bare_expression_statement']
    digests=sorted({x['context_sha256'] for x in frame})
    if len(digests)>4 or len(frame)>20:raise ValueError('Predeclared four-context/20-site budget exceeded; explicit migration required')
    output.mkdir(parents=True,exist_ok=False);compiler=output/'compiler';compiler.mkdir();results={};site_evidence=defaultdict(list);failures=[]
    version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=30,check=True);(compiler/'version.txt').write_text(version.stdout+version.stderr)
    if '0.4.25+' not in version.stdout:raise ValueError('Compiler version differs')
    for digest in digests:
        raw=safe_file(contexts,'contexts/'+digest+'.sol').read_bytes()
        if text_hash(raw.decode('utf-8'))!=digest:raise ValueError('Raw context hash differs')
        source=raw.decode('utf-8')
        if masked_source_and_imports(source)[1]:results[digest]={'status':'imports_blocked'};continue
        data={'language':'Solidity','sources':{'context.sol':{'content':source}},'settings':{'optimizer':{'enabled':False,'runs':200},'evmVersion':'byzantium','outputSelection':{'*':{'':['ast']}}}}
        serialized=json.dumps(data,sort_keys=True)+'\n';(compiler/(digest+'_input.json')).write_text(serialized)
        process=subprocess.run([str(binary),'--standard-json'],input=serialized,capture_output=True,text=True,timeout=30,check=False)
        (compiler/(digest+'_output.json')).write_text(process.stdout);(compiler/(digest+'_stderr.txt')).write_text(process.stderr)
        parsed=json.loads(process.stdout);diagnostics=parsed.get('errors',[])
        result={'status':'compiled' if process.returncode==0 and not any(e.get('severity')=='error' for e in diagnostics) else 'compiler_failed','returncode':process.returncode,'diagnostics':diagnostics};results[digest]=result
        if result['status']!='compiled':continue
        ast=parsed['sources']['context.sol']['ast'];sid=parsed['sources']['context.sol']['id']
        for site in [x for x in frame if x['context_sha256']==digest]:
            t=targets[site['record_id']];e=t['mapping_evidence'];row=originals[site['record_id']]
            if e['context_sha256']!=digest or e['qualified_identifier']!=row['function_identifier'] or e['function_span']['raw_fragment_sha256']!=row['raw_sha256']:raise ValueError('Native/function/context binding differs')
            try:
                proof=typed_send_evidence(source,ast,sid,row['function_identifier'],e['function_span'],site['member_span'])
                start,end=proof['call_byte_span']
                if not retains_call(inputs[site['record_id']]['model_code'],raw[start:end].decode('utf-8')):raise ValueError('Typed call not retained in current code-only view')
            except ValueError as error:failures.append({'site_id':site['site_id'],'record_id':site['record_id'],'status':'typed_site_unresolved','reason':str(error)});continue
            site_evidence[site['record_id']].append({'site_id':site['site_id'],'context_sha256':digest,'native_source_identity':row['source_identity'],**proof})
    output_targets=[];models=[]
    for identity,row in sorted(originals.items()):
        t=targets[identity];m=inputs[identity];masked_source_example(m,t);ux=native_binary(row['source_metadata'].get('UX'))
        if t['native_source_assessments']['UX'] is not ux or t['source_revision']!=row['source_revision'] or text_hash(m['model_code'])!=m['model_code_sha256'] or m['model_code_sha256']!=t['model_code_sha256']:raise ValueError('Native/code lineage differs')
        labels=dict.fromkeys(CATEGORIES);masks=dict.fromkeys(CATEGORIES,False);evidence=site_evidence[identity]
        if evidence:labels['unchecked_external_call']=True;masks['unchecked_external_call']=True
        target={'record_id':identity,'protocol':VERSION,'target_definition_id':DEFINITION,'source':SOURCE,'source_revision':row['source_revision'],'source_identity':row['source_identity'],'native_UX':ux,'source_native_UX_raw':row['source_metadata'].get('UX'),'model_code_sha256':m['model_code_sha256'],'source_proxy_labels':labels,'source_proxy_observation_mask':masks,'typed_site_evidence':evidence,'group_constraints':t['group_constraints'],'supervision_tier':'native_source_claim_plus_compiler_pattern_weak_proxy' if evidence else 'unobserved','independent_gold':False,'human_review_status':'unreviewed','training_eligible':False,'evaluation_eligible':False,'split':None,'readiness_blockers':['targeted_source_quality_and_target_alignment','independent_grouped_evaluation','provenance_license_eligibility','frozen_grouped_cohorts','reviewed_multisource_training_protocol']}
        send_proxy_example(m,target);output_targets.append(target);models.append(m)
    if verify_delivery(pilot)!=delivered or sha256(binary)!=COMPILER_SHA256:raise ValueError('Original/compiler changed')
    for root,name,digest,manifest in parents.values():
        if sha256(root/name)!=digest:raise ValueError('Parent changed')
        verify_artifacts(root,name,'artifact_sha256')
    mapped=[t for t in output_targets if t['typed_site_evidence']]
    summary={'protocol':VERSION,'native_source_rows':len(output_targets),'native_UX_counts':dict(Counter(str(t['native_UX']) for t in output_targets)),'fixed_frame_functions':len({s['record_id'] for s in frame}),'fixed_frame_sites':len(frame),'fixed_contexts':len(digests),'compiler_status_counts':dict(Counter(x['status'] for x in results.values())),'observed_positive_cells':len(mapped),'observed_negative_cells':0,'confirmed_typed_discarded_send_sites':sum(len(t['typed_site_evidence']) for t in mapped),'unresolved_sites':len(failures),'mapped_existing_corpus_overlap_rows':sum(t['group_constraints']['direct_verified_context_corpus_overlap'] for t in mapped),'independent_gold':0,'training_eligible':0,'evaluation_eligible':0,'assigned_splits':0,'training_or_inference':False}
    for name,rows in [('source_proxy_targets.jsonl',output_targets),('model_inputs.jsonl',models),('site_failures.jsonl',failures)]:write_jsonl(output/name,rows)
    for name,data in [('summary.json',summary),('compiler_report.json',results),('data_card.json',{'protocol':VERSION,'definition':DEFINITION,'meaning':'Native UX=1 source claim with compiler-typed address.send bool discarded as an exact local expression statement. Weak source/pattern proxy only; not proven reachable vulnerability, exploitability, native UX wholesale mapping or independent evaluation truth.','unknown_policy':'All unsupported/absent/negative/ambiguous native claims and other categories remain null with false masks. No safe or repaired negatives.','inputs':'Only unchanged current annotation-v2 code; source comments, compiler proofs, labels and rationales are separate from model inputs.','limits':'Compiler0.4.25AST conditional on declared settings, not deployed bytecode or full failure propagation/reachability. Four retained existing-corpus contexts are not independent generalization data. Do not evaluate correctness against the same compiler extraction rule. Old loaders/loss do not grant eligibility or namespace harmonization.'})]:
        (output/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'helper_sha256':{Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (verify_delivery,verify_artifacts,write_jsonl,native_binary,literal_tokens)},'compiler_version':'0.4.25','compiler_sha256':COMPILER_SHA256,'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,'parent_manifest_sha256':{k:v[2] for k,v in parents.items()},'artifact_sha256':{str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*')) if f.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','source-layer-dir','context-release-dir','call-release-dir','compiler-binary','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.pilot_dir,a.source_layer_dir,a.context_release_dir,a.call_release_dir,a.compiler_binary,a.output_dir),indent=2))

if __name__=='__main__':main()
