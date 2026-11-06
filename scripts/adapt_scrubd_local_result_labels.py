#!/usr/bin/env python3
"""Export scoped weak local-result labels; checked negatives are never global CLEAN."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
from audit_function_pilot_handoff import jsonl_rows,sha256,verify_delivery
from prepare_context_review import safe_file,verify_artifacts,masked_source_and_imports
from prepare_function_review import CATEGORIES,text_hash,write_jsonl
from adapt_scrubd_send_claims import SOURCE,COMPILER_SHA256,VERSION_V2,DEFINITION_V2,walk,bounds,retains_call

VERSION='scrubd-source-local-native-result-binary-proxy-v3'
DEFINITION='source-UX-supported-local-native-result-discard-with-strong-checked-contrast-v3'

def native_call(node):
    """Resolve only native address send/call bool invocations, not arbitrary wrappers."""
    if node.get('nodeType')!='FunctionCall' or node.get('kind')!='functionCall' or node.get('typeDescriptions',{}).get('typeIdentifier')!='t_bool':return None
    access=node.get('expression',{});wrappers=[]
    while access.get('nodeType')=='FunctionCall':
        setter=access.get('expression',{});name=setter.get('memberName')
        if name not in {'value','gas'} or name in wrappers or len(access.get('arguments',[]))!=1 or access.get('kind')!='functionCall' or setter.get('nodeType')!='MemberAccess' or not setter.get('typeDescriptions',{}).get('typeIdentifier','').startswith('t_function_set'+str(name)+'_') or not access.get('typeDescriptions',{}).get('typeIdentifier','').startswith('t_function_barecall_'):
            return None
        wrappers.append(name);access=setter.get('expression',{})
    kind=access.get('memberName');receiver=access.get('expression',{}).get('typeDescriptions',{})
    if access.get('nodeType')!='MemberAccess' or kind not in {'send','call'} or receiver.get('typeIdentifier')!='t_address' or receiver.get('typeString')!='address':return None
    if kind=='send' and wrappers:return None
    prefix='t_function_send_' if kind=='send' else 't_function_barecall_'
    if not access.get('typeDescriptions',{}).get('typeIdentifier','').startswith(prefix):return None
    return access,kind,wrappers

def unconditional_failure(node):
    if not isinstance(node,dict):return False
    if node.get('nodeType')=='Block':
        statements=node.get('statements',[])
        return len(statements)==1 and unconditional_failure(statements[0])
    if node.get('nodeType')=='Throw':return True
    expression=node.get('expression',{});callee=expression.get('expression',{})
    return (node.get('nodeType')=='ExpressionStatement' and expression.get('nodeType')=='FunctionCall' and expression.get('kind')=='functionCall' and expression.get('arguments')==[] and callee.get('nodeType')=='Identifier' and callee.get('name')=='revert' and callee.get('typeDescriptions',{}).get('typeIdentifier','').startswith('t_function_revert_'))

def strong_guards(function,call_id):
    matches=[]
    for node in walk(function.get('body')):
        expr=node.get('expression',{});callee=expr.get('expression',{});args=expr.get('arguments',[])
        if (node.get('nodeType')=='ExpressionStatement' and expr.get('nodeType')=='FunctionCall' and expr.get('kind')=='functionCall' and len(args)==1 and args[0].get('id')==call_id and callee.get('nodeType')=='Identifier' and callee.get('name')=='require' and callee.get('typeDescriptions',{}).get('typeIdentifier','').startswith('t_function_require_')):
            matches.append(('direct_builtin_require',node))
        condition=node.get('condition',{})
        if (node.get('nodeType')=='IfStatement' and condition.get('nodeType')=='UnaryOperation' and condition.get('operator')=='!' and condition.get('prefix') is True and condition.get('subExpression',{}).get('id')==call_id and unconditional_failure(node.get('trueBody'))):matches.append(('negated_if_unconditional_failure',node))
    return matches

def assess_body(source,ast,source_id,qualified,span,model_code):
    raw=source.encode();a,b=span['start_character'],span['end_character']
    if not 0<=a<b<=len(source) or text_hash(source[a:b])!=span['raw_fragment_sha256']:raise ValueError('Original function span differs')
    lo,hi=len(source[:a].encode()),len(source[:b].encode());owner,sep,name=qualified.rpartition('.')
    if not sep:raise ValueError('Qualified function required')
    owners=[x for x in walk(ast) if x.get('nodeType')=='ContractDefinition' and x.get('name')==owner]
    if len(owners)!=1:raise ValueError('Unique compiler owner required')
    functions=[x for x in owners[0].get('nodes',[]) if x.get('nodeType')=='FunctionDefinition' and x.get('name')==name and bounds(x,source_id,len(raw))==(lo,hi)]
    if len(functions)!=1:raise ValueError('Unique exact compiler function required')
    fn=functions[0];calls=[];unresolved=[];unsupported_body_forms=sorted({x['nodeType'] for x in walk(functions[0].get('body')) if x.get('nodeType')=='InlineAssembly'})
    for node in walk(fn.get('body')):
        resolved=native_call(node)
        if resolved is None:
            # A bool invocation whose callee expression contains a native target is ambiguous.
            if node.get('nodeType')=='FunctionCall' and node.get('typeDescriptions',{}).get('typeIdentifier')=='t_bool' and (node.get('expression',{}).get('typeDescriptions',{}).get('typeIdentifier','').startswith(('t_function_barecall_','t_function_send_')) or any(x.get('nodeType')=='MemberAccess' and x.get('memberName') in {'call','send'} and x.get('expression',{}).get('typeDescriptions',{}).get('typeIdentifier')=='t_address' for x in walk(node.get('expression',{})))):unresolved.append(node.get('id'))
            continue
        access,kind,wrappers=resolved;cs,ce=bounds(node,source_id,len(raw));ms,me=bounds(access,source_id,len(raw))
        if not lo<=cs<=ms<me<=ce<=hi or raw[me-len(kind):me]!=kind.encode():raise ValueError('Native call span differs')
        guards=strong_guards(fn,node['id']);evidence={'compiler_call_id':node['id'],'native_member':kind,'wrappers_outer_to_inner':wrappers,'call_byte_span':[cs,ce],'call_sha256':text_hash(raw[cs:ce].decode()),'strong_guard_status':'unresolved_or_nonreverting','guard_evidence':None}
        if len(guards)==1:
            guard_kind,guard=guards[0];gs,ge=bounds(guard,source_id,len(raw))
            if not lo<=gs<=cs<ce<=ge<=hi:raise ValueError('Guard outside function/call')
            if retains_call(model_code,raw[gs:ge].decode()):evidence.update(strong_guard_status=guard_kind,guard_evidence={'byte_span':[gs,ge],'raw_sha256':text_hash(raw[gs:ge].decode())})
        calls.append(evidence)
    status='supported_source_checked_contrast' if calls and not unresolved and not unsupported_body_forms and all(c['guard_evidence'] for c in calls) else ('no_local_target_call_unknown' if not calls and not unresolved and not unsupported_body_forms else 'unsupported_or_nonreverting_handling_unknown')
    return {'status':status,'typed_target_calls':calls,'unresolved_native_call_ids':unresolved,'unsupported_body_forms':unsupported_body_forms,'compiler_function_id':fn['id'],'function_byte_span':[lo,hi],'scope':'named_function_body_only;not_modifiers_callees_creation_or_global_UX_safety'}

def local_binary_example(model,target,*,for_training=False):
    if for_training:raise ValueError('Quality/evaluation/group/license/baseline gates not passed')
    if target['protocol']!=VERSION or target['target_definition_id']!=DEFINITION or target['source']!=SOURCE:raise ValueError('Explicit local-binary weak namespace required')
    if set(model)!={'record_id','model_code','model_code_sha256','input_protocol'} or model['record_id']!=target['record_id'] or text_hash(model['model_code'])!=model['model_code_sha256'] or target['model_code_sha256']!=model['model_code_sha256']:raise ValueError('Code-only identity differs')
    if target['independent_gold'] is not False or target['training_eligible'] is not False or target['evaluation_eligible'] is not False or target['split'] is not None or target['human_review_status']!='unreviewed':raise ValueError('No gold/eligibility/split promotion')
    labels=target['source_proxy_labels'];masks=target['source_proxy_observation_mask']
    if set(labels)!=set(CATEGORIES) or set(masks)!=set(CATEGORIES):raise ValueError('Partial three-category schema required')
    for c in CATEGORIES:
        if type(masks[c]) is not bool:raise ValueError('Boolean mask required')
        if not masks[c]:
            if labels[c] is not None:raise ValueError('Unknown is null, not negative')
        elif c!='unchecked_external_call' or type(labels[c]) is not bool:raise ValueError('Only local native-result target available')
        elif labels[c]:
            if target['native_UX'] is not True or not target['typed_site_evidence']:raise ValueError('Positive needs native source/evidence')
        else:
            proof=target['negative_admission_evidence']
            if target['native_UX'] is not False or not proof or proof['status']!='supported_source_checked_contrast' or not proof['typed_target_calls'] or proof['unresolved_native_call_ids'] or proof.get('unsupported_body_forms') or not all(x['guard_evidence'] for x in proof['typed_target_calls']):raise ValueError('Negative needs explicit source0 and complete strong local handling evidence')
    return model['model_code'],[float(labels[c]) if masks[c] else 0. for c in CATEGORIES],[masks[c] for c in CATEGORIES]

def prepare(pilot,source,positive,contexts,old_compiler,binary,output):
    pilot,source,positive,contexts,old_compiler,binary,output=[Path(x).resolve() for x in (pilot,source,positive,contexts,old_compiler,binary,output)]
    if output.exists():raise FileExistsError(output)
    roots=(pilot,source,positive,contexts,old_compiler,binary.parent)
    if any(root==output or root.is_relative_to(output) for root in roots) or any(output.is_relative_to(root) for root in roots[1:]) or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')):raise ValueError('Preserve originals/parents/compiler')
    delivered=verify_delivery(pilot);parents={}
    for key,root,name in [('source',source,'manifest.json'),('positive',positive,'manifest.json'),('contexts',contexts,'recovery_manifest.json'),('old_compiler',old_compiler,'manifest.json')]:
        m=verify_artifacts(root,name,'artifact_sha256')
        if m['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Delivery lineage differs')
        parents[key]=(root,name,sha256(root/name),m)
    if parents['positive'][3]['protocol']!=VERSION_V2 or parents['positive'][3]['parent_manifest_sha256']['source']!=parents['source'][2] or parents['positive'][3]['parent_manifest_sha256']['contexts']!=parents['contexts'][2] or parents['old_compiler'][3]['parent_manifest_sha256']!={'source':parents['source'][2],'contexts':parents['contexts'][2]} or parents['old_compiler'][3]['compiler_sha256']['0.4.25']!=COMPILER_SHA256:raise ValueError('Explicit parent ancestry differs')
    if sha256(binary)!=COMPILER_SHA256:raise ValueError('Installed compiler identity differs')
    source_rows={t['record_id']:t for t in jsonl_rows(source/'source_targets.jsonl') if t['source']==SOURCE};targets=list(jsonl_rows(positive/'source_proxy_targets.jsonl'));models={m['record_id']:m for m in jsonl_rows(positive/'model_inputs.jsonl')}
    if len(targets)!=len(models) or len(targets)>5000 or len({t['record_id'] for t in targets})!=len(targets) or set(models)!=set(source_rows):raise ValueError('Unique native-source frame required')
    frame=[t for t in source_rows.values() if t['mapping_available_for_source_supervision'] and t['native_source_assessments']['UX'] is False];digests=sorted({t['mapping_evidence']['context_sha256'] for t in frame})
    if len(frame)>7 or len(digests)>6:raise ValueError('Predeclared7function/6context budget exceeded')
    output.mkdir(parents=True,exist_ok=False);compdir=output/'compiler';compdir.mkdir();version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True,timeout=30);(compdir/'version.txt').write_text(version.stdout+version.stderr)
    if '0.4.25+' not in version.stdout:raise ValueError('Compiler version differs')
    cached_input=json.loads(safe_file(old_compiler,'actual_0.4.25_input.json').read_text());cached_source=cached_input['sources']['context.sol']['content'];cached_digest=text_hash(cached_source);reports={};asts={};decisions=[]
    for digest in digests:
        raw=safe_file(contexts,'contexts/'+digest+'.sol').read_bytes();text=raw.decode()
        if text_hash(text)!=digest:raise ValueError('Pinned raw context differs')
        if masked_source_and_imports(text)[1]:reports[digest]={'status':'imports_blocked'};continue
        reused=digest==cached_digest
        if reused:
            settings=cached_input['settings']
            if settings['optimizer']!={'enabled':False,'runs':200} or settings['evmVersion']!='byzantium' or 'ast' not in settings['outputSelection']['*'][''] or set(cached_input['sources'])!={'context.sol'}:raise ValueError('Cached semantic settings/AST request differ')
            for suffix in ['_input.json','_output.json','_stderr.txt']:shutil.copyfile(safe_file(old_compiler,'actual_0.4.25'+suffix),compdir/(digest+suffix))
            stdout=(compdir/(digest+'_output.json')).read_text();returncode=0
        else:
            data={'language':'Solidity','sources':{'context.sol':{'content':text}},'settings':{'optimizer':{'enabled':False,'runs':200},'evmVersion':'byzantium','outputSelection':{'*':{'':['ast']}}}};serialized=json.dumps(data,sort_keys=True)+'\n';(compdir/(digest+'_input.json')).write_text(serialized)
            process=subprocess.run([str(binary),'--standard-json'],input=serialized,capture_output=True,text=True,timeout=30,check=False);stdout=process.stdout;returncode=process.returncode;(compdir/(digest+'_output.json')).write_text(stdout);(compdir/(digest+'_stderr.txt')).write_text(process.stderr)
        parsed=json.loads(stdout);errors=parsed.get('errors',[]);ok=returncode==0 and not any(e.get('severity')=='error' for e in errors)
        reports[digest]={'status':'compiled' if ok else 'compiler_failed','returncode':returncode,'diagnostics':errors,'reused_verified_prior_AST':reused,'output_selection':'original_AST_plus_bytecode_superset_retained' if reused else 'AST_only'}
        if ok:asts[digest]=(text,parsed['sources']['context.sol'])
    positive_hashes={t['model_code_sha256'] for t in targets if t['source_proxy_observation_mask']['unchecked_external_call'] and t['source_proxy_labels']['unchecked_external_call'] is True};decision_by={}
    for t in sorted(frame,key=lambda t:t['record_id']):
        e=t['mapping_evidence'];digest=e['context_sha256'];status=reports[digest]['status'];decision={'record_id':t['record_id'],'native_UX':False,'qualified_identifier':e['qualified_identifier'],'context_sha256':digest,'source_mapping_evidence':e}
        if digest in asts:
            text,entry=asts[digest]
            try:decision.update(assess_body(text,entry['ast'],entry['id'],e['qualified_identifier'],e['function_span'],models[t['record_id']]['model_code']))
            except ValueError as error:decision.update(status='typed_function_unresolved_unknown',reason=str(error))
            if decision['status']=='supported_source_checked_contrast' and t['model_code_sha256'] in positive_hashes:decision['status']='exact_input_source_conflict_requires_adjudication'
        else:decision['status']=status+'_unknown'
        decisions.append(decision);decision_by[t['record_id']]=decision
    emitted=[]
    for original in sorted(targets,key=lambda t:t['record_id']):
        t=copy.deepcopy(original);t.update(protocol=VERSION,target_definition_id=DEFINITION,negative_admission_evidence=None,migration_parent_protocol=VERSION_V2)
        d=decision_by.get(t['record_id'])
        if d and d['status']=='supported_source_checked_contrast':
            if t['native_UX'] is not False or t['source_proxy_observation_mask']['unchecked_external_call']:raise ValueError('Explicit unobserved native0 contrast required')
            t['source_proxy_labels']['unchecked_external_call']=False;t['source_proxy_observation_mask']['unchecked_external_call']=True;t['negative_admission_evidence']=d;t['supervision_tier']='native_source0_plus_strong_local_check_weak_contrast'
        local_binary_example(models[t['record_id']],t);emitted.append(t)
    if verify_delivery(pilot)!=delivered or sha256(binary)!=COMPILER_SHA256:raise ValueError('Original/compiler changed')
    for root,name,digest,m in parents.values():
        if sha256(root/name)!=digest:raise ValueError('Parent changed')
        verify_artifacts(root,name,'artifact_sha256')
    observed=[t for t in emitted if t['source_proxy_observation_mask']['unchecked_external_call']];negative=[t for t in observed if t['source_proxy_labels']['unchecked_external_call'] is False]
    summary={'protocol':VERSION,'native_rows':len(emitted),'native0_frame':len(frame),'context_frame':len(digests),'decision_counts':dict(Counter(x['status'] for x in decisions)),'compiler_status_counts':dict(Counter(x['status'] for x in reports.values())),'cached_contexts_reused':sum(x.get('reused_verified_prior_AST',False) for x in reports.values()),'new_context_compilations':sum(not x.get('reused_verified_prior_AST',False) for x in reports.values()),'source_positive_cells':len(observed)-len(negative),'source_negative_cells':len(negative),'negative_checked_sites':sum(len(t['negative_admission_evidence']['typed_target_calls']) for t in negative),'observed_existing_corpus_overlap_rows':sum(t['group_constraints']['direct_verified_context_corpus_overlap'] for t in observed),'independent_gold':0,'training_eligible':0,'evaluation_eligible':0,'assigned_splits':0,'training_or_inference':False}
    for name,rows in [('source_proxy_targets.jsonl',emitted),('model_inputs.jsonl',[models[k] for k in sorted(models)]),('negative_admission.jsonl',decisions)]:write_jsonl(output/name,rows)
    for name,data in [('summary.json',summary),('compiler_report.json',reports),('data_card.json',{'protocol':VERSION,'definition':DEFINITION,'target':'Local named-function-body native send/call bool-result discard only; no modifier/callee/creation/global-security assertion.','positive':'Preserved nativeUX1+typeddiscardedcall evidence from explicit v2 parent; no new positive labels.','negative':'Explicit nativeUX0, at least one local target call, complete typed body enumeration, every target immediately checked by direct builtin require or negated-if unconditional throw/revert, exact guard retained in input, no unresolved calls/input conflict. Weak source-specific contrast, not independent truth.','unknown':'No-call cases, inline assembly in the body, other handling styles, incomplete/compiler-error/conflict evidence, absent native annotations and other categories stay null/masked. No blanket safe or patched labels.','policy_limit':'Source-visible conservative admission, not blinded/random sample or measured label accuracy. Strong local checks do not establish broader UX source-label correctness, modifier/callee/creation safety or exploitability. Independent evaluation must not use this extraction rule as its truth oracle.','readiness':'No independent labels, eligibility, splits, model training or performance gain. Source-quality/adjudication, legitimate class/project support, license/provenance and independent grouped evaluation/baselines remain gates.'})]:(output/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    manifest={'protocol':VERSION,'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'delivered_file_hashes':delivered,'compiler_sha256':COMPILER_SHA256,'generator_sha256':sha256(Path(__file__)),'parent_manifest_sha256':{k:v[2] for k,v in parents.items()},'helper_sha256':{Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (walk,verify_delivery,verify_artifacts,write_jsonl)},'artifact_sha256':{str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*')) if f.is_file()}};(output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['pilot-dir','source-layer-dir','positive-layer-dir','context-release-dir','old-compiler-release-dir','compiler-binary','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.pilot_dir,a.source_layer_dir,a.positive_layer_dir,a.context_release_dir,a.old_compiler_release_dir,a.compiler_binary,a.output_dir),indent=2))

if __name__=='__main__':main()
