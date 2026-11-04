#!/usr/bin/env python3
"""Offline, hash-bound compiler call-site diagnostic; never execute contracts or relabel."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from adapt_function_source_labels import VERSION as SOURCE_VERSION, DEFINITION
from prepare_context_review import safe_file, verify_artifacts, masked_source_and_imports

VERSION = 'function-compiler-call-evidence-v1'
VERSIONS = ('0.4.24','0.4.25','0.4.26','0.5.3')
CALL_OPS = {0xf1:'CALL',0xfa:'STATICCALL',0xf2:'CALLCODE',0xf4:'DELEGATECALL'}
CONTROL = ('pragma solidity >=0.4.24 <0.6.0;\n'
           'interface Book { function getPlayerID(address a) external returns(uint); '
           'function getPlayerName(uint p) external view returns(bytes32); }\n'
           'contract Probe { function probe(Book book,uint pid) public returns(bytes32) { '
           'book.getPlayerID(msg.sender); return book.getPlayerName(pid); } }\n')
CONTROL_EXPRESSIONS = ('book.getPlayerID(msg.sender)','book.getPlayerName(pid)')


def expanded_source_maps(encoded):
    if not isinstance(encoded,str) or not encoded:raise ValueError('Missing source map')
    previous=[-1,-1,-1,'-',0];result=[]
    for entry in encoded.split(';'):
        fields=entry.split(':')
        if len(fields)>5:raise ValueError('Unsupported source-map fields')
        for index,value in enumerate(fields):
            if value:
                previous[index]=value if index==3 else int(value)
        if min(previous[:3]) < -1 or previous[3] not in ('i','o','-'):
            raise ValueError('Invalid source-map entry')
        result.append(tuple(previous[:3]))
    return result


def instruction_prefix(bytecode,count):
    """Decode mapped instruction prefix, skipping PUSH data; ignore unmapped metadata."""
    if not isinstance(bytecode,str) or type(count) is not int or count<1:raise ValueError('Invalid bytecode/count')
    data=bytes.fromhex(bytecode);pc=0;instructions=[]
    for _ in range(count):
        if pc>=len(data):raise ValueError('More map entries than instructions')
        opcode=data[pc];width=1+(opcode-0x5f if 0x60<=opcode<=0x7f else 0)
        if pc+width>len(data):raise ValueError('Truncated PUSH instruction')
        instructions.append((pc,opcode));pc+=width
    return instructions,len(data)-pc


def mapped_calls(source,bytecode,source_map,source_id,expressions, *, character_bounds=None):
    masked,imports,_=masked_source_and_imports(source)
    if imports:raise ValueError('Imports prohibited in this offline diagnostic')
    lo,hi=character_bounds if character_bounds is not None else (0,len(source))
    if not 0<=lo<hi<=len(source):raise ValueError('Invalid source bounds')
    maps=expanded_source_maps(source_map);instructions,tail=instruction_prefix(bytecode,len(maps));result=[]
    for expression in expressions:
        if not expression or masked[lo:hi].count(expression)!=1:
            raise ValueError('Call expression must occur exactly once in the selected raw function')
        start=lo+masked[lo:hi].index(expression)
        if source[start:start+len(expression)]!=expression:raise ValueError('Expression is not exact source code')
        begin=len(source[:start].encode('utf-8'));end=begin+len(expression.encode('utf-8'));matches=[]
        for (pc,opcode),(offset,length,file_id) in zip(instructions,maps):
            if opcode in CALL_OPS and file_id==source_id and length>0 and begin<=offset and offset+length<=end:
                matches.append({'pc':pc,'opcode':CALL_OPS[opcode],'source_map_byte_start':offset,'source_map_byte_length':length,'source_id':file_id})
        if len(matches)!=1:raise ValueError('Require one unambiguous mapped external-call instruction per expression')
        result.append({'expression':expression,'expression_byte_start':begin,'expression_byte_length':end-begin,
                       'expression_start_line':source[:start].count('\n')+1,**matches[0]})
    return result,{'mapped_instructions':len(instructions),'unmapped_tail_bytes_not_analyzed':tail}


def compile_case(binary,version,source,contract,stem,output):
    if masked_source_and_imports(source)[1]:raise ValueError('Do not resolve any source imports')
    data={'language':'Solidity','sources':{'context.sol':{'content':source}},
          'settings':{'optimizer':{'enabled':False,'runs':200},'evmVersion':'byzantium',
                      'outputSelection':{'*':{contract:['evm.deployedBytecode.object','evm.deployedBytecode.sourceMap','evm.methodIdentifiers'],'':['ast']}}}}
    serialized=json.dumps(data,sort_keys=True)+'\n';(output/(stem+'_input.json')).write_text(serialized)
    process=subprocess.run([str(binary),'--standard-json'],input=serialized,capture_output=True,text=True,timeout=30,check=False)
    (output/(stem+'_output.json')).write_text(process.stdout);(output/(stem+'_stderr.txt')).write_text(process.stderr)
    parsed=json.loads(process.stdout);errors=parsed.get('errors',[])
    if process.returncode or any(e.get('severity')=='error' for e in errors):
        raise ValueError('Compiler failed; retain complete diagnostics, never patch or ignore errors')
    runtime=parsed['contracts']['context.sol'][contract]['evm']['deployedBytecode']
    return runtime,parsed['sources']['context.sol']['id'],{'compiler':version,'returncode':process.returncode,'warnings':errors,'runtime_sha256':hashlib.sha256(bytes.fromhex(runtime['object'])).hexdigest(),'input_sha256':sha256(output/(stem+'_input.json'))}


def prepare(pilot,source_release,contexts,compiler_root,config_path,output):
    pilot,source_release,contexts,compiler_root,config_path,output=[Path(p).resolve() for p in (pilot,source_release,contexts,compiler_root,config_path,output)]
    if output.exists():raise FileExistsError(output)
    if (output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives'))
        or any(output==p or output.is_relative_to(p) for p in (source_release,contexts,compiler_root))):
        raise ValueError('Preserve originals, ancestors and compiler installation')
    delivered=verify_delivery(pilot);sm=verify_artifacts(source_release,'manifest.json','artifact_sha256');cm=verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256')
    parents={'source':sha256(source_release/'manifest.json'),'contexts':sha256(contexts/'recovery_manifest.json')}
    if (sm['protocol']!=SOURCE_VERSION or sm['target_definition_id']!=DEFINITION
        or sm['parent_manifest_sha256']['contexts']!=parents['contexts']
        or sm['delivery_manifest_sha256']!=cm['delivery_manifest_sha256']
        or cm['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json')):
        raise ValueError('Inconsistent source/context lineage')
    config=json.loads(config_path.read_text());config_hash=sha256(config_path)
    if set(config['compiler_sha256'])!=set(VERSIONS):raise ValueError('Exact predeclared compiler matrix required')
    matches=[r for r in jsonl_rows(source_release/'source_targets.jsonl') if r['record_id']==config['record_id']]
    if len(matches)!=1 or not matches[0]['mapping_available_for_source_supervision']:raise ValueError('Unique mapped source target required')
    target=matches[0];evidence=target['mapping_evidence'];path=safe_file(contexts,evidence['context_artifact']);source_bytes=path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=evidence['context_sha256']:raise ValueError('Raw context hash changed')
    source=source_bytes.decode('utf-8');span=evidence['function_span'];fragment=source[span['start_character']:span['end_character']]
    if hashlib.sha256(fragment.encode()).hexdigest()!=span['raw_fragment_sha256']:raise ValueError('Raw target span changed')
    output.mkdir(parents=True,exist_ok=False);results=[];report={'status':'running','protocol':VERSION}
    try:
        (output/'probe_config.json').write_bytes(config_path.read_bytes())
        for version in VERSIONS:
            binary=(compiler_root/f'solc-{version}'/f'solc-{version}').resolve()
            if not binary.is_relative_to(compiler_root) or not binary.is_file() or sha256(binary)!=config['compiler_sha256'][version]:raise ValueError('Installed compiler identity mismatch')
            process=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True,timeout=30)
            (output/f'compiler_{version}_version.txt').write_text(process.stdout+process.stderr)
            if not re.search(r'\b'+re.escape(version)+r'\+',process.stdout):raise ValueError('Compiler version output differs')
            cases=[('control',CONTROL,'Probe',CONTROL_EXPRESSIONS,None)]
            if version!='0.5.3':cases.insert(0,('actual',source,config['contract'],config['expressions'],(span['start_character'],span['end_character'])))
            for name,text,contract,expressions,bounds in cases:
                runtime,sid,details=compile_case(binary,version,text,contract,f'{name}_{version}',output)
                calls,decoding=mapped_calls(text,runtime['object'],runtime['sourceMap'],sid,expressions,character_bounds=bounds)
                results.append({'case':name,**details,'calls':calls,'decoding':decoding})
            if sha256(binary)!=config['compiler_sha256'][version]:raise ValueError('Compiler changed during probe')
        if verify_delivery(pilot)!=delivered or sha256(path)!=evidence['context_sha256'] or sha256(config_path)!=config_hash:raise ValueError('Input changed during probe')
        for root,name,key in [(source_release,'manifest.json','source'),(contexts,'recovery_manifest.json','contexts')]:
            if sha256(root/name)!=parents[key]:raise ValueError('Parent changed')
            verify_artifacts(root,name,'artifact_sha256')
        expected_control=all([c['opcode'] for c in r['calls']]==(['CALL','STATICCALL'] if r['compiler']=='0.5.3' else ['CALL','CALL']) for r in results if r['case']=='control')
        all_old_actual_call=all(c['opcode']=='CALL' for r in results if r['case']=='actual' for c in r['calls'])
        report={'status':'completed','protocol':VERSION,'record_id':target['record_id'],'qualified_identifier':evidence['qualified_identifier'],
                'mapping_evidence':evidence,'source_claim_unchanged':True,'native_RE':target['native_source_assessments']['RE'],
                'independent_adjudication':None,'training_eligible':False,'evaluation_eligible':False,
                'compiler_flags':{'optimizer_enabled':False,'optimizer_runs':200,'evm_version':'byzantium'},
                'compilations':results,'control_checks_passed':expected_control,'all_three_old_actual_calls_use_CALL':all_old_actual_call,
                'limits':['Conditional recompilation/code-generation evidence; deployed compiler, bytecode, optimizer settings, callee behavior, caller reachability and security truth are not verified.',
                          'A CALL instruction is not proof of reentry or exploitability; prior source claims and independent review gates remain unchanged.']}
    except Exception as error:
        report={'status':'failed','protocol':VERSION,'completed_compilations':results,'error':str(error)};raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),
              'parent_manifest_sha256':parents,'config_sha256':config_hash,'compiler_sha256':config['compiler_sha256'],
              'helper_sha256':{Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (verify_delivery,verify_artifacts)},
              'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','source-release-dir','context-release-dir','compiler-root','config','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();report=prepare(a.pilot_dir,a.source_release_dir,a.context_release_dir,a.compiler_root,a.config,a.output_dir)
    print(json.dumps({'status':report['status'],'control_checks_passed':report['control_checks_passed'],'all_three_old_actual_calls_use_CALL':report['all_three_old_actual_calls_use_CALL'],'compilations':len(report['compilations'])}))

if __name__=='__main__':main()
