#!/usr/bin/env python3
"""Build auditable split constraints and template sensitivity groups, never splits."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from audit_function_pilot_handoff import inspect_staging_rows, jsonl_rows, sha256, verify_delivery
from prepare_function_review import text_hash, write_jsonl
from prepare_context_review import masked_source_and_imports, verify_artifacts, safe_file

VERSION='function-constraint-groups-v1'
# A frozen heuristic token recipe, not a compiler lexer or semantic-clone oracle.
TOKEN=re.compile(r'''(?P<literal>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?P<number>0[xX][0-9a-fA-F_]+|(?:[0-9][0-9_]*(?:\.[0-9_]+)?|\.[0-9_]+)(?:[eE][+-]?[0-9][0-9_]*)?)|(?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)|(?P<operator>>>=|<<=|\*\*=|=>|:=|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|&&|\|\||<<|>>|\*\*)|(?P<other>\S)''',re.DOTALL)
PRESERVE=set('function constructor fallback receive if else for while do break continue return returns public private internal external view pure payable constant immutable override virtual memory storage calldata address bool string mapping struct enum event error emit revert require assert new delete unchecked assembly let switch case default leave contract library interface is import from as pragma solidity using try catch anonymous indexed this super msg tx block abi type keccak256 sha256 ripemd160 ecrecover addmod mulmod selfdestruct suicide now gasleft unicode hex true false'.split())
TYPE=re.compile(r'(?:u?int|bytes)[0-9]*$')


def signature_hash(value):
    return text_hash(json.dumps(value,ensure_ascii=False,separators=(',',':')))


def signatures(code):
    masked,_,_=masked_source_and_imports(code)
    tokens=[(m.lastgroup,m.group()) for m in TOKEN.finditer(masked)]
    if not tokens:raise ValueError('No lexical tokens')
    template=[];identities={};previous=None
    for kind,value in tokens:
        if kind=='identifier' and value not in PRESERVE and not TYPE.fullmatch(value) and previous!='.':
            if value not in identities:identities[value]=len(identities)
            template.append((kind,'<id:'+str(identities[value])+'>'))
        else:template.append((kind,value))
        previous=value
    return signature_hash(tokens),signature_hash(template),len(tokens)


class UnionFind:
    def __init__(self,ids):self.parent={identity:identity for identity in ids}
    def find(self,value):
        while self.parent[value]!=value:
            self.parent[value]=self.parent[self.parent[value]];value=self.parent[value]
        return value
    def union(self,left,right):
        left,right=self.find(left),self.find(right)
        if left!=right:self.parent[max(left,right)]=min(left,right)
    def components(self):
        groups=defaultdict(list)
        for identity in sorted(self.parent):groups[self.find(identity)].append(identity)
        return sorted(groups.values(),key=lambda members:members[0])


def build_groups(rows,direct_overlap):
    ids=[row['record_id'] for row in rows]
    if not ids or len(set(ids))!=len(ids) or len(ids)>5000:raise ValueError('Expected 1-5000 unique candidate records')
    if not direct_overlap.issubset(ids):raise ValueError('Overlap seeds are not candidate identities')
    if sum(len(row.get('model_code','')) for row in rows)>20_000_000:raise ValueError('Text inventory exceeds 20M-character budget')
    by_id={row['record_id']:row for row in rows};buckets=defaultdict(list);features={}
    for row in sorted(rows,key=lambda r:r['record_id']):
        identity=row['record_id'];code=row.get('model_code')
        if not isinstance(code,str) or not code:raise ValueError('Missing function input')
        features[identity]={'exact_input_sha256':text_hash(code),'token_sha256':None,'template_sha256':None,'token_count':0,'signature_status':'unassessed'}
        buckets['exact_input_bytes',text_hash(code)].append(identity)
        try:
            token,template,count=signatures(code);features[identity].update(token_sha256=token,template_sha256=template,token_count=count,signature_status='computed')
            buckets['lexical_tokens',token].append(identity);buckets['identifier_template_heuristic',template].append(identity)
        except ValueError as error:features[identity]['signature_status']='lexical_error: '+str(error)
        project=row.get('project_id')
        if isinstance(project,str) and project.strip():buckets['source_project_metadata',signature_hash([row['source'],project])].append(identity)
        address=row.get('contract_address')
        if isinstance(address,str) and re.fullmatch('0x[0-9a-fA-F]{40}',address):buckets['source_recorded_address',signature_hash([row['source'],row['source_revision'],address.lower()])].append(identity)
        context=row.get('context_sha256')
        if context:
            if not isinstance(context,str) or not re.fullmatch('[0-9a-f]{64}',context):raise ValueError('Malformed pinned context identity')
            buckets['pinned_producer_context_hash',context].append(identity)
    evidence=UnionFind(ids);strict=UnionFind(ids);edges=[];edge_counts=Counter();group_counts=Counter()
    for (reason,key),members in sorted(buckets.items()):
        members=sorted(set(members))
        if len(members)<2:continue
        group_counts[reason]+=1
        for other in members[1:]:
            left=members[0];strict.union(left,other)
            if reason!='identifier_template_heuristic':evidence.union(left,other)
            edges.append({'left':left,'right':other,'reason':reason,'constraint_key_sha256':key,'graph':'strict_only' if reason=='identifier_template_heuristic' else 'evidence_and_strict'})
            edge_counts[reason]+=1
    maps={};component_rows=[];summaries={}
    for name,graph in [('evidence',evidence),('strict_template',strict)]:
        mapping={};components=[]
        for members in graph.components():
            component_id=signature_hash(members);projects=sorted({(by_id[x]['source'],by_id[x]['project_id']) for x in members if by_id[x].get('project_id')})
            seeds=sorted(set(members)&direct_overlap)
            entry={'graph':name,'component_id':component_id,'members':members,'size':len(members),
                   'source_counts':dict(Counter(by_id[x]['source'] for x in members)),'source_project_metadata_anchors':[list(x) for x in projects],
                   'direct_verified_context_overlap_records':seeds,'conservative_quarantine':bool(seeds),
                   'cross_source':len({by_id[x]['source'] for x in members})>1,'multiple_source_project_anchors':len(projects)>1}
            components.append(entry)
            for identity in members:mapping[identity]=(component_id,bool(seeds))
        component_rows.extend(components);maps[name]=mapping
        sizes=sorted((x['size'] for x in components),reverse=True)
        summaries[name]={'components':len(components),'non_singleton_components':sum(x['size']>1 for x in components),'largest_component':sizes[0],
                         'top_ten_sizes':sizes[:10],'cross_source_components':sum(x['cross_source'] for x in components),
                         'multi_project_metadata_components':sum(x['multiple_source_project_anchors'] for x in components),
                         'conservative_quarantine_candidates':sum(x['size'] for x in components if x['conservative_quarantine']),
                         'newly_propagated_quarantine_candidates':sum(x['size']-len(x['direct_verified_context_overlap_records']) for x in components if x['conservative_quarantine'])}
    membership=[]
    for identity in sorted(ids):
        row=by_id[identity];membership.append({'record_id':identity,'source':row['source'],'original_project_id':row.get('project_id'),
            'provenance_status':'source_project_metadata_only' if row.get('project_id') else ('source_address_identity_only' if row.get('contract_address') else 'unresolved'),
            'evidence_component':maps['evidence'][identity][0],'strict_template_component':maps['strict_template'][identity][0],
            'direct_verified_context_corpus_overlap':identity in direct_overlap,'evidence_conservative_quarantine':maps['evidence'][identity][1],
            'strict_template_conservative_quarantine':maps['strict_template'][identity][1],**features[identity],
            'pattern_labels':row.get('pattern_labels'),'training_eligible':False,'split':None})
    summary={'protocol':VERSION,'records':len(rows),'signature_status_counts':dict(Counter(x['signature_status'] for x in membership)),
             'unique_exact_input_hashes':len({x['exact_input_sha256'] for x in membership}),
             'unique_token_signatures':len({x['token_sha256'] for x in membership if x['token_sha256']}),
             'unique_identifier_template_signatures':len({x['template_sha256'] for x in membership if x['template_sha256']}),
             'records_without_original_project_id':sum(not r.get('project_id') for r in rows),'source_project_metadata_anchors':len({(r['source'],r['project_id']) for r in rows if r.get('project_id')}),
             'direct_verified_context_corpus_overlap':len(direct_overlap),'edge_counts':dict(edge_counts),'constraint_bucket_counts':dict(group_counts),'graphs':summaries,
             'new_project_ids_inferred':0,'pattern_labels_promoted':0,'training_eligible_records':0,'assigned_splits':0,
             'limitations':['Evidence graph mixes exact input identities with explicitly source-asserted provenance/context metadata; it is not verified true-project or semantic-clone ground truth.',
                           'Identifier normalization is only a strict sensitivity constraint. Common boilerplate or different callees/types may overmerge unrelated projects; it never establishes shared provenance or label truth.',
                           'The frozen heuristic token recipe is not a Solidity compiler lexer. It preserves literal values, maximal operators and identifiers, but does not establish AST validity or semantic equivalence.',
                           'Exact/template grouping is not a comprehensive near-clone audit; different constants/edits/dependencies may remain unlinked.',
                           f'{len(direct_overlap)} direct corpus-presence seeds are distinct from conservative component quarantine propagation; propagated membership is not independently confirmed raw-source overlap.',
                           'No split, fresh holdout, license clearance, expert label or model performance is established. Unknown provenance/labels remain unresolved.']}
    return membership,edges,component_rows,summary


def prepare(pilot,recovery,output):
    pilot,recovery,output=[p.resolve() for p in (pilot,recovery,output)]
    if output==pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')) or output==recovery or output.is_relative_to(recovery):raise ValueError('Preserve originals and input release; use new versioned output')
    delivered=verify_delivery(pilot);input_identity=sha256(safe_file(recovery,'recovery_manifest.json'));manifest=verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    if manifest['delivery_manifest_sha256']!=sha256(pilot/'DELIVERY_SHA256.json'):raise ValueError('Recovery release belongs to different pilot')
    rows=list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'));inspect_staging_rows(rows)
    links=list(jsonl_rows(recovery/'candidate_context_links.jsonl'));by_id={row['record_id']:row for row in rows}
    if len(links)!=len(rows) or {row['record_id'] for row in links}!=set(by_id):raise ValueError('Recovery membership differs from candidate inventory')
    seeds=set();context_hash_cache={}
    for link in links:
        if not link.get('context_file'):continue
        original=by_id[link['record_id']];digest=link.get('pinned_context_sha256')
        if digest!=original.get('context_sha256'):raise ValueError('Overlap seed does not match original pinned context')
        path=safe_file(recovery,link['context_file'])
        if path not in context_hash_cache:context_hash_cache[path]=sha256(path)
        if context_hash_cache[path]!=digest:raise ValueError('Overlap seed lacks exact source-text evidence')
        seeds.add(link['record_id'])
    members,edges,components,summary=build_groups(rows,seeds)
    if verify_delivery(pilot)!=delivered or sha256(recovery/'recovery_manifest.json')!=input_identity:raise ValueError('Input identities changed during grouping')
    verify_artifacts(recovery,'recovery_manifest.json','artifact_sha256')
    output.mkdir(parents=True,exist_ok=False);write_jsonl(output/'candidate_groups.jsonl',members);write_jsonl(output/'constraint_edges.jsonl',edges);write_jsonl(output/'components.jsonl',components)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    release={'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),'comment_masker_sha256':sha256(Path(masked_source_and_imports.__code__.co_filename)),'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),'recovery_manifest_sha256':input_identity,
             'tokenizer_version':'literal-operator-preserving-heuristic-v1','template_preserved_identifiers':sorted(PRESERVE),'template_member_selectors_preserved':True,
             'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(release,indent=2,sort_keys=True)+'\n');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','recovery-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.pilot_dir,args.recovery_dir,args.output_dir),indent=2))

if __name__=='__main__':main()
