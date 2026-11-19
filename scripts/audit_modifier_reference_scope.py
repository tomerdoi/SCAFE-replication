"""Audit compiler reference locality; not a security detector or name resolver."""
from collections import Counter,defaultdict

def audit(payload):
 if any(e.get('severity')=='error' for e in payload.get('errors',[])):raise ValueError('Compiler errors: no qualified oracle')
 sources=payload.get('sources',{});nodes={};owner={};source_ids={};invocations=[]
 if not sources:raise ValueError('No compiler sources')
 def walk(x,contract=None):
  if isinstance(x,list):
   for y in x:walk(y,contract)
  elif isinstance(x,dict):
   if x.get('nodeType')=='ContractDefinition':contract=x.get('id')
   if 'nodeType' in x and 'id' in x:
    i=x['id']
    if not isinstance(i,int) or i in nodes:raise ValueError('Invalid or duplicate AST node ID')
    nodes[i]=x;owner[i]=contract
   if x.get('nodeType')=='ModifierInvocation':invocations.append((x,contract))
   for y in x.values():
    if isinstance(y,(dict,list)):walk(y,contract)
 for name,data in sources.items():
  i=data.get('id')
  if not isinstance(i,int) or i in source_ids or 'ast' not in data:raise ValueError('Missing/duplicate source mapping')
  if not isinstance(data['ast'],dict) or 'nodeType' not in data['ast'] or 'id' not in data['ast']:raise ValueError('Unobserved AST root is not genuine absence')
  source_ids[i]=name;walk(data['ast'])
 def location(n):
  try:a,b,s=map(int,n['src'].split(':'))
  except (KeyError,ValueError,TypeError):raise ValueError('Invalid source span')
  if a<0 or b<0 or s not in source_ids:raise ValueError('Unbound source span')
  return (s,a,b)
 defs=defaultdict(list)
 for i,n in nodes.items():
  if n.get('nodeType')=='ModifierDefinition':defs[location(n)[0],n.get('name')].append(i)
 counts=Counter();rows=[]
 for n,c in invocations:
  counts['all_invocation_nodes']+=1;ref=n.get('modifierName',{}).get('referencedDeclaration')
  if not isinstance(ref,int) or ref not in nodes:
   counts['reference_unknown']+=1;continue
  target=nodes[ref];kind=target.get('nodeType')
  if kind=='ContractDefinition':counts['base_constructor_excluded']+=1;continue
  if kind!='ModifierDefinition':counts['unsupported_reference_kind']+=1;continue
  qloc=location(n);dloc=location(target);counts['genuine_modifier_references']+=1
  local=qloc[0]==dloc[0];counts['same_source' if local else 'cross_source']+=1
  counts['own_contract' if owner[ref]==c else 'other_contract']+=1
  query_name=n.get('modifierName',{}).get('name') or n.get('modifierName',{}).get('namePath')
  candidates=defs[qloc[0],query_name] if isinstance(query_name,str) else []
  naive='unknown' if len(candidates)!=1 else ('correct' if candidates[0]==ref else 'wrong')
  counts['file_name_'+naive]+=1
  direct=[i for i in candidates if owner[i]==c]
  abstain='unknown' if len(direct)!=1 else ('correct' if direct[0]==ref else 'wrong')
  counts['direct_only_'+abstain]+=1
  rows.append({'query_node':n['id'],'target_node':ref,'query_source_id':qloc[0],'target_source_id':dloc[0],'query_contract':c,'target_contract':owner[ref],'declaration_span':list(dloc),'file_name_status':naive,'direct_only_status':abstain})
 keys=['all_invocation_nodes','base_constructor_excluded','reference_unknown','unsupported_reference_kind','genuine_modifier_references','same_source','cross_source','own_contract','other_contract','file_name_correct','file_name_wrong','file_name_unknown','direct_only_correct','direct_only_wrong','direct_only_unknown']
 out={k:counts[k] for k in keys};out['source_units']=len(sources);out['modifier_declarations']=sum(len(v) for v in defs.values());out['rows']=rows
 assert out['genuine_modifier_references']==out['same_source']+out['cross_source']==sum(out['file_name_'+k] for k in ['correct','wrong','unknown'])
 answered=out['file_name_correct']+out['file_name_wrong']
 out['file_name_answered_fraction']=answered/out['genuine_modifier_references'] if out['genuine_modifier_references'] else None
 out['file_name_accuracy_on_answered']=out['file_name_correct']/answered if answered else None
 out['role']='Compiler-oracle locality and two explicitly limited file-local policies, not selective recovery/full-project algorithm accuracy, security labels or end-to-end cost'
 return out

if __name__=='__main__':
 import argparse,json,hashlib
 from pathlib import Path
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists() or args.output.is_symlink():raise FileExistsError('Output must be new')
 if args.compiler_output.stat().st_size>32*1024*1024:raise ValueError('Oversized compiler artifact')
 raw=args.compiler_output.read_bytes();result=audit(json.loads(raw));result['compiler_output_sha256']=hashlib.sha256(raw).hexdigest();result['oracle_lineage_warning']='Caller must separately qualify exact source/compiler manifest; AST data alone does not clear source use or independent evaluation.'
 with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
