#!/usr/bin/env python3
"""Paired fixed-score retrospective nuisance contrast; no models or sources loaded."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS=('DIVE','Empirical-Analysis')
SEEDS=(42,43,44,45,46)
NUISANCE_FEATURES=('n_files','loc','comment_ratio','n_total_tokens','n_unique_tokens','n_duplicated_lines')

def digest(path: Path) -> str:
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load_targets(path: Path) -> dict:
 result={name:{} for name in TARGETS}
 with path.open(encoding='utf-8',newline='') as stream:
  reader=csv.reader(stream);header=next(reader)
  required=('dataset','model','contract_id','y_true','y_prob','random_state')
  if len(header)!=len(set(header)) or not set(required).issubset(header):raise ValueError('Invalid prediction header')
  idx={key:header.index(key) for key in required}
  for row in reader:
   # Exclude all other datasets before interpreting or retaining outcome fields.
   dataset=row[idx['dataset']]
   if dataset not in result:continue
   if len(row)!=len(header) or row[idx['model']]!='et':raise ValueError('Unexpected target row/model')
   cid=row[idx['contract_id']];seed=int(row[idx['random_state']]);label=float(row[idx['y_true']]);score=float(row[idx['y_prob']])
   if not cid or seed not in SEEDS or label not in (0.,1.) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid observed target prediction')
   key=(cid,seed)
   if key in result[dataset]:raise ValueError('Duplicate target/identity/seed')
   result[dataset][key]=(int(label),score)
 return result

def align(full: dict, nuisance: dict) -> dict:
 if full.keys()!=nuisance.keys() or set(full)!=set(TARGETS):raise ValueError('Target sets differ')
 aligned={}
 for name in TARGETS:
  a,b=full[name],nuisance[name]
  if not a or a.keys()!=b.keys():raise ValueError('Cohort or seed keys differ; no intersection permitted')
  ids=sorted({cid for cid,_ in a})
  if set(a)!={(cid,seed) for cid in ids for seed in SEEDS}:raise ValueError('Incomplete seed coverage')
  labels=[];x=[];z=[]
  for cid in ids:
   observed={a[cid,seed][0] for seed in SEEDS}|{b[cid,seed][0] for seed in SEEDS}
   if len(observed)!=1:raise ValueError('Label conflict between seeds or views')
   labels.append(observed.pop());x.append([a[cid,s][1] for s in SEEDS]);z.append([b[cid,s][1] for s in SEEDS])
  y=np.asarray(labels,dtype=np.int8)
  if len(np.unique(y))!=2:raise ValueError('Both observed classes required')
  aligned[name]={'y':y,'full':np.asarray(x),'nuisance':np.asarray(z),'ordered_identity_sha256':hashlib.sha256('\0'.join(ids).encode()).hexdigest()}
 return aligned

def metrics(y: np.ndarray, score: np.ndarray) -> np.ndarray:
 return np.asarray([roc_auc_score(y,score),average_precision_score(y,score)],dtype=float)

def paired(y: np.ndarray, full: np.ndarray, nuisance: np.ndarray, *, repetitions: int, rng: np.random.Generator, deadline: float) -> dict:
 if len(y)!=len(full) or len(y)!=len(nuisance) or repetitions<100:raise ValueError('Invalid paired dimensions/budget')
 samples=np.empty((repetitions,2));attempts=0
 for i in range(repetitions):
  if time.process_time()>deadline:raise TimeoutError('Predeclared bootstrap CPU budget exceeded')
  while True:
   index=rng.integers(0,len(y),size=len(y));attempts+=1
   if len(np.unique(y[index]))==2:break
   if attempts>repetitions*10:raise RuntimeError('Degenerate bootstrap cohort')
  samples[i]=metrics(y[index],full[index])-metrics(y[index],nuisance[index])
 return {'full':metrics(y,full).tolist(),'nuisance':metrics(y,nuisance).tolist(),'difference':(metrics(y,full)-metrics(y,nuisance)).tolist(),'lower':np.percentile(samples,2.5,axis=0).tolist(),'upper':np.percentile(samples,97.5,axis=0).tolist(),'draw_attempts':attempts,'repetitions':repetitions}

def run(root: Path, output: Path, *, repetitions: int=1000, seed: int=20261117) -> dict:
 if output.exists():raise FileExistsError('Output must be new')
 base=root/'data/derived/sanitized_v1/ml_results';paths=[base/'primary_lodo/scafe_repeated_lodo_predictions.csv',base/'nuisance_baseline/predictions.csv']
 hashes=[digest(p) for p in paths];p=json.loads((base/'primary_lodo/scafe_repeated_lodo.json').read_text());q=json.loads((base/'nuisance_baseline/protocol.json').read_text())
 if p['input_sha256']!=q['input_sha256'] or p['seeds']!=list(SEEDS) or q['seeds']!=list(SEEDS) or p['feature_count']!=244 or tuple(q['features'])!=NUISANCE_FEATURES:raise ValueError('Protocol/input mismatch')
 started=time.process_time();a=align(load_targets(paths[0]),load_targets(paths[1]));assert {name:len(a[name]['y']) for name in TARGETS}=={'DIVE':21939,'Empirical-Analysis':2172},'Current target counts differ from pinned manuscript cohorts';rng=np.random.default_rng(seed);results=[]
 for name in TARGETS:
  y=a[name]['y'];full=a[name]['full'];nuisance=a[name]['nuisance'];x=full.mean(axis=1);z=nuisance.mean(axis=1)
  result=paired(y,x,z,repetitions=repetitions,rng=rng,deadline=started+120.)
  seed_diff=[float(roc_auc_score(y,full[:,i])-roc_auc_score(y,nuisance[:,i])) for i in range(len(SEEDS))]
  results.append({'dataset':name,'n_records':len(y),'n_positive':int(y.sum()),'positive_rate':float(y.mean()),'ordered_identity_sha256':a[name]['ordered_identity_sha256'],'all_paired_labels_match':True,'seed_count':len(SEEDS),'paired_per_seed_auc_differences':seed_diff,'per_seed_difference_range':[min(seed_diff),max(seed_diff)],**result})
  print('Completed paired target',name,flush=True)
 # Invariance check on one declared bounded subset, no outcome-selected cases.
 item=a[TARGETS[0]];y=item['y'];control=paired(y,item['full'].mean(axis=1),item['full'].mean(axis=1),repetitions=100,rng=np.random.default_rng(seed+1),deadline=started+120.)
 assert control['difference']==control['lower']==control['upper']==[0.,0.]
 assert hashes==[digest(p) for p in paths]
 output.mkdir(parents=True)
 report={'profile':'retrospective-fixed-score-paired-nuisance-v1','input_files':[{'repository_path':str(p.relative_to(root)),'sha256':h} for p,h in zip(paths,hashes)],'original_scored_input_sha256':p['input_sha256'],'targets':list(TARGETS),'seed':seed,'repetitions':repetitions,'metric_order':['ROC-AUC','average_precision'],'estimand':'difference of mean-probability fixed-score metrics, full minus nuisance; distinct from mean per-seed metrics','sampling':'paired iid record resampling, marginal percentile 95% intervals; descriptive four contrasts, no multiplicity-adjusted or project/corpus/annotation/training/generalization uncertainty','cohort_role':'previously inspected retrospective development/regression evidence, not fresh confirmation','model_fitting_or_selection':False,'source_code_or_finding_access':False,'dappscan_or_other_untouched_reserved_outcomes_retained_or_used':False,'new_labels_or_eligibility':0,'identity_control_zero':True,'input_hashes_unchanged':True,'cpu_seconds':time.process_time()-started,'results':results}
 (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 return report

def main() -> None:
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=run(args.root,args.output);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
