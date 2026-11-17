#!/usr/bin/env python3
"""Fixed-score retrospective observed-label ranking at prespecified record budgets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from paired_nuisance import TARGETS, SEEDS, NUISANCE_FEATURES, digest, load_targets, align

FRACTIONS=(.01,.05,.10,.20)

class FixedRanking:
    """Exact score groups; boundary ties never use label-favorable ordering."""
    def __init__(self, labels, scores):
        y=np.asarray(labels);s=np.asarray(scores,dtype=float)
        if y.ndim!=1 or s.ndim!=1 or not len(y) or len(y)!=len(s):raise ValueError('Nonempty aligned vectors required')
        if not np.all(np.isin(y,[0,1])) or not np.all(np.isfinite(s)):raise ValueError('Observed binary labels and finite scores required')
        self.y=y.astype(np.int8);self.n=len(y);self.p=int(self.y.sum())
        order=np.argsort(-s,kind='stable');new=np.r_[True,s[order][1:]!=s[order][:-1]]
        self.groups=int(new.sum());self.group=np.empty(self.n,dtype=np.int64);self.group[order]=np.cumsum(new)-1

    def at(self, ks, weights=None):
        ks=np.asarray(ks)
        if ks.ndim!=1 or not len(ks) or not np.all(np.isfinite(ks)) or not np.all(ks==np.floor(ks)) or np.any(ks<1) or np.any(ks>self.n):raise ValueError('Integer budgets in [1,N] required')
        ks=ks.astype(np.int64)
        if weights is None:w=np.ones(self.n,dtype=np.int64)
        else:
            w=np.asarray(weights)
            if w.shape!=(self.n,) or not np.all(np.isfinite(w)) or np.any(w<0) or not np.all(w==np.floor(w)) or w.sum()!=self.n:raise ValueError('Bootstrap integer multiplicities must sum to N')
        count=np.bincount(self.group,weights=w,minlength=self.groups)
        positive=np.bincount(self.group,weights=w*self.y,minlength=self.groups)
        cumulative=np.cumsum(count);cum_positive=np.cumsum(positive);out=[]
        for k in ks:
            boundary=int(np.searchsorted(cumulative,k,side='left'))
            higher=int(cumulative[boundary-1]) if boundary else 0
            higher_positive=float(cum_positive[boundary-1]) if boundary else 0.
            slots=int(k)-higher;size=int(count[boundary]);positives=int(positive[boundary])
            assert size>0 and 0<slots<=size
            expected=higher_positive+slots*positives/size
            out.append({'k':int(k),'expected_observed_positive':float(expected),'precision':float(expected/k),'strictly_higher_records':higher,'boundary_tie_records':size,'boundary_tie_positive':positives,'boundary_slots':slots})
        return out

def budgets(n, fractions=FRACTIONS):
    if type(n) is not int or n<1 or tuple(fractions)!=FRACTIONS:raise ValueError('Frozen positive N/four-budget protocol required')
    return [max(1,int(np.floor(f*n))) for f in fractions]

def compare(y, full, nuisance, *, repetitions=1000, seed=2026111703, deadline=float('inf')):
    a=FixedRanking(y,full);b=FixedRanking(y,nuisance)
    if a.p==0:raise ValueError('Observed-positive recall needs positive support')
    if type(repetitions) is not int or repetitions<100:raise ValueError('At least100 prespecified paired draws required')
    ks=budgets(a.n);x=a.at(ks);z=b.at(ks);rng=np.random.default_rng(seed);samples=np.empty((repetitions,len(ks)))
    for i in range(repetitions):
        if time.process_time()>deadline:raise TimeoutError('Prespecified CPU budget exceeded')
        weights=np.bincount(rng.integers(0,a.n,size=a.n),minlength=a.n)
        samples[i]=[u['precision']-v['precision'] for u,v in zip(a.at(ks,weights),b.at(ks,weights))]
    rows=[]
    for j,(fraction,k,u,v) in enumerate(zip(FRACTIONS,ks,x,z)):
        prevalence=a.p/a.n
        for q in [u,v]:
            q['recall']=q['expected_observed_positive']/a.p
            q['precision_lift_over_random']=q['precision']/prevalence
            q['positive_count_gain_over_random']=q['expected_observed_positive']-k*prevalence
        rows.append({'requested_fraction':fraction,'k':k,'actual_fraction':k/a.n,'full':u,'nuisance':v,'random':{'expected_observed_positive':k*prevalence,'precision':prevalence,'recall':k/a.n},'precision_difference':u['precision']-v['precision'],'precision_difference_lower':float(np.percentile(samples[:,j],2.5)),'precision_difference_upper':float(np.percentile(samples[:,j],97.5))})
    return {'n_records':a.n,'n_positive':a.p,'positive_rate':a.p/a.n,'repetitions':repetitions,'seed':seed,'rows':rows}

def run(root:Path, output:Path, protocol_path:Path):
    if output.exists():raise FileExistsError('Output must be new')
    protocol=json.loads(protocol_path.read_text());frozen=digest(protocol_path)
    if protocol['protocol']!='fixed-score-review-budget-v1' or tuple(protocol['budget_fractions'])!=FRACTIONS or protocol['rng_seed']!=2026111703:raise ValueError('Unexpected frozen protocol')
    base=root/'data/derived/sanitized_v1/ml_results'
    paths=[base/'primary_lodo/scafe_repeated_lodo_predictions.csv',base/'nuisance_baseline/predictions.csv']
    hashes=[digest(p) for p in paths]
    if hashes!=[protocol['inputs']['full'],protocol['inputs']['nuisance']]:raise ValueError('Cache identity changed')
    meta=json.loads((base/'primary_lodo/scafe_repeated_lodo.json').read_text());other=json.loads((base/'nuisance_baseline/protocol.json').read_text())
    if meta['input_sha256']!=other['input_sha256'] or meta['input_sha256']!=protocol['inputs']['original_scored_input'] or meta['feature_count']!=244 or meta['seeds']!=list(SEEDS) or other['seeds']!=list(SEEDS) or tuple(other['features'])!=NUISANCE_FEATURES:raise ValueError('Original scored-input/feature/seed lineage differs')
    start=time.process_time();aligned=align(load_targets(paths[0]),load_targets(paths[1]))
    if {k:len(v['y']) for k,v in aligned.items()}!={'DIVE':21939,'Empirical-Analysis':2172}:raise ValueError('Pinned target cohorts differ')
    results=[]
    for i,name in enumerate(TARGETS):
        item=aligned[name]
        r=compare(item['y'],item['full'].mean(axis=1),item['nuisance'].mean(axis=1),seed=protocol['rng_seed'],deadline=start+120.)
        r.update(dataset=name,ordered_identity_sha256=item['ordered_identity_sha256'],all_paired_labels_seeds_match=True);results.append(r)
        print('Completed fixed review budgets',name,flush=True)
    control_item=aligned[TARGETS[0]];control_scores=control_item['full'].mean(axis=1)
    control=compare(control_item['y'],control_scores,control_scores,repetitions=100,seed=protocol['rng_seed'],deadline=start+120.)
    if not all(row['precision_difference']==row['precision_difference_lower']==row['precision_difference_upper']==0. for row in control['rows']):raise AssertionError('Real identical-score control failed')
    if digest(protocol_path)!=frozen or hashes!=[digest(p) for p in paths]:raise ValueError('Input changed during analysis')
    report={'protocol':'fixed-score-review-budget-v1','protocol_sha256':frozen,'input_files':[{'path':str(p.relative_to(root)),'sha256':h} for p,h in zip(paths,hashes)],'original_scored_input_sha256':meta['input_sha256'],'results':results,'cpu_seconds':time.process_time()-start,'cohort_role':'Previously inspected retrospective/regression evidence; not fresh independent confirmation','outcomes':'Native observed binary labels, not independent vulnerability/exploit/security truth or measured human-review success','uncertainty':'Eight marginal conditional paired iid-record percentile95% precision-difference intervals; no joint test, multiplicity adjustment or project/corpus/annotation/training uncertainty','ties':'Exact fixed mean-score boundary ties, expected count under uniform selection within boundary tie','dappscan_outcomes_used_or_retained':False,'new_source_labels_eligibility_or_model_fits':0,'input_hashes_unchanged':True,'real_identical_score_100draw_control_zero':True}
    output.mkdir(parents=True);(output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();r=run(a.root,a.output,a.protocol);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
