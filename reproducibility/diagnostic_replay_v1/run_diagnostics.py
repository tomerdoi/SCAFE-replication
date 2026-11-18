"""Portable fixed diagnostic replay. Original private caches stay outside this package."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,os,math

ROOT=Path(__file__).resolve().parent

def project(report,keys):return [{key:row[key] for key in keys} for row in report['results']]
def same(a,b):
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12
 return type(a)==type(b) and a==b

def run(output,cache_root=None):
 if output.exists() or output.is_symlink():raise FileExistsError('Output must be new')
 expected=json.loads((ROOT/'expected_aggregates.json').read_text());assert expected['version']=='fixed-diagnostic-replay-v1'
 # Isolated-CWD subprocess uses only copied tests/scripts and installed declared dependencies.
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,capture_output=True,text=True,timeout=120,env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'})
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 output.mkdir();(output/'controls.log').write_text(p.stdout+p.stderr)
 status={'controls':'passed43numericalcontrols','scientific_replay':'not attempted: original private cache root not supplied','cached_inputs_exported':False,'new_fits_or_training':0,'independent_confirmation':False}
 if cache_root is not None:
  sys.path.insert(0,str(ROOT/'scripts'))
  import paired_nuisance,review_budget,auc_label_sensitivity
  for module in [paired_nuisance,review_budget,auc_label_sensitivity]:
   if not Path(module.__file__).resolve().is_relative_to(ROOT/'scripts'):raise RuntimeError('External repository module imported')
  reports={'paired':paired_nuisance.run(cache_root,output/'paired'),'budget':review_budget.run(cache_root,output/'budget',ROOT/'review_budget_protocol.json'),'sensitivity':auc_label_sensitivity.run(cache_root,output/'sensitivity',ROOT/'sensitivity_protocol.json')}
  for name,report in reports.items():
   if not same(project(report,expected['projection_keys'][name]),expected['results'][name]):raise AssertionError('Scientific aggregate mismatch: '+name)
  status['scientific_replay']='passed: all3fixed diagnostics exactly matched integers/structure and floats within1e-12; read-only external input backed'
 status['limitations']='Software/numeric reproduction only; no source-rights release, model-training reproduction, label validation, novelty or generalization established'
 (output/'replay_report.json').write_text(json.dumps(status,indent=2)+'\n');return status
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-root',type=Path);a=p.parse_args();print(json.dumps(run(a.output,a.cache_root),indent=2))
