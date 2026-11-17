#!/usr/bin/env python3
"""Build a current, local-only aggregate-evidence review package; never copy corpora."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / 'sc_risk_score_overleaf'
BASE = 'data/derived/sanitized_v1/'
CSV_PATHS = (
 'ml_results/primary_lodo/scafe_repeated_lodo.csv',
 'ml_results/primary_lodo/scafe_repeated_lodo_summary.csv',
 'ml_results/primary_lodo/primary_record_bootstrap.csv',
 'ml_results/primary_development/scafe_within_corpus_summary.csv',
 'ml_results/nuisance_baseline/summary.csv',
 'ml_results/feature_ablations/scafe_feature_ablations.csv',
 'ml_results/lexical_extension/scafe_repeated_lodo_summary.csv',
 'ml_results/sampled_ensemble/ensemble_metrics.csv',
)
# Only metadata consumed by the two canonical aggregate validators is exported.
JSON_FIELDS = {
 'manifest.json': ('scored_csv_sha256',),
 'ml_results/primary_lodo/scafe_repeated_lodo.json': ('feature_count', 'protocol', 'input_sha256'),
 'ml_results/sampled_ensemble/ensemble_lock.json': ('scafe_model',),
 'shap_results/primary/metadata.json': ('model', 'feature_count'),
}
PROTOCOL_FIELDS = (
 'canonicalize_within_dataset','exclude_within_dataset_label_conflicts',
 'held_out_dataset','exclude_cross_dataset_source_duplicates',
 'exclude_cross_training_label_conflicts','collapse_cross_training_same_label_duplicates',
 'test_threshold_tuning','primary_model','external_targets',
)
LOCAL_NUMERIC_SHA = '720f1d86a195cfc5cb623877c9427ca544bc7a58406641b4d2473c26635293be'
FORBIDDEN_COLUMNS = {'source','code','source_code','source_path','path','contract_id','project_id','y_true','y_score','prediction','label','address','fingerprint'}
ALLOWED_COLUMNS = frozenset(('FN', 'FP', 'TN', 'TP', 'coverage', 'coverage_mean', 'coverage_seed_resample95_high', 'coverage_seed_resample95_low', 'coverage_std', 'dataset', 'delegation_fraction', 'experiment', 'f1', 'f1_mean', 'f1_seed_resample95_high', 'f1_seed_resample95_low', 'f1_std', 'feature_count', 'feature_variant', 'lift_at_10pct_mean', 'lift_at_1pct_mean', 'lift_at_5pct_mean', 'method', 'model', 'n_contracts', 'n_cross_training_conflict_rows_excluded', 'n_cross_training_duplicates_collapsed', 'n_duplicate_train_excluded', 'n_neg', 'n_pos', 'n_records', 'n_samples', 'n_seeds', 'n_train', 'positive_only', 'positive_prevalence', 'positive_rate', 'pr_auc', 'pr_auc_bootstrap95_high', 'pr_auc_bootstrap95_low', 'pr_auc_mean', 'pr_auc_seed_resample95_high', 'pr_auc_seed_resample95_low', 'pr_auc_std', 'precision', 'precision_at_10pct_mean', 'precision_at_1pct_mean', 'precision_at_5pct_mean', 'precision_mean', 'precision_seed_resample95_high', 'precision_seed_resample95_low', 'precision_std', 'prevalence', 'random_state', 'recall', 'recall_at_10pct_mean', 'recall_at_1pct_mean', 'recall_at_5pct_mean', 'recall_mean', 'recall_seed_resample95_high', 'recall_seed_resample95_low', 'recall_std', 'roc_auc', 'roc_auc_bootstrap95_high', 'roc_auc_bootstrap95_low', 'roc_auc_mean', 'roc_auc_seed_resample95_high', 'roc_auc_seed_resample95_low', 'roc_auc_std', 'role', 'threshold', 'train_datasets'))
TEXT_SUFFIXES = {'.py','.tex','.bib','.csv','.json','.md','.txt'}
PRIVATE = re.compile(r'/Users/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
# Fixed immutable numeric slice: 19 source snapshots, two nested compiler sources.
VERIFY_NUMERIC = '''from pathlib import Path
import json
d=json.loads(Path(__file__).with_name("structural_numeric_evidence.json").read_text())
s=d["comment_sources"]; a=d["ast_sources"]
assert len(s)==19 and len(a)==2
assert len({r["id"] for r in s})==19
assert {r["id"] for r in a}<={r["id"] for r in s}
n=[sum(r["legacy"][k]!=r["masked"][k] for k in r["legacy"]) for r in s]
assert all(len(r["legacy"])==14 for r in s)
assert all(r["tokens_preserved"] and r["noncomment_positions_literals_preserved"] and r["line_breaks_preserved"] for r in s)
assert all(r["controlled_pair_changed_focus_cells"]==0 and r["relational_negative_control_changed_cells"]==0 for r in s)
u=[v for r in a for v in r["counters"].values()]
x=[sum(i>0 for i in n),sum(n),len(s)*14,sum(v["original"]==v["reference"] for v in u),sum(v["comment_controlled"]==v["reference"] for v in u),len(u),sum(abs(v["original"]-v["reference"]) for v in u),sum(abs(v["comment_controlled"]-v["reference"]) for v in u)]
assert x==[14,31,266,5,8,8,13,0]
print("PASS: fixed local structural table arithmetic; not independent source or model validation")
'''
README = '''# SCAFE local aggregate-evidence review candidate

This is a LOCAL review package, not a published or submission-cleared archive.
It preserves the current manuscript structure and resolves its literal TeX/BibTeX/
figure dependencies. It contains canonical sanitized-v1 Extra Trees aggregate
metrics, minimal declared metadata projections and the fixed local structural
numeric slice. Historical DAppSCAN development/reverse-transfer statistics remain
historical evidence, never fresh confirmation or function-model training input.

With the existing pinned pandas environment, run from this directory:

    cd sc_review
    python scripts/validate_experiment_integrity.py
    python scripts/validate_submission_artifacts.py
    cd ../local_extraction
    python verify.py

No installation or model/source download is performed by these checks. The full
upstream reproduction environment is described separately in the owning repo.

## Scope and rights

No raw Solidity/corpus/report, structured per-record prediction/label exports,
source/project identity tables, feature matrix, checkpoints, credentials, Git
history, or private research ledger is included. The full manuscript retains its
existing local attribution-case illustrations; this package is NOT case-free and
those illustrated outputs still require distribution review. Metadata projections omit source IDs, local paths,
SHAP cases and unneeded fields; SOURCE_LINEAGE.json records original-file hashes,
projection rules and packaged hashes. Aggregate exports and source-code provenance
still require artifact-level distribution review; absence of raw code is not a
blanket rights grant. Every release/publication eligibility flag remains false.

These checks establish selected aggregate consistency and arithmetic. They do not
rerun training, independently establish labels, detect every near clone, reconstruct
source rights, reproduce plots from per-record data, prove anonymity of binary
assets, or validate all manuscript claims. Full per-record/source reproduction and
substantive contribution review remain separate unresolved gates. The retained PDF
is a local review output, not evidence that the paper is submission-ready.
'''

def sha(path: Path) -> str:
 return hashlib.sha256(path.read_bytes()).hexdigest()

def source_file(root: Path, relative: str) -> Path:
 path = root / relative
 if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
  raise ValueError('Missing, symbolic or escaping input: '+relative)
 if path.stat().st_size > 4*1024*1024:
  raise ValueError('Oversized review input: '+relative)
 return path

def manuscript_closure(root: Path) -> list[str]:
 pending = ['sc_vulnerability_detection.tex']; seen: set[str] = set()
 while pending:
  relative = pending.pop()
  if relative in seen: continue
  path = source_file(root, relative); seen.add(relative)
  if path.suffix != '.tex': continue
  text = path.read_text(encoding='utf-8')
  for command, raw in re.findall(r'\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^{}]+)\}', text):
   if any(x in raw for x in ('\\','#','$')): raise ValueError('Dynamic TeX dependency')
   names = raw.split(',') if command == 'bibliography' else [raw]
   for name in names:
    suffix = '.bib' if command == 'bibliography' else '.tex'
    candidate = name if Path(name).suffix else name+suffix
    source_file(root, candidate); pending.append(candidate)
 return sorted(seen | {'sc_vulnerability_detection.pdf'})

def metadata_projection(relative: str, value: dict) -> dict:
 projected = {k: value[k] for k in JSON_FIELDS[relative]}
 if 'protocol' in projected:
  projected['protocol'] = {k:value['protocol'][k] for k in PROTOCOL_FIELDS}
 return projected

def validate_aggregate_csv(path: Path) -> None:
 with path.open(encoding='utf-8', newline='') as stream:
  reader=csv.DictReader(stream); fields=reader.fieldnames or []
  if not fields or len(set(fields)) != len(fields) or not set(fields).issubset(ALLOWED_COLUMNS) or FORBIDDEN_COLUMNS.intersection(k.lower() for k in fields):
   raise ValueError('Per-record or unsupported CSV schema')
  count=0
  for row in reader:
   count+=1
   if count>200 or None in row: raise ValueError('Unexpected aggregate CSV shape')
   if any(v is None or len(v)>256 or PRIVATE.search(v) for v in row.values()):
    raise ValueError('Unexpected aggregate CSV content')
  if count==0: raise ValueError('Empty aggregate CSV')

def build(output: Path, numeric: Path, repo: Path=REPO, paper: Path=PAPER) -> dict:
 output=output.absolute()
 if output.exists() or output.is_symlink(): raise FileExistsError('Output must be new; never overwrite a review artifact')
 if output.resolve().is_relative_to(repo.resolve()) or output.resolve().is_relative_to(paper.resolve()):
  raise ValueError('Review output must be outside source repositories')
 if numeric.is_symlink() or sha(numeric)!=LOCAL_NUMERIC_SHA: raise ValueError('Unrecognized fixed local numeric evidence')
 closure=manuscript_closure(paper)
 # Check all declared inputs before creating output; never run downloaded/source code.
 inputs=[(source_file(repo,BASE+rel),'sc_review/'+BASE+rel,'aggregate_csv') for rel in CSV_PATHS]
 inputs += [(source_file(repo,BASE+rel),'sc_review/'+BASE+rel,'metadata_projection') for rel in JSON_FIELDS]
 inputs += [(source_file(repo,'scripts/'+name),'sc_review/scripts/'+name,'own_validator') for name in ('validate_experiment_integrity.py','validate_submission_artifacts.py')]
 inputs += [(source_file(paper,rel),'sc_risk_score_overleaf/'+rel,'manuscript_dependency') for rel in closure]
 for src,_,kind in inputs:
  if kind=='aggregate_csv':validate_aggregate_csv(src)
 output.mkdir(parents=True);lineage=[]
 for src,relative,kind in inputs:
  dst=output/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  original=sha(src)
  if kind=='metadata_projection':
   key=str(src.relative_to(repo/BASE));dst.write_text(json.dumps(metadata_projection(key,json.loads(src.read_text())),indent=2)+'\n')
  else:shutil.copyfile(src,dst)
  if sha(src)!=original:raise RuntimeError('Input changed during packaging')
  lineage.append({'package_path':relative,'kind':kind,'original_sha256':original,'package_sha256':sha(dst),'projection':'JSON field allowlist; omitted fields are not cleared for release' if kind=='metadata_projection' else 'byte-identical'})
 local=output/'local_extraction';local.mkdir();shutil.copyfile(numeric,local/'structural_numeric_evidence.json');(local/'verify.py').write_text(VERIFY_NUMERIC)
 (output/'README.md').write_text(README)
 (output/'SOURCE_LINEAGE.json').write_text(json.dumps({'files':lineage,'fixed_numeric_sha256':LOCAL_NUMERIC_SHA},indent=2)+'\n')
 meta={'profile':'current-sanitized-v1-aggregate-review-v2','publication_eligible':False,'source_use_clearance':False,'full_training_reproduction':False,'structured_per_record_prediction_or_label_exports_included':False,'manuscript_case_illustrations_retained':True,'case_free':False,'local_structural_snapshot_rows':19,'local_structural_ast_subset_rows':2,'raw_sources_included':False,'manuscript_dependency_count':len(closure),'copied_or_projected_inputs':len(inputs),'metadata_projection_count':len(JSON_FIELDS)}
 (output/'ARTIFACT_METADATA.json').write_text(json.dumps(meta,indent=2)+'\n')
 files=[p for p in output.rglob('*') if p.is_file()]
 if len(files)>100 or sum(p.stat().st_size for p in files)>30*1024*1024:raise ValueError('Package exceeds bounded review budget')
 for p in files:
  if p.suffix.lower() in TEXT_SUFFIXES and PRIVATE.search(p.read_text(encoding='utf-8')):
   raise ValueError('Private text detected in '+str(p.relative_to(output)))
 (output/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(output))+'\n' for p in sorted(files)))
 return meta

def main() -> None:
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--local-extraction-evidence',type=Path,required=True)
 args=parser.parse_args()
 print(json.dumps(build(args.output,args.local_extraction_evidence),indent=2))
if __name__=='__main__':main()
