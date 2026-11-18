#!/usr/bin/env python3
"""Current local aggregate package plus portable, externally input-backed replay tools."""
from pathlib import Path
import argparse,hashlib,json,shutil
from build_anonymous_artifact import build,sha,source_file,PRIVATE,TEXT_SUFFIXES,manuscript_closure

REPLAY_FILES=('run_diagnostics.py','README.md')
LOCAL_INPUTS=('expected_aggregates.json','review_budget_protocol.json','sensitivity_protocol.json')
SCRIPTS=('paired_nuisance.py','review_budget.py','auc_label_sensitivity.py')
TESTS=('test_paired_nuisance.py','test_review_budget.py','test_auc_label_sensitivity.py')

def verify_build(paper,receipt):
 if receipt is None:raise ValueError('Fresh exact-input build receipt required')
 report=json.loads(receipt.read_text());inventory=json.loads((receipt.parent/'repository_before_sha256.json').read_text())
 if not report['repository_files_unchanged'] or report['undefined_citations_or_references'] or report['outstanding_rerun_requests'] or not all(c['exit_code']==0 for c in report['commands']):raise ValueError('Incomplete manuscript validation')
 closure=manuscript_closure(paper)
 for name in closure:
  expected=report['pdf_sha256'] if name.endswith('.pdf') and name=='sc_vulnerability_detection.pdf' else inventory[name]
  if sha(source_file(paper,name))!=expected:raise ValueError('Stale or changed manuscript dependency: '+name)
 return {'pdf_sha256':report['pdf_sha256'],'pages':report['pdfkit_page_count'],'source_sha256':{n:inventory[n] for n in closure if n!='sc_vulnerability_detection.pdf'},'build_report_sha256':sha(receipt),'scope':'Exact PDF/source/asset lineage; compilation is not scientific validity or distribution clearance'}

def build_current(output,numeric,repo=None,paper=None,build_report=None,diagnostic_evidence=None):
 if output.exists() or output.is_symlink():raise FileExistsError("Output must be new")
 import build_anonymous_artifact as base
 repo=repo or base.REPO;paper=paper or base.PAPER
 # Fail before output creation on missing/escaping source inputs.
 build_lineage=verify_build(paper,build_report)
 replay=repo/'reproducibility/diagnostic_replay_v1'
 if diagnostic_evidence is None:raise ValueError('Local frozen diagnostic evidence directory required; not tracked research data')
 inputs=[(source_file(replay,n),'diagnostic_replay/'+n) for n in REPLAY_FILES]
 inputs += [(source_file(diagnostic_evidence,n),'diagnostic_replay/'+n) for n in LOCAL_INPUTS]
 inputs += [(source_file(repo,'scripts/'+n),'diagnostic_replay/scripts/'+n) for n in SCRIPTS]
 inputs += [(source_file(repo,'tests/'+n),'diagnostic_replay/tests/'+n) for n in TESTS]
 for p,_ in inputs:
  if p.suffix in TEXT_SUFFIXES and PRIVATE.search(p.read_text()):raise ValueError('Private replay source text')
 meta=build(output,numeric,repo,paper);lineage=json.loads((output/'SOURCE_LINEAGE.json').read_text())
 for src,name in inputs:
  h=sha(src);dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
  if sha(src)!=h or sha(dst)!=h:raise RuntimeError('Changed replay input during packaging')
  lineage['files'].append({'package_path':name,'kind':'own_protocol_or_numeric_replay_tool','original_sha256':h,'package_sha256':h,'projection':'fixed filename allowlist; no private cache rows'})
 meta.update(profile='current-sanitized-v1-aggregate-review-v3',portable_fixed_diagnostic_replay=True,private_inputs_required_for_real_replay=True,scientific_replay_run_by_builder=False)
 (output/'BUILD_LINEAGE.json').write_text(json.dumps(build_lineage,indent=2)+'\n')
 (output/'ARTIFACT_METADATA.json').write_text(json.dumps(meta,indent=2)+'\n');(output/'SOURCE_LINEAGE.json').write_text(json.dumps(lineage,indent=2)+'\n')
 with (output/'README.md').open('a') as f:f.write('\n## Current portable diagnostic replay\n\nThe diagnostic_replay directory includes the three fixed comparison tools,43numerical controls, original frozen protocols and allowlisted aggregate expectations. See its README. Original private prediction caches are deliberately external and remain held; no-input tests do not imply real-data reproduction. No release eligibility changes.\n')
 files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
 if len(files)>120 or sum(p.stat().st_size for p in files)>30*1024*1024:raise ValueError('Review size budget exceeded')
 for p in files:
  if p.suffix in TEXT_SUFFIXES and PRIVATE.search(p.read_text()):raise ValueError('Private package text detected')
 (output/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(output))+'\n' for p in files));return meta
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--local-extraction-evidence',type=Path,required=True);p.add_argument('--build-report',type=Path,required=True);p.add_argument('--diagnostic-evidence',type=Path,required=True);a=p.parse_args();print(json.dumps(build_current(a.output,a.local_extraction_evidence,build_report=a.build_report,diagnostic_evidence=a.diagnostic_evidence),indent=2))
