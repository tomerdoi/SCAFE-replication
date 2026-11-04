#!/usr/bin/env python3
"""Bind preliminary AI source-quality observations to an immutable audit frame.

This exports evidence, never labels, training eligibility or independent gold.
It verifies artifact integrity; it cannot verify the truth of reviewer reasoning.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import safe_file, verify_artifacts
from prepare_function_review import text_hash, write_jsonl
from adapt_function_source_labels import VERSION as SOURCE_VERSION, DEFINITION

VERSION = 'function-source-quality-ai-pilot-v1'
CODES = {'interaction_before_state_update', 'guard_outside_function',
         'dependency_implementation_missing', 'execution_semantics_required',
         'source_rationale_incomplete', 'source_rationale_call_mismatch',
         'no_storage_write_in_function'}
KINDS = {'lexical_observation', 'ai_interpretation'}


def validate_reviews(reviews, queue, targets, candidates, contexts):
    """Require one explicitly non-gold review for every frozen sampled ID."""
    queue_by = {r['record_id']: r for r in queue}
    if len(queue_by) != len(queue):
        raise ValueError('Duplicate sample IDs')
    identities = [r['record_id'] for r in reviews]
    if len(set(identities)) != len(identities) or set(identities) != set(queue_by):
        raise ValueError('Review IDs must exactly cover the frozen sample')
    out = []
    for review in reviews:
        required = {'record_id','protocol','reviewer_kind','review_status','independent_gold',
                    'independent_decision','training_eligible','evaluation_eligible',
                    'mapping_assessment','source_comment_sha256','findings','remaining_questions'}
        if set(review) != required:
            raise ValueError('Unexpected review schema; no replacement labels or model input')
        rid = review['record_id']; sample = queue_by[rid]; target = targets[rid]
        if (review['protocol'] != VERSION or review['reviewer_kind'] != 'ai_assisted_single_pass'
            or review['review_status'] != 'preliminary_completed'
            or review['independent_gold'] is not False
            or review['independent_decision'] is not None
            or review['training_eligible'] is not False
            or review['evaluation_eligible'] is not False):
            raise ValueError('AI pilot cannot claim independent review, gold or eligibility')
        if review['mapping_assessment'] != 'consistent_with_pinned_qualified_span':
            raise ValueError('A different mapping result requires an explicit protocol revision')
        if (sample['mapping_evidence'] != target['mapping_evidence']
            or sample['source_proxy_label'] is not target['labels']['reentrancy']
            or not target['mapping_available_for_source_supervision']
            or target['supervision_protocol'] != SOURCE_VERSION
            or target['target_definition_id'] != DEFINITION):
            raise ValueError('Frozen sample/source target mismatch')
        candidate = candidates[rid]; mapping = target['mapping_evidence']
        path = safe_file(contexts, mapping['context_artifact'])
        if sha256(path) != mapping['context_sha256']:
            raise ValueError('Context changed')
        # Read exact UTF-8 bytes: universal-newline conversion would break raw spans.
        raw = path.read_bytes().decode('utf-8'); span = mapping['function_span']
        fragment = raw[span['start_character']:span['end_character']]
        if text_hash(fragment) != span['raw_fragment_sha256'] or text_hash(fragment) != candidate['raw_sha256']:
            raise ValueError('Function span/source claim binding changed')
        if mapping['qualified_identifier'] != candidate['function_identifier']:
            raise ValueError('Qualified identity mismatch')
        metadata = candidate['source_metadata']
        if review['source_comment_sha256'] != text_hash(metadata.get('Comments', '')):
            raise ValueError('Source rationale changed')
        if not review['findings'] or not review['remaining_questions'] or not all(isinstance(q,str) and q.strip() for q in review['remaining_questions']):
            raise ValueError('Require explicit findings and unresolved questions')
        lines = raw.splitlines(keepends=True)
        for finding in review['findings']:
            if finding['code'] not in CODES or finding['kind'] not in KINDS or not isinstance(finding['observation'],str) or not finding['observation'].strip():
                raise ValueError('Unsupported observation')
            evidence = finding['evidence']
            if not evidence:
                raise ValueError('Every observation requires source evidence')
            for item in evidence:
                lo, hi = item['start_line'], item['end_line']
                if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi <= len(lines):
                    raise ValueError('Invalid evidence range')
                if text_hash(''.join(lines[lo-1:hi])) != item['raw_excerpt_sha256']:
                    raise ValueError('Evidence excerpt changed')
        if any(k in review for k in ('labels','label_observation_mask','model_code','source_proxy_label')):
            raise ValueError('Audit must not export replacement targets or model inputs')
        out.append({**review, 'mapping_evidence': mapping,
                    'source_native_RE': target['native_source_assessments']['RE'],
                    'source_identity': target['source_identity'],
                    'source_revision': target['source_revision'],
                    'group_constraints': target['group_constraints'],
                    'stratum': sample['stratum'],
                    'stratum_population': sample['stratum_population'],
                    'stratum_selected': sample['stratum_selected'],
                    'source_labels_unchanged': True})
    return sorted(out, key=lambda r:r['record_id'])


def prepare(pilot, source, contexts, reviews_path, output):
    pilot, source, contexts, reviews_path, output = [p.resolve() for p in (pilot,source,contexts,reviews_path,output)]
    if (output == pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives'))
        or any(output == p or output.is_relative_to(p) for p in (source, contexts))):
        raise ValueError('Preserve originals and parents; use a new versioned output')
    if output.exists(): raise FileExistsError(output)
    delivered = verify_delivery(pilot)
    sm = verify_artifacts(source,'manifest.json','artifact_sha256')
    cm = verify_artifacts(contexts,'recovery_manifest.json','artifact_sha256')
    parent_hashes = {'source':sha256(source/'manifest.json'), 'contexts':sha256(contexts/'recovery_manifest.json')}
    if (sm['protocol'] != SOURCE_VERSION or sm['target_definition_id'] != DEFINITION
        or sm['parent_manifest_sha256']['contexts'] != parent_hashes['contexts']
        or sm['delivery_manifest_sha256'] != sha256(pilot/'DELIVERY_SHA256.json')
        or cm['delivery_manifest_sha256'] != sm['delivery_manifest_sha256']):
        raise ValueError('Inconsistent immutable parents')
    review_hash = sha256(reviews_path)
    targets = {r['record_id']:r for r in jsonl_rows(source/'source_targets.jsonl')}
    candidates = {r['record_id']:r for r in jsonl_rows(pilot/'public_output/function_candidates.jsonl')}
    queue = list(jsonl_rows(source/'stratified_audit_queue.jsonl'))
    results = validate_reviews(list(jsonl_rows(reviews_path)),queue,targets,candidates,contexts)
    summary = {'protocol':VERSION, 'preliminary_ai_review_records':len(results),
               'local_mapping_bindings_checked':len(results),
               'distinct_contexts':len({r['mapping_evidence']['context_sha256'] for r in results}),
               'distinct_evidence_components':len({r['group_constraints']['evidence_component'] for r in results}),
               'finding_record_counts':dict(Counter(code for r in results for code in {f['code'] for f in r['findings']})),
               'independent_semantic_reviews':0, 'adjudicated_labels':0,
               'training_eligible_records':0, 'evaluation_eligible_records':0,
               'assigned_splits':0, 'source_target_changes':0,
               'source_label_accuracy_estimated':False, 'training_or_inference_run':False,
               'decision':'Retain native claims and proxy namespace; do not use these as syntactic pattern negatives or independent evaluation. Resolve the cited semantic/callee/guard questions in targeted qualified adjudication.',
               'limits':['Single AI pass with source claims visible; evidence hashes verify identity, not semantic correctness.',
                         'Eight selected rows in four RE/student strata, with correlated contexts; not eight independent projects or a calibrated noise estimate.',
                         'No source claim is changed, no gate automatically passes, and neither accuracy nor safe exploitation conclusions are established.']}
    if verify_delivery(pilot) != delivered or sha256(reviews_path) != review_hash:
        raise ValueError('Input changed during compilation')
    for root, name, key in [(source,'manifest.json','source'),(contexts,'recovery_manifest.json','contexts')]:
        if sha256(root/name) != parent_hashes[key]:raise ValueError('Parent changed')
        verify_artifacts(root,name,'artifact_sha256')
    output.mkdir(parents=True,exist_ok=False)
    (output/'review_submissions.jsonl').write_bytes(reviews_path.read_bytes())
    write_jsonl(output/'preliminary_reviews.jsonl',results)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest = {'protocol':VERSION,'generator_sha256':sha256(Path(__file__)),
                'review_submission_sha256':review_hash,'parent_manifest_sha256':parent_hashes,
                'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),
                'helper_sha256':{Path(fn.__code__.co_filename).name:sha256(Path(fn.__code__.co_filename)) for fn in (verify_delivery,verify_artifacts,write_jsonl)},
                'artifact_sha256':{p.name:sha256(p) for p in sorted(output.iterdir())}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','source-release-dir','context-release-dir','reviews','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(prepare(a.pilot_dir,a.source_release_dir,a.context_release_dir,a.reviews,a.output_dir),indent=2))

if __name__=='__main__':main()
