#!/usr/bin/env python3
"""Bind retained source claims to delivery records; never assign review or eligibility."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_function_pilot_handoff import verify_delivery, sha256

SOURCE = 'sujeetc/SCRUBD'
VERSION = 'retained-scrubd-review-provenance-v1'
PROTOCOL = 'https://arxiv.org/html/2412.09935v1#S3.SS3'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def indexed_rows(path):
    """Line anchors refer to the exact hashed JSONL, never reconstructed CSV bytes."""
    result = {}
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            raise ValueError('Blank JSONL row; explicit migration required')
        row = json.loads(line)
        identity = row.get('record_id')
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError('Missing or duplicate record identity')
        result[identity] = (row, number, hashlib.sha256(line).hexdigest())
    return result


def bind_records(candidates, staging, revision, annotation_entry):
    if annotation_entry.get('revision') != revision or annotation_entry.get('source') != SOURCE:
        raise ValueError('Annotation manifest revision/source mismatch')
    if annotation_entry.get('path') != 'SCRUBD-CD/data/labels.csv':
        raise ValueError('Only the CD source protocol is in scope')
    bindings = []
    for identity, (row, line, line_hash) in sorted(candidates.items()):
        if row['source'] != SOURCE:
            continue
        if identity not in staging:
            raise ValueError('Candidate has no retained staging record')
        original, original_line, original_hash = staging[identity]
        # Require full equality, not merely a same-named function or label.
        if row != original or row.get('source_revision') != revision:
            raise ValueError('Retained record or revision differs')
        metadata = row['source_metadata']
        address = row['contract_address']
        function = row['function_identifier']
        if (metadata.get('Smart Contract', '').lower() != address.lower()
                or metadata.get('Function Name') != function
                or row['source_identity'] != address.lower() + '|' + function
                or row['source_path'] != 'SCRUBD-CD/data/solidity_codes/' + address.lower() + '.sol'):
            raise ValueError('Native function/source identity mismatch')
        for native, preserved in [('RE', 'reentrancy'), ('UX', 'unhandled_exceptions')]:
            value = metadata.get(native)
            if value not in ('0', '1', '', None):
                raise ValueError('Unsupported native label')
            expected = int(value) if value in ('0', '1') else None
            if row['source_assessments'].get(preserved) != expected:
                raise ValueError('Native label contradicts retained assessment')
        if (row.get('training_eligible') is not False or row.get('split') is not None
                or row.get('human_review_status') != 'unreviewed'
                or any(value is not None for value in row['pattern_labels'].values())):
            raise ValueError('Annotated or assigned delivery requires explicit migration')
        bindings.append({
            'record_id': identity, 'source': SOURCE, 'source_revision_claim': revision,
            'source_identity': row['source_identity'],
            'native_claims': {key: metadata.get(key) for key in ('RE', 'UX')},
            'native_observation_mask': {key: metadata.get(key) in ('0', '1') for key in ('RE', 'UX')},
            'retained_is_student': metadata.get('is_student'),
            'native_metadata_sha256': canonical_hash(metadata),
            'raw_function_sha256': row['raw_sha256'],
            'retained_model_code_sha256': hashlib.sha256(row['model_code'].encode()).hexdigest(),
            'context_sha256_claim': row['context_sha256'],
            'candidate_anchor': {'line_1based': line, 'line_bytes_sha256': line_hash},
            'staging_anchor': {'line_1based': original_line, 'line_bytes_sha256': original_hash},
            'upstream_annotation_artifact_claim': {
                'path': annotation_entry['path'], 'sha256': annotation_entry['sha256'],
                'exact_upstream_bytes_verified_this_run': False},
            'review_evidence': {
                'protocol_reference': PROTOCOL,
                'level': 'documented_upstream_protocol_not_record_adjudication',
                'role_flag_is_reviewer_identity_or_adjudication_proof': False,
                'review_coverage_at_retained_revision': 'unverified',
                'fresh_independent_evaluation_status': 'not_established'},
            'independent_gold': False, 'training_eligible': False,
            'evaluation_eligible': False, 'split': None,
        })
    if not bindings:
        raise ValueError('No retained SCRUBD candidates to bind')
    return bindings


def prepare(pilot, output):
    pilot, output = pilot.resolve(), output.resolve()
    if (output == pilot or output.exists()
            or (output.is_relative_to(pilot) and not output.is_relative_to(pilot / 'derivatives'))):
        raise ValueError('Use a new versioned derivative; preserve originals')
    verify_delivery(pilot)
    names = ['DELIVERY_SHA256.json', 'source_manifest.json',
             'public_output/function_candidates.jsonl', 'public_output/candidate_records.jsonl']
    before = {name: sha256(pilot / name) for name in names}
    manifest = json.loads((pilot / 'source_manifest.json').read_text())
    revision = manifest['sources'][SOURCE]['revision']
    entries = [entry for entry in manifest['files'] if entry.get('source') == SOURCE
               and entry.get('path') == 'SCRUBD-CD/data/labels.csv']
    if len(entries) != 1:
        raise ValueError('Ambiguous annotation artifact manifest')
    rows = bind_records(indexed_rows(pilot / names[2]), indexed_rows(pilot / names[3]), revision, entries[0])
    if any(sha256(pilot / name) != digest for name, digest in before.items()):
        raise ValueError('Inputs changed during binding')
    report = {'protocol': VERSION, 'bound_retained_candidates': len(rows),
              'retained_role_flag_counts': dict(sorted(Counter(str(r['retained_is_student']) for r in rows).items())),
              'native_RE_counts': dict(sorted(Counter(str(r['native_claims']['RE']) for r in rows).items())),
              'native_UX_counts': dict(sorted(Counter(str(r['native_claims']['UX']) for r in rows).items())),
              'certified_record_level_review_at_retained_revision': 0,
              'new_labels_or_eligibility': 0,
              'limitation': 'Lossless delivered-record binding, not exact upstream CSV verification or independent semantic adjudication.'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'review_provenance_links.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    result = {'protocol': VERSION, 'input_sha256': before,
              'implementation_sha256': sha256(Path(__file__)),
              'artifact_sha256': {name: sha256(output / name) for name in ['review_provenance_links.jsonl', 'report.json']}}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pilot, args.output), indent=2))
