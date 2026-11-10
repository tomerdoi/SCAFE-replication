#!/usr/bin/env python3
"""Export development-only native alarm-review claims; no model/security-gold output."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

PIN = 'e3e089c4d119ac4de1398345c525f06dc4f2ef96'
INPUT_SHA256 = '42c3b98ccd77f51dbb561ab7178425ea69e10abbfe10a8dad3133bbb018e0073'
TOOLS = ('oyente', 'mythril', 'securify1', 'securify2', 'smartian', 'sailfish')
HEADER = ['address', 'true_positive', *TOOLS]
VERSION = 'reentrancy-study-alarm-review-claims-v1'


def address(value):
    if not isinstance(value, str) or not re.fullmatch(r'0x[0-9a-fA-F]{40}', value.strip()):
        raise ValueError('Expected an Ethereum address')
    return value.strip().lower()


def role(value):
    key = address(value)
    digest = hashlib.sha256(('SCAFE-RS-route-v1:' + key).encode()).hexdigest()
    return 'provisional_reserve' if int(digest[:8], 16) % 5 < 2 else 'development'


def source_claim(row):
    """A native zero is observed only inside the publisher's alarm-review frame."""
    key = address(row.get('address'))
    if set(row) != set(HEADER):
        raise ValueError('Unexpected upstream schema')
    values = {name: row[name].strip() for name in HEADER[1:] if isinstance(row[name], str)}
    if len(values) != len(HEADER) - 1 or any(v not in {'0', '1', 'N/A'} for v in values.values()):
        raise ValueError('Unsupported native value; do not coerce to negative')
    alarm = any(values[name] == '1' for name in TOOLS)
    native = values['true_positive']
    if native == '1' and not alarm:
        raise ValueError('Positive claim outside declared alarm frame needs adjudication')
    observed = alarm and native in {'0', '1'}
    return {
        'record_id': 'InPlusLab/ReentrancyStudy-Data:' + PIN + ':' + key,
        'address': key,
        'native_true_positive': native,
        'native_tool_results': {name: values[name] for name in TOOLS},
        'alarm_union': alarm,
        'target_definition': 'publisher_contract_level_reentrancy_alarm_review;not_function_pattern_or_universal_safety',
        'source_target': int(native) if observed else None,
        'source_observation_mask': observed,
        'mask_reason': 'alarm_linked_native_claim' if observed else 'no_alarm_outside_reported_review_frame' if native == '0' else 'native_unknown',
        'supervision_tier': 'publisher_cohort_manual_review_claim;case_review_source_linkage_unverified',
        'independent_gold': False,
        'training_eligible': False,
        'evaluation_eligible': False,
        'frozen_split': None,
        'canonical_function_category_mapping': None,
    }


def export(input_file: Path, output: Path):
    if not input_file.is_file() or input_file.stat().st_size > 16 * 1024**2:
        raise ValueError('Expected bounded upstream CSV')
    raw = input_file.read_bytes()
    if len(raw) > 16 * 1024**2 or hashlib.sha256(raw).hexdigest() != INPUT_SHA256:
        raise ValueError('Upstream revision/content mismatch')
    # Validate before creating outputs. Reserve labels/tools are never inspected/exported.
    seen = set(); claims = []; counts = Counter()
    with io.StringIO(raw.decode('utf-8-sig'), newline='') as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != HEADER:
            raise ValueError('Unexpected CSV header')
        for row in rows:
            if set(row) != set(HEADER):
                raise ValueError('Malformed upstream row')
            key = address(row['address'])
            if key in seen:
                raise ValueError('Duplicate address requires provenance review')
            seen.add(key); selected = role(key); counts[selected] += 1
            if selected != 'development':
                continue
            record = source_claim(row)
            record['selection_role'] = 'development'
            claims.append(record)
    summary = {
        'version': VERSION, 'upstream_repository': 'InPlusLab/ReentrancyStudy-Data',
        'pin': PIN, 'input_sha256': INPUT_SHA256, 'unit': 'contract alarm-review native claim',
        'role_counts': dict(counts), 'development_records': len(claims),
        'observed_claim_records_exported': sum(r['source_observation_mask'] for r in claims),
        'observed_positive_claims': sum(r['source_target'] == 1 for r in claims),
        'observed_negative_claims': sum(r['source_target'] == 0 for r in claims),
        'masked_no_alarm_native_zeros': sum(r['native_true_positive'] == '0' and not r['alarm_union'] for r in claims),
        'unknown_claims': sum(not r['source_observation_mask'] for r in claims),
        'reserve_label_values_inspected': False, 'independent_gold_added': 0,
        'training_eligible_added': 0, 'evaluation_eligible_added': 0,
        'role_is_frozen_grouped_split': False,
        'forbidden_model_input_fields': ['native_true_positive', 'native_tool_results', 'address', 'record_id', 'alarm_union', 'all source/review metadata'],
        'source_rights_and_code_binding': 'not_qualified;no_contract_source_acquired',
    }
    output.mkdir(parents=False, exist_ok=False)
    with (output / 'development_source_claims.jsonl').open('w') as stream:
        for record in claims:
            if record['source_observation_mask']:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
    with (output / 'development_unobserved.jsonl').open('w') as stream:
        for record in claims:
            if not record['source_observation_mask']:
                stream.write(json.dumps({key: record[key] for key in ['address', 'native_true_positive', 'mask_reason', 'source_observation_mask', 'source_target', 'selection_role', 'training_eligible', 'evaluation_eligible']}, sort_keys=True) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
