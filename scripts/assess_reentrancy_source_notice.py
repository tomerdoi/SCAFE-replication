#!/usr/bin/env python3
"""Compare literal provenance evidence; never infer model eligibility or legal grants."""
from __future__ import annotations
import re


def notice_assessment(prefix, explorer):
    if not isinstance(prefix, dict) or not isinstance(explorer, dict):
        raise ValueError('Expected provenance records')
    key = prefix.get('address')
    declared_address = explorer.get('ContractAddress')
    if not isinstance(key, str) or not re.fullmatch(r'0x[0-9a-f]{40}', key):
        raise ValueError('Expected canonical source address')
    if not isinstance(declared_address, str) or key != declared_address.strip().lower():
        raise ValueError('Source-header and explorer address mismatch')
    if prefix.get('source_path') != 'deduplicated_smart_contracts/' + key + '.sol':
        raise ValueError('Source path does not bind the selected address')
    if type(prefix.get('native_source_target')) is not int or prefix['native_source_target'] not in (0, 1):
        raise ValueError('Expected preserved native binary source claim')
    if not isinstance(explorer.get('LicenseType'), str):
        raise ValueError('Expected literal explorer license claim')
    notices = prefix.get('spdx_expressions')
    if not isinstance(notices, list) or any(not isinstance(x, str) for x in notices):
        raise ValueError('Expected explicit notice observations')
    result = {
        'source_path': prefix['source_path'], 'address': prefix['address'],
        'native_source_target': prefix['native_source_target'],
        'source_header_spdx': notices, 'explorer_license_claim': explorer.get('LicenseType', ''),
        'source_binding_candidate': False, 'training_eligible': False,
        'evaluation_eligible': False, 'independent_gold': False,
        'claim_unit': 'publisher_contract_alarm_review;not_function_gold_or_safety',
    }
    if prefix.get('status') not in {'notice_observed', 'no_spdx_in_bounded_prefix'}:
        return {**result, 'status': 'held_header_unavailable'}
    if not notices:
        return {**result, 'status': 'held_no_notice_in_bounded_prefix;not_proof_of_no_license'}
    # Compare explicit family names only; do not guess GPL/LGPL version/exception terms.
    explicit_family_conflict = (
        any(x.startswith('GPL-') for x in notices) and explorer.get('LicenseType', '').startswith('GNU LGPL')
    ) or (
        any(x.startswith('LGPL-') for x in notices) and explorer.get('LicenseType', '').startswith('GNU GPL')
    )
    if explicit_family_conflict:
        return {**result, 'status': 'held_literal_GPL_LGPL_family_discrepancy;no_adjudication'}
    if notices == ['MIT'] and explorer.get('LicenseType') == 'MIT':
        size = prefix.get('reported_total_bytes')
        if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 128 * 1024:
            return {**result, 'status': 'held_source_size_unqualified'}
        return {**result, 'status': 'consistent_MIT_evidence_for_bounded_private_binding', 'source_binding_candidate': True}
    return {**result, 'status': 'held_unresolved_notice_comparison;no_automatic_license_mapping'}
