"""Consolidate source assertions without inventing canonical labels or eligibility."""
from __future__ import annotations
import hashlib

CATEGORIES = ('reentrancy', 'unchecked_external_call', 'authorization')
VERSION = 'function-unified-source-claims-v1'


def consolidate(candidates, model_views, source_layers):
    """Join exact IDs; retain per-definition labels and model-binding evidence.

    This is an inspection release. Model consumers must explicitly project code;
    metadata/rationales are never input text. Existing training gates are unchanged.
    """
    def index(rows):
        result = {}
        for row in rows:
            identity = row.get('record_id')
            if not isinstance(identity, str) or not identity or identity in result:
                raise ValueError('Missing or duplicate record ID')
            result[identity] = row
        return result
    base, views = index(candidates), index(model_views)
    if set(base) != set(views):
        raise ValueError('Candidate/model inventories differ')
    result = {}
    for identity, candidate in base.items():
        view = views[identity]
        code = view.get('model_code')
        if not isinstance(code, str) or not code.strip():
            raise ValueError('Missing model code')
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest != view.get('model_code_sha256'):
            raise ValueError('Model code hash mismatch')
        if (candidate['source'], candidate['source_revision']) != (view['source'], view['source_revision']):
            raise ValueError('Model provenance mismatch')
        if set(candidate['pattern_labels']) != set(CATEGORIES) or any(v is not None for v in candidate['pattern_labels'].values()):
            raise ValueError('Unexpected canonical annotations require a new protocol')
        result[identity] = {
            'record_id': identity, 'schema_version': VERSION,
            'model_input': {'model_code': code, 'model_code_sha256': digest,
                            'input_protocol': view['view_protocol']},
            'provenance': {k: candidate.get(k) for k in
                           ('source', 'source_revision', 'source_identity', 'project_id', 'license_status')},
            'original_source_claim': candidate.get('original_claim'),
            'canonical_labels': dict(candidate['pattern_labels']),
            'canonical_observation_mask': {k: False for k in CATEGORIES},
            'source_label_assertions': [],
            'ai_assessments': {k: {'status': 'not_assessed', 'label': None, 'label_origin': None} for k in CATEGORIES},
            'reserved_evaluation_overlap': 'unresolved',
            'group_constraints_by_release': [],
            'training_eligible': False, 'evaluation_eligible': False,
            'independent_gold': False, 'split': None,
            'eligibility_blockers': ['reserved_DAppSCAN_project_clone_exclusion_unresolved',
                                     'source_rights_and_target_quality_protocol_pending',
                                     'grouped_split_and_independent_evaluation_pending'],
        }
    seen = set()
    for release, rows in source_layers:
        index(rows)  # Fail closed on duplicate IDs even if they have no labels.
        for line_number, row in enumerate(rows, 1):
            identity = row['record_id']
            if identity not in result:
                raise ValueError('Source assertion has no candidate')
            candidate, out = base[identity], result[identity]
            if row.get('source') != candidate['source'] or row.get('source_revision') != candidate['source_revision']:
                raise ValueError('Source assertion provenance mismatch')
            labels = row.get('labels', row.get('source_proxy_labels'))
            masks = row.get('label_observation_mask', row.get('source_proxy_observation_mask'))
            if set(labels or {}) != set(CATEGORIES) or set(masks or {}) != set(CATEGORIES):
                raise ValueError('Require explicit labels and partial masks')
            groups = row.get('group_constraints', {})
            if groups:
                out['group_constraints_by_release'].append({'release': release, 'constraints': groups})
            if groups.get('direct_verified_context_corpus_overlap') or groups.get('evidence_conservative_quarantine') or groups.get('strict_template_conservative_quarantine'):
                out['reserved_evaluation_overlap'] = 'existing_corpus_overlap_or_quarantine;exact_reserved_membership_pending'
            for category in CATEGORIES:
                value, observed = labels[category], masks[category]
                if type(observed) is not bool or (observed and type(value) is not bool) or (not observed and value is not None):
                    raise ValueError('Invalid observed/unknown label pairing')
                if not observed:
                    continue
                definition = row.get('target_definition_ids', {}).get(category) or row.get('target_definition_id')
                if not isinstance(definition, str) or not definition:
                    raise ValueError('Observed assertion needs its exact target definition')
                key = (identity, category, definition)
                if key in seen:
                    raise ValueError('Duplicate/conflicting definition assertion; adjudication required')
                seen.add(key)
                out['source_label_assertions'].append({
                    'category': category, 'label': value, 'observed': True, 'label_origin': 'source-provided',
                    'target_definition_id': definition,
                    'supervision_tier': row.get('supervision_tier', 'upstream_source_proxy'),
                    'evidence_ref': {'release': release, 'line': line_number},
                    'mapping_evidence': row.get('mapping_evidence'),
                    'finding_support': row.get('finding_support'),
                    'typed_site_evidence': row.get('typed_site_evidence'),
                    'negative_admission_evidence': row.get('negative_admission_evidence'),
                    'bound_model_code_sha256': row.get('model_code_sha256'),
                    'matches_consolidated_model_view': row.get('model_code_sha256') == out['model_input']['model_code_sha256'],
                    'human_review_status': row.get('human_review_status', 'unreviewed'),
                    'independent_gold': False,
                })
    return [result[key] for key in sorted(result)]
