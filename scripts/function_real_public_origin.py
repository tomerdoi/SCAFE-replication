"""Validate a reviewed, identity-bound origin decision; never infer origin from hosting.

This is a provenance-decision schema check, not independent verification of its
truth, reviewer qualifications, license, labels, splitting or training readiness.
No URL is fetched and no example is made eligible by passing this check.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

POLICY = 'real-public-function-origin-v1'
EVIDENCE_KINDS = {'upstream_real_world_origin_documentation',
                  'real_project_source_lineage', 'on_chain_source_provenance'}


def require_real_public_origin(model, target, *, purpose):
    """Require an accepted provenance decision bound to this input and source.

    The caller must obtain the accepted decision through the declared review
    protocol and pin it in the source release. Strings cannot prove semantic
    truth or review independence. Public URL shape alone never qualifies a row.
    """
    if purpose not in ('training', 'evaluation'):
        raise ValueError('Explicit research training/evaluation purpose required')
    decision = target.get('real_public_origin_decision')
    if not isinstance(decision, dict):
        raise ValueError('Real-public origin decision missing; unknown origin is excluded')
    if (decision.get('policy') != POLICY or decision.get('status') != 'accepted'
            or decision.get('origin') != 'real_world'
            or decision.get('public_dataset') is not True):
        raise ValueError('Require accepted real-world public-dataset origin; synthetic/unknown excluded')
    for key in ('record_id', 'source', 'source_revision', 'model_code_sha256'):
        expected = target.get(key)
        if not isinstance(expected, str) or not expected or decision.get(key) != expected:
            raise ValueError('Origin decision has missing or mismatched record/source/input binding')
    if (model.get('record_id') != target['record_id']
            or model.get('model_code_sha256') != target['model_code_sha256']
            or not re.fullmatch('[0-9a-f]{64}', target['model_code_sha256'])):
        raise ValueError('Origin decision is bound to a different model input')
    for key in ('decision_id', 'reviewer_id'):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError('Require traceable origin review decision metadata')
    url = decision.get('public_dataset_url')
    if not isinstance(url, str):
        raise ValueError('Public dataset reference required')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or any(character.isspace() for character in url)):
        raise ValueError('Require a credential-free HTTPS public dataset reference')
    evidence = decision.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Public hosting alone is not real-world origin evidence')
    for item in evidence:
        if (not isinstance(item, dict) or item.get('kind') not in EVIDENCE_KINDS
                or not isinstance(item.get('artifact_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', item['artifact_sha256'])
                or not isinstance(item.get('locator'), str) or not item['locator'].strip()):
            raise ValueError('Require pinned origin evidence references; hosting/category labels are insufficient')
    return None
