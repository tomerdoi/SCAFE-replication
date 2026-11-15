"""Separate, fail-closed export of evidence-backed contextual AI proposals.

This is not a reviewer, vulnerability verifier, or training-admission gate.
Legacy pattern labels are never promoted by this adapter.
"""
from __future__ import annotations
import hashlib
from prepare_function_review import CATEGORIES

VERSION = 'AI-function-context-assessment-v1'
TARGET = 'function-context-suspiciousness-v1'
DEFINITIONS = {
    'reentrancy': 'A supported feasible callback path with relevant inconsistent state or invariant exposure, assessed with guards and caller/callee assumptions; an external call alone is insufficient.',
    'unchecked_external_call': 'A supported external-call failure or return-value handling defect under the inspected callee/API semantics and subsequent control flow; a call or empty catch alone is insufficient.',
    'authorization': 'A supported sensitive operation reachable without its required authorization under the inspected entry points and guards; public visibility or a modifier name alone is insufficient.',
}

def validate_context_assessment(code, record, contexts):
    """Verify exact evidence binding and export ONLY explicit contextual decisions.

    contexts is an in-memory {logical_id: source_text}; no file/network loading.
    A valid record is still AI-proposed and requires separate quality/admission gates.
    """
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    if not isinstance(code, str) or not code.strip() or record.get('model_code_sha256') != digest(code):
        raise ValueError('Model input identity mismatch')
    required = {'protocol': VERSION, 'target_definition_id': TARGET,
                'label_origin': 'AI-assigned', 'code_origin': 'real-public'}
    if any(record.get(k) != v for k,v in required.items()):
        raise ValueError('Explicit context protocol, target and origins required')
    if any(record.get(k) is not False for k in ('training_eligible','evaluation_eligible','independent_gold')) or record.get('split') is not None:
        raise ValueError('Proposal cannot grant admission or independent truth')
    for key in ('record_id','method_model_version','source_provenance_ref','source_claim_disposition','admission_screen_ref','parent_release_sha256'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError('Missing lineage or source-claim disposition')
    manifest = record.get('context_hashes', {})
    if not manifest or set(manifest) != set(contexts):
        raise ValueError('Exact context inventory required')
    if any(not isinstance(v,str) or digest(v) != manifest[k] for k,v in contexts.items()):
        raise ValueError('Context identity mismatch')
    if set(record.get('decisions',{})) != set(CATEGORIES):
        raise ValueError('Every category must be explicitly assessed')
    labels, masks = {}, {}
    for category, decision in record['decisions'].items():
        status = decision.get('status')
        if status not in ('positive','negative','unknown','context-required'):
            raise ValueError('Unknown is not a negative')
        for key in ('behavior','protections','scope','rationale'):
            if not isinstance(decision.get(key),str) or not decision[key].strip():
                raise ValueError('Behavior, protections, scope and rationale required')
        evidence = decision.get('evidence',[])
        observed = status in ('positive','negative')
        if observed and not evidence:
            raise ValueError('Observed contextual decision needs evidence')
        if not observed and not decision.get('missing_evidence'):
            raise ValueError('Unknown requires an actionable evidence gap')
        if observed and decision.get('missing_evidence'):
            raise ValueError('Unresolved evidence cannot support an observed label')
        for span in evidence:
            source = contexts.get(span.get('context_id'))
            start,end = span.get('start'),span.get('end')
            if source is None or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source) or source[start:end] != span.get('quote'):
                raise ValueError('Evidence must bind exact source characters')
        labels[category] = {'positive': True,'negative':False,'unknown':None,'context-required':None}[status]
        masks[category] = observed
    return {**required,'labels':labels,'observation_mask':masks,'training_eligible':False,
            'evaluation_eligible':False,'independent_gold':False,'split':None}
