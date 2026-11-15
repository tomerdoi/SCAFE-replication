"""Validate AI-proposed pattern assessments; never grant training or gold status."""
from __future__ import annotations
import hashlib
from prepare_function_review import CATEGORIES, AXES

VERSION = 'AI-function-pattern-assessment-v1'
TARGET_DEFINITION = 'function-review-pilot-v1:pattern_presence'


def validated_assessment(code, assessment):
    """Require origin, source/context binding and evidence for every observed cell.

    The label is the declared pattern-presence axis, not contextual suspicion,
    an upstream vulnerability label or independently adjudicated truth.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError('Nonempty code required')
    digest = hashlib.sha256(code.encode()).hexdigest()
    if assessment.get('model_code_sha256') != digest:
        raise ValueError('Assessment/code identity mismatch')
    if (assessment.get('protocol') != VERSION or assessment.get('label_origin') != 'AI-assigned'
        or assessment.get('code_origin') != 'real-public'
        or assessment.get('target_definition_id') != TARGET_DEFINITION):
        raise ValueError('Explicit AI origin, real-code origin and target definition required')
    if any(assessment.get(k) is not False for k in ('training_eligible','evaluation_eligible','independent_gold')) or assessment.get('split') is not None:
        raise ValueError('Proposal cannot grant eligibility, splits or independent truth')
    method = assessment.get('method', {})
    if not all(isinstance(method.get(k),str) and method[k].strip() for k in
               ('assessor','model_family','exact_checkpoint_status','rubric_version','assessed_utc')):
        raise ValueError('Method/version and explicit checkpoint availability required')
    provenance = assessment.get('provenance', {})
    if not all(isinstance(provenance.get(k),str) and provenance[k].strip() for k in
               ('official_source_url','official_revision','context_sha256','scope','source_notice')):
        raise ValueError('Source/context provenance required')
    if provenance['source_notice'] != 'MIT' or provenance['scope'] != 'passed_declared_screen_for_AI_annotation':
        raise ValueError('This protocol supports only the qualified MIT assessment cohort')
    if set(assessment.get('categories',{})) != set(CATEGORIES):
        raise ValueError('All three categories must be assessed explicitly')
    labels, mask = {}, {}
    for category, axes in assessment['categories'].items():
        if set(axes) != set(AXES):
            raise ValueError('Pattern, contextual suspicion and vulnerability axes must stay separate')
        for axis, decision in axes.items():
            status = decision.get('status')
            if status not in ('present','absent','unknown') or not isinstance(decision.get('reason'),str) or not decision['reason'].strip():
                raise ValueError('Explicit supported decision/reason required')
            spans = decision.get('evidence_spans',[])
            if status != 'unknown' and not spans:
                raise ValueError('Observed decision needs code evidence')
            for span in spans:
                start,end,quote = span.get('start'),span.get('end'),span.get('quote')
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(code) or code[start:end] != quote:
                    raise ValueError('Evidence must match exact code character offsets')
        status=axes['pattern_presence']['status'];labels[category]={'present':True,'absent':False,'unknown':None}[status];mask[category]=status!='unknown'
    return {'labels':labels,'observation_mask':mask,'target_definition_id':TARGET_DEFINITION,
            'label_origin':'AI-assigned','code_origin':'real-public','independent_gold':False,
            'training_eligible':False,'evaluation_eligible':False,'split':None}
