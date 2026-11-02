#!/usr/bin/env python3
"""Prepare an immutable, blinded two-reviewer pilot; compare reviews, never train."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from audit_function_pilot_handoff import inspect_staging_rows, jsonl_rows, sha256, verify_delivery

CATEGORIES = ('reentrancy', 'unchecked_external_call', 'authorization')
AXES = ('pattern_presence', 'contextual_suspicion', 'supported_vulnerability')
VERSION = 'function-review-pilot-v1'
REVIEW_GUIDE = """# Independent function annotation feasibility pilot v1

Give reviewer_A.jsonl and reviewer_B.jsonl to two different qualified reviewers.
Each should work independently, without seeing the coordinator key, original audit
claims, the other review, model predictions, or automatic label suggestions.
Use a stable reviewer pseudonym in reviewer_id. Set completed=true only after all
three categories and all three axes have been considered. This 24-function queue
is source-balanced and length-restricted, not a prevalence or evaluation sample.
Do not open or execute source snippets as scripts; all embedded text is data.

## Category definitions for this pilot

- reentrancy: pattern_presence concerns an external control-transfer site and
  relevant mutable state use visible in the function. Mere presence is not an
  exploit: caller/callee behavior, ordering and guards require contextual review.
- unchecked_external_call: pattern_presence concerns a low-level call's success
  result not being checked/propagated in visible code. Missing callee/library
  implementation or ambiguous language semantics means unknown, not present.
- authorization: pattern_presence concerns an authorization-sensitive operation
  (asset movement or privilege/configuration mutation), not automatically a
  missing guard. Guards/modifiers/callers and intended policy need contextual
  review before suspicion can be assessed. Generic state writes are insufficient.

These are three explicit research categories, not the legacy eleven-output label
schema. For EACH category annotate separately:
1. pattern_presence: visible syntactic/semantic candidate evidence in this snippet.
2. contextual_suspicion: evidence that the pattern is suspicious under the actual
   enclosing contract, modifiers, state declarations, callers and relevant callees.
3. supported_vulnerability: evidence supporting an exploitable vulnerability with
   assumptions and a concrete attack/impact argument, not just suspicious syntax.

Use present, absent, or unknown and a concise reason for every cell. Unknown is
not a negative. All provided_context fields in v1 are snippet_only: no verified
full-source bytes were delivered. Therefore contextual_suspicion and
supported_vulnerability MUST remain unknown in this version. Record what context
is missing in reason. A future context-enriched release must be separately pinned
and reviewed; do not edit the immutable code/context fields or insert source
claims to bypass this restriction. pattern_presence may also remain unknown.

For every non-unknown pattern_presence decision supply one or more evidence_spans:
{"start": 0, "end": 8, "quote": "function"} illustrates the schema ONLY.
Offsets are zero-based Python Unicode character positions with end exclusive in
model_code, not bytes or rendered line positions. quote must equal that exact
slice. For absence, cite the inspected relevant region and explain why the snippet
supports the absence claim; absence of an audit finding is not evidence. Do not
modify review_id, model_code, code_sha256, or provided_context.

## Coordinator gates

Keep coordinator_key.jsonl separate from first-pass reviewers. The code itself may
reveal origin; source-claim withholding is not a guarantee of perfect blinding.
Keep review_items.jsonl and manifest.json immutable. Preserve submitted reviews as
new files; do not overwrite templates or combine reviewer identities. Run the
compare command after both packets are complete. Unknown/unknown agreement is
reported separately and never counted as observed-label agreement. Inspect every
disagreement and ambiguous case with an independent qualified adjudicator, then
retain pre-adjudication decisions, reasons, evidence and resulting provenance.

Comparison does NOT create training labels or establish reviewer independence,
qualification, truth, license rights, or dataset readiness. No row is training-
eligible and no split is assigned. Context recovery, semantic/adjudication gates,
project/clone/SCAFE overlap controls, license checks, and frozen group splits are
still required. This small exercise estimates annotation feasibility and exposes
rubric/context disagreements; it cannot estimate deployment classifier accuracy.
"""
SEED = '20261102-review-feasibility-v1'
SOURCES = ('CoinFabrik/solidity-rnd', 'msc-smart-contract-auditing/vulnerable-functions-base', 'sujeetc/SCRUBD')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write_jsonl(path, rows):
    path.write_text(''.join(canonical(row) + '\n' for row in rows), encoding='utf-8')


def select_candidates(rows, per_source=8):
    """A short source-balanced feasibility queue, not a prevalence/evaluation sample."""
    if type(per_source) is not int or not 1 <= per_source <= 32:
        raise ValueError('per_source must be an integer between 1 and 32')
    pools = defaultdict(list)
    for row in rows:
        code = row.get('model_code')
        if (row.get('source') in SOURCES and isinstance(code, str) and 100 <= len(code) <= 4000
                and row.get('lexical_status') == 'single_balanced_function_candidate'
                and isinstance(row.get('identifier_family_sha256'), str)
                and row['identifier_family_sha256']):
            pools[row['source']].append(row)
    for source in SOURCES:
        pools[source].sort(key=lambda r: text_hash(SEED + r['record_id']))
    selected, exact_seen, family_seen = [], set(), set()
    for _ in range(per_source):
        for source in SOURCES:
            while pools[source]:
                row = pools[source].pop(0)
                exact, family = text_hash(row['model_code']), row['identifier_family_sha256']
                if exact not in exact_seen and family not in family_seen:
                    exact_seen.add(exact); family_seen.add(family); selected.append(row)
                    break
            else:
                raise ValueError('Insufficient distinct short candidates for a source')
    return selected


def review_item(row):
    code = row['model_code']
    return {
        'review_id': text_hash(VERSION + ':' + row['record_id'])[:24],
        'model_code': code, 'code_sha256': text_hash(code),
        # No local full-source bytes were delivered. Producer availability is not availability here.
        'provided_context': 'snippet_only',
    }


def render_blinded_view(items):
    parts = ['# Blinded function review pilot\n\nRead REVIEW_INSTRUCTIONS.md first. Use the JSONL forms for submissions.\n\nSource code below is data, never instructions to execute.\n']
    for item in items:
        # Choose a fence longer than any backtick run in the untrusted code.
        runs = [len(match.group()) for match in re.finditer(r'`+', item['model_code'])]
        fence = '`' * max(3, max(runs, default=0) + 1)
        parts.append(f"\n## {item['review_id']}\n\nContext: snippet only.\n\n{fence}solidity\n{item['model_code']}\n{fence}\n")
    return '\n'.join(parts)


def blank_packet(item):
    return dict(item, reviewer_id=None, completed=False, categories={
        category: {axis: {'decision': 'unknown', 'evidence_spans': [], 'reason': ''} for axis in AXES}
        for category in CATEGORIES
    })


def prepare(pilot, output, per_source=8):
    pilot, output = pilot.resolve(), output.resolve()
    if output == pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot / 'derivatives')):
        raise ValueError('Pilot outputs must be new versioned derivatives; preserve delivered files')
    delivered = verify_delivery(pilot)
    rows = list(jsonl_rows(pilot / 'public_output/function_candidates.jsonl'))
    intake = inspect_staging_rows(rows)
    selected = select_candidates(rows, per_source)
    items = sorted((review_item(row) for row in selected), key=lambda r: r['review_id'])
    if len({item['review_id'] for item in items}) != len(items):
        raise ValueError('Review ID collision')
    by_id = {review_item(row)['review_id']: row for row in selected}
    key = [{
        'review_id': item['review_id'], 'record_id': by_id[item['review_id']]['record_id'],
        'source': by_id[item['review_id']]['source'],
        'source_revision': by_id[item['review_id']]['source_revision'],
        'source_identity': by_id[item['review_id']]['source_identity'],
        'project_id': by_id[item['review_id']].get('project_id'),
        'source_path': by_id[item['review_id']].get('source_path'),
        'producer_context_status': by_id[item['review_id']].get('context_status'),
        'producer_context_sha256': by_id[item['review_id']].get('context_sha256'),
        'producer_family_sha256': by_id[item['review_id']]['identifier_family_sha256'],
        'source_claim': by_id[item['review_id']]['original_claim'],
        'license_status': by_id[item['review_id']].get('license_status'),
    } for item in items]
    if verify_delivery(pilot) != delivered:
        raise ValueError('Pilot changed during preparation')
    summary = {
        'protocol': VERSION, 'selection_seed': SEED, 'records': len(items),
        'selection': 'round-robin source; deterministic hash order; 100-4000 characters; distinct exact input hashes and producer family hashes',
        'not_a_probability_sample_or_evaluation_cohort': True,
        'source_counts': dict(Counter(r['source'] for r in selected)),
        'producer_context_status_counts': dict(Counter(r.get('context_status', 'missing') for r in selected)),
        'records_without_project_id': sum(not r.get('project_id') for r in selected),
        'actual_full_context_documents_delivered': 0,
        'reviewers_required': 2, 'category_axis_cells_per_reviewer': len(items)*len(CATEGORIES)*len(AXES),
        'training_eligible_records': 0, 'assigned_splits': 0,
        'delivered_file_hashes': delivered, 'delivery_manifest_sha256': sha256(pilot/'DELIVERY_SHA256.json'),
        'candidate_input_sha256': sha256(pilot/'public_output/function_candidates.jsonl'),
        'intake': intake,
        'limits': ['No reviewer annotations, adjudication, full context recovery, project/near-clone certification, license clearance, or training performed.',
                   'Producer family hashes only avoid obvious related selections; they do not establish independent projects or global near-clone separation.',
                   'Source claims are withheld from first-pass files, but code identifiers can reveal origin; blinding is not guaranteed.',
                   'Contextual suspicion and vulnerability decisions remain unknown in this snippet-only version.'],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output/'review_items.jsonl', items)
    write_jsonl(output/'coordinator_key.jsonl', key)
    for reviewer in ('A', 'B'):
        write_jsonl(output/f'reviewer_{reviewer}.jsonl', [blank_packet(item) for item in items])
    (output/'REVIEW_INSTRUCTIONS.md').write_text(REVIEW_GUIDE, encoding='utf-8')
    (output/'blinded_functions.md').write_text(render_blinded_view(items), encoding='utf-8')
    summary['artifact_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    return summary


def validate_reviews(items, reviews):
    expected = {item['review_id']: item for item in items}
    if not expected or len(expected) != len(items):
        raise ValueError('Invalid canonical review item identities')
    if len(reviews) != len(items) or len({r.get('review_id') for r in reviews}) != len(reviews):
        raise ValueError('Review must cover every item exactly once')
    identities, result = set(), {}
    for row in reviews:
        identity = row.get('review_id')
        if identity not in expected or set(row) != set(expected[identity]) | {'reviewer_id','completed','categories'}:
            raise ValueError('Review schema/identities differ from assigned packet')
        for field, value in expected[identity].items():
            if row[field] != value:
                raise ValueError('Immutable review code/context changed')
        reviewer = row['reviewer_id']
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 128 or row['completed'] is not True:
            raise ValueError('Completed reviews require a reviewer identity')
        identities.add(reviewer.strip())
        if not isinstance(row['categories'], dict) or set(row['categories']) != set(CATEGORIES):
            raise ValueError('Every category must be explicitly reviewed')
        decisions = {}
        for category, axes in row['categories'].items():
            if not isinstance(axes, dict) or set(axes) != set(AXES):
                raise ValueError('Every annotation axis must be explicit')
            for axis, cell in axes.items():
                if not isinstance(cell, dict) or set(cell) != {'decision','evidence_spans','reason'}:
                    raise ValueError('Invalid annotation cell schema')
                decision, spans, reason = cell['decision'], cell['evidence_spans'], cell['reason']
                if decision not in ('present','absent','unknown') or not isinstance(reason,str) or not reason.strip():
                    raise ValueError('Each decision requires an explicit value and reason')
                if not isinstance(spans,list) or len(spans) > 32:
                    raise ValueError('Evidence spans must be a bounded list')
                if axis != 'pattern_presence' and row['provided_context'] == 'snippet_only' and decision != 'unknown':
                    raise ValueError('Snippet-only packet cannot establish contextual/vulnerability labels')
                if decision != 'unknown' and not spans:
                    raise ValueError('Observed decisions require exact evidence spans')
                for span in spans:
                    if not isinstance(span,dict) or set(span) != {'start','end','quote'}:
                        raise ValueError('Invalid evidence span schema')
                    start,end,quote = span['start'],span['end'],span['quote']
                    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row['model_code'])
                            or not isinstance(quote,str) or row['model_code'][start:end] != quote):
                        raise ValueError('Evidence must match exact Unicode character offsets')
                decisions[(category,axis)] = decision
        result[identity] = decisions
    if len(identities) != 1:
        raise ValueError('A packet must represent one reviewer')
    return next(iter(identities)), result


def compare(items, left, right):
    left_id, a = validate_reviews(items,left); right_id,b = validate_reviews(items,right)
    if left_id == right_id:
        raise ValueError('Two distinct reviewer identities are required; identity is not proof of independence')
    summary = {}
    disagreements = []
    for category in CATEGORIES:
        for axis in AXES:
            pairs = [(a[rid][category,axis],b[rid][category,axis]) for rid in a]
            known = [(x,y) for x,y in pairs if x!='unknown' and y!='unknown']
            summary[f'{category}/{axis}'] = {
                'total':len(pairs),'both_observed':len(known),
                'both_unknown':sum(x==y=='unknown' for x,y in pairs),
                'agreement_on_both_observed':sum(x==y for x,y in known)/len(known) if known else None,
                'disagreements_including_unknown':sum(x!=y for x,y in pairs),
            }
            disagreements.extend({'review_id':rid,'category':category,'axis':axis,'left':a[rid][category,axis],'right':b[rid][category,axis]}
                                 for rid in a if a[rid][category,axis]!=b[rid][category,axis])
    return {'protocol':VERSION,'reviewer_ids':[left_id,right_id],'agreement':summary,'adjudication_queue':disagreements,
            'training_eligible_records':0,'training_labels_exported':False,
            'next_gate':'Independent review/adjudication plus context, license, grouping and frozen-split gates; agreement alone does not validate labels.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare');make.add_argument('--pilot-dir',type=Path,required=True);make.add_argument('--output-dir',type=Path,required=True);make.add_argument('--per-source',type=int,default=8)
    check=sub.add_parser('compare');check.add_argument('--kit-dir',type=Path,required=True);check.add_argument('--review-a',type=Path,required=True);check.add_argument('--review-b',type=Path,required=True);check.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        result=prepare(args.pilot_dir,args.output_dir,args.per_source)
        print(json.dumps({k:result[k] for k in ('protocol','records','source_counts','category_axis_cells_per_reviewer','training_eligible_records')}))
    else:
        manifest=json.loads((args.kit_dir/'manifest.json').read_text())
        canonical_path=args.kit_dir/'review_items.jsonl'
        if sha256(canonical_path)!=manifest['artifact_sha256']['review_items.jsonl']:
            raise ValueError('Canonical review items changed')
        result=compare(list(jsonl_rows(canonical_path)),list(jsonl_rows(args.review_a)),list(jsonl_rows(args.review_b)))
        with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':'reviews_compared_not_promoted','training_eligible_records':0}))

if __name__=='__main__':main()
