#!/usr/bin/env python3
"""Extend pinned source recovery using a bounded, filename-independent local scan."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import group_function_candidates as grouping
import recover_function_context as recovery
from audit_function_pilot_handoff import jsonl_rows, sha256, verify_delivery
from prepare_context_review import safe_file, verify_artifacts

VERSION = 'bounded-content-context-extension-v1'
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_LISTED = 500_000


def corpus_key(relative):
    parts = Path(relative).parts
    return '/'.join(parts[:2]) if parts[0] == 'paper_benchmarks' else parts[0]


def ordered_paths(raw_root):
    raw_root = raw_root.resolve()
    args = ['rg', '--files', '--null', '-g', '*.sol']
    for name in sorted(recovery.SKIP_DIRS):
        args.extend(['-g', '!**/' + name + '/**'])
    result = subprocess.run(args + [str(raw_root)], capture_output=True, timeout=40, check=False)
    if result.returncode not in (0, 1) or result.stderr:
        raise ValueError('Local listing failed; scope is not established')
    paths = [Path(os.fsdecode(value)) for value in result.stdout.split(b'\0') if value]
    if len(paths) > MAX_LISTED:
        raise ValueError('Path inventory exceeds listing limit')
    buckets = defaultdict(list)
    for path in paths:
        # Do not follow even internal symlinks: a path must be a regular contained input.
        if path.is_symlink() or not path.resolve().is_relative_to(raw_root):
            continue
        buckets[corpus_key(path.relative_to(raw_root))].append(path)
    queues = {key: deque(sorted(values)) for key, values in sorted(buckets.items())}
    ordered = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.popleft())
    return ordered, len(paths)


def scan(raw_root, targets, *, max_bytes, max_files):
    raw_root = raw_root.resolve()
    if type(max_bytes) is not int or not 0 < max_bytes <= 2 * 1024**3 or type(max_files) is not int or not 0 < max_files <= 90_000:
        raise ValueError('Expected positive bounded byte/file budgets')
    paths, listed = ordered_paths(raw_root)
    checked = []; skipped = Counter(); texts = {}; matches = defaultdict(list); total = 0; stop = 'listed_scope_exhausted'
    for path in paths:
        if len(checked) >= max_files:
            stop = 'read_file_budget'; break
        if path.is_symlink() or not path.resolve().is_relative_to(raw_root):
            raise ValueError('Source path changed containment during scan')
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped['oversized'] += 1; continue
        if total + size > max_bytes:
            stop = 'total_byte_budget'; break
        with path.open('rb') as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) != size or len(data) > MAX_FILE_BYTES:
            raise ValueError('Source changed size during scan')
        total += len(data)
        entry = {'local_relative_path': str(path.relative_to(raw_root)), 'local_bytes_sha256': hashlib.sha256(data).hexdigest(),
                 'bytes': len(data), 'decoded_text_sha256': None, 'decoding': 'invalid_utf8'}
        try:
            text = data.decode('utf-8-sig')
            entry.update(decoded_text_sha256=recovery.text_hash(text), decoding='utf-8-sig')
        except UnicodeDecodeError:
            skipped['invalid_utf8'] += 1
        checked.append(entry)
        digest = entry['decoded_text_sha256']
        if digest in targets:
            matches[digest].append(entry); texts[digest] = text
    summary = {'ordering': 'sorted-corpus-round-robin-sorted-relative-path-v1', 'listed_solidity_paths': listed,
               'contained_non_symlink_paths': len(paths), 'checked_files': len(checked), 'read_bytes': total,
               'max_read_bytes': max_bytes, 'max_read_files': max_files, 'max_file_bytes': MAX_FILE_BYTES,
               'stop_reason': stop, 'checked_corpus_counts': dict(Counter(corpus_key(e['local_relative_path']) for e in checked)),
               'skipped': dict(skipped), 'scope_exhaustive': stop == 'listed_scope_exhausted'}
    return checked, dict(matches), texts, summary


def verify_checked(raw_root, checked):
    raw_root = raw_root.resolve()
    for entry in checked:
        original_path = raw_root / entry['local_relative_path']
        path = safe_file(raw_root, entry['local_relative_path'])
        if original_path.is_symlink() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['local_bytes_sha256']:
            raise ValueError('Inventoried source changed after scan')


def prepare(pilot, raw_root, prior, view, output, *, max_bytes=2 * 1024**3, max_files=90_000):
    pilot, raw_root, prior, view, output = [p.resolve() for p in (pilot, raw_root, prior, view, output)]
    if output == pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot / 'derivatives')) or any(output == root or output.is_relative_to(root) for root in (raw_root, prior, view)):
        raise ValueError('Preserve originals and ancestors; use new versioned output')
    if output.exists():
        raise FileExistsError(output)
    delivered = verify_delivery(pilot)
    prior_id = sha256(safe_file(prior, 'recovery_manifest.json')); pm = verify_artifacts(prior, 'recovery_manifest.json', 'artifact_sha256')
    view_id = sha256(safe_file(view, 'manifest.json')); vm = verify_artifacts(view, 'manifest.json', 'artifact_sha256')
    if pm['delivery_manifest_sha256'] != sha256(pilot / 'DELIVERY_SHA256.json') or vm['delivery_manifest_sha256'] != pm['delivery_manifest_sha256']:
        raise ValueError('Parent delivery identity differs')
    if pm.get('raw_root') != str(raw_root):
        raise ValueError('Raw root differs from pinned prior recovery')
    rows = list(jsonl_rows(pilot / 'public_output/function_candidates.jsonl')); by_id = {r['record_id']: r for r in rows}
    views = list(jsonl_rows(view / 'function_model_views.jsonl')); links = list(jsonl_rows(prior / 'candidate_context_links.jsonl'))
    if len(by_id) != len(rows) or any(len(data) != len(rows) or {r['record_id'] for r in data} != set(by_id) for data in (views, links)):
        raise ValueError('Parent candidate inventories differ')
    if any(r['model_code_sha256'] != recovery.text_hash(r['model_code']) or r.get('parent_model_code_sha256') != recovery.text_hash(by_id[r['record_id']]['model_code']) for r in views):
        raise ValueError('Current model view is not bound to delivered input')
    for row in [*rows, *views, *links]:
        if any(value is not None for value in row['pattern_labels'].values()) or row['training_eligible'] is not False or row['split'] is not None or any(row.get('label_observation_mask', {}).values()) or row.get('human_review_status', 'unreviewed') != 'unreviewed':
            raise ValueError('Reviewed or assigned records require explicit migration')
    context_rows = list(jsonl_rows(prior / 'context_manifest.jsonl')); old_texts = {}
    for row in context_rows:
        path = safe_file(prior, row['context_file']); digest = row['context_sha256']
        if sha256(path) != digest:
            raise ValueError('Prior raw context hash differs')
        old_texts[digest] = path.read_bytes().decode('utf-8')
    for link in links:
        if link['context_file']:
            original = by_id[link['record_id']]
            if link['pinned_context_sha256'] != original.get('context_sha256') or sha256(safe_file(prior, link['context_file'])) != original['context_sha256']:
                raise ValueError('Prior context seed is not original evidence')
    targets = {r['context_sha256'] for r in rows if r.get('context_sha256')} - set(old_texts)
    checked, matches, texts, inventory = scan(raw_root, targets, max_bytes=max_bytes, max_files=max_files)
    changed_ids = []
    for link in links:
        original = by_id[link['record_id']]; digest = original.get('context_sha256')
        if link['context_file'] or digest not in texts:
            continue
        spans, status = recovery.exact_function_spans(original, texts[digest])
        link.update(context_status='verified_local_exact_text', context_file='contexts/' + digest + '.sol',
                    fragment_localization=status, function_spans=spans,
                    local_corpus_overlap_hints=[recovery.local_overlap_hint(e['local_relative_path']) for e in matches[digest]],
                    evaluation_status='quarantine_existing_corpus_overlap', semantic_context_sufficiency='not_reviewed')
        changed_ids.append(link['record_id'])
    for digest in sorted(texts):
        pins = sorted({(r['source'], r['source_revision'], r['source_path']) for r in rows if r.get('context_sha256') == digest})
        context_rows.append({'context_sha256': digest, 'context_file': 'contexts/' + digest + '.sol', 'stored_encoding': 'utf-8',
                             'pins': [{'source': a, 'revision': b, 'source_path': c} for a, b, c in pins], 'local_evidence': matches[digest],
                             'semantic_context_sufficiency': 'not_reviewed', 'warning': 'Raw source may reveal labels; not a blinded reviewer/model input. Corpus presence is not fresh holdout evidence.'})
    seeds = {r['record_id'] for r in links if r['context_file']}
    members, edges, components, group_summary = grouping.build_groups(views, seeds)
    prior_seeds = seeds - set(changed_ids)
    before_groups = grouping.build_groups(views, prior_seeds)[3]
    # Freeze a rechecked inventory, not an inference from similar names or token hashes.
    verify_checked(raw_root, checked)
    if verify_delivery(pilot) != delivered or sha256(prior / 'recovery_manifest.json') != prior_id or sha256(view / 'manifest.json') != view_id:
        raise ValueError('Input identities changed during extension')
    verify_artifacts(prior, 'recovery_manifest.json', 'artifact_sha256'); verify_artifacts(view, 'manifest.json', 'artifact_sha256')
    summary = {'protocol': VERSION, 'candidate_count': len(rows), 'prior_context_documents': len(old_texts),
               'new_context_documents': len(texts), 'recovered_context_documents': len(old_texts) + len(texts),
               'prior_covered_candidates': len(prior_seeds), 'new_covered_candidates': len(changed_ids), 'candidates_with_verified_context': len(seeds),
               'newly_covered_ids': sorted(changed_ids), 'new_fragment_localization_counts': dict(Counter(r['fragment_localization'] for r in links if r['record_id'] in set(changed_ids))),
               'new_covered_source_counts': dict(Counter(by_id[x]['source'] for x in changed_ids)), 'scan': inventory,
               'prior_current_view_groups': before_groups, 'recomputed_current_view_groups': group_summary,
               'independent_reviews_completed': 0, 'pattern_labels_promoted': 0, 'training_eligible_records': 0, 'assigned_splits': 0,
               'limitations': ['Exact whole-file text and original function-span hashes verify lexical identity, not compiler validity, dependency closure, license clearance, independent label truth or source authenticity beyond pinned producer evidence.',
                              'All newly recovered local-source records are direct corpus-overlap seeds, not fresh evaluation data. Group-propagated quarantine is a conservative constraint, not independently verified additional overlap.',
                              'A deterministic byte/file-bounded scan using default rg ignore rules is not an exhaustive disk search. Unknown labels remain unobserved, never negative. No model training or performance gain.']}
    output.mkdir(parents=True, exist_ok=False); (output / 'contexts').mkdir(); (output / 'groups').mkdir(); (output / 'checked_inventory').mkdir()
    for digest, text in {**old_texts, **texts}.items():
        path = output / 'contexts' / (digest + '.sol'); path.write_bytes(text.encode('utf-8'))
        if sha256(path) != digest: raise ValueError('Stored context hash differs')
    for name, data in [('candidate_context_links.jsonl', links), ('context_manifest.jsonl', sorted(context_rows, key=lambda r:r['context_sha256'])),
                       ('groups/candidate_groups.jsonl', members), ('groups/constraint_edges.jsonl', edges), ('groups/components.jsonl', components)]:
        recovery.write_jsonl(output / name, data)
    for start in range(0, len(checked), 5000):
        recovery.write_jsonl(output / 'checked_inventory' / ('part_%04d.jsonl' % (start // 5000)), checked[start:start+5000])
    (output / 'groups/summary.json').write_text(json.dumps(group_summary, indent=2, sort_keys=True) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    manifest = {'protocol': VERSION, 'generator_sha256': sha256(Path(__file__)),
                'helper_sha256': {Path(module.__file__).name: sha256(Path(module.__file__)) for module in (recovery, grouping)},
                'grouping_comment_masker_sha256':sha256(Path(grouping.masked_source_and_imports.__code__.co_filename)),
                'delivery_manifest_sha256': sha256(pilot / 'DELIVERY_SHA256.json'), 'delivered_file_hashes': delivered,
                'prior_recovery_manifest_sha256': prior_id, 'model_view_manifest_sha256': view_id, 'raw_root': str(raw_root),
                'artifact_sha256': {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    (output / 'recovery_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir', 'raw-root', 'prior-recovery-dir', 'view-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=2 * 1024**3); parser.add_argument('--max-files', type=int, default=90_000)
    args = parser.parse_args()
    result = prepare(args.pilot_dir, args.raw_root, args.prior_recovery_dir, args.view_dir, args.output_dir, max_bytes=args.max_bytes, max_files=args.max_files)
    print(json.dumps({k:v for k,v in result.items() if k not in ('newly_covered_ids','prior_current_view_groups','recomputed_current_view_groups')}, indent=2))

if __name__ == '__main__': main()
