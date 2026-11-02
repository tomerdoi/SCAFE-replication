#!/usr/bin/env python3
"""Recover only hash-pinned function context from existing local Solidity corpora."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from audit_function_pilot_handoff import inspect_staging_rows, jsonl_rows, sha256, verify_delivery

VERSION = 'local-exact-context-recovery-v1'
MAX_BYTES = 16 * 1024 * 1024
SKIP_DIRS = {'.git', 'node_modules', 'venv', '.venv', '__pycache__'}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def scan_local(raw_root, names):
    """Bounded filename prefilter; only selected files are read, never executed."""
    raw_root = raw_root.resolve()
    args = ['rg', '--files', '--null', '-g', '*.sol']
    for directory in sorted(SKIP_DIRS):
        args.extend(['-g', '!**/'+directory+'/**'])
    args.append(str(raw_root))
    listing = subprocess.run(args, capture_output=True, timeout=40, check=False)
    if listing.returncode not in (0, 1) or listing.stderr:
        raise ValueError('Bounded local file listing failed; do not claim complete search')
    paths = [Path(os.fsdecode(value)) for value in listing.stdout.split(b'\0') if value]
    if len(paths) > 500000:
        raise ValueError('Solidity inventory exceeds bounded budget')
    checked, skipped = [], Counter()
    for path in sorted(paths):
        if path.name not in names:
            continue
        if not path.resolve().is_relative_to(raw_root):
            skipped['path_outside_root'] += 1
            continue
        if path.stat().st_size > MAX_BYTES:
            skipped['oversized'] += 1
            continue
        data = path.read_bytes()
        if len(data) > MAX_BYTES:
            raise ValueError('Local source exceeded size budget while reading')
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            skipped['invalid_utf8'] += 1
            continue
        relative = str(path.relative_to(raw_root))
        checked.append({'local_relative_path': relative, 'local_bytes_sha256': hashlib.sha256(data).hexdigest(),
                        'decoded_text_sha256': text_hash(text), 'bytes': len(data), 'decoding': 'utf-8-sig'})
    return checked, {'solidity_paths_enumerated':len(paths),'listing_method':'rg --files --null; default ignore rules plus explicit directory exclusions','skipped':dict(skipped)}


def exact_function_spans(row, text):
    """Verify existing fragment hashes/lengths, without pretending to parse Solidity."""
    size, expected = row.get('characters'), row.get('raw_sha256')
    if type(size) is not int or not 0 < size <= MAX_BYTES or not isinstance(expected,str):
        return [], 'missing_fragment_identity'
    starts = []
    if row['source'] == 'CoinFabrik/solidity-rnd':
        suffix = row['source_identity'].rpartition(':')[2]
        if suffix.isdecimal():
            starts = [int(suffix)]
    elif row['source'] == 'sujeetc/SCRUBD':
        name = str(row.get('function_identifier','')).rpartition('.')[2]
        if re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', name):
            starts = [match.start() for match in re.finditer(r'\bfunction\s+' + re.escape(name) + r'\s*\(', text)]
    spans=[]
    for start in starts:
        end = start + size
        if 0 <= start < end <= len(text) and text_hash(text[start:end]) == expected:
            spans.append({'start_character':start,'end_character':end,'raw_fragment_sha256':expected,
                          'start_line':text.count('\n',0,start)+1,'end_line':text.count('\n',0,end-1)+1})
    return spans, 'exact_unique_fragment' if len(spans)==1 else ('ambiguous_identical_fragments' if spans else 'fragment_not_localized')


def local_overlap_hint(relative):
    parts = Path(relative).parts
    if parts[0] == 'FORGE-Artifacts-main':
        return {'local_corpus':'FORGE-Artifacts-main','audit_directory_hint':parts[3] if len(parts)>3 and parts[1:3]==('dataset','contracts') else None}
    if len(parts)>1 and parts[0]=='paper_benchmarks':
        return {'local_corpus':'paper_benchmarks/'+parts[1], 'audit_directory_hint':None}
    return {'local_corpus':parts[0], 'audit_directory_hint':None}


def recover(pilot, raw_root, output):
    pilot,raw_root,output = pilot.resolve(),raw_root.resolve(),output.resolve()
    if output == pilot or (output.is_relative_to(pilot) and not output.is_relative_to(pilot/'derivatives')):
        raise ValueError('Only new versioned pilot derivatives may be written')
    if output == raw_root or output.is_relative_to(raw_root):
        raise ValueError('Original local corpora must remain read-only')
    if not raw_root.is_dir():
        raise ValueError('Authorized raw source root is unavailable')
    delivered = verify_delivery(pilot)
    rows = list(jsonl_rows(pilot/'public_output/function_candidates.jsonl'))
    intake = inspect_staging_rows(rows)
    source_manifest = json.loads((pilot/'source_manifest.json').read_text())
    for row in rows:
        if row['source_revision'] != source_manifest['sources'][row['source']]['revision']:
            raise ValueError('Candidate source revision differs from pinned manifest')
    names={Path(row['source_path']).name for row in rows if row.get('source_path')}
    expected={row['context_sha256'] for row in rows if row.get('context_sha256')}
    if any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in expected):
        raise ValueError('Invalid pinned context hash')
    checked, inventory = scan_local(raw_root,names)
    matches = defaultdict(list)
    for evidence in checked:
        if evidence['decoded_text_sha256'] in expected:
            matches[evidence['decoded_text_sha256']].append(evidence)
    texts = {}
    for digest,evidence in matches.items():
        data=(raw_root/evidence[0]['local_relative_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=evidence[0]['local_bytes_sha256']:
            raise ValueError('Local context changed after inventory')
        text=data.decode('utf-8-sig')
        if text_hash(text)!=digest:
            raise ValueError('Context identity changed after decoding')
        texts[digest]=text
    links=[]
    for row in rows:
        digest=row.get('context_sha256');evidence=matches.get(digest,[])
        spans,status=exact_function_spans(row,texts[digest]) if evidence else ([], 'context_unavailable')
        links.append({'record_id':row['record_id'],'source':row['source'],'source_revision':row['source_revision'],
                      'source_identity':row['source_identity'],'original_project_id':row.get('project_id'),
                      'license_status':row.get('license_status'),'pinned_context_sha256':digest,
                      'context_status':'verified_local_exact_text' if evidence else ('no_exact_local_match' if digest else 'no_pinned_context_identity'),
                      'context_file':'contexts/'+digest+'.sol' if evidence else None,
                      'fragment_localization':status,'function_spans':spans,
                      'local_corpus_overlap_hints':[local_overlap_hint(e['local_relative_path']) for e in evidence],
                      'evaluation_status':'quarantine_existing_corpus_overlap' if evidence else 'unassessed_not_evaluation_ready',
                      'semantic_context_sufficiency':'not_reviewed','pattern_labels':row['pattern_labels'],
                      'training_eligible':False,'split':None})
    context_rows=[]
    for digest in sorted(matches):
        pins=sorted({(row['source'],row['source_revision'],row['source_path']) for row in rows if row.get('context_sha256')==digest})
        context_rows.append({'context_sha256':digest,'context_file':'contexts/'+digest+'.sol','stored_encoding':'utf-8',
                             'pins':[{'source':source,'revision':revision,'source_path':path} for source,revision,path in pins],
                             'local_evidence':matches[digest], 'semantic_context_sufficiency':'not_reviewed',
                             'warning':'Source comments may reveal claims; raw evidence is not a sanitized model input or a blinded annotation packet.'})
    # Check all original files and every inventoried selected source again, not just copied matches.
    if verify_delivery(pilot)!=delivered:
        raise ValueError('Delivered pilot changed during recovery')
    for evidence in checked:
        path=(raw_root/evidence['local_relative_path']).resolve()
        if not path.is_relative_to(raw_root) or sha256(path)!=evidence['local_bytes_sha256']:
            raise ValueError('Inventoried local source changed during recovery')
    covered=[link for link in links if link['context_file']]
    summary={'protocol':VERSION,'candidate_count':len(rows),'unique_pinned_context_hashes':len(expected),
             'recovered_context_documents':len(matches),'matching_local_files':sum(map(len,matches.values())),
             'candidates_with_verified_context':len(covered),'candidates_without_verified_context':len(rows)-len(covered),
             'covered_source_counts':dict(Counter(link['source'] for link in covered)),
             'fragment_localization_counts':dict(Counter(link['fragment_localization'] for link in covered)),
             'existing_corpus_overlap_quarantined_candidates':len(covered),'training_eligible_records':0,'assigned_splits':0,
             'intake':intake,'inventory':inventory,'basename_matched_files_checked':len(checked),
             'limitations':['Exact text identity verifies bytes, not source authenticity beyond the pinned producer manifest, semantic context sufficiency, compiler validity or annotation truth.',
                            'Recovered files may still lack imported declarations/callers/callees. Function span identity is a lexical exact-hash check, not AST or overload/contract resolution.',
                            'Local corpus presence is flagged; actual historical training/test membership and all clone/project relationships remain unaudited. These are not fresh independent holdouts.',
                            'No new project IDs, license clearance, pattern labels, splits or eligibility were inferred. Existing review-pilot-v1 packets remain immutable and snippet-only.',
                            'Filename-prefiltered, pruned local search is bounded and not exhaustive content recovery. No downloads or source execution occurred.']}
    output.mkdir(parents=True,exist_ok=False);(output/'contexts').mkdir()
    for digest,text in texts.items():
        path=output/'contexts'/f'{digest}.sol';path.write_bytes(text.encode('utf-8'));assert sha256(path)==digest
    write_jsonl(output/'candidate_context_links.jsonl',links)
    write_jsonl(output/'context_manifest.jsonl',context_rows)
    write_jsonl(output/'local_source_inventory.jsonl',checked)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    artifacts={str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest={'protocol':VERSION,'delivery_manifest_sha256':sha256(pilot/'DELIVERY_SHA256.json'),
              'source_manifest_sha256':sha256(pilot/'source_manifest.json'),'delivered_file_hashes':delivered,
              'raw_root':str(raw_root),'search_basename_count':len(names),'skip_directories':sorted(SKIP_DIRS),
              'artifact_sha256':artifacts}
    (output/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir',type=Path,required=True);parser.add_argument('--raw-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();result=recover(args.pilot_dir,args.raw_root,args.output_dir)
    print(json.dumps({key:result[key] for key in ('recovered_context_documents','matching_local_files','candidates_with_verified_context','covered_source_counts','fragment_localization_counts','training_eligible_records')},indent=2))

if __name__=='__main__':main()
