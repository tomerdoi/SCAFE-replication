"""Audit a source-free compiler-identity projection; this never admits a source."""
from __future__ import annotations
import re

MINIMAL = {'match', 'creationMatch', 'runtimeMatch', 'chainId', 'address', 'verifiedAt', 'matchId'}
COMPILATION = {'language', 'compiler', 'compilerVersion', 'compilerSettings', 'name', 'fullyQualifiedName'}
VERSION = re.compile(r'^v?(\d+\.\d+\.\d+\+commit\.[0-9a-f]{8})$')

def audit(payload, *, expected_chain, expected_address, approved_builds):
    if not isinstance(payload, dict) or set(payload) - (MINIMAL | {'compilation', 'sourceIds'}):
        raise ValueError('Unexpected fields: response is not the declared source-free projection')
    if payload.get('chainId') != expected_chain or str(payload.get('address', '')).lower() != expected_address.lower():
        raise ValueError('Wrong contract identity')
    compilation = payload.get('compilation')
    if compilation is not None and (not isinstance(compilation, dict) or set(compilation) - COMPILATION):
        raise ValueError('Unexpected compilation fields')
    c = compilation or {}
    settings = c.get('compilerSettings')
    if settings is not None and not isinstance(settings, dict):
        raise ValueError('Invalid compiler settings')
    def check_no_implementation(value):
        if isinstance(value, dict):
            if {'content', 'ast', 'sources', 'stdJsonInput', 'stdJsonOutput'} & set(value):
                raise ValueError('Unexpected implementation-bearing field')
            for v in value.values(): check_no_implementation(v)
        elif isinstance(value, list):
            for v in value: check_no_implementation(v)
    check_no_implementation(settings)
    source_ids = payload.get('sourceIds')
    if source_ids is not None:
        if not isinstance(source_ids, dict): raise ValueError('Invalid source-ID mapping')
        ids = []
        for path, item in source_ids.items():
            if not isinstance(path, str) or not isinstance(item, dict) or set(item) != {'id'} or type(item['id']) is not int or item['id'] < 0:
                raise ValueError('Invalid source-ID entry')
            ids.append(item['id'])
        if len(ids) != len(set(ids)): raise ValueError('Duplicate source IDs')
    v = c.get('compilerVersion')
    exact = VERSION.fullmatch(v) if isinstance(v, str) else None
    build = exact.group(1) if exact else None
    local = (c.get('compiler') == 'solc' and build in approved_builds) if build else None
    target = c.get('fullyQualifiedName')
    target_path = target.rsplit(':', 1)[0] if isinstance(target, str) and ':' in target else None
    observed_sources = len(source_ids) if source_ids is not None else None
    return {
        'compiler': c.get('compiler'), 'language': c.get('language'), 'reported_version': v,
        'exact_build_observed': build, 'approved_local_exact_build': local,
        'approved_build_match_basis': 'Caller-supplied approved-build list; binary presence/hash/version require separate verification.',
        'compiler_settings_observed': settings is not None,
        'source_id_entries_observed': observed_sources,
        'target_path_in_observed_source_ids': (target_path in source_ids) if target_path is not None and source_ids is not None else None,
        'source_ids_are_structural_asts': False,
        'per_source_content_hashes_observed': False, 'per_source_licenses_observed': False,
        'independent_project_family_admission_observed': False,
        'structural_reference_oracle_observed': False, 'positive_source_admission': False,
        'missingness_policy': 'These facts are absent from the requested projection, not demonstrated absent from the contract or service.',
        'decision': 'Projection alone is insufficient for positive source admission or a structural reference AST.'
    }

if __name__ == '__main__':
    import argparse, hashlib, json
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--chain', required=True)
    p.add_argument('--address', required=True)
    p.add_argument('--approved-build', action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r'\d{1,20}', args.chain) or not re.fullmatch(r'0x[0-9a-fA-F]{40}', args.address):
        raise ValueError('Invalid expected identity')
    if any(not VERSION.fullmatch(x) or VERSION.fullmatch(x).group(1) != x for x in args.approved_build):
        raise ValueError('Approved build must be an exact canonical version+commit')
    if args.input.stat().st_size > 250000: raise ValueError('Oversized metadata projection')
    if args.output.exists() or args.output.is_symlink(): raise FileExistsError('Output must be new')
    raw = args.input.read_bytes()
    result = audit(json.loads(raw), expected_chain=args.chain, expected_address=args.address, approved_builds=args.approved_build)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
