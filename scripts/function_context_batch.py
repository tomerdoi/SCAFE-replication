"""Hash-bound diagnostic batching of contextual proposals; never training intake.

Reuses the contextual evidence validator and existing observed-cell loss. Source
claims and assessment prose remain outside model input strings. This interface
does not certify source rights, grouping, quality, or independent evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import torch

from audit_function_pilot_handoff import jsonl_rows, sha256
from function_context_assessment import TARGET, validate_context_assessment
from prepare_context_review import safe_file, verify_artifacts
from prepare_function_review import CATEGORIES

VERSION = 'context-proposal-diagnostic-batch-v1'
PROFILE = 'auth-source-pool-admission-v1'
INPUT_PROTOCOLS = frozenset({
    'post-statement-and-if-block-inline-audit-tail-v2',
    'comment-masked-source-CRLF-preserved-v1',
    'comment-masked-exact-span-v1',
    'exact-native-function-CRLF-preserved-no-annotation-v1',
})
MODEL_FIELDS = {'record_id', 'model_code', 'model_code_sha256', 'input_protocol'}


@dataclass(frozen=True)
class ContextDiagnosticBatch:
    record_ids: tuple[str, ...]
    model_codes: tuple[str, ...]
    input_protocols: tuple[str, ...]
    targets: torch.Tensor
    observation_mask: torch.Tensor
    categories: tuple[str, ...] = CATEGORIES
    target_definition_id: str = TARGET
    for_training: bool = False


def _diagnostic_only(for_training):
    if for_training is not False:
        raise ValueError('Context proposals are diagnostic only; training admission is not implemented')


def collate_context_examples(examples, *, for_training=True):
    """Validate (code-only view, assessment, exact contexts) triples on CPU.

    Explicit diagnostic purpose is mandatory. Unknown/context-required cells
    have inert zero placeholders AND false masks; zero is not a negative label.
    """
    _diagnostic_only(for_training)
    ids, codes, protocols, targets, masks = [], [], [], [], []
    for model, assessment, contexts in examples:
        if set(model) != MODEL_FIELDS or model['input_protocol'] not in INPUT_PROTOCOLS:
            raise ValueError('Pinned code-only model view required')
        identity = model['record_id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Nonempty unique record IDs required')
        if identity != assessment.get('record_id') or model['model_code_sha256'] != assessment.get('model_code_sha256'):
            raise ValueError('Assessment/input identity mismatch')
        export = validate_context_assessment(model['model_code'], assessment, contexts)
        if 'export' in assessment and assessment['export'] != export:
            raise ValueError('Cached export differs from verified contextual decisions')
        ids.append(identity)
        codes.append(model['model_code'])
        protocols.append(model['input_protocol'])
        targets.append([float(export['labels'][c]) if export['observation_mask'][c] else 0.0 for c in CATEGORIES])
        masks.append([export['observation_mask'][c] for c in CATEGORIES])
    if not ids:
        raise ValueError('Empty diagnostic batch')
    return ContextDiagnosticBatch(tuple(ids), tuple(codes), tuple(protocols),
                                  torch.tensor(targets, dtype=torch.float32),
                                  torch.tensor(masks, dtype=torch.bool))


def load_context_batch(release, *, expected_manifest_sha256, record_ids, for_training=True):
    """Read only selected existing records from an explicitly pinned release.

    Contexts must be manifest-listed and match both logical name and content
    hash. No current/upstream substitute, network access, write, or model call.
    """
    _diagnostic_only(for_training)
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256):
        raise ValueError('Explicit expected manifest SHA256 required')
    ids = tuple(record_ids)
    if not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Explicit nonempty unique selection required')
    root = Path(release).resolve()
    manifest_path = safe_file(root, 'manifest.json')
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError('Release identity mismatch')
    manifest = verify_artifacts(root, 'manifest.json', 'files')
    if manifest.get('profile') != PROFILE:
        raise ValueError('Unsupported contextual release profile')
    for name in ('unified_functions.jsonl', 'context_assessments.jsonl'):
        if name not in manifest['files']:
            raise ValueError('Required inventory is not manifest-bound')

    def index(name):
        result = {}
        for row in jsonl_rows(safe_file(root, name)):
            identity = row.get('record_id')
            if not isinstance(identity, str) or not identity or identity in result:
                raise ValueError('Invalid or duplicate inventory record ID')
            result[identity] = row
        return result

    models = index('unified_functions.jsonl')
    assessments = index('context_assessments.jsonl')
    if not set(ids) <= models.keys() or not set(ids) <= assessments.keys():
        raise ValueError('Selected ID missing from input or assessment inventory')
    examples = []
    for identity in ids:
        view = models[identity].get('model_input', {})
        if set(view) != MODEL_FIELDS - {'record_id'}:
            raise ValueError('Unexpected model-input fields')
        assessment = assessments[identity]
        contexts = {}
        for name, digest in assessment.get('context_hashes', {}).items():
            matches = [path for path, value in manifest['files'].items()
                       if Path(path).name == name and value == digest]
            if not matches:
                raise ValueError('Exact declared context is not manifest-bound')
            # Identical named bytes may be inherited in multiple evidence folders.
            text = safe_file(root, sorted(matches)[0]).read_bytes().decode('utf-8')
            if hashlib.sha256(text.encode('utf-8')).hexdigest() != digest:
                raise ValueError('Context changed while loading')
            contexts[name] = text
        examples.append(({'record_id': identity, **view}, assessment, contexts))
    batch = collate_context_examples(examples, for_training=False)
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError('Manifest changed while loading')
    verify_artifacts(root, 'manifest.json', 'files')
    return batch
