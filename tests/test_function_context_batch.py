"""Software fixtures only; no research labels, model fitting, or evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from function_context_assessment import VERSION, TARGET, validate_context_assessment
from function_context_batch import (CATEGORIES, PROFILE, collate_context_examples,
                                    load_context_batch)
from function_partial_supervision import observed_bce_with_logits

torch.set_num_threads(1)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fixture(identity='fixture'):
    code = 'function f() external {\r\n    sensitive();\r\n}'
    model = dict(record_id=identity, model_code=code, model_code_sha256=digest(code),
                 input_protocol='comment-masked-source-CRLF-preserved-v1')
    assessment = dict(protocol=VERSION, target_definition_id=TARGET,
        label_origin='AI-assigned', code_origin='real-public',
        model_code_sha256=digest(code), training_eligible=False,
        evaluation_eligible=False, independent_gold=False, split=None,
        record_id=identity, method_model_version='software-fixture-only',
        source_provenance_ref='software fixture, not real data',
        source_claim_disposition='original claim MUST NOT ENTER MODEL INPUT',
        admission_screen_ref='software fixture, no admission',
        parent_release_sha256='0' * 64, context_hashes={'unit.sol': digest(code)},
        decisions={})
    for category, status in zip(CATEGORIES, ['negative', 'context-required', 'positive']):
        assessment['decisions'][category] = dict(status=status,
            behavior='Software boundary fixture only', protections='Fixture only',
            scope='No research claim', rationale='MUST NOT ENTER MODEL INPUT',
            missing_evidence=['callee context'] if status == 'context-required' else [],
            evidence=[dict(context_id='unit.sol', start=0, end=len(code), quote=code)])
    contexts = {'unit.sol': code}
    assessment['export'] = validate_context_assessment(code, assessment, contexts)
    return model, assessment, contexts


class ContextBatchTests(unittest.TestCase):
    def batch(self, rows):
        return collate_context_examples(rows, for_training=False)

    def release(self, root, rows=None):
        rows = rows or [fixture()]
        (root / 'unit.sol').write_bytes(rows[0][2]['unit.sol'].encode('utf-8'))
        for filename, values in [('unified_functions.jsonl', [
            dict(record_id=m['record_id'], model_input={k:v for k,v in m.items() if k != 'record_id'})
            for m, _, _ in rows]), ('context_assessments.jsonl', [a for _, a, _ in rows])]:
            (root / filename).write_text(''.join(json.dumps(v) + '\n' for v in values))
        files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
        (root / 'manifest.json').write_text(json.dumps(dict(profile=PROFILE, files=files)))
        return hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()

    def test_masks_code_only_crlf_and_input_immutability(self):
        rows = [fixture()]; before = copy.deepcopy(rows)
        b = self.batch(rows)
        self.assertEqual(b.model_codes, (rows[0][0]['model_code'],))
        self.assertIn('\r\n', b.model_codes[0])
        self.assertNotIn('MUST NOT ENTER', b.model_codes[0])
        self.assertEqual(b.targets.tolist(), [[0., 0., 1.]])
        self.assertEqual(b.observation_mask.tolist(), [[True, False, True]])
        self.assertEqual(b.target_definition_id, TARGET)
        self.assertFalse(b.for_training); self.assertEqual(rows, before)

    def test_existing_loss_gives_no_unknown_gradient(self):
        b = self.batch([fixture()]); logits = torch.zeros_like(b.targets, requires_grad=True)
        result = observed_bce_with_logits(logits, b.targets, b.observation_mask, categories=b.categories)
        result.loss.backward()
        self.assertEqual(result.observed_cells, 2)
        self.assertEqual(logits.grad.tolist(), [[.25, 0., -.25]])

    def test_training_refused_before_filesystem_access(self):
        for purpose in [True, None, 0, 'diagnostic']:
            with self.subTest(purpose=purpose), self.assertRaisesRegex(ValueError, 'diagnostic only'):
                load_context_batch('/nonexistent', expected_manifest_sha256='0'*64,
                                   record_ids=['fixture'], for_training=purpose)
        with self.assertRaises(ValueError): collate_context_examples([fixture()])

    def test_swapped_id_or_code_hash_rejected(self):
        for key, value in [('record_id', 'other'), ('model_code_sha256', '1'*64), ('model_code', 'changed')]:
            m, a, c = fixture(); m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.batch([(m,a,c)])

    def test_stale_export_or_forged_evidence_rejected(self):
        m, a, c = fixture(); a['export']['labels']['authorization'] = False
        with self.assertRaisesRegex(ValueError, 'Cached export'): self.batch([(m,a,c)])
        m, a, c = fixture(); a['decisions']['authorization']['evidence'][0]['quote'] = 'wrong'
        with self.assertRaises(ValueError): self.batch([(m,a,c)])

    def test_changed_context_and_promoted_flags_rejected(self):
        m, a, c = fixture(); c['unit.sol'] += ' '
        with self.assertRaises(ValueError): self.batch([(m,a,c)])
        for key in ['training_eligible', 'evaluation_eligible', 'independent_gold']:
            m,a,c = fixture(); a[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): self.batch([(m,a,c)])

    def test_extra_cue_fields_unsupported_protocol_duplicates_and_empty_rejected(self):
        m,a,c = fixture(); m['audit_comment'] = 'positive'
        with self.assertRaises(ValueError): self.batch([(m,a,c)])
        m,a,c = fixture(); m['input_protocol'] = 'legacy'
        with self.assertRaises(ValueError): self.batch([(m,a,c)])
        for rows in [[], [fixture(), fixture()]]:
            with self.assertRaises(ValueError): self.batch(rows)

    def test_pinned_loader_read_only_and_selection_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); h = self.release(root, [fixture('a'), fixture('b')])
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            b = load_context_batch(root, expected_manifest_sha256=h, record_ids=['b','a'], for_training=False)
            self.assertEqual(b.record_ids, ('b','a'))
            self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})
            for ids in [[], ['a','a'], ['missing']]:
                with self.assertRaises(ValueError):
                    load_context_batch(root, expected_manifest_sha256=h, record_ids=ids, for_training=False)

    def test_release_tampering_and_unlisted_context_rejected(self):
        for change in ['code', 'manifest', 'unlisted_context', 'path_escape']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); h=self.release(root)
                if change == 'code': (root/'unit.sol').write_text('changed')
                else:
                    p=json.loads((root/'manifest.json').read_text())
                    if change=='manifest': p['profile']='other'
                    elif change=='unlisted_context': del p['files']['unit.sol']
                    else: p['files']['../outside.sol']='0'*64
                    (root/'manifest.json').write_text(json.dumps(p))
                    if change != 'manifest': h=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    load_context_batch(root, expected_manifest_sha256=h, record_ids=['fixture'], for_training=False)

    def test_duplicate_inventory_rows_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); h=self.release(root, [fixture(), fixture()])
            with self.assertRaisesRegex(ValueError, 'duplicate inventory'):
                load_context_batch(root, expected_manifest_sha256=h, record_ids=['fixture'], for_training=False)


if __name__ == '__main__': unittest.main()
