"""Metadata-only software tests; no research examples, model training or evaluation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from function_real_public_origin import POLICY, require_real_public_origin


def metadata_fixture():
    model = {'record_id': 'metadata-test', 'model_code_sha256': 'a' * 64}
    target = {**model, 'source': 'metadata-only-test', 'source_revision': 'pinned-test'}
    target['real_public_origin_decision'] = {
        **target, 'policy': POLICY, 'status': 'accepted', 'origin': 'real_world',
        'public_dataset': True, 'public_dataset_url': 'https://example.invalid/dataset',
        'decision_id': 'schema-test-only', 'reviewer_id': 'schema-test-only',
        'evidence': [{'kind': 'real_project_source_lineage', 'artifact_sha256': 'b' * 64,
                      'locator': 'metadata fixture only; not a real accepted data decision'}]}
    return model, target


class RealPublicOriginTests(unittest.TestCase):
    def check(self, model, target):
        return require_real_public_origin(model, target, purpose='training')

    def test_complete_metadata_does_not_mutate_or_grant_eligibility(self):
        model, target = metadata_fixture(); before = copy.deepcopy((model, target))
        self.assertIsNone(self.check(model, target))
        self.assertEqual((model, target), before)
        self.assertNotIn('training_eligible', target)

    def test_missing_decision_fails(self):
        model, target = metadata_fixture(); target.pop('real_public_origin_decision')
        with self.assertRaises(ValueError): self.check(model, target)

    def test_synthetic_unknown_and_unreviewed_fail(self):
        for key, value in [('origin', 'synthetic'), ('origin', 'unknown'),
                           ('status', 'proposed'), ('public_dataset', False),
                           ('public_dataset', 'true')]:
            model, target = metadata_fixture(); target['real_public_origin_decision'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(model, target)

    def test_each_identity_binding_is_required(self):
        for key in ['record_id', 'source', 'source_revision', 'model_code_sha256']:
            model, target = metadata_fixture(); target['real_public_origin_decision'][key] = 'other'
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(model, target)

    def test_public_hosting_does_not_supply_origin_evidence(self):
        for evidence in [[], [{'kind': 'public_hosting', 'artifact_sha256': 'b' * 64, 'locator': 'url'}],
                         [{'kind': 'real_project_source_lineage', 'artifact_sha256': 'bad', 'locator': 'x'}]]:
            model, target = metadata_fixture(); target['real_public_origin_decision']['evidence'] = evidence
            with self.assertRaises(ValueError): self.check(model, target)

    def test_missing_review_identity_is_not_accepted(self):
        for key in ['reviewer_id', 'decision_id']:
            model, target = metadata_fixture(); target['real_public_origin_decision'][key] = ''
            with self.assertRaises(ValueError): self.check(model, target)

    def test_credential_bearing_or_non_https_reference_rejected(self):
        for url in ['http://example.invalid/data', 'https://user:secret@example.invalid/data',
                    'file:///data', 'https://example.invalid/has space']:
            model, target = metadata_fixture(); target['real_public_origin_decision']['public_dataset_url'] = url
            with self.subTest(url=url), self.assertRaises(ValueError): self.check(model, target)

    def test_evaluation_requires_the_same_origin_boundary(self):
        model, target = metadata_fixture(); target['real_public_origin_decision']['origin'] = 'unknown'
        with self.assertRaises(ValueError): require_real_public_origin(model, target, purpose='evaluation')


if __name__ == '__main__':
    unittest.main()
