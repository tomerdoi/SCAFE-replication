import copy
from pathlib import Path
import tempfile
import unittest

from bind_function_review_provenance import bind_records, indexed_rows


class RetainedReviewBindingTests(unittest.TestCase):
    def setUp(self):
        self.row = {'record_id': 'id', 'source': 'sujeetc/SCRUBD',
                    'source_revision': 'pin', 'source_identity': '0x123|C.f',
                    'contract_address': '0x123', 'function_identifier': 'C.f',
                    'source_path': 'SCRUBD-CD/data/solidity_codes/0x123.sol',
                    'source_metadata': {'Smart Contract': '0x123', 'Function Name': 'C.f',
                                        'RE': '0', 'UX': '', 'is_student': '1'},
                    'source_assessments': {'reentrancy': 0, 'unhandled_exceptions': None},
                    'pattern_labels': {'reentrancy': None, 'authorization': None,
                                       'unchecked_external_call': None},
                    'training_eligible': False, 'human_review_status': 'unreviewed', 'split': None,
                    'raw_sha256': 'raw', 'model_code': 'function f() {}', 'context_sha256': 'context'}
        self.entry = {'source': 'sujeetc/SCRUBD', 'revision': 'pin',
                      'path': 'SCRUBD-CD/data/labels.csv', 'sha256': 'claimed-csv-hash'}

    def bind(self, row=None, original=None, entry=None):
        row = self.row if row is None else row
        original = copy.deepcopy(row) if original is None else original
        return bind_records({'id': (row, 2, 'candidate-line')},
                            {'id': (original, 10, 'staging-line')}, 'pin',
                            self.entry if entry is None else entry)[0]

    def test_link_preserves_unknown_mask_and_never_promotes_role_flag(self):
        result = self.bind()
        self.assertEqual(result['native_claims'], {'RE': '0', 'UX': ''})
        self.assertEqual(result['native_observation_mask'], {'RE': True, 'UX': False})
        self.assertEqual(result['staging_anchor']['line_1based'], 10)
        self.assertFalse(result['independent_gold'])
        self.assertFalse(result['training_eligible'])
        self.assertFalse(result['evaluation_eligible'])
        self.assertEqual(result['review_evidence']['review_coverage_at_retained_revision'], 'unverified')

    def test_changed_rationale_or_role_is_not_a_lossless_binding(self):
        original = copy.deepcopy(self.row)
        original['source_metadata']['is_student'] = '0'
        with self.assertRaises(ValueError): self.bind(original=original)

    def test_conflicting_native_label_rejected_even_if_records_agree(self):
        row = copy.deepcopy(self.row)
        row['source_metadata']['RE'] = '1'
        with self.assertRaises(ValueError): self.bind(row=row)

    def test_wrong_source_revision_rejected(self):
        row = copy.deepcopy(self.row); row['source_revision'] = 'other'
        with self.assertRaises(ValueError): self.bind(row=row)

    def test_synthetic_source_path_cannot_inherit_cd_protocol(self):
        row = copy.deepcopy(self.row); row['source_path'] = 'SCRUBD-SD/data/solidity_codes/x.sol'
        with self.assertRaises(ValueError): self.bind(row=row)

    def test_assigned_eligibility_cannot_be_silently_overwritten(self):
        row = copy.deepcopy(self.row); row['training_eligible'] = True
        with self.assertRaises(ValueError): self.bind(row=row)

    def test_annotation_manifest_revision_must_match(self):
        entry = dict(self.entry, revision='other')
        with self.assertRaises(ValueError): self.bind(entry=entry)

    def test_missing_staging_identity_rejected(self):
        with self.assertRaises(ValueError):
            bind_records({'id': (self.row, 2, 'line')}, {}, 'pin', self.entry)

    def test_duplicate_jsonl_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_text('{"record_id":"x"}\n{"record_id":"x"}\n')
            with self.assertRaises(ValueError): indexed_rows(path)


if __name__ == '__main__':
    unittest.main()
