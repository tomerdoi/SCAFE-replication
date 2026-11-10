"""Regression checks for native-zero scope and strict upstream claim semantics."""
import unittest
from scripts.export_reentrancy_study_claims import HEADER, TOOLS, address, role, source_claim


class NativeReviewClaimsTest(unittest.TestCase):
    def row(self, label='0', alarm=False):
        row = dict.fromkeys(HEADER, '0')
        row.update(address='0x' + 'a' * 40, true_positive=label)
        if alarm: row['mythril'] = '1'
        return row

    def test_no_alarm_zero_is_unknown(self):
        r = source_claim(self.row())
        self.assertIsNone(r['source_target']); self.assertFalse(r['source_observation_mask'])
        self.assertEqual(r['mask_reason'], 'no_alarm_outside_reported_review_frame')

    def test_alarm_zero_is_source_specific_negative_claim(self):
        r = source_claim(self.row(alarm=True))
        self.assertEqual(r['source_target'], 0); self.assertTrue(r['source_observation_mask'])
        self.assertFalse(r['independent_gold']); self.assertFalse(r['training_eligible'])
        self.assertFalse(r['evaluation_eligible']); self.assertIsNone(r['canonical_function_category_mapping'])

    def test_alarm_positive_preserved(self):
        self.assertEqual(source_claim(self.row('1', True))['source_target'], 1)

    def test_unknown_not_negative_even_with_alarm(self):
        r = source_claim(self.row('N/A', True))
        self.assertIsNone(r['source_target']); self.assertFalse(r['source_observation_mask'])

    def test_positive_without_alarm_rejects(self):
        with self.assertRaises(ValueError): source_claim(self.row('1'))

    def test_all_unanalyzed_stays_unknown(self):
        r = self.row('N/A')
        for name in TOOLS: r[name] = 'N/A'
        self.assertFalse(source_claim(r)['source_observation_mask'])

    def test_unsupported_blank_and_float_values_reject(self):
        for value in ['', 'nan', '0.0', '-1', 'safe']:
            with self.subTest(value=value):
                r = self.row(); r['true_positive'] = value
                with self.assertRaises(ValueError): source_claim(r)

    def test_schema_change_rejects(self):
        r = self.row(); r['inferred_label'] = '0'
        with self.assertRaises(ValueError): source_claim(r)

    def test_bad_address_rejects(self):
        for value in ['../source', '0x1', None]:
            with self.assertRaises(ValueError): address(value)

    def test_role_is_deterministic_and_case_normalized(self):
        value = '0x' + 'a' * 40
        self.assertEqual(role(value), role('0x' + 'A' * 40))
        self.assertIn(role(value), ['development', 'provisional_reserve'])


if __name__ == '__main__': unittest.main()
