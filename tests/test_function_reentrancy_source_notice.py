"""Regression tests for the observed provenance conflict, not legal decisions."""
import unittest
from scripts.assess_reentrancy_source_notice import notice_assessment


class SourceNoticeEvidenceTest(unittest.TestCase):
    def pair(self, notices=None, license_claim='MIT'):
        key = '0x' + 'a' * 40
        return ({'address': key, 'source_path': 'deduplicated_smart_contracts/' + key + '.sol',
                 'native_source_target': 0, 'status': 'notice_observed',
                 'spdx_expressions': ['MIT'] if notices is None else notices,
                 'reported_total_bytes': 1000},
                {'ContractAddress': key, 'LicenseType': license_claim})

    def test_observed_GPL_LGPL_discrepancy_stays_held(self):
        a, b = self.pair(['GPL-3.0'], 'GNU LGPLv3')
        r = notice_assessment(a, b)
        self.assertIn('discrepancy', r['status']); self.assertFalse(r['source_binding_candidate'])
        self.assertFalse(r['training_eligible']); self.assertFalse(r['evaluation_eligible'])

    def test_reverse_family_discrepancy_also_held(self):
        a, b = self.pair(['LGPL-3.0'], 'GNU GPLv3')
        self.assertIn('discrepancy', notice_assessment(a, b)['status'])

    def test_matching_MIT_only_allows_private_binding(self):
        r = notice_assessment(*self.pair())
        self.assertTrue(r['source_binding_candidate']); self.assertFalse(r['independent_gold'])
        self.assertFalse(r['training_eligible']); self.assertFalse(r['evaluation_eligible'])

    def test_missing_prefix_notice_not_inferred_from_metadata(self):
        r = notice_assessment(*self.pair([], 'MIT'))
        self.assertFalse(r['source_binding_candidate']); self.assertIn('not_proof_of_no_license', r['status'])

    def test_missing_explorer_claim_does_not_promote_MIT_header(self):
        self.assertFalse(notice_assessment(*self.pair(['MIT'], ''))['source_binding_candidate'])

    def test_size_guard_rejects_oversize_or_boolean(self):
        for size in [0, 128 * 1024 + 1, None, True]:
            a, b = self.pair(); a['reported_total_bytes'] = size
            self.assertFalse(notice_assessment(a, b)['source_binding_candidate'])

    def test_address_and_source_path_mismatch_reject(self):
        a, b = self.pair(); b['ContractAddress'] = '0x' + 'b' * 40
        with self.assertRaises(ValueError): notice_assessment(a, b)
        a, b = self.pair(); a['source_path'] = '../other.sol'
        with self.assertRaises(ValueError): notice_assessment(a, b)

    def test_unsupported_evidence_and_native_claim_reject(self):
        a, b = self.pair(); a['spdx_expressions'] = 'MIT'
        with self.assertRaises(ValueError): notice_assessment(a, b)
        a, b = self.pair(); a['native_source_target'] = True
        with self.assertRaises(ValueError): notice_assessment(a, b)

    def test_non_MIT_terms_never_automatically_normalized_into_admission(self):
        r = notice_assessment(*self.pair(['GPL-3.0-or-later'], 'GNU GPLv3'))
        self.assertFalse(r['source_binding_candidate']); self.assertIn('no_automatic_license_mapping', r['status'])


if __name__ == '__main__': unittest.main()
