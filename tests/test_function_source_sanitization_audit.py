import unittest

from scripts.audit_function_source_sanitization import inspect_source


class FunctionSourceSanitizationAuditTests(unittest.TestCase):
    def test_audit_separates_annotation_exposure_from_string_literals(self):
        source = (
            'contract C { function f() external {\n// SWC-107-Reentrancy\n'
            'string memory doc = "SWC-107 literal";\n'
            'uint x = 1; // SWC-101-Overflow\n} }'
        )
        audit = inspect_source(source)
        self.assertEqual(audit['source_changed'], 1)
        self.assertEqual(audit['legacy_functions_with_annotation_comments'], 1)
        self.assertEqual(audit['sanitized_functions_with_annotation_comments'], 0)
        self.assertEqual(audit['matched_functions_text_changed'], 1)
        self.assertEqual(audit['matched_functions_normalized_identity_changed'], 0)

    def test_audit_reports_sequence_changes_without_forcing_function_matches(self):
        source = (
            'contract C { function f() external {\n'
            '// SWC-107 } function bogus() external {\nuint x = 1;\n} }'
        )
        audit = inspect_source(source)
        self.assertEqual(audit['function_sequence_changed'], 1)
        self.assertEqual(audit['matched_functions_text_changed'], 0)

    def test_no_annotation_source_has_no_claimed_exposure(self):
        self.assertFalse(any(inspect_source('contract C { function f() external {} }').values()))
