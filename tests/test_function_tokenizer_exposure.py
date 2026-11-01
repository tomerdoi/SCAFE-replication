import unittest

from scripts.audit_function_tokenizer_exposure import (
    annotation_spans,
    span_fully_visible,
    span_visible,
    window_counts,
)


class FunctionTokenizerExposureTests(unittest.TestCase):
    def test_annotation_scanner_ignores_quoted_comments_and_ordinary_mentions(self):
        source = (
            'string a = "// SWC-107"; string b = \'/* SWC-101 */\';\n'
            '// ordinary discussion of SWC-107\n'
            '// SWC-107-Reentrancy\nuint x; /* SWC-101 overflow */'
        )
        comments = [source[start:end] for start, end in annotation_spans(source)]
        self.assertEqual(comments, ['// SWC-107-Reentrancy', '/* SWC-101 overflow */'])

    def test_escaped_quote_does_not_create_a_fake_comment(self):
        source = 'string a = "escaped \\" // SWC-107"; // SWC-101 real'
        comments = [source[start:end] for start, end in annotation_spans(source)]
        self.assertEqual(comments, ['// SWC-101 real'])

    def test_special_token_zero_offsets_do_not_expose_text(self):
        self.assertFalse(span_visible((2, 9), [(0, 0), (0, 0)]))
        self.assertFalse(span_fully_visible((2, 9), [(0, 0), (0, 0)]))

    def test_partial_and_complete_marker_coverage_are_separate(self):
        self.assertTrue(span_visible((10, 17), [(10, 13)]))
        self.assertFalse(span_fully_visible((10, 17), [(10, 13)]))
        self.assertTrue(span_fully_visible((10, 17), [(10, 13), (13, 14), (14, 17)]))
        self.assertFalse(span_fully_visible((10, 17), [(10, 13), (14, 17)]))

    def test_outside_window_marker_is_not_visible(self):
        source = 'uint x;\n// SWC-107-Reentrancy\n'
        counts = window_counts(source, [(0, 4), (5, 6), (6, 7)])
        self.assertEqual(counts['annotation_markers'], 1)
        self.assertEqual(counts['markers_with_tokens_in_window'], 0)
        self.assertEqual(counts['annotation_comments_with_tokens_in_window'], 0)

    def test_full_window_counts_only_annotation_markers(self):
        source = 'string doc = "SWC-999"; // SWC-107-Reentrancy'
        counts = window_counts(source, [(0, len(source))])
        self.assertEqual(counts['annotation_markers'], 1)
        self.assertEqual(counts['complete_markers_in_window'], 1)
