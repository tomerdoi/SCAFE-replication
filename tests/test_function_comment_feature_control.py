"""Software regression fixtures only; never research training/evaluation examples."""
import unittest
from src.ml.comment_feature_control import mask_comments, extract_comment_control_features
from src.risk.contract_risk import ContractRiskUtil


class CommentControlTest(unittest.TestCase):
    def test_preserves_literal_comment_delimiters_and_escape(self):
        source = 'contract C { string a = "https://x/*y*/"; string b = "say \\\"//ok\\\""; } // remove'
        result = mask_comments(source)
        self.assertIn('"https://x/*y*/"', result.text)
        self.assertIn('"say \\\"//ok\\\""', result.text)
        self.assertEqual(len(result.spans), 1); self.assertNotIn('remove', result.text)

    def test_preserves_unicode_code_characters_CR_LF_and_offsets(self):
        source = 'contract C { string a = unicode"β"; } /* note α\r\nmore */\r\n'
        result = mask_comments(source)
        self.assertEqual(len(result.text), len(source))
        self.assertEqual([(i,c)for i,c in enumerate(source)if c in '\r\n'],
                         [(i,c)for i,c in enumerate(result.text)if c in '\r\n'])
        self.assertIn('unicode"β"', result.text)

    def test_unterminated_block_or_quote_rejects(self):
        for source in ['contract C { /* bad', 'contract C { string a = "bad', '"bad\\']:
            with self.subTest(source=source):
                with self.assertRaises(ValueError): mask_comments(source)

    def test_comment_separates_tokens_instead_of_concatenating(self):
        self.assertEqual(mask_comments('uint/*gap*/value;').text, 'uint       value;')

    def test_line_comment_at_EOF_and_CR_only(self):
        self.assertEqual(mask_comments('// x\rcontract C {}').text, '    \rcontract C {}')
        self.assertEqual(mask_comments('contract C {}//x').text, 'contract C {}   ')

    def test_mask_is_idempotent(self):
        view = mask_comments('/*first*/contract C {} //last\n')
        self.assertEqual(mask_comments(view.text).text, view.text)
        self.assertFalse(mask_comments(view.text).spans)

    def test_rejects_nontext_or_oversized_input(self):
        for source in [None, b'code', 'x' * (2 * 1024**2 + 1)]:
            with self.assertRaises(ValueError): mask_comments(source)

    def test_comment_function_and_delegatecall_mentions_do_not_count_in_new_view(self):
        source = '// function delegatecall\ncontract C { function f() public {} }'
        base = ContractRiskUtil._empty_feature_base('regression')
        legacy = ContractRiskUtil._extract_source_features(base, source, 1)
        controlled = extract_comment_control_features(source)
        self.assertEqual(legacy['n_functions'], 2)
        self.assertEqual(legacy['n_delegatecall'], 1)
        self.assertEqual(controlled['features']['n_functions'], 1)
        self.assertEqual(controlled['features']['n_delegatecall'], 0)
        self.assertNotIn('raw_source_sha256', controlled['features'])
        self.assertIn('not_legacy_cache_or_model_compatible', controlled['metadata']['compatibility'])

    def test_wrapper_does_not_change_on_comment_only_preprocessing(self):
        source = '/* doc function */contract C { function f() public {} }'
        a = extract_comment_control_features(source)
        b = extract_comment_control_features(mask_comments(source).text)
        self.assertEqual(a['features'], b['features'])
        self.assertNotEqual(a['metadata']['raw_source_sha256'], b['metadata']['raw_source_sha256'])

    def test_file_count_rejects_boolean_and_out_of_range(self):
        for count in [True, 0, 1001, 1.5]:
            with self.assertRaises(ValueError): extract_comment_control_features('contract C {}', n_files=count)


if __name__ == '__main__': unittest.main()
