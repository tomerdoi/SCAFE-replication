import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.run_function_semantic_transfer import (
    LABELS,
    SCORING_PROTOCOL,
    _normalise,
    _registry_function_hashes,
    _sha256,
    _source_text,
    _validate_scoring_protocol,
    extract_functions,
    score,
)
from src.source_sanitization import SOURCE_SANITIZATION_VERSION


class FunctionSemanticTransferTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_extract_functions_keeps_nested_block_and_following_function(self):
        source = (
            'contract C { function first(uint value) external { '
            'if (value > 0) { value -= 1; } } function second() external {} }'
        )
        extracted = extract_functions(source)
        self.assertEqual([name for name, _ in extracted], ['first', 'second'])
        self.assertIn('value -= 1;', extracted[0][1])
        self.assertTrue(extracted[1][1].endswith('{}'))

    def test_function_reader_removes_annotations_before_extraction(self):
        path = self.tmp_path / 'contract.sol'
        raw = (
            'contract C { function withdraw() external {\n'
            '// SWC-107-Reentrancy: L3-L6\n'
            'uint value = 1; // SWC-101-Overflow: L4\n'
            'string memory doc = "SWC-107 is a literal";\n'
            '// ordinary comment\n} }'
        )
        path.write_text(raw)
        clean = _source_text(str(path), 'DAppSCAN')
        functions = extract_functions(clean)
        self.assertEqual(len(functions), 1)
        self.assertNotIn('Reentrancy: L3-L6', functions[0][1])
        self.assertNotIn('Overflow: L4', functions[0][1])
        self.assertIn('"SWC-107 is a literal"', functions[0][1])
        self.assertIn('uint value = 1;', functions[0][1])
        self.assertIn('ordinary comment', functions[0][1])
        self.assertEqual(_source_text(str(path), 'DIVE'), raw)
        self.assertEqual(_normalise(clean), _normalise(raw))

    def test_directory_reader_and_legacy_dappscan_path_use_sanitizer(self):
        root = self.tmp_path / 'DAppSCAN-main'
        root.mkdir()
        (root / 'b.sol').write_text('contract B {}')
        (root / 'a.sol').write_text('// SWC-107-Reentrancy\ncontract A {}')
        (root / 'metadata.json').write_text('not Solidity')
        self.assertEqual(_source_text(str(root)), 'contract A {}\ncontract B {}')
        self.assertEqual(_source_text(str(root / 'metadata.json'), 'DAppSCAN'), '')
        self.assertEqual(_source_text(str(root / 'missing.sol'), 'DAppSCAN'), '')

    def test_overlap_filter_keeps_dataset_specific_source_variants(self):
        path = self.tmp_path / 'contract.sol'
        # A comment brace exposes the legacy extractor's approximation. Dataset
        # variants must not be collapsed by a path-only cache/deduplication key.
        path.write_text('contract C { function f() external { // SWC-107 }\n uint x = 1; } }')
        frame = pd.DataFrame({
            'source_dataset': ['DAppSCAN', 'DIVE', 'DAppSCAN'],
            'local_source_path': [str(path)] * 3,
        })
        expected = {
            _sha256(_normalise(function))
            for dataset in ('DAppSCAN', 'DIVE')
            for _, function in extract_functions(_source_text(str(path), dataset))
        }
        self.assertEqual(len(expected), 2)
        self.assertEqual(_registry_function_hashes(frame), expected)

    def test_evaluation_rejects_unproven_or_changed_aggregates(self):
        for case in ('missing', 'legacy', 'wrong_version', 'tampered'):
            with self.subTest(case=case):
                root = self.tmp_path / case
                root.mkdir()
                aggregates = root / 'contract_aggregates.csv'
                aggregates.write_text('dataset,contract_id\nDAppSCAN,c1\n')
                protocol = {
                    'protocol': SCORING_PROTOCOL,
                    'source_sanitization_version': SOURCE_SANITIZATION_VERSION,
                    'aggregates_sha256': hashlib.sha256(aggregates.read_bytes()).hexdigest(),
                }
                if case == 'legacy':
                    protocol = {'functions_scored': 1}
                if case == 'wrong_version':
                    protocol['source_sanitization_version'] = 'old'
                if case != 'missing':
                    (root / 'scoring_protocol.json').write_text(json.dumps(protocol))
                if case == 'tampered':
                    aggregates.write_text('dataset,contract_id\nDAppSCAN,changed\n')
                with self.assertRaises(ValueError):
                    _validate_scoring_protocol(aggregates)

    def test_scoring_sanitizes_by_dataset_before_tokenization_and_caching(self):
        import argparse
        from unittest.mock import patch
        import torch

        texts_seen = []

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                texts_seen.extend(texts)
                return {
                    'input_ids': torch.zeros((len(texts), 1), dtype=torch.long),
                    'attention_mask': torch.ones((len(texts), 1), dtype=torch.long),
                }

        class Model:
            def __call__(self, ids, mask):
                return torch.zeros((len(ids), len(LABELS)))

        source = self.tmp_path / 'contract.sol'
        source.write_text('contract C { function f() external {\n// SWC-107-Reentrancy\nuint x = 1; } }')
        registry = self.tmp_path / 'registry.csv'
        pd.DataFrame({
            'source_dataset': ['DAppSCAN', 'DIVE'],
            'contract_id': ['c1', 'c2'],
            'local_source_path': [str(source)] * 2,
        }).to_csv(registry, index=False)
        args = argparse.Namespace(
            device='cpu', registry=registry, checkpoint=self.tmp_path,
            datasets=['DAppSCAN', 'DIVE'], batch_size=2, flush_functions=2,
            output=self.tmp_path / 'output',
        )
        with patch('scripts.run_function_semantic_transfer._load_model', return_value=(
            Model(), Tokenizer(), {'max_length': 512}, torch.device('cpu'),
        )):
            score(args)
        self.assertEqual(len(texts_seen), 2)
        self.assertNotIn('SWC-107', texts_seen[0])
        self.assertIn('SWC-107', texts_seen[1])
        protocol = _validate_scoring_protocol(args.output / 'contract_aggregates.csv')
        self.assertEqual(protocol['records_scored'], 2)
