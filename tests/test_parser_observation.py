"""Software fixtures only; these are never function-model data."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if (HERE / "parser_observation.py").is_file():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parents[0] / "src" / "ml"))
from parser_observation import FIELDS, MAX_BYTES, _counts, observe_ast


class ParserObservationTests(unittest.TestCase):
    def test_real_zero_is_observed(self):
        r = observe_ast("contract Empty {}")
        self.assertTrue(all(r["observation_mask"].values()))
        self.assertEqual(set(r["values"].values()), {0})

    def test_empty_is_unknown(self):
        r = observe_ast(" \n")
        self.assertFalse(any(r["observation_mask"].values()))
        self.assertEqual(set(r["values"].values()), {None})

    def test_lexer_recovery_is_unknown(self):
        r = observe_ast("contract Empty {} @")
        self.assertGreater(r["provenance"]["diagnostic_counts"].get("lexer", 0), 0)
        self.assertFalse(any(r["observation_mask"].values()))
        self.assertEqual(set(r["values"].values()), {None})

    def test_parser_recovery_is_unknown(self):
        r = observe_ast("contract Empty {")
        self.assertGreater(r["provenance"]["diagnostic_counts"].get("parser", 0), 0)
        self.assertFalse(any(r["observation_mask"].values()))

    def test_exception_is_atomic(self):
        with patch("solidity_parser.parser.AstVisitor.visit", side_effect=RuntimeError("private text")):
            r = observe_ast("contract Empty {}")
        self.assertEqual(r["provenance"]["status"], "visitor_exception")
        self.assertEqual(r["provenance"]["exception_class"], "RuntimeError")
        self.assertNotIn("private text", str(r))
        self.assertEqual(set(r["values"].values()), {None})

    def test_parse_exception(self):
        with patch("solidity_parser.parser.SolidityParser.sourceUnit", side_effect=RuntimeError):
            r = observe_ast("contract Empty {}")
        self.assertEqual(r["provenance"]["status"], "parse_exception")

    def test_count_exception_is_atomic(self):
        with patch("parser_observation._counts", side_effect=ValueError):
            r = observe_ast("contract Empty {}")
        self.assertEqual(r["provenance"]["status"], "count_exception")
        self.assertEqual(set(r["values"].values()), {None})

    def test_field_schema_and_metadata_separation(self):
        r = observe_ast("contract Empty {}")
        self.assertEqual(tuple(r["values"]), FIELDS)
        self.assertEqual(tuple(r["observation_mask"]), FIELDS)
        self.assertNotIn("status", r["values"])
        self.assertFalse(r["provenance"]["model_input_approved"])

    def test_hash_and_diagnostic_privacy(self):
        s = "contract PrivateName {} @"
        r = observe_ast(s)
        self.assertEqual(r["provenance"]["source_sha256"], hashlib.sha256(s.encode()).hexdigest())
        self.assertNotIn("PrivateName", str(r))
        self.assertNotIn("@", str(r))

    def test_invalid_input_bound(self):
        for s in [None, b"source", "x" * (MAX_BYTES + 1)]:
            with self.assertRaises(ValueError):
                observe_ast(s)

    def test_known_legacy_units(self):
        r = observe_ast("contract C { function f() public view returns(uint) { return 1; } }")
        self.assertTrue(all(r["observation_mask"].values()))
        self.assertEqual(r["values"]["ast_n_functions"], 1)
        self.assertEqual(r["values"]["ast_pure_view_ratio"], 1)

    def test_unknown_ast_shape_rejected(self):
        with self.assertRaises(ValueError):
            _counts({"type": "unexpected"})


if __name__ == "__main__":
    unittest.main()
