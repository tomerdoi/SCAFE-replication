import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.run_function_semantic_transfer import LABELS, _auxiliary_input_fingerprint, _multihot, _validate_auxiliary_frame, train
from scripts.audit_function_pilot_handoff import inspect_staging_rows, verify_delivery


def valid_frame():
    return pd.DataFrame({
        "id": ["a", "b", "c"],
        "statement": ["function a() external {}", "function b() external {}", "function c() external {}"],
        "label": ["CLEAN", LABELS[0], LABELS[1]],
        "split": ["train", "val", "test"],
    })


class FunctionLabelIntakeTests(unittest.TestCase):
    def test_explicit_clean_and_known_multilabel_encoding(self):
        encoded = _multihot(pd.Series(["CLEAN", f" {LABELS[0]} , {LABELS[1]} ", LABELS[0]]))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded.sum(axis=1).tolist(), [0, 2, 1])
        self.assertEqual(encoded[1, :2].tolist(), [1, 1])

    def test_unknown_context_missing_and_typo_labels_never_become_negatives(self):
        for value in ["UNKNOWN", "CONTEXT_REQUIRED", "UNREVIEWED", "UNCERTAIN", None, np.nan, pd.NA, "", "  ", "TYPO", f"{LABELS[0]},UNKNOWN"]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError): _multihot(pd.Series([value]))

    def test_clean_plus_pattern_and_empty_components_are_rejected(self):
        for value in [f"CLEAN,{LABELS[0]}", "CLEAN,CLEAN", f"{LABELS[0]},", f",{LABELS[0]}"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): _multihot(pd.Series([value]))

    def test_validated_frame_is_a_copy_without_source_mutation(self):
        frame = valid_frame()
        original = frame.copy(deep=True)
        result = _validate_auxiliary_frame(frame)
        self.assertTrue(frame.equals(original))
        self.assertTrue(result.equals(original))
        self.assertIsNot(result, frame)

    def test_required_fields_empty_source_and_duplicate_ids_fail(self):
        variants = [valid_frame().drop(columns="label"), valid_frame().iloc[:0]]
        for column in ("id", "statement"):
            frame = valid_frame(); frame.loc[0, column] = " "
            variants.append(frame)
        frame = valid_frame(); frame.loc[1, "id"] = "a"; variants.append(frame)
        for frame in variants:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                with self.assertRaises(ValueError): _validate_auxiliary_frame(frame)

    def test_bad_or_incomplete_splits_fail_before_training(self):
        for split in ["development", None, "train"]:
            frame = valid_frame(); frame.loc[1, "split"] = split
            with self.subTest(split=split):
                with self.assertRaises(ValueError): _validate_auxiliary_frame(frame)

    def test_identical_text_crossing_splits_is_rejected(self):
        frame = valid_frame(); frame.loc[1, "statement"] = frame.loc[0, "statement"]
        with self.assertRaisesRegex(ValueError, "crosses auxiliary splits"):
            _validate_auxiliary_frame(frame)

    def test_conflicting_claims_on_same_text_require_context_review(self):
        frame = valid_frame()
        extra = frame.iloc[[0]].copy(); extra["id"] = "new"; extra["label"] = LABELS[0]
        with self.assertRaisesRegex(ValueError, "context review required"):
            _validate_auxiliary_frame(pd.concat([frame, extra], ignore_index=True))

    def test_literal_whitespace_is_not_collapsed_into_a_false_duplicate(self):
        frame = valid_frame()
        frame.loc[0, "statement"] = 'function f() external { string memory s = "a b"; }'
        frame.loc[1, "statement"] = 'function f() external { string memory s = "a  b"; }'
        self.assertEqual(len(_validate_auxiliary_frame(frame)), 3)

    def test_bad_labels_rejected_before_registry_output_or_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "auxiliary.csv"
            frame = valid_frame(); frame.loc[0, "label"] = "CONTEXT_REQUIRED"
            frame.to_csv(path, index=False)
            args = argparse.Namespace(auxiliary=path, output=root / "output")
            with patch.dict("sys.modules", {"torch": None, "transformers": None}):
                with self.assertRaisesRegex(ValueError, "Unresolved pattern annotation"):
                    train(args)
            self.assertFalse(args.output.exists())

    def test_fingerprint_binds_content_labels_and_splits_at_same_row_count(self):
        frame = valid_frame()
        baseline = _auxiliary_input_fingerprint(frame)
        self.assertEqual(baseline, _auxiliary_input_fingerprint(frame.copy()))
        for column, value in [("statement", "function altered() external {}"), ("label", LABELS[2]), ("id", "revised")]:
            changed = frame.copy(); changed.loc[0, column] = value
            with self.subTest(column=column):
                self.assertNotEqual(baseline, _auxiliary_input_fingerprint(changed))
        changed = frame.copy(); changed.loc[[0, 1], "split"] = ["val", "train"]
        self.assertNotEqual(baseline, _auxiliary_input_fingerprint(changed))

    def test_fingerprint_preserves_training_row_order(self):
        frame = valid_frame()
        self.assertNotEqual(_auxiliary_input_fingerprint(frame), _auxiliary_input_fingerprint(frame.iloc[::-1]))


class PilotHandoffTests(unittest.TestCase):
    def row(self):
        return {"record_id": "x", "source": "fixture", "source_revision": "fixed-revision",
                "pattern_labels": {key: None for key in ("reentrancy", "unchecked_external_call", "authorization")},
                "training_eligible": False, "split": None, "human_review_status": "unreviewed", "context_status": "missing"}

    def test_unknowns_preserved_and_missing_project_reported(self):
        report = inspect_staging_rows([self.row()])
        self.assertEqual(report["unknown_pattern_cells"], 3)
        self.assertEqual(report["records_without_project_id"], 1)
        self.assertEqual(report["training_eligible_records"], 0)

    def test_promotion_or_label_change_requires_explicit_review(self):
        for field, value in [("training_eligible", True), ("split", "train"), ("human_review_status", "reviewed")]:
            row = self.row(); row[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError): inspect_staging_rows([row])
        row = self.row(); row["pattern_labels"]["reentrancy"] = 0
        with self.assertRaises(ValueError): inspect_staging_rows([row])

    def test_duplicate_ids_and_missing_revisions_are_rejected(self):
        row = self.row()
        with self.assertRaises(ValueError): inspect_staging_rows([row, row])
        row["source_revision"] = ""
        with self.assertRaises(ValueError): inspect_staging_rows([row])

    def test_delivery_rejects_changed_file_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.txt").write_text("changed")
            (root / "DELIVERY_SHA256.json").write_text(json.dumps({"data.txt": "0" * 64}))
            with self.assertRaises(ValueError): verify_delivery(root)
            (root / "DELIVERY_SHA256.json").write_text(json.dumps({"../outside": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "escapes the pilot"):
                verify_delivery(root)
