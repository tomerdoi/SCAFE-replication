import unittest

import numpy as np
import pandas as pd

from scripts.run_frozen_annotation_diagnostic import annotation_token_mask, select_project_sample, summarize_pairs
from scripts.run_function_semantic_transfer import LABELS


class FrozenAnnotationDiagnosticTests(unittest.TestCase):
    def test_selection_is_order_independent_and_project_separated(self):
        data = pd.DataFrame({"project": ["a", "a", "b", "c"], "contract_id": ["a1", "a2", "b1", "c1"], "function_ordinal": [0, 1, 0, 0]})
        selected = select_project_sample(data, 3)
        other = select_project_sample(data.sample(frac=1, random_state=7), 3)
        self.assertEqual(selected.contract_id.tolist(), other.contract_id.tolist())
        self.assertEqual(selected.project.nunique(), 3)

    def test_missing_provenance_and_oversized_pilot_are_rejected(self):
        frame = pd.DataFrame({"project": [""], "contract_id": ["a"], "function_ordinal": [0]})
        with self.assertRaises(ValueError): select_project_sample(frame, 1)
        with self.assertRaises(ValueError): select_project_sample(frame, 25)

    def test_mask_changes_only_annotation_token_positions(self):
        self.assertEqual(annotation_token_mask([(0, 0), (0, 4), (5, 8), (8, 11), (12, 15)], [(6, 10)]), [False, False, True, True, False])

    def test_identity_control_has_zero_shift_and_degenerate_interval(self):
        raw = np.full((4, len(LABELS)), 0.7)
        summary = summarize_pairs(raw, raw)
        self.assertEqual(summary["mean_max_absolute_output_change"], 0)
        self.assertEqual(summary["conditional_project_resample95"], [0, 0])
        self.assertEqual(summary["functions_crossing_uncalibrated_0_5_boundary"], 0)

    def test_known_paired_change_reports_magnitude_and_boundary_crossing(self):
        raw = np.full((3, len(LABELS)), 0.4)
        changed = raw.copy(); changed[:, 0] = 0.6
        summary = summarize_pairs(raw, changed)
        self.assertAlmostEqual(summary["mean_max_absolute_output_change"], 0.2)
        self.assertEqual(summary["functions_crossing_uncalibrated_0_5_boundary"], 3)
        self.assertAlmostEqual(summary["per_label_mean_signed_change"][LABELS[0]], 0.2)
