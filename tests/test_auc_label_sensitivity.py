"""Numerical software controls; no generated function/model examples."""
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if SCRIPTS.is_dir():
    sys.path.insert(0, str(SCRIPTS))
from auc_label_sensitivity import extrema, summarize, twice_midrank, run


def auc(y, s):
    pairs = [(s[i] > s[j]) + .5*(s[i] == s[j]) for i in range(len(y))
             if y[i] == 1 for j in range(len(y)) if y[j] == 0]
    return sum(pairs)/len(pairs)


class LabelSensitivityTests(unittest.TestCase):
    def test_midrank_ties(self):
        self.assertEqual(twice_midrank([3, 1, 1, 2]).tolist(), [8, 3, 3, 6])

    def test_exhaustive_assignments(self):
        for a, b in [([0, 0, 1, 2, 2, 3], [3, 1, 1, 0, 2, 2]),
                     (list(range(6)), list(range(5, -1, -1))),
                     ([1]*6, [0, 1, 0, 1, 2, 2])]:
            for native in itertools.combinations(range(6), 3):
                y = [int(i in native) for i in range(6)]; e = extrema(y, a, b)
                actual = {j: [] for j in range(4)}
                for alternative in itertools.combinations(range(6), 3):
                    other = [int(i in alternative) for i in range(6)]
                    swaps = sum(x != z for x, z in zip(y, other))//2
                    delta = auc(other, a)-auc(other, b)
                    for k in range(swaps, 4):
                        actual[k].append(delta)
                for k in range(4):
                    self.assertAlmostEqual(e["lower"][k]/e["denominator"], min(actual[k]))
                    self.assertAlmostEqual(e["upper"][k]/e["denominator"], max(actual[k]))

    def test_identical_scores_zero(self):
        e = extrema([1, 0, 1, 0], [1, 2, 2, 4], [1, 2, 2, 4])
        self.assertFalse(e["lower"].any()); self.assertFalse(e["upper"].any())

    def test_all_tied_zero(self):
        self.assertEqual(extrema([1, 0, 1], [1]*3, [2]*3)["base"], 0)

    def test_swap_views(self):
        y = [1, 0, 1, 0]; a = [0, 1, 3, 3]; b = [4, 1, 2, 0]
        e = extrema(y, a, b); f = extrema(y, b, a)
        np.testing.assert_array_equal(e["lower"], -f["upper"])
        np.testing.assert_array_equal(e["upper"], -f["lower"])

    def test_label_complement(self):
        y = np.array([1, 0, 1, 0]); a = [0, 1, 3, 3]; b = [4, 1, 2, 0]
        e = extrema(y, a, b); f = extrema(1-y, a, b)
        np.testing.assert_array_equal(e["lower"], -f["upper"])

    def test_row_and_monotone_invariance(self):
        y = np.array([1, 0, 1, 0]); a = np.array([0, 1, 3, 3]); b = np.array([4, 1, 2, 0]); o = [3, 1, 0, 2]
        e = extrema(y, a, b); f = extrema(y[o], (2*a+5)[o], (3*b-2)[o])
        np.testing.assert_array_equal(e["lower"], f["lower"])
        np.testing.assert_array_equal(e["upper"], f["upper"])

    def test_nested_at_most_bounds(self):
        e = extrema([1, 1, 0, 0], [0, 2, 3, 1], [4, 2, 3, 1])
        self.assertTrue(np.all(np.diff(e["lower"]) <= 0))
        self.assertTrue(np.all(np.diff(e["upper"]) >= 0))

    def test_minimum_reversal(self):
        e = extrema([1, 1, 0, 0], [4, 3, 2, 1], [1, 2, 3, 4]); s = summarize(e)
        j = s["minimum_reach_zero_or_reverse"]["swaps"]
        self.assertGreater(e["lower"][j-1], 0); self.assertLessEqual(e["lower"][j], 0)

    def test_budget_and_class_denominators(self):
        s = summarize(extrema([1]*80+[0]*20, list(range(100)), list(range(99, -1, -1))))
        self.assertEqual([r["swaps"] for r in s["budget_bounds"]], [0, 2, 5])
        self.assertEqual(s["budget_bounds"][2]["fraction_native_negatives"], .25)

    def test_inputs_unchanged(self):
        y = np.array([1, 0, 1]); a = np.array([0., 1., 2.]); b = a[::-1].copy()
        old = [x.copy() for x in (y, a, b)]; extrema(y, a, b)
        for x, z in zip((y, a, b), old):
            np.testing.assert_array_equal(x, z)

    def test_invalid_vectors(self):
        for y, a, b in [([], [], []), ([0, 0], [1, 2], [2, 1]),
                        ([0, 1], [1], [1, 2]), ([0, 2], [1, 2], [2, 1]),
                        ([0, 1], [np.nan, 0], [0, 1])]:
            with self.assertRaises(ValueError):
                extrema(y, a, b)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                run(Path('/absent'), Path(d), Path('/absent'))


if __name__ == "__main__":
    unittest.main()
