import itertools
from pathlib import Path
import tempfile
import time
import unittest
import sys
SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
if SCRIPTS.is_dir():sys.path.insert(0,str(SCRIPTS))
import numpy as np
from review_budget import FixedRanking, budgets, compare, run

class ReviewBudgetTests(unittest.TestCase):
    def test_boundary_tie_expectation(self):
        # Highest record positive; one of two boundary records is positive.
        r=FixedRanking([1,1,0,0],[.9,.5,.5,.1]).at([2])[0]
        self.assertEqual(r['expected_observed_positive'],1.5)
        self.assertEqual((r['boundary_slots'],r['boundary_tie_records']),(1,2))
        self.assertEqual(r['precision'],.75)

    def test_all_tied_equals_analytic_random(self):
        for r in FixedRanking([1,0,1,0],[.5]*4).at([1,2,3,4]):
            self.assertEqual(r['precision'],.5)
            self.assertEqual(r['expected_observed_positive'],r['k']*.5)

    def test_review_all_recovers_all_positives(self):
        r=FixedRanking([1,0,1,1],[.1,.4,.4,.6]).at([4])[0]
        self.assertEqual((r['expected_observed_positive'],r['precision']),(3.,.75))

    def test_perfect_and_reversed_order(self):
        self.assertEqual(FixedRanking([1,1,0,0],[4,3,2,1]).at([2])[0]['expected_observed_positive'],2.)
        self.assertEqual(FixedRanking([1,1,0,0],[1,2,3,4]).at([2])[0]['expected_observed_positive'],0.)

    def test_tie_expectation_matches_all_boundary_permutations(self):
        labels=[1,1,0];actual=FixedRanking(labels,[.2]*3).at([2])[0]['expected_observed_positive']
        brute=np.mean([sum(p[:2]) for p in itertools.permutations(labels)])
        self.assertAlmostEqual(actual,brute)

    def test_weighted_bootstrap_matches_expanded_sample(self):
        y=np.array([1,0,1,0]);s=np.array([.9,.5,.5,.1]);w=np.array([0,1,2,1]);index=np.repeat(np.arange(4),w)
        a=FixedRanking(y,s).at([1,2,3,4],w);b=FixedRanking(y[index],s[index]).at([1,2,3,4])
        self.assertEqual(a,b)

    def test_missing_high_score_bootstrap_group(self):
        r=FixedRanking([1,0,1],[.9,.5,.1]).at([1,3],[0,0,3])
        self.assertEqual([x['precision'] for x in r],[1.,1.])

    def test_row_permutation_invariance(self):
        y=np.array([1,0,1,0,1]);s=np.array([.8,.8,.3,.3,.1]);order=[4,1,3,0,2]
        self.assertEqual(FixedRanking(y,s).at([1,3,5]),FixedRanking(y[order],s[order]).at([1,3,5]))

    def test_strict_monotone_score_invariance(self):
        self.assertEqual(FixedRanking([1,0,1],[1,2,2]).at([1,2,3]),FixedRanking([1,0,1],[3,5,5]).at([1,2,3]))

    def test_same_scores_paired_intervals_zero(self):
        y=np.array([1,0]*50);s=np.linspace(0,1,100)
        result=compare(y,s,s,repetitions=100,seed=5)
        self.assertTrue(all(r['precision_difference']==r['precision_difference_lower']==r['precision_difference_upper']==0 for r in result['rows']))

    def test_swap_views_negates_point_difference(self):
        y=np.array([1,0]*50);s=np.arange(100)
        a=compare(y,s,-s,repetitions=100,seed=3);b=compare(y,-s,s,repetitions=100,seed=3)
        for x,z in zip(a['rows'],b['rows']):
            self.assertAlmostEqual(x['precision_difference'],-z['precision_difference'])
            self.assertAlmostEqual(x['precision_difference_lower'],-z['precision_difference_upper'])

    def test_native_positive_recall_denominator(self):
        r=compare(np.array([1,0]*50),np.ones(100),np.ones(100),repetitions=100)['rows'][2]
        self.assertEqual(r['full']['recall'],.1)
        self.assertEqual(r['full']['precision_lift_over_random'],1.)

    def test_budget_floor_counts(self):
        self.assertEqual(budgets(21939),[219,1096,2193,4387])
        self.assertEqual(budgets(2172),[21,108,217,434])

    def test_invalid_labels_or_scores_rejected(self):
        for y,s in [([.5,1],[0,1]),([0,1],[0,np.nan]),([0,1],[0,np.inf]),([],[]),([0,1],[0])]:
            with self.subTest(y=y,s=s),self.assertRaises(ValueError):FixedRanking(y,s)

    def test_invalid_multiplicities_rejected(self):
        a=FixedRanking([0,1],[0,1])
        for w in [[-1,3],[1,0],[.5,1.5],[np.nan,2],[2]]:
            with self.subTest(w=w),self.assertRaises(ValueError):a.at([1],w)

    def test_invalid_budgets_rejected(self):
        a=FixedRanking([0,1],[0,1])
        for ks in [[0],[3],[1.5],[]]:
            with self.subTest(ks=ks),self.assertRaises(ValueError):a.at(ks)

    def test_no_positive_recall_rejected(self):
        with self.assertRaises(ValueError):compare([0,0],[0,1],[1,0],repetitions=100)

    def test_cpu_deadline_and_output_refusal(self):
        with self.assertRaises(TimeoutError):compare([0,1],[0,1],[1,0],repetitions=100,deadline=time.process_time()-1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):run(Path('/nonexistent'),Path(d),Path('/nonexistent'))

if __name__=='__main__':unittest.main()
