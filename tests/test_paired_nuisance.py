"""Software-only cohort and paired-statistic controls; no research examples."""
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/paired_nuisance.py'
if not p.exists():p=Path(__file__).with_name('paired_nuisance.py')
spec=importlib.util.spec_from_file_location('paired_nuisance',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def frame():
 return {target:{(str(i),seed):(i%2,.2+.6*(i%2)) for i in range(4) for seed in m.SEEDS} for target in m.TARGETS}

class PairedNuisanceControls(unittest.TestCase):
 def test_tied_and_perfect_metrics_have_known_values(self):
  y=np.array([0,1,0,1]);np.testing.assert_allclose(m.metrics(y,np.array([.5]*4)),[.5,.5]);np.testing.assert_allclose(m.metrics(y,y.astype(float)),[1,1])
 def test_no_favorable_intersection_on_missing_key(self):
  a=frame();b=frame();b[m.TARGETS[0]].pop(('0',42))
  with self.assertRaises(ValueError):m.align(a,b)
 def test_missing_seed_in_both_still_fails(self):
  a=frame();b=frame();a[m.TARGETS[0]].pop(('0',42));b[m.TARGETS[0]].pop(('0',42))
  with self.assertRaises(ValueError):m.align(a,b)
 def test_conflicting_labels_fail(self):
  a=frame();b=frame();b[m.TARGETS[0]][('0',42)]=(1,.2)
  with self.assertRaises(ValueError):m.align(a,b)
 def test_reordering_does_not_change_aligned_arrays(self):
  a=frame();b={k:dict(reversed(list(v.items()))) for k,v in frame().items()};x=m.align(a,a);z=m.align(b,b)
  for target in m.TARGETS:
   np.testing.assert_array_equal(x[target]['full'],z[target]['full']);self.assertEqual(x[target]['ordered_identity_sha256'],z[target]['ordered_identity_sha256'])
 def test_identity_control_has_zero_interval(self):
  y=np.array([0,1,0,1]);p=np.array([.1,.9,.5,.4]);r=m.paired(y,p,p,repetitions=100,rng=np.random.default_rng(17),deadline=time.process_time()+30)
  self.assertEqual(r['difference'],[0.,0.]);self.assertEqual(r['lower'],[0.,0.]);self.assertEqual(r['upper'],[0.,0.])
 def test_swap_reverses_paired_difference_and_interval(self):
  y=np.array([0,1,0,1]);a=np.array([.1,.9,.2,.8]);b=np.array([.6,.3,.7,.2]);kw={'repetitions':100,'deadline':time.process_time()+30}
  x=m.paired(y,a,b,rng=np.random.default_rng(17),**kw);z=m.paired(y,b,a,rng=np.random.default_rng(17),**kw)
  np.testing.assert_allclose(x['difference'],-np.array(z['difference']));np.testing.assert_allclose(x['lower'],-np.array(z['upper']));np.testing.assert_allclose(x['upper'],-np.array(z['lower']))
 def csv_load(self,rows):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.csv';p.write_text('dataset,model,contract_id,y_true,y_prob,random_state\n'+rows);return m.load_targets(p)
 def test_non_target_outcomes_are_not_interpreted(self):
  x=self.csv_load('DAppSCAN,not-a-model,software-fixture,NOT_A_LABEL,NOT_A_SCORE,NOT_A_SEED\n')
  self.assertEqual(x,{k:{} for k in m.TARGETS})
 def test_duplicate_target_key_rejected(self):
  with self.assertRaises(ValueError):self.csv_load('DIVE,et,fixture,0,.2,42\nDIVE,et,fixture,0,.2,42\n')
 def test_nonfinite_target_score_rejected(self):
  with self.assertRaises(ValueError):self.csv_load('DIVE,et,fixture,0,nan,42\n')
 def test_wrong_target_model_rejected(self):
  with self.assertRaises(ValueError):self.csv_load('DIVE,rf,fixture,0,.2,42\n')
 def test_existing_output_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'sentinel').write_text('keep')
   with self.assertRaises(FileExistsError):m.run(p,p)
   self.assertEqual((p/'sentinel').read_text(),'keep')
if __name__=='__main__':unittest.main()
