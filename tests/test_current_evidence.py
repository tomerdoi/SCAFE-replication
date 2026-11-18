"""Portable artifact boundary controls; software fixtures only."""
from pathlib import Path
import importlib.util,tempfile,unittest,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from build_current_evidence import build_current,verify_build
p=ROOT/'reproducibility/diagnostic_replay_v1/run_diagnostics.py';s=importlib.util.spec_from_file_location('portable_runner',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class PortableEvidence(unittest.TestCase):
 def test_existing_artifact_not_touched(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'sentinel').write_text('keep')
   with self.assertRaises(FileExistsError):build_current(p,p/'absent')
   self.assertEqual((p/'sentinel').read_text(),'keep')
 def test_scientific_comparison_exact_structure(self):
  self.assertFalse(r.same({'x':1},{'x':1,'identity':'hidden'}));self.assertFalse(r.same([1],[1,2]));self.assertFalse(r.same(1,True))
 def test_nonfinite_or_material_change_rejected(self):
  self.assertFalse(r.same(0.1,float('nan')));self.assertFalse(r.same(.1,.2));self.assertTrue(r.same(.1,.1+1e-14))
 def test_projection_excludes_unlisted_identity(self):
  self.assertEqual(r.project({'results':[{'n':2,'contract_id':'not exported'}]},['n']),[{'n':2}])
 def test_missing_fresh_build_refused(self):
  with self.assertRaises(ValueError):verify_build(Path("/absent"),None)
 def test_replay_existing_output_refused(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileExistsError):r.run(Path(d))
if __name__=='__main__':unittest.main()
