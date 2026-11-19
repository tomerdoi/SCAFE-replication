import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
 from audit_structural_oracle_metadata import audit
except ImportError:
 sys.path.insert(0,str(Path(__file__).resolve().parent));from audit_structural_oracle_metadata import audit
ADDRESS='0x'+'1'*40
BUILD='0.8.18+commit.87f61d96'
def payload():return {'chainId':'1','address':ADDRESS,'compilation':{'language':'Solidity','compiler':'solc','compilerVersion':BUILD,'compilerSettings':{},'fullyQualifiedName':'a.sol:A'},'sourceIds':{'a.sol':{'id':0}}}
def run(p):return audit(p,expected_chain='1',expected_address=ADDRESS,approved_builds=[BUILD])
class Metadata(unittest.TestCase):
 def test_source_ids_never_become_asts_or_admission(self):
  r=run(payload());self.assertEqual(r['source_id_entries_observed'],1);self.assertFalse(r['source_ids_are_structural_asts']);self.assertFalse(r['positive_source_admission'])
 def test_missing_ids_unknown_not_zero(self):
  p=payload();del p['sourceIds'];self.assertIsNone(run(p)['source_id_entries_observed'])
 def test_empty_ids_are_observed_zero_only(self):
  p=payload();p['sourceIds']={};r=run(p);self.assertEqual(r['source_id_entries_observed'],0);self.assertFalse(r['structural_reference_oracle_observed'])
 def test_unrequested_source_content_refused(self):
  p=payload();p['sources']={}
  with self.assertRaises(ValueError):run(p)
 def test_ast_in_id_mapping_refused(self):
  p=payload();p['sourceIds']['a.sol']['ast']={}
  with self.assertRaises(ValueError):run(p)
 def test_duplicate_ids_refused(self):
  p=payload();p['sourceIds']['b.sol']={'id':0}
  with self.assertRaises(ValueError):run(p)
 def test_boolean_id_refused(self):
  p=payload();p['sourceIds']['a.sol']['id']=True
  with self.assertRaises(ValueError):run(p)
 def test_other_contract_refused(self):
  p=payload();p['address']='0x'+'2'*40
  with self.assertRaises(ValueError):run(p)
 def test_unexpected_compilation_payload_refused(self):
  p=payload();p['compilation']['sources']={}
  with self.assertRaises(ValueError):run(p)
 def test_missing_build_remains_unknown(self):
  p=payload();del p['compilation']['compilerVersion'];self.assertIsNone(run(p)['approved_local_exact_build'])
 def test_exact_commit_not_semver_alone(self):
  p=payload();p['compilation']['compilerVersion']='0.8.18+commit.00000000';self.assertFalse(run(p)['approved_local_exact_build'])
 def test_metadata_cannot_assert_absent_rights(self):
  r=run(payload());self.assertFalse(r['per_source_licenses_observed']);self.assertIn('not demonstrated absent',r['missingness_policy']);self.assertTrue(r['approved_local_exact_build'])
if __name__=='__main__':unittest.main()
