import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import measure_coinfabrik_pairs as m

def row(i,role='vulnerable',project='p',symbol='C::f',suffix='src/C.sol',revision='pin'):
 return {'record_id':i,'source':m.SOURCE,'source_revision':revision,'project_id':project,'source_path':f'{project}/{role}/{suffix}','function_identifier':symbol,'version_role':role}

class PairEvidenceTests(unittest.TestCase):
 def test_exact_unique_pair(self):
  pairs,unpaired=m.pair_frame([row('v'),row('p','patched')])
  self.assertEqual(len(pairs),1);self.assertEqual(unpaired,[]);self.assertEqual(pairs[0]['role_stripped_path'],'p/src/C.sol')
 def test_missing_is_not_fuzzy_paired(self):
  pairs,unpaired=m.pair_frame([row('v'),row('p','patched',suffix='other/C.sol')])
  self.assertEqual(pairs,[]);self.assertEqual(len(unpaired),2);self.assertTrue(all(r['status']=='missing_counterpart' for r in unpaired))
 def test_projects_and_symbols_not_conflated(self):
  pairs,unpaired=m.pair_frame([row('v'),row('p','patched',project='q'),row('p2','patched',symbol='D::f')])
  self.assertEqual(pairs,[]);self.assertEqual(len(unpaired),3)
 def test_overloads_are_ambiguous_not_selected(self):
  pairs,unpaired=m.pair_frame([row('v1'),row('v2'),row('p','patched')])
  self.assertEqual(pairs,[]);self.assertEqual(unpaired[0]['status'],'ambiguous_endpoint')
 def test_duplicate_identity_and_revision_rejected(self):
  with self.assertRaises(ValueError):m.pair_frame([row('x'),row('x','patched')])
  with self.assertRaises(ValueError):m.pair_frame([row('x'),row('y','patched',revision='other')])
 def test_bad_role_path_and_source_rejected(self):
  for patch in [{'source_path':'/p/vulnerable/C.sol'},{'source_path':'p/vulnerable/../C.sol'},{'source_path':'q/vulnerable/C.sol'},{'version_role':'safe'},{'source':'other'}]:
   r=row('x');r.update(patch)
   with self.subTest(patch=patch),self.assertRaises(ValueError):m.pair_frame([r])
 def test_whitespace_outside_tokens_ignored(self):
  self.assertEqual(m.token_hash('function f(){return 10;}'),m.token_hash(' function  f ( ) {\nreturn 10 ; } '))
 def test_literals_identifiers_and_numbers_preserved(self):
  base='function f(){return "a b";}'
  for code in ['function g(){return "a b";}','function f(){return "ab";}','function f(){return "a  b";}']:
   self.assertNotEqual(m.token_hash(base),m.token_hash(code))
  self.assertNotEqual(m.token_hash('return 1;'),m.token_hash('return 2;'))
 def test_escaped_quotes_and_comment_like_literal_preserved(self):
  code=r'''return "a\" b /*not comment*/ //still literal";'''
  tokens=m.literal_tokens(code)
  self.assertEqual([v for k,v in tokens if k=='literal'],[r'''"a\" b /*not comment*/ //still literal"'''])
  self.assertNotEqual(m.token_hash(code),m.token_hash(code.replace(' b','b')))
 def test_unclosed_literals_empty_and_oversize_refused(self):
  for code in ['','   ','return "unterminated;',"return 'unterminated;",'x'*1_000_001]:
   with self.subTest(length=len(code)),self.assertRaises(ValueError):m.literal_tokens(code)
 def test_pairing_never_assigns_labels_or_repair(self):
  rows=[row('v'),row('p','patched')];before=copy.deepcopy(rows)
  pairs,_=m.pair_frame(rows)
  self.assertEqual(rows,before)
  self.assertFalse(any('label' in k or 'repair' in k for k in pairs[0]))

if __name__=='__main__':unittest.main()
