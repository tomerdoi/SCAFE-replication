import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:from adjudicate_cached_query_scope import modifier_shapes,aggregate
except ImportError:
 sys.path.insert(0,str(Path(__file__).resolve().parent));from adjudicate_cached_query_scope import modifier_shapes,aggregate

def row(shapes=None,hits=None,status='parsed'):return {'modifier_shapes':shapes or [],'hits':hits or [],'header_status':status}
class QueryScope(unittest.TestCase):
 def test_plain_function_has_observed_empty_header_shapes(self):self.assertEqual(modifier_shapes('function f() external {}'),[])
 def test_role_call_is_a_syntactic_anchor(self):self.assertEqual(modifier_shapes('function f(address a) external onlyRole(ROLE) {}'),['onlyRole'])
 def test_override_returns_not_modifiers(self):self.assertEqual(modifier_shapes('function f() public virtual override(A,B) returns (uint) {}'),[])
 def test_nested_function_type_returns_are_skipped(self):self.assertEqual(modifier_shapes('function f() external guard returns (function(uint) external returns (uint)) {}'),['guard'])
 def test_qualified_multiple_modifiers(self):self.assertEqual(modifier_shapes('function f() external Auth.guard(1) ready {}'),['Auth.guard','ready'])
 def test_header_comments_do_not_create_queries(self):self.assertEqual(modifier_shapes('function f() /* fake() */ external {}'),[])
 def test_body_names_do_not_create_header_queries(self):self.assertEqual(modifier_shapes('function f() external { guard(); }'),[])
 def test_special_function_not_silently_counted(self):
  with self.assertRaises(ValueError):modifier_shapes('constructor() Base() {}')
 def test_bad_returns_not_zero(self):
  with self.assertRaises(ValueError):modifier_shapes('function f() public returns {}')
 def test_positive_direct_blocks_complete_scope_even_with_unknown_other(self):
  r=aggregate([row(['guard'],['tokens']),row(status='unknown')]);self.assertEqual(r['decision'],'exclude_current_unchanged_query_scope');self.assertFalse(r['positive_source_admission'])
 def test_template_is_quarantine_not_proven_identity(self):self.assertEqual(aggregate([row(['guard'],['template'])])['decision'],'quarantine_current_unchanged_query_scope')
 def test_nonmatch_and_empty_never_admit(self):
  for rs in [[],[row(['guard'])],[row()],[row(status='unknown')]]:self.assertEqual(aggregate(rs)['decision'],'unknown_never_positive_admission')
 def test_match_outside_query_member_is_not_query_exclusion(self):self.assertEqual(aggregate([row([],['tokens']),row(['guard'])])['decision'],'unknown_never_positive_admission')
if __name__=='__main__':unittest.main()
