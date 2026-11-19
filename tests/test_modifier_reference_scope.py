import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_modifier_reference_scope import audit

def node(i,t,src='0:1:0',**kw):return {'id':i,'nodeType':t,'src':src,**kw}
def fixture(second=False):
 d=node(2,'ModifierDefinition',name='guard');q=node(3,'ModifierInvocation',modifierName={'referencedDeclaration':2,'name':'guard'});a=node(1,'ContractDefinition',nodes=[d,q]);r={'sources':{'a.sol':{'id':0,'ast':a}}}
 if second:
  a['nodes'].remove(d);r['sources']['b.sol']={'id':1,'ast':node(4,'ContractDefinition',src='0:1:1',nodes=[{**d,'src':'0:1:1'}])}
 return r
class Scope(unittest.TestCase):
 def test_local(self):
  r=audit(fixture());self.assertEqual((r['same_source'],r['file_name_correct'],r['direct_only_correct']),(1,1,1))
 def test_query_name_not_oracle_name(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'][1]['modifierName']['name']='different';self.assertEqual(audit(p)['file_name_unknown'],1)
 def test_missing_context(self):
  r=audit(fixture(True));self.assertEqual((r['cross_source'],r['file_name_unknown']),(1,1))
 def test_wrong_local_match_retained(self):
  p=fixture(True);p['sources']['a.sol']['ast']['nodes'].append(node(5,'ModifierDefinition',name='guard'));r=audit(p);self.assertEqual(r['file_name_wrong'],1)
 def test_ambiguity_abstains(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'].append(node(5,'ModifierDefinition',name='guard'));self.assertEqual(audit(p)['file_name_unknown'],1)
 def test_constructor_not_modifier(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'][1]['modifierName']['referencedDeclaration']=1;self.assertEqual(audit(p)['base_constructor_excluded'],1)
 def test_missing_reference_unknown(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'][1]['modifierName']['referencedDeclaration']=99;self.assertEqual(audit(p)['reference_unknown'],1)
 def test_empty_support_is_not_perfect_accuracy(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes']=[];r=audit(p);self.assertIsNone(r['file_name_answered_fraction']);self.assertIsNone(r['file_name_accuracy_on_answered'])
 def test_missing_ast_not_absence(self):
  p=fixture();p['sources']['a.sol']['ast']={}
  with self.assertRaises(ValueError):audit(p)
 def test_error_refuses_oracle(self):
  p=fixture();p['errors']=[{'severity':'error'}]
  with self.assertRaises(ValueError):audit(p)
 def test_duplicate_id_refused(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'].append(node(2,'ModifierDefinition',name='x'))
  with self.assertRaises(ValueError):audit(p)
 def test_unbound_span_refused(self):
  p=fixture();p['sources']['a.sol']['ast']['nodes'][0]['src']='0:1:99'
  with self.assertRaises(ValueError):audit(p)
if __name__=='__main__':unittest.main()
