import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import adapt_scrubd_send_claims as m
import test_function_scrubd_send as baseline

def call_fixture(wrappers=(),prefix=''):
 chain='target.call'+''.join('.'+w+'(1)' for w in wrappers)
 source=prefix+'contract C { function f() public { '+chain+'(); } }'
 start=source.index('target.call');member=start+len('target.');lo=source.index('function');hi=source.index('}',lo)+1
 def loc(a,b):return f'{len(source[:a].encode())}:{len(source[a:b].encode())}:0'
 bare={'typeIdentifier':'t_function_barecall_payable$__$returns$_t_bool_$','typeString':'function () payable returns (bool)'}
 access={'nodeType':'MemberAccess','memberName':'call','src':loc(start,member+4),'expression':{'typeDescriptions':{'typeIdentifier':'t_address','typeString':'address'}},'typeDescriptions':copy.deepcopy(bare)}
 base=access;end=member+4;setters=[]
 for w in wrappers:
  end+=len(w)+1
  setter={'nodeType':'MemberAccess','memberName':w,'expression':access,'typeDescriptions':{'typeIdentifier':'t_function_set'+w+'_nonpayable','typeString':'function setting'},'src':loc(start,end)}
  end+=3;access={'nodeType':'FunctionCall','kind':'functionCall','expression':setter,'arguments':[{'nodeType':'Literal','value':'1'}],'typeDescriptions':copy.deepcopy(bare),'src':loc(start,end)};setters.append(setter)
 outer={'nodeType':'FunctionCall','kind':'functionCall','expression':access,'arguments':[],'typeDescriptions':{'typeIdentifier':'t_bool','typeString':'bool'},'src':loc(start,end+2)}
 statement={'nodeType':'ExpressionStatement','expression':outer,'src':loc(start,end+2)}
 function={'nodeType':'FunctionDefinition','name':'f','id':7,'src':loc(lo,hi),'body':{'nodeType':'Block','statements':[statement]}}
 ast={'nodes':[{'nodeType':'ContractDefinition','name':'C','nodes':[function]}]}
 span={'start_character':lo,'end_character':hi,'raw_fragment_sha256':m.text_hash(source[lo:hi])};anchor={'start_character':member,'end_character':member+4}
 return source,ast,span,anchor,base,outer,statement,setters

class LowlevelCallTests(unittest.TestCase):
 def proof(self,d):return m.typed_lowlevel_evidence(d[0],d[1],0,'C.f',d[2],d[3],'call')
 def test_direct_native_call(self):self.assertEqual(self.proof(call_fixture())['native_member'],'call')
 def test_legacy_value_and_gas_chains(self):
  for order in [('value',),('gas',),('value','gas'),('gas','value')]:
   with self.subTest(order=order):self.assertEqual(self.proof(call_fixture(order))['gas_value_wrappers_outer_to_inner'],list(reversed(order)))
 def test_duplicate_or_unrecognized_setting_refused(self):
  for order in [('value','value'),('delegate',)]:
   with self.subTest(order=order),self.assertRaises(ValueError):self.proof(call_fixture(order))
 def test_custom_address_like_call_refused(self):
  d=call_fixture();d[4]['expression']['typeDescriptions']={'typeIdentifier':'t_contract$_C_$1','typeString':'contract C'}
  with self.assertRaises(ValueError):self.proof(d)
 def test_wrong_native_function_type_refused(self):
  d=call_fixture();d[4]['typeDescriptions']['typeIdentifier']='t_function_external'
  with self.assertRaises(ValueError):self.proof(d)
 def test_wrapper_types_and_argument_count_checked(self):
  d=call_fixture(('value',));d[7][0]['typeDescriptions']['typeIdentifier']='t_function_external'
  with self.assertRaises(ValueError):self.proof(d)
  d=call_fixture(('gas',));d[5]['expression']['arguments']=[]
  with self.assertRaises(ValueError):self.proof(d)
 def test_checked_or_assigned_result_not_discarded(self):
  for kind in ['Assignment','FunctionCall']:
   d=call_fixture(('value',));d[6]['expression']={'nodeType':kind,'expression':{'nodeType':'Identifier','name':'require'},'arguments':[d[5]]}
   with self.subTest(kind=kind),self.assertRaises(ValueError):self.proof(d)
 def test_new_tuple_return_semantics_refused(self):
  d=call_fixture();d[5]['typeDescriptions']={'typeIdentifier':'t_tuple$_t_bool_$_t_bytes_memory_ptr_$','typeString':'tuple(bool,bytes memory)'}
  with self.assertRaises(ValueError):self.proof(d)
 def test_non_ascii_offset_and_call_span(self):
  d=call_fixture(('value',),'// π\r\n');p=self.proof(d);a,b=p['call_byte_span'];self.assertEqual(d[0].encode()[a:b],b'target.call.value(1)()')
 def test_send_v1_wrapper_remains_unchanged(self):
  d=baseline.fixture();p=m.typed_send_evidence(d[0],d[1],0,'C.f',d[2],d[3]);self.assertNotIn('native_member',p)
  with self.assertRaises(ValueError):m.typed_lowlevel_evidence(d[0],d[1],0,'C.f',d[2],d[3],'call')
 def test_explicit_v2_namespace_and_no_training(self):
  c,t=baseline.ProxyBoundaryTests().target();t['protocol']=m.VERSION_V2;t['target_definition_id']=m.DEFINITION_V2;t['typed_site_evidence']=[{'native_member':'call'}]
  self.assertEqual(sum(m.send_proxy_example(c,t)[2]),1)
  with self.assertRaises(ValueError):m.send_proxy_example(c,t,for_training=True)
  t['typed_site_evidence']=[{'native_member':'delegatecall'}]
  with self.assertRaises(ValueError):m.send_proxy_example(c,t)

if __name__=='__main__':unittest.main()
