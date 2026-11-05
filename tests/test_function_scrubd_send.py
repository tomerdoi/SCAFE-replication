import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import adapt_scrubd_send_claims as m

def fixture(prefix=''):
 source=prefix+'contract C { function f() public { target.send(1); } }'
 a=source.index('function');b=source.index('}',a)+1;call=source.index('target.send');end=source.index(';',call);member=source.index('send',call)
 def src(start,end):return f'{len(source[:start].encode())}:{len(source[start:end].encode())}:0'
 receiver={'typeDescriptions':{'typeIdentifier':'t_address','typeString':'address'}}
 access={'nodeType':'MemberAccess','memberName':'send','src':src(call,member+4),'expression':receiver,'typeDescriptions':{'typeIdentifier':'t_function_send_nonpayable$_t_uint256_$returns$_t_bool_$','typeString':'function (uint256) returns (bool)'}}
 expression={'nodeType':'FunctionCall','kind':'functionCall','src':src(call,end),'expression':access,'typeDescriptions':{'typeIdentifier':'t_bool','typeString':'bool'}}
 statement={'nodeType':'ExpressionStatement','src':src(call,end),'expression':expression}
 function={'nodeType':'FunctionDefinition','name':'f','id':10,'src':src(a,b),'body':{'nodeType':'Block','statements':[statement]}}
 ast={'nodeType':'SourceUnit','nodes':[{'nodeType':'ContractDefinition','name':'C','nodes':[function]}]}
 span={'start_character':a,'end_character':b,'raw_fragment_sha256':m.text_hash(source[a:b])};anchor={'start_character':member,'end_character':member+4}
 return source,ast,span,anchor,function,statement,expression,access

class TypedSendTests(unittest.TestCase):
 def evidence(self,data):return m.typed_send_evidence(data[0],data[1],0,'C.f',data[2],data[3])
 def test_retained_call_preserves_literals_and_ignores_external_whitespace(self):
  self.assertTrue(m.retains_call('function f(){ target.send ( 1 ); }','target.send(1)'))
  self.assertFalse(m.retains_call('function f(){ other.send(1); }','target.send(1)'))
  self.assertFalse(m.retains_call('function f(){ string s="target.send(1)"; }','target.send(1)'))
 def test_exact_typed_discarded_send(self):
  d=fixture();e=self.evidence(d);self.assertEqual(e['result_type']['typeString'],'bool');self.assertEqual(e['function_byte_span'][0],d[2]['start_character'])
 def test_utf8_byte_offsets_not_character_offsets(self):
  d=fixture('// π\n');e=self.evidence(d)
  self.assertEqual(e['function_byte_span'][0],len(d[0][:d[2]['start_character']].encode()))
  self.assertNotEqual(e['function_byte_span'][0],d[2]['start_character'])
 def test_contract_member_named_send_not_native_address(self):
  d=fixture();d[7]['expression']['typeDescriptions']={'typeIdentifier':'t_contract$_C_$1','typeString':'contract C'}
  with self.assertRaises(ValueError):self.evidence(d)
 def test_member_function_type_and_result_type_required(self):
  for index,key,value in [(7,'typeDescriptions',{'typeIdentifier':'t_function_external','typeString':'function (uint) returns (bool)'}),(6,'typeDescriptions',{'typeIdentifier':'t_tuple$__$','typeString':'tuple()'})]:
   d=fixture();d[index][key]=value
   with self.subTest(index=index),self.assertRaises(ValueError):self.evidence(d)
 def test_assignment_and_guard_not_discarded(self):
  for kind in ['Assignment','FunctionCall']:
   d=fixture();inner=copy.deepcopy(d[6]);d[5]['expression']={'nodeType':kind,'expression':{'nodeType':'Identifier','name':'require'},'arguments':[inner]}
   with self.subTest(kind=kind),self.assertRaises(ValueError):self.evidence(d)
 def test_mismatched_owner_function_and_span_refused(self):
  for qual in ['D.f','C.g','f']:
   d=fixture()
   with self.subTest(qual=qual),self.assertRaises(ValueError):m.typed_send_evidence(d[0],d[1],0,qual,d[2],d[3])
  d=fixture();d[4]['src']='0:1:0'
  with self.assertRaises(ValueError):self.evidence(d)
 def test_ambiguous_owner_or_statement_refused(self):
  d=fixture();d[1]['nodes']*=2
  with self.assertRaises(ValueError):self.evidence(d)
  d=fixture();d[4]['body']['statements']*=2
  with self.assertRaises(ValueError):self.evidence(d)
 def test_changed_raw_hash_anchor_and_source_id_refused(self):
  d=fixture();d[2]['raw_fragment_sha256']='0'*64
  with self.assertRaises(ValueError):self.evidence(d)
  d=fixture();d[3]['start_character']-=1
  with self.assertRaises(ValueError):self.evidence(d)
  with self.assertRaises(ValueError):m.bounds({'src':'0:2:1'},0,10)
  with self.assertRaises(ValueError):m.bounds({'src':'0:20:0'},0,10)
 def test_crlf_source_is_not_normalized(self):
  d=fixture('// line\r\n');e=self.evidence(d);self.assertEqual(e['function_byte_span'][0],len(d[0][:d[2]['start_character']].encode()))

class ProxyBoundaryTests(unittest.TestCase):
 def target(self):
  labels=dict.fromkeys(m.CATEGORIES);masks=dict.fromkeys(m.CATEGORIES,False);labels['unchecked_external_call']=True;masks['unchecked_external_call']=True
  code={'record_id':'x','model_code':'function f() {}','model_code_sha256':m.text_hash('function f() {}'),'input_protocol':'v2'}
  t={'record_id':'x','protocol':m.VERSION,'target_definition_id':m.DEFINITION,'source':m.SOURCE,'model_code_sha256':code['model_code_sha256'],'independent_gold':False,'training_eligible':False,'evaluation_eligible':False,'split':None,'human_review_status':'unreviewed','source_proxy_labels':labels,'source_proxy_observation_mask':masks,'native_UX':True,'typed_site_evidence':[{'basis':'software_fixture_only'}]};return code,t
 def test_partial_positive_mask(self):
  c,t=self.target();_,y,mask=m.send_proxy_example(c,t);self.assertEqual(sum(mask),1);self.assertEqual(sum(y),1)
 def test_unknown_not_negative(self):
  c,t=self.target();t['source_proxy_labels']['authorization']=False
  with self.assertRaises(ValueError):m.send_proxy_example(c,t)
 def test_no_negative_or_unsupported_positive(self):
  c,t=self.target();t['source_proxy_labels']['unchecked_external_call']=False
  with self.assertRaises(ValueError):m.send_proxy_example(c,t)
  c,t=self.target();t['source_proxy_labels']['authorization']=True;t['source_proxy_observation_mask']['authorization']=True
  with self.assertRaises(ValueError):m.send_proxy_example(c,t)
 def test_native_claim_and_proof_required(self):
  for field,value in [('native_UX',False),('typed_site_evidence',[])]:
   c,t=self.target();t[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.send_proxy_example(c,t)
 def test_no_gold_split_eligibility_or_training(self):
  for field,value in [('independent_gold',True),('training_eligible',True),('evaluation_eligible',True),('split','train'),('human_review_status','reviewed')]:
   c,t=self.target();t[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.send_proxy_example(c,t)
  c,t=self.target()
  with self.assertRaises(ValueError):m.send_proxy_example(c,t,for_training=True)
 def test_code_only_identity_and_namespace(self):
  for field,value in [('model_code','changed'),('rationale','audit prose')]:
   c,t=self.target();c[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.send_proxy_example(c,t)
  c,t=self.target();t['target_definition_id']='legacy11'
  with self.assertRaises(ValueError):m.send_proxy_example(c,t)

if __name__=='__main__':unittest.main()
