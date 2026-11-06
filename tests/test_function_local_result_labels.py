from pathlib import Path
import copy
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import adapt_scrubd_local_result_labels as m
import test_function_scrubd_send as baseline

def body_fixture(style='require',extra_unchecked=False,prefix=''):
 call='target.call.value(1)()'
 if style in {'require','custom_require'}:body='require('+call+');'
 elif style=='throw':body='if (!'+call+') throw;'
 elif style=='revert':body='if (!'+call+') { revert(); }'
 elif style=='fallback':body='if (!'+call+') { value = 0; }'
 elif style=='success':body='if ('+call+') { value = 0; }'
 elif style=='none':body='return;'
 elif style=='alias':body='f();'
 else:raise ValueError(style)
 if extra_unchecked:body+=' target.call.value(1)();'
 source=prefix+'contract C { function f() public { '+body+' } }';lo=source.index('function');hi=source.rindex('}')-1
 while source[hi-1].isspace():hi-=1
 # hi points immediately after the function's closing brace.
 hi=source.rindex('}',0,source.rindex('}'))+1
 loc=lambda a,b:f'{len(source[:a].encode())}:{len(source[a:b].encode())}:0'
 def invocation(start,identity):
  member=start+len('target.');end=start+len(call)
  base={'nodeType':'MemberAccess','memberName':'call','src':loc(start,member+4),'expression':{'typeDescriptions':{'typeIdentifier':'t_address','typeString':'address'}},'typeDescriptions':{'typeIdentifier':'t_function_barecall_payable$__$returns$_t_bool_$'}}
  setter={'nodeType':'MemberAccess','memberName':'value','expression':base,'typeDescriptions':{'typeIdentifier':'t_function_setvalue_nonpayable'}}
  wrapper={'nodeType':'FunctionCall','kind':'functionCall','expression':setter,'arguments':[{'value':'1'}],'typeDescriptions':{'typeIdentifier':'t_function_barecall_payable$__$returns$_t_bool_$value'}}
  return {'id':identity,'nodeType':'FunctionCall','kind':'functionCall','expression':wrapper,'arguments':[],'typeDescriptions':{'typeIdentifier':'t_bool'},'src':loc(start,end)}
 statements=[]
 if style=='none':statements=[{'nodeType':'Return'}]
 elif style=='alias':statements=[{'nodeType':'ExpressionStatement','expression':{'id':9,'nodeType':'FunctionCall','kind':'functionCall','typeDescriptions':{'typeIdentifier':'t_bool'},'expression':{'nodeType':'Identifier','name':'f','typeDescriptions':{'typeIdentifier':'t_function_barecall_payable'}}}}]
 else:
  node=invocation(source.index(call),9);gs=source.index(body);ge=gs+len(body.split(';')[0])+1
  if style in {'require','custom_require'}:
   statement={'nodeType':'ExpressionStatement','src':loc(gs,ge),'expression':{'nodeType':'FunctionCall','kind':'functionCall','arguments':[node],'expression':{'nodeType':'Identifier','name':'require','typeDescriptions':{'typeIdentifier':'t_function_require_pure' if style=='require' else 't_function_internal_pure'}}}}
  else:
   condition={'nodeType':'UnaryOperation','operator':'!','prefix':True,'subExpression':node} if style!='success' else node
   if style=='throw':failure={'nodeType':'Throw'}
   elif style=='revert':failure={'nodeType':'Block','statements':[{'nodeType':'ExpressionStatement','expression':{'nodeType':'FunctionCall','kind':'functionCall','arguments':[],'expression':{'nodeType':'Identifier','name':'revert','typeDescriptions':{'typeIdentifier':'t_function_revert_pure'}}}}]}
   else:failure={'nodeType':'Block','statements':[{'nodeType':'ExpressionStatement','expression':{'nodeType':'Assignment'}}]}
   # Complete conditional span, including the brace for a revert/fallback block.
   ge=source.index(' } }') if style in {'revert','fallback','success'} else source.index(';',gs)+1
   statement={'nodeType':'IfStatement','src':loc(gs,ge),'condition':condition,'trueBody':failure}
  statements=[statement]
  if extra_unchecked:
   start=source.index(call,source.index(call)+1);statements.append({'nodeType':'ExpressionStatement','src':loc(start,start+len(call)),'expression':invocation(start,10)})
 fn={'nodeType':'FunctionDefinition','name':'f','id':1,'src':loc(lo,hi),'body':{'nodeType':'Block','statements':statements}}
 ast={'nodes':[{'nodeType':'ContractDefinition','name':'C','nodes':[fn]}]};span={'start_character':lo,'end_character':hi,'raw_fragment_sha256':m.text_hash(source[lo:hi])}
 return source,ast,span,fn

class BodyContrastTests(unittest.TestCase):
 def assess(self,fixture,model=None):
  source,ast,span,_=fixture;return m.assess_body(source,ast,0,'C.f',span,source if model is None else model)
 def test_direct_builtin_require(self):self.assertEqual(self.assess(body_fixture())['status'],'supported_source_checked_contrast')
 def test_negated_throw_or_revert(self):
  for style in ['throw','revert']:
   with self.subTest(style=style):self.assertEqual(self.assess(body_fixture(style))['status'],'supported_source_checked_contrast')
 def test_no_call_never_becomes_negative(self):self.assertEqual(self.assess(body_fixture('none'))['status'],'no_local_target_call_unknown')
 def test_nonreverting_and_success_only_handling_stay_unknown(self):
  for style in ['fallback','success']:
   with self.subTest(style=style):self.assertEqual(self.assess(body_fixture(style))['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_user_function_named_require_is_not_builtin_guard(self):self.assertEqual(self.assess(body_fixture('custom_require'))['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_all_calls_must_be_checked(self):
  result=self.assess(body_fixture(extra_unchecked=True));self.assertEqual(len(result['typed_target_calls']),2);self.assertEqual(result['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_alias_call_is_unresolved_not_absent(self):
  result=self.assess(body_fixture('alias'));self.assertTrue(result['unresolved_native_call_ids']);self.assertEqual(result['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_inline_assembly_body_requires_review(self):
  f=body_fixture();f[3]['body']['statements'].append({'nodeType':'InlineAssembly','id':90})
  self.assertEqual(self.assess(f)['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_guard_must_survive_model_view(self):
  f=body_fixture();self.assertEqual(self.assess(f,'target.call.value(1)();')['status'],'unsupported_or_nonreverting_handling_unknown')
 def test_original_hash_owner_and_byte_offsets(self):
  f=body_fixture(prefix='// π\r\n');result=self.assess(f);self.assertEqual(result['function_byte_span'][0],len(f[0][:f[2]['start_character']].encode()))
  bad=copy.deepcopy(f);bad[2]['raw_fragment_sha256']='0'*64
  with self.assertRaises(ValueError):self.assess(bad)
  bad=copy.deepcopy(f);bad[1]['nodes'][0]['name']='Other'
  with self.assertRaises(ValueError):self.assess(bad)

class BinaryMaskBoundaryTests(unittest.TestCase):
 def target(self,negative=True):
  code,t=baseline.ProxyBoundaryTests().target();t.update(protocol=m.VERSION,target_definition_id=m.DEFINITION,native_UX=False if negative else True,negative_admission_evidence={'status':'supported_source_checked_contrast','typed_target_calls':[{'guard_evidence':{'byte_span':[0,1]}}],'unresolved_native_call_ids':[]});t['source_proxy_labels']['unchecked_external_call']=not negative
  return code,t
 def test_observed_zero_is_distinct_from_unknown_placeholder(self):
  c,t=self.target();_,values,masks=m.local_binary_example(c,t);self.assertEqual(sum(values),0);self.assertEqual(sum(masks),1)
  t['source_proxy_labels']['unchecked_external_call']=None;t['source_proxy_observation_mask']['unchecked_external_call']=False;_,values,masks=m.local_binary_example(c,t);self.assertEqual(sum(values),0);self.assertEqual(sum(masks),0)
 def test_negative_needs_explicit_source_zero_and_complete_proof(self):
  for value in [None,True,0]:
   c,t=self.target();t['native_UX']=value
   with self.subTest(value=value),self.assertRaises(ValueError):m.local_binary_example(c,t)
  for field,value in [('status','no_local_target_call_unknown'),('typed_target_calls',[]),('unresolved_native_call_ids',[1])]:
   c,t=self.target();t['negative_admission_evidence'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.local_binary_example(c,t)
 def test_other_axes_and_unknowns_not_negative(self):
  c,t=self.target();t['source_proxy_labels']['authorization']=False
  with self.assertRaises(ValueError):m.local_binary_example(c,t)
 def test_no_gold_eligibility_split_training_or_wrong_namespace(self):
  for field,value in [('independent_gold',True),('training_eligible',True),('evaluation_eligible',True),('split','train'),('protocol','old')]:
   c,t=self.target();t[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.local_binary_example(c,t)
  c,t=self.target()
  with self.assertRaises(ValueError):m.local_binary_example(c,t,for_training=True)
 def test_positive_requires_source_and_prior_evidence(self):
  c,t=self.target(False);m.local_binary_example(c,t);t['native_UX']=False
  with self.assertRaises(ValueError):m.local_binary_example(c,t)
 def test_extra_prose_or_changed_code_rejected(self):
  for field,value in [('rationale','audit text'),('model_code','changed')]:
   c,t=self.target();c[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.local_binary_example(c,t)

if __name__=='__main__':unittest.main()
