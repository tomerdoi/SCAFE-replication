"""Synthetic source-linkage fixtures, not audited annotations or detector scores."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adapt_coinfabrik_source_findings import (SOURCE, VERSION, DEFINITIONS, CATEGORIES,
    crosswalk, anchored_pointer_pairs, localized_evidence, source_finding_example)
from adapt_function_source_labels import masked_source_example
from prepare_function_review import text_hash


class CoinFabrikSourceTests(unittest.TestCase):
    def setUp(self):
        self.raw='contract C {\n function f(uint x) internal {\n  state = x;\n }\n}\n'
        start=self.raw.index('function');end=self.raw.index(' }')+2;fragment=self.raw[start:end]
        self.span={'start_character':start,'end_character':end,'start_line':2,'end_line':4,'raw_fragment_sha256':text_hash(fragment)}
        self.row={'version_role':'vulnerable','context_sha256':text_hash(self.raw),'function_identifier':'C::f','source_identity':'proj/vulnerable/C.sol:'+str(start),'raw_sha256':text_hash(fragment)}
        self.scope={'scope_status':'unique_qualified_scope_match','pinned_context_sha256':text_hash(self.raw),'source_qualified_identifier':'C::f','context_artifact':'contexts/x.sol','selected_span':self.span,'selected_declaration':{'name':'C','body_open':self.raw.index('{'),'body_close':self.raw.rfind('}')}}
        self.model={'record_id':'a','model_code':fragment,'model_code_sha256':text_hash(fragment),'input_protocol':'fixture_code_only'}
        self.target={'record_id':'a','source':SOURCE,'supervision_protocol':VERSION,'model_code_sha256':text_hash(fragment),'independent_gold':False,'human_review_status':'unreviewed','training_eligible':False,'evaluation_eligible':False,'split':None,'version_role':'vulnerable','source_proxy_labels':dict.fromkeys(CATEGORIES),'source_proxy_observation_mask':dict.fromkeys(CATEGORIES,False),'target_definition_ids':dict.fromkeys(CATEGORIES),'mapping_evidence':None,'finding_support':[]}

    def positive(self,category='authorization'):
        self.target['source_proxy_labels'][category]=True;self.target['source_proxy_observation_mask'][category]=True;self.target['target_definition_ids'][category]=DEFINITIONS[category];self.target['mapping_evidence']={'fixture_localized':True};self.target['finding_support']=[{'category':category}]

    def test_crosswalk_requires_explicit_class_and_supported_integer_cwe(self):
        self.assertEqual(crosswalk({'vulnerability_class':'Access Control','cwe_classification':284}),'authorization')
        self.assertEqual(crosswalk({'vulnerability_class':'Reentrancy','cwe_classification':367}),'reentrancy')
        for finding in [{'vulnerability_class':'Access Control','cwe_classification':640},{'vulnerability_class':'Validation','cwe_classification':252},{'vulnerability_class':'Reentrancy','cwe_classification':'367'},{'vulnerability_class':'Authorization','cwe_classification':True}]:self.assertIsNone(crosswalk(finding))

    def test_positional_line_anchors_and_sol_suffix(self):
        f={'vulnerable_functions':['C.sol::f()','D::g()'],'lines':'2-4, 9'}
        self.assertEqual(anchored_pointer_pairs(f),[('C::f',{'start_line':2,'end_line':4},'C.sol::f()'),('D::g',{'start_line':9,'end_line':9},'D::g()')])

    def test_incomplete_contract_only_malformed_or_untyped_overload_pointers_refused(self):
        for f in [{'vulnerable_functions':['C::'],'lines':'2'},{'vulnerable_functions':['C::f()','D::g()'],'lines':'2'},{'vulnerable_functions':['C::f(uint)'],'lines':'2'},{'vulnerable_functions':['C::f()'],'lines':'4-2'},{'vulnerable_functions':['C:f()'],'lines':'2'}]:self.assertEqual(anchored_pointer_pairs(f),[])

    def test_exact_qualified_context_and_raw_span_localized(self):
        before=copy.deepcopy((self.row,self.scope));e=localized_evidence(self.row,self.scope,self.raw)
        self.assertEqual(e['function_span'],self.span);self.assertEqual(before,(self.row,self.scope))

    def test_patched_missing_scope_and_overloaded_symbol_not_accepted(self):
        self.row['version_role']='patched';self.assertIsNone(localized_evidence(self.row,self.scope,self.raw));self.row['version_role']='vulnerable'
        self.scope['scope_status']='ambiguous';self.assertIsNone(localized_evidence(self.row,self.scope,self.raw))
        self.scope['scope_status']='unique_qualified_scope_match';raw=self.raw[:self.raw.rfind('}')]+ 'function f() internal {}\n'+self.raw[self.raw.rfind('}'):]
        self.row['context_sha256']=self.scope['pinned_context_sha256']=text_hash(raw);self.scope['selected_declaration']['body_close']=len(raw)-2
        self.assertIsNone(localized_evidence(self.row,self.scope,raw))

    def test_context_fragment_offset_or_owner_tampering_rejected(self):
        for which,value in [('context','changed'),('offset','wrong'),('owner','Other')]:
            row=copy.deepcopy(self.row);scope=copy.deepcopy(self.scope);raw=self.raw
            if which=='context':raw=value
            elif which=='offset':row['source_identity']='proj/vulnerable/C.sol:999'
            else:scope['selected_declaration']['name']=value
            with self.subTest(which=which),self.assertRaises(ValueError):localized_evidence(row,scope,raw)

    def test_positive_only_partial_mask_unknown_is_not_negative(self):
        self.positive();code,values,masks=source_finding_example(self.model,self.target)
        self.assertEqual(code,self.model['model_code']);self.assertEqual(values,[0.,0.,1.]);self.assertEqual(masks,[False,False,True])
        self.target['source_proxy_labels']['reentrancy']=False
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)

    def test_patched_source_support_and_unsupported_category_cannot_promote(self):
        self.positive();self.target['version_role']='patched'
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)
        self.target['version_role']='vulnerable';self.target['finding_support']=[]
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)

    def test_gold_readiness_training_and_legacy_namespace_fail_closed(self):
        self.positive()
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target,for_training=True)
        with self.assertRaises(ValueError):masked_source_example(self.model,self.target)
        self.target['independent_gold']=True
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)

    def test_code_hash_and_extra_finding_prose_input_fields_rejected(self):
        self.model['audit_description']='source explanation'
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)
        del self.model['audit_description'];self.model['model_code']='changed'
        with self.assertRaises(ValueError):source_finding_example(self.model,self.target)


if __name__=='__main__':unittest.main()
