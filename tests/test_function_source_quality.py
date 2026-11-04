"""Synthetic regression fixtures; no semantic correctness or accuracy tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_function_source_quality import VERSION, validate_reviews
from adapt_function_source_labels import VERSION as SOURCE_VERSION, DEFINITION
from prepare_function_review import text_hash


class SourceQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.code='contract C {\r\nfunction f() public {}\r\n}\r\n'
        (self.root/'context.sol').write_bytes(self.code.encode())
        fragment='function f() public {}';start=self.code.index(fragment)
        mapping={'context_artifact':'context.sol','context_sha256':text_hash(self.code),
                 'function_span':{'start_character':start,'end_character':start+len(fragment),
                                  'raw_fragment_sha256':text_hash(fragment)},
                 'qualified_identifier':'C.f'}
        self.candidates={'a':{'raw_sha256':text_hash(fragment),'function_identifier':'C.f',
                              'source_metadata':{'Comments':'source claim'}}}
        self.targets={'a':{'mapping_evidence':mapping,'labels':{'reentrancy':False},
                           'mapping_available_for_source_supervision':True,
                           'supervision_protocol':SOURCE_VERSION,'target_definition_id':DEFINITION,
                           'native_source_assessments':{'RE':False},'source_identity':'addr|C.f',
                           'source_revision':'revision','group_constraints':{'evidence_component':'group'}}}
        self.queue=[{'record_id':'a','mapping_evidence':mapping,'source_proxy_label':False,
                     'stratum':{'source_RE':False,'is_student':'0'},'stratum_population':4,'stratum_selected':1}]
        self.reviews=[{'record_id':'a','protocol':VERSION,'reviewer_kind':'ai_assisted_single_pass',
                       'review_status':'preliminary_completed','independent_gold':False,
                       'independent_decision':None,'training_eligible':False,'evaluation_eligible':False,
                       'mapping_assessment':'consistent_with_pinned_qualified_span',
                       'source_comment_sha256':text_hash('source claim'),
                       'findings':[{'code':'execution_semantics_required','kind':'ai_interpretation',
                                    'observation':'A compiler-specific check remains necessary.',
                                    'evidence':[{'start_line':2,'end_line':2,
                                                 'raw_excerpt_sha256':text_hash(self.code.splitlines(keepends=True)[1])}]}],
                       'remaining_questions':['Compiler behavior?']}]

    def check(self):
        return validate_reviews(self.reviews,self.queue,self.targets,self.candidates,self.root)

    def test_retains_native_negative_without_promoting_gold_or_mutating_inputs(self):
        before=copy.deepcopy((self.reviews,self.targets,self.candidates,self.queue))
        result=self.check()[0]
        self.assertIs(result['source_native_RE'],False)
        self.assertIs(result['independent_gold'],False)
        self.assertIsNone(result['independent_decision'])
        self.assertIs(result['training_eligible'],False)
        self.assertEqual(before,(self.reviews,self.targets,self.candidates,self.queue))

    def test_duplicate_missing_or_extra_sample_rejected(self):
        for bad in [[],self.reviews*2,[dict(self.reviews[0],record_id='b')]]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_reviews(bad,self.queue,self.targets,self.candidates,self.root)

    def test_ai_cannot_be_gold_human_or_eligible(self):
        for key,val in [('reviewer_kind','human'),('independent_gold',True),
                        ('independent_decision',False),('training_eligible',True),
                        ('evaluation_eligible',True),('review_status','adjudicated')]:
            with self.subTest(key=key):
                old=self.reviews[0][key];self.reviews[0][key]=val
                with self.assertRaises(ValueError):self.check()
                self.reviews[0][key]=old

    def test_cannot_export_labels_masks_or_model_input(self):
        for key in ['labels','label_observation_mask','model_code','source_proxy_label']:
            self.reviews[0][key]={}
            with self.assertRaises(ValueError):self.check()
            del self.reviews[0][key]

    def test_evidence_context_and_line_hashes_fail_closed(self):
        item=self.reviews[0]['findings'][0]['evidence'][0]
        for key,val in [('start_line',0),('end_line',999),('start_line',True),('raw_excerpt_sha256','wrong')]:
            old=item[key];item[key]=val
            with self.assertRaises(ValueError):self.check()
            item[key]=old
        (self.root/'context.sol').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check()

    def test_qualified_span_and_source_rationale_must_match(self):
        self.candidates['a']['function_identifier']='Other.f'
        with self.assertRaises(ValueError):self.check()
        self.candidates['a']['function_identifier']='C.f'
        self.candidates['a']['source_metadata']['Comments']='other claim'
        with self.assertRaises(ValueError):self.check()

    def test_source_target_and_frozen_queue_must_agree(self):
        self.targets['a']['labels']['reentrancy']=True
        with self.assertRaises(ValueError):self.check()

    def test_no_unanchored_or_unsupported_findings(self):
        f=self.reviews[0]['findings'][0]
        f['code']='verified_exploitable'
        with self.assertRaises(ValueError):self.check()
        f['code']='execution_semantics_required';f['evidence']=[]
        with self.assertRaises(ValueError):self.check()

    def test_path_traversal_not_allowed(self):
        for row in [self.targets['a'],self.queue[0]]:
            row['mapping_evidence']=dict(row['mapping_evidence'],context_artifact='../outside.sol')
        with self.assertRaises(ValueError):self.check()

    def test_raw_crlf_bound_span_not_normalized(self):
        self.targets['a']['mapping_evidence']['function_span']['raw_fragment_sha256']='wrong'
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
