"""Metadata validator fixtures only, not synthetic model examples."""
from pathlib import Path
import sys,unittest,copy,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from function_ai_assessment import validated_assessment,VERSION,TARGET_DEFINITION
from prepare_function_review import CATEGORIES,AXES

class AIAssessmentTest(unittest.TestCase):
    def setUp(self):
        self.code='code'
        self.a={'protocol':VERSION,'model_code_sha256':hashlib.sha256(self.code.encode()).hexdigest(),'label_origin':'AI-assigned','code_origin':'real-public','target_definition_id':TARGET_DEFINITION,'training_eligible':False,'evaluation_eligible':False,'independent_gold':False,'split':None,'method':{k:'explicit' for k in ('assessor','model_family','exact_checkpoint_status','rubric_version','assessed_utc')},'provenance':{'official_source_url':'https://example.test/source','official_revision':'pinned','context_sha256':'hash','scope':'passed_declared_screen_for_AI_annotation','source_notice':'MIT'},'categories':{c:{a:{'status':'unknown','reason':'Missing necessary context','evidence_spans':[]} for a in AXES} for c in CATEGORIES}}
    def test_unknown_is_unobserved_not_negative(self):
        out=validated_assessment(self.code,self.a);self.assertTrue(all(x is None for x in out['labels'].values()));self.assertFalse(any(out['observation_mask'].values()))
    def test_observed_absence_has_mask_and_evidence(self):
        self.a['categories']['reentrancy']['pattern_presence']={'status':'absent','reason':'Inspected relevant body','evidence_spans':[{'start':0,'end':4,'quote':'code'}]};o=validated_assessment(self.code,self.a);self.assertIs(o['labels']['reentrancy'],False);self.assertTrue(o['observation_mask']['reentrancy'])
    def test_wrong_hash_rejects(self):
        with self.assertRaises(ValueError):validated_assessment('other',self.a)
    def test_claimed_gold_or_eligibility_rejects(self):
        for key in ['training_eligible','evaluation_eligible','independent_gold']:
            a=copy.deepcopy(self.a);a[key]=True
            with self.assertRaises(ValueError):validated_assessment(self.code,a)
    def test_missing_origin_rejects(self):
        self.a['label_origin']='human'
        with self.assertRaises(ValueError):validated_assessment(self.code,self.a)
    def test_axes_must_not_collapse(self):
        self.a['categories']['authorization'].pop('contextual_suspicion')
        with self.assertRaises(ValueError):validated_assessment(self.code,self.a)
    def test_forged_span_and_missing_evidence_reject(self):
        for spans in [[],[{'start':0,'end':4,'quote':'fake'}],[{'start':True,'end':4,'quote':'code'}]]:
            self.a['categories']['reentrancy']['pattern_presence']={'status':'present','reason':'Claim','evidence_spans':spans}
            with self.assertRaises(ValueError):validated_assessment(self.code,self.a)
    def test_source_scope_and_checkpoint_status_required(self):
        self.a['method'].pop('exact_checkpoint_status')
        with self.assertRaises(ValueError):validated_assessment(self.code,self.a)
        self.setUp();self.a['provenance']['scope']='nonmatch_only'
        with self.assertRaises(ValueError):validated_assessment(self.code,self.a)

if __name__=='__main__':unittest.main()
