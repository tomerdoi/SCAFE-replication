import copy,hashlib,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from function_context_assessment import VERSION,TARGET,validate_context_assessment

class ContextAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.code='function compare(uint a,uint b) internal pure returns(bool){return a<b;}'
        self.context={'unit':self.code}; h=hashlib.sha256(self.code.encode()).hexdigest()
        self.record=dict(protocol=VERSION,target_definition_id=TARGET,label_origin='AI-assigned',code_origin='real-public',model_code_sha256=h,training_eligible=False,evaluation_eligible=False,independent_gold=False,split=None,record_id='fixture',method_model_version='test-only',source_provenance_ref='test-only',source_claim_disposition='test-only; not dataset evidence',admission_screen_ref='test-only',parent_release_sha256='0'*64,context_hashes={'unit':h},decisions={})
        for category in ('reentrancy','unchecked_external_call','authorization'):
            self.record['decisions'][category]=dict(status='negative',behavior='Memory comparison only',protections='No sensitive operation or call in this bounded body',scope='Exact test body only',rationale='Fixture for structural validation, not a real-code annotation',missing_evidence=[],evidence=[dict(context_id='unit',start=0,end=len(self.code),quote=self.code)])
    def run_record(self):return validate_context_assessment(self.code,self.record,self.context)
    def test_negative_and_no_admission(self):
        result=self.run_record(); self.assertFalse(result['labels']['reentrancy']);self.assertFalse(result['training_eligible'])
    def test_unknown_is_masked(self):
        self.record['decisions']['reentrancy'].update(status='context-required',missing_evidence=['caller']);r=self.run_record();self.assertIsNone(r['labels']['reentrancy']);self.assertFalse(r['observation_mask']['reentrancy'])
    def test_legacy_pattern_target_rejected(self):
        self.record['target_definition_id']='function-review-pilot-v1:pattern_presence'
        with self.assertRaises(ValueError):self.run_record()
    def test_changed_context_rejected(self):
        self.context['unit']+=' '
        with self.assertRaises(ValueError):self.run_record()
    def test_forged_span_rejected(self):
        self.record['decisions']['authorization']['evidence'][0]['quote']='other'
        with self.assertRaises(ValueError):self.run_record()
    def test_missing_evidence_blocks_observed(self):
        self.record['decisions']['authorization']['missing_evidence']=['policy']
        with self.assertRaises(ValueError):self.run_record()
    def test_promotion_rejected(self):
        self.record['training_eligible']=True
        with self.assertRaises(ValueError):self.run_record()
    def test_missing_source_disposition_rejected(self):
        del self.record['source_claim_disposition']
        with self.assertRaises(ValueError):self.run_record()
    def test_missing_evidence_for_unknown_rejected(self):
        self.record['decisions']['authorization']['status']='unknown'
        with self.assertRaises(ValueError):self.run_record()
    def test_pattern_fields_not_used(self):
        self.record['pattern_labels']={k:True for k in self.record['decisions']}
        self.assertFalse(self.run_record()['labels']['authorization'])
if __name__=='__main__':unittest.main()
