"""Software metadata fixtures only; not model training/evaluation data."""
import copy, hashlib, unittest
from scripts.consolidate_function_labels import consolidate, CATEGORIES

class ConsolidationTest(unittest.TestCase):
    def setUp(self):
        self.base={'record_id':'id','source':'source','source_revision':'pin','pattern_labels':dict.fromkeys(CATEGORIES)}
        self.view={'record_id':'id','source':'source','source_revision':'pin','model_code':'code','model_code_sha256':hashlib.sha256(b'code').hexdigest(),'view_protocol':'p'}
        self.target={'record_id':'id','source':'source','source_revision':'pin','labels':dict.fromkeys(CATEGORIES),'label_observation_mask':dict.fromkeys(CATEGORIES,False),'target_definition_id':'native-definition','model_code_sha256':self.view['model_code_sha256']}
        self.target['labels']['reentrancy']=False;self.target['label_observation_mask']['reentrancy']=True
    def run_case(self,target=None):
        return consolidate([self.base],[self.view],[('source', [target or self.target])])[0]
    def test_negative_is_observed_unknowns_stay_unknown(self):
        out=self.run_case();self.assertFalse(out['source_label_assertions'][0]['label']);self.assertTrue(all(v is None for v in out['canonical_labels'].values()));self.assertFalse(out['training_eligible']);self.assertEqual(out['ai_assessments']['authorization']['status'],'not_assessed')
    def test_definitions_not_collapsed(self):
        t=copy.deepcopy(self.target);t['target_definition_id']='different';t['labels']['reentrancy']=True
        out=consolidate([self.base],[self.view],[('a',[self.target]),('b',[t])])[0]
        self.assertEqual(len(out['source_label_assertions']),2)
    def test_same_definition_duplicate_or_conflict_rejects(self):
        with self.assertRaises(ValueError):consolidate([self.base],[self.view],[('a',[self.target]),('b',[self.target])])
    def test_hash_mismatch_not_silently_rebound(self):
        self.target['model_code_sha256']='old';self.assertFalse(self.run_case()['source_label_assertions'][0]['matches_consolidated_model_view'])
        self.view['model_code']='changed'
        with self.assertRaises(ValueError):self.run_case()
    def test_unknown_never_becomes_negative(self):
        self.target['label_observation_mask']['reentrancy']=False
        with self.assertRaises(ValueError):self.run_case()
    def test_missing_definition_rejects(self):
        self.target.pop('target_definition_id')
        with self.assertRaises(ValueError):self.run_case()
    def test_duplicate_and_foreign_ids_reject(self):
        with self.assertRaises(ValueError):consolidate([self.base,self.base],[self.view],[])
        self.target['record_id']='foreign'
        with self.assertRaises(ValueError):self.run_case()
    def test_reserved_overlap_and_rationales_not_model_input(self):
        self.target['group_constraints']={'direct_verified_context_corpus_overlap':True};self.target['mapping_evidence']={'rationale':'source claim'}
        out=self.run_case();self.assertIn('overlap',out['reserved_evaluation_overlap']);self.assertEqual(set(out['model_input']),{'model_code','model_code_sha256','input_protocol'})
    def test_source_revision_mismatch_rejects(self):
        self.target['source_revision']='other'
        with self.assertRaises(ValueError):self.run_case()
    def test_native_canonical_promotion_requires_new_protocol(self):
        self.base['pattern_labels']['reentrancy']=True
        with self.assertRaises(ValueError):self.run_case()

if __name__=='__main__':unittest.main()
