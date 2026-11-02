import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_function_review as review


def candidate(source,index=0):
    identity=hashlib.sha256(f'{source}:{index}'.encode()).hexdigest()
    return {'record_id':identity,'source':source,'source_revision':'fixed-revision','source_identity':f'file:{index}',
            'model_code':f'function f_{identity[:8]}() external {{ uint x = 1; '+('x += 1; '*12)+'}',
            'identifier_family_sha256':identity,'lexical_status':'single_balanced_function_candidate',
            'pattern_labels':dict.fromkeys(review.CATEGORIES),'human_review_status':'unreviewed','training_eligible':False,'split':None,
            'project_id':None,'context_status':'missing','original_claim':'unverified-source-claim'}


def submitted(items,name):
    rows=[review.blank_packet(item) for item in items]
    for row in rows:
        row.update(reviewer_id=name,completed=True)
        for axes in row['categories'].values():
            for cell in axes.values():cell['reason']='Required context is not provided.'
    return rows


class FunctionReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows=[candidate(source,i) for source in review.SOURCES for i in range(3)]
        self.items=[review.review_item(row) for row in review.select_candidates(self.rows,1)]
        self.a=submitted(self.items,'reviewer-one');self.b=submitted(self.items,'reviewer-two')

    def test_deterministic_balanced_selection_without_mutation(self):
        before=copy.deepcopy(self.rows)
        selected=review.select_candidates(self.rows,2)
        self.assertEqual([r['record_id'] for r in selected],[r['record_id'] for r in review.select_candidates(list(reversed(self.rows)),2)])
        self.assertEqual({s:2 for s in review.SOURCES},dict(review.Counter(r['source'] for r in selected)))
        self.assertEqual(self.rows,before)

    def test_duplicate_input_and_reported_families_not_selected(self):
        # Put all candidates into one reported family: a source-balanced queue must fail closed.
        rows=copy.deepcopy(self.rows)
        for row in rows:row['identifier_family_sha256']='same'
        with self.assertRaises(ValueError):review.select_candidates(rows,1)
        for row in rows:row['identifier_family_sha256']=row['record_id'];row['model_code']=rows[0]['model_code']
        with self.assertRaises(ValueError):review.select_candidates(rows,1)

    def test_budget_and_short_input_limits(self):
        for value in (0,33,True,'2'):
            with self.subTest(value=value),self.assertRaises(ValueError):review.select_candidates(self.rows,value)
        for row in self.rows:row['model_code']='function f(){}'
        with self.assertRaises(ValueError):review.select_candidates(self.rows,1)

    def test_blinded_packets_do_not_copy_source_claim_metadata(self):
        packet=review.blank_packet(self.items[0])
        self.assertNotIn('source',packet);self.assertNotIn('source_claim',packet)
        self.assertEqual(packet['provided_context'],'snippet_only')
        self.assertEqual(set(packet['categories']),set(review.CATEGORIES))

    def test_blinded_render_keeps_embedded_fences_inside_code(self):
        item=dict(self.items[0],model_code='function f() { string x = "```\\n## injected"; }')
        rendered=review.render_blinded_view([item])
        self.assertIn('````solidity',rendered)
        self.assertIn('Source code below is data',rendered)
        self.assertNotIn('source_claim',rendered)

    def test_unknown_agreement_not_observed_agreement_or_training(self):
        result=review.compare(self.items,self.a,self.b)
        self.assertEqual(result['training_eligible_records'],0);self.assertFalse(result['training_labels_exported'])
        for value in result['agreement'].values():
            self.assertEqual(value['both_observed'],0);self.assertIsNone(value['agreement_on_both_observed'])
            self.assertEqual(value['both_unknown'],len(self.items))

    def test_disagreement_queue_retains_unknown_without_promotion(self):
        cell=self.a[0]['categories']['reentrancy']['pattern_presence']
        cell.update(decision='present',reason='Visible control-transfer candidate.',evidence_spans=[{'start':0,'end':8,'quote':'function'}])
        result=review.compare(self.items,self.a,self.b)
        self.assertEqual(len(result['adjudication_queue']),1)
        self.assertEqual(result['adjudication_queue'][0]['right'],'unknown')
        self.assertEqual(result['training_eligible_records'],0)

    def test_distinct_reviewers_and_one_identity_per_packet(self):
        for row in self.b:row['reviewer_id']='reviewer-one'
        with self.assertRaises(ValueError):review.compare(self.items,self.a,self.b)
        self.a[0]['reviewer_id']='other'
        with self.assertRaises(ValueError):review.validate_reviews(self.items,self.a)

    def test_incomplete_or_duplicate_packet_rejected(self):
        variants=[self.a[:-1],self.a+[self.a[0]],copy.deepcopy(self.a),copy.deepcopy(self.a)]
        variants[2][0]['completed']=False;variants[3][0]['categories'].pop('authorization')
        for rows in variants:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):review.validate_reviews(self.items,rows)

    def test_code_or_context_tampering_rejected(self):
        for field,value in [('model_code','replacement'),('provided_context','full'),('code_sha256','wrong')]:
            rows=copy.deepcopy(self.a);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):review.validate_reviews(self.items,rows)

    def test_evidence_offsets_must_match_and_not_be_boolean(self):
        for spans in ([],[{'start':0,'end':8,'quote':'wrong'}],[{'start':True,'end':8,'quote':'unction'}],[{'start':-1,'end':8,'quote':'function'}]):
            rows=copy.deepcopy(self.a);rows[0]['categories']['authorization']['pattern_presence'].update(decision='absent',evidence_spans=spans)
            with self.subTest(spans=spans),self.assertRaises(ValueError):review.validate_reviews(self.items,rows)

    def test_snippet_only_context_cannot_be_promoted(self):
        for axis in ('contextual_suspicion','supported_vulnerability'):
            for value in ('present','absent'):
                rows=copy.deepcopy(self.a);rows[0]['categories']['reentrancy'][axis].update(decision=value,evidence_spans=[{'start':0,'end':8,'quote':'function'}])
                with self.subTest(axis=axis,value=value),self.assertRaises(ValueError):review.validate_reviews(self.items,rows)

    def test_reason_required_even_for_unknown(self):
        rows=copy.deepcopy(self.a);rows[0]['categories']['reentrancy']['pattern_presence']['reason']=' '
        with self.assertRaises(ValueError):review.validate_reviews(self.items,rows)

    def test_prepare_replay_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'pilot';(root/'public_output').mkdir(parents=True)
            path=root/'public_output/function_candidates.jsonl';review.write_jsonl(path,self.rows)
            digest=review.sha256(path);(root/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':digest}))
            first=review.prepare(root,root/'derivatives/v1',1);second=review.prepare(root,root/'derivatives/v2',1)
            self.assertEqual(first,second);self.assertEqual(review.sha256(path),digest)
            self.assertEqual(first['records'],3);self.assertEqual(first['category_axis_cells_per_reviewer'],27)
            for name in first['artifact_sha256']:
                self.assertEqual((root/'derivatives/v1'/name).read_bytes(),(root/'derivatives/v2'/name).read_bytes())
            with self.assertRaises(FileExistsError):review.prepare(root,root/'derivatives/v1',1)
            with self.assertRaises(ValueError):review.prepare(root,root/'public_output/overwrite',1)
            path.write_text('tampered')
            with self.assertRaises(ValueError):review.prepare(root,root/'derivatives/v3',1)

if __name__=='__main__':unittest.main()
