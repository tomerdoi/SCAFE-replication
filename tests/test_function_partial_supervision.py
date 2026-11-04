"""Fixed CPU tensor/gradient checks, not model training or evaluation."""
import copy
import json
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from function_partial_supervision import (CATEGORIES, INPUT_PROTOCOL, MODEL_FIELDS,
    collate_source_examples, load_source_batch, observed_bce_with_logits)
from adapt_function_source_labels import VERSION, DEFINITION, SOURCE
from prepare_function_review import text_hash

torch.set_num_threads(1)


def example(identity='a', label=False, eligible=False):
    code='function f() public {}'
    model={'record_id':identity,'model_code':code,'model_code_sha256':text_hash(code),'input_protocol':INPUT_PROTOCOL}
    target={'record_id':identity,'model_code_sha256':text_hash(code),
            'supervision_protocol':VERSION,'target_definition_id':DEFINITION,
            'labels':dict(zip(CATEGORIES,[label,None,None])),
            'label_observation_mask':dict(zip(CATEGORIES,[label is not None,False,False])),
            'independent_gold':False,'human_review_status':'unreviewed',
            'training_eligible':eligible,'training_readiness_blockers':[] if eligible else ['quality_pending'],
            'source':SOURCE,'mapping_status':'mapped_source_proxy','mapping_evidence':{'fixture':True},
            'native_source_assessments':{'RE':label}}
    return model,target


class PartialSourceLossTests(unittest.TestCase):
    def loss(self,l,y,m,**kw):return observed_bce_with_logits(l,y,m,categories=CATEGORIES,**kw)

    def test_manual_observed_loss_and_analytic_gradients(self):
        l=torch.tensor([[0.,2.,-1.],[1.,-2.,0.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1.,0.,0.],[0.,1.,1.]],dtype=torch.float64)
        m=torch.tensor([[True,False,True],[True,True,False]])
        out=self.loss(l,y,m)
        reference=sum(math.log1p(math.exp(float(l[i,j].detach())))-float(y[i,j])*float(l[i,j].detach()) for i,j in [(0,0),(0,2),(1,0),(1,1)])/4
        self.assertAlmostEqual(out.loss.item(),reference,places=12)
        out.loss.backward();expected=torch.zeros_like(l)
        expected[m]=(torch.sigmoid(l.detach()[m])-y[m])/4
        torch.testing.assert_close(l.grad,expected)
        self.assertEqual((out.observed_cells,out.observed_rows,out.observed_per_category),(4,2,(2,1,1)))
        self.assertFalse(out.must_skip)

    def test_nonfinite_unknown_cells_have_zero_effect_and_gradient(self):
        m=torch.tensor([[True,False,False]])
        gradients=[];losses=[]
        for unknown in [0.,1e30,float('nan'),float('inf')]:
            l=torch.tensor([[0.,unknown,unknown]],requires_grad=True)
            y=torch.tensor([[1.,unknown,unknown]])
            out=self.loss(l,y,m);out.loss.backward();losses.append(out.loss.item());gradients.append(l.grad)
            self.assertEqual(l.grad[0,1:].tolist(),[0.,0.])
        self.assertEqual(len(set(losses)),1)
        for g in gradients:torch.testing.assert_close(g,gradients[0])

    def test_observed_negative_is_not_unknown(self):
        l=torch.zeros((1,3),requires_grad=True);y=torch.zeros_like(l);m=torch.tensor([[True,False,False]])
        out=self.loss(l,y,m);out.loss.backward()
        self.assertEqual(l.grad.tolist(),[[0.5,0.,0.]])

    def test_unobserved_rows_do_not_dilute_objective_or_gradient(self):
        values=[]
        for size in [1,100]:
            l=torch.zeros((size,3),requires_grad=True);y=torch.zeros_like(l);m=torch.zeros_like(l,dtype=torch.bool);m[0,0]=True
            out=self.loss(l,y,m);out.loss.backward();values.append((out.loss.item(),l.grad[0,0].item()))
        self.assertEqual(values[0],values[1])

    def test_all_unknown_requires_skip_without_loss_or_gradients(self):
        l=torch.full((2,3),float('nan'),requires_grad=True)
        out=self.loss(l,torch.full_like(l,float('nan')),torch.zeros_like(l,dtype=torch.bool))
        self.assertIsNone(out.loss);self.assertTrue(out.must_skip);self.assertIsNone(l.grad)
        self.assertEqual(out.observed_per_category,(0,0,0))

    def test_positive_weights_apply_only_to_observed_positive_terms(self):
        l=torch.zeros((1,3),dtype=torch.float64,requires_grad=True);y=torch.tensor([[1.,0.,1.]],dtype=torch.float64);m=torch.tensor([[True,True,False]])
        out=self.loss(l,y,m,positive_class_weights=torch.tensor([3.,7.,99.],dtype=torch.float64))
        self.assertAlmostEqual(out.loss.item(),2*math.log(2),places=12)
        out.loss.backward();torch.testing.assert_close(l.grad,torch.tensor([[-.75,.25,0.]],dtype=torch.float64))

    def test_legacy_width_order_nonboolean_masks_and_dtype_rejected(self):
        l=torch.zeros((1,3));y=l.clone();m=torch.ones_like(l,dtype=torch.bool)
        bad=[(torch.zeros((1,11)),torch.zeros((1,11)),torch.ones((1,11),dtype=torch.bool),CATEGORIES),
             (l,y,m,tuple(reversed(CATEGORIES))),(l,y,m.float(),CATEGORIES),(l,y.double(),m,CATEGORIES)]
        for a,b,c,categories in bad:
            with self.subTest(shape=a.shape,categories=categories),self.assertRaises(ValueError):
                observed_bce_with_logits(a,b,c,categories=categories)

    def test_invalid_observed_values_and_learnable_labels_rejected(self):
        m=torch.tensor([[True,False,False]])
        for logit,label in [(float('nan'),0.),(float('inf'),1.),(0.,float('nan')),(0.,.5),(0.,-1.)]:
            with self.subTest(logit=logit,label=label),self.assertRaises(ValueError):
                self.loss(torch.tensor([[logit,0.,0.]]),torch.tensor([[label,0.,0.]]),m)
        with self.assertRaises(ValueError):self.loss(torch.zeros((1,3)),torch.zeros((1,3),requires_grad=True),m)

    def test_invalid_or_overflowing_weights_rejected(self):
        l=torch.zeros((1,3));m=torch.tensor([[True,False,False]])
        for w in [torch.tensor([1.,0.,1.]),torch.tensor([float('nan'),1.,1.]),torch.ones(11),torch.ones(3,dtype=torch.float64),torch.ones(3,requires_grad=True)]:
            with self.assertRaises(ValueError):self.loss(l,l.clone(),m,positive_class_weights=w)
        with self.assertRaises(ValueError):self.loss(torch.tensor([[-1e38,0.,0.]]),torch.tensor([[1.,0.,0.]]),m,positive_class_weights=torch.tensor([3e38,1.,1.]))

    def test_batch_preserves_code_only_inputs_unknown_mask_and_negative(self):
        pairs=[example('a',False),example('b',None)];before=copy.deepcopy(pairs)
        batch=collate_source_examples(pairs,for_training=False)
        self.assertEqual(batch.model_codes,tuple(p[0]['model_code'] for p in pairs))
        self.assertEqual(batch.observation_mask.tolist(),[[True,False,False],[False,False,False]])
        self.assertEqual(batch.targets.tolist(),[[0.,0.,0.],[0.,0.,0.]])
        self.assertFalse(batch.for_training);self.assertEqual(pairs,before)

    def test_default_training_gate_and_input_identity_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'readiness'):collate_source_examples([example()])
        model,target=example(eligible=True)
        self.assertTrue(collate_source_examples([(model,target)]).for_training)  # synthetic approved fixture only
        model['model_code']='changed'
        with self.assertRaises(ValueError):collate_source_examples([(model,target)])

    def test_extra_audit_input_fields_duplicates_and_unsupported_views_rejected(self):
        for change in [('audit_comment','source claim'),('input_protocol','old_view')]:
            model,target=example();model[change[0]]=change[1]
            with self.assertRaises(ValueError):collate_source_examples([(model,target)],for_training=False)
        with self.assertRaises(ValueError):collate_source_examples([example(),example()],for_training=False)
        with self.assertRaises(ValueError):collate_source_examples([],for_training=False)

    def test_pinned_release_loader_gate_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model,target=example()
            for name,row in [('model_inputs.jsonl',model),('source_targets.jsonl',target)]:
                (root/name).write_text(json.dumps(row)+'\n')
            def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
            manifest={'protocol':VERSION,'target_definition_id':DEFINITION,'model_input_fields':sorted(MODEL_FIELDS),
                      'artifact_sha256':{n:digest(root/n) for n in ['model_inputs.jsonl','source_targets.jsonl']}}
            (root/'manifest.json').write_text(json.dumps(manifest));h=digest(root/'manifest.json')
            with self.assertRaisesRegex(ValueError,'readiness'):load_source_batch(root,expected_manifest_sha256=h)
            b=load_source_batch(root,expected_manifest_sha256=h,for_training=False);self.assertEqual(b.record_ids,('a',))
            with self.assertRaises(ValueError):load_source_batch(root,expected_manifest_sha256='0'*64,for_training=False)
            (root/'model_inputs.jsonl').write_text('changed')
            with self.assertRaises(ValueError):load_source_batch(root,expected_manifest_sha256=h,for_training=False)


if __name__=='__main__':unittest.main()
