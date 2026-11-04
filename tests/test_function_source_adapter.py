import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import adapt_function_source_labels as adapter

CODE='function f() external { recipient.call(""); }'
TEXT='contract C { '+CODE+' }'


def fixture(root, assessments=('1',), localized=True):
    pilot=root/'pilot';(pilot/'public_output').mkdir(parents=True);records=[]
    for i,value in enumerate(assessments):
        records.append({'record_id':str(i),'source':adapter.SOURCE,'source_revision':'pin','source_identity':'address'+str(i)+'|C.f','function_identifier':'C.f',
                        'model_code':CODE,'raw_sha256':adapter.text_hash(CODE),'characters':len(CODE),'context_sha256':adapter.text_hash(TEXT),
                        'pattern_labels':dict.fromkeys(adapter.CATEGORIES),'training_eligible':False,'split':None,'human_review_status':'unreviewed',
                        'source_metadata':{'RE':value,'UX':'','is_student':'1','Comments':'SECRET AUDIT EXPLANATION'},
                        'source_assessments':{'reentrancy':int(value) if value else None,'unhandled_exceptions':None}})
    adapter.write_jsonl(pilot/'public_output/function_candidates.jsonl',records);(pilot/'source_manifest.json').write_text(json.dumps({'sources':{adapter.SOURCE:{'revision':'pin'}}}))
    (pilot/'DELIVERY_SHA256.json').write_text(json.dumps({n:adapter.sha256(pilot/n) for n in ['source_manifest.json','public_output/function_candidates.jsonl']}))
    views=pilot/'derivatives/views';views.mkdir(parents=True)
    model=[{**r,'model_code_sha256':adapter.text_hash(CODE),'parent_model_code_sha256':adapter.text_hash(CODE),'view_protocol':'pinned-v2','label_observation_mask':dict.fromkeys(adapter.CATEGORIES,False)} for r in records]
    adapter.write_jsonl(views/'function_model_views.jsonl',model);delivery=adapter.sha256(pilot/'DELIVERY_SHA256.json')
    (views/'manifest.json').write_text(json.dumps({'delivery_manifest_sha256':delivery,'artifact_sha256':{'function_model_views.jsonl':adapter.sha256(views/'function_model_views.jsonl')}}))
    context=pilot/'derivatives/contexts';(context/'contexts').mkdir(parents=True);(context/'groups').mkdir();(context/'contexts/C.sol').write_bytes(TEXT.encode())
    groups=[{'record_id':r['record_id'],'exact_input_sha256':adapter.text_hash(CODE),'evidence_component':'e','strict_template_component':'s','token_sha256':'t','template_sha256':'p','direct_verified_context_corpus_overlap':True,'evidence_conservative_quarantine':True,'strict_template_conservative_quarantine':True} for r in records]
    adapter.write_jsonl(context/'groups/candidate_groups.jsonl',groups)
    (context/'recovery_manifest.json').write_text(json.dumps({'delivery_manifest_sha256':delivery,'model_view_manifest_sha256':adapter.sha256(views/'manifest.json'),'artifact_sha256':{str(p.relative_to(context)):adapter.sha256(p) for p in context.rglob('*') if p.is_file()}}))
    scopes=pilot/'derivatives/scopes';scopes.mkdir();start=TEXT.index(CODE);span={'start_character':start,'end_character':start+len(CODE),'raw_fragment_sha256':adapter.text_hash(CODE)}
    mappings=[{**r,'scope_status':'unique_qualified_scope_match' if localized else 'context_unavailable','pinned_context_sha256':adapter.text_hash(TEXT),'context_artifact':'contexts/C.sol','selected_span':span if localized else None,'original_candidate_spans':[span] if localized else [],'source_qualified_identifier':'C.f','selected_declaration':{'name':'C'} if localized else None} for r in records]
    adapter.write_jsonl(scopes/'function_scope_links.jsonl',mappings);(scopes/'manifest.json').write_text(json.dumps({'delivery_manifest_sha256':delivery,'context_release_manifest_sha256':adapter.sha256(context/'recovery_manifest.json'),'artifact_sha256':{'function_scope_links.jsonl':adapter.sha256(scopes/'function_scope_links.jsonl')}}))
    return pilot,context,scopes,views


def update_manifest(root,name,file):
    m=json.loads((root/name).read_text());m['artifact_sha256'][file]=adapter.sha256(root/file);(root/name).write_text(json.dumps(m))


class FunctionSourceAdapterTests(unittest.TestCase):
    def test_native_binary_unknown_and_invalid_are_not_synthetic_negatives(self):
        for raw,value in [('1',True),('0',False),(1,True),(0,False),('',None),(None,None)]:self.assertIs(adapter.native_binary(raw),value)
        for raw in [True,False,'UNKNOWN',' 0 ',2,0.0]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):adapter.native_binary(raw)

    def test_source_proxy_and_unobserved_other_categories_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,('0',));summary=adapter.prepare(*args,root/'out');row=list(adapter.jsonl_rows(root/'out/source_targets.jsonl'))[0]
            self.assertEqual(summary['mapped_RE_counts'],{'False':1});self.assertIs(row['labels']['reentrancy'],False);self.assertTrue(row['label_observation_mask']['reentrancy'])
            for category in ['unchecked_external_call','authorization']:self.assertIsNone(row['labels'][category]);self.assertFalse(row['label_observation_mask'][category])
            self.assertFalse(row['training_eligible']);self.assertFalse(row['independent_gold']);self.assertIsNone(row['split'])

    def test_missing_exact_scope_preserves_native_claim_without_target_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,localized=False);adapter.prepare(*args,root/'out');row=list(adapter.jsonl_rows(root/'out/source_targets.jsonl'))[0]
            self.assertIs(row['native_source_assessments']['RE'],True);self.assertIsNone(row['labels']['reentrancy']);self.assertFalse(any(row['label_observation_mask'].values()))

    def test_unknown_native_RE_is_unobserved_even_with_a_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,('',));result=adapter.prepare(*args,root/'out');self.assertEqual(result['mapped_source_proxy_rows'],0);self.assertEqual(result['mapping_status_counts'],{'native_RE_unknown':1})

    def test_identical_input_conflicting_claims_are_adjudication_not_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,('0','1'));result=adapter.prepare(*args,root/'out');self.assertEqual(result['exact_input_conflict_groups'],1);self.assertEqual(result['mapped_source_proxy_rows'],0)
            for row in adapter.jsonl_rows(root/'out/source_targets.jsonl'):self.assertIsNone(row['labels']['reentrancy']);self.assertEqual(row['mapping_status'],'identical_input_RE_conflict_requires_adjudication')

    def test_masked_join_distinguishes_unknown_from_observed_negative_and_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,('0',));adapter.prepare(*args,root/'out');row=list(adapter.jsonl_rows(root/'out/source_targets.jsonl'))[0];model=list(adapter.jsonl_rows(root/'out/model_inputs.jsonl'))[0]
            code,targets,masks=adapter.masked_source_example(model,row);self.assertEqual(code,CODE);self.assertEqual(targets,[0.,0.,0.]);self.assertEqual(masks,[True,False,False])
            with self.assertRaisesRegex(ValueError,'readiness'):adapter.masked_source_example(model,row,for_training=True)
            for change in ['hash','category','mask','gold']:
                bad=copy.deepcopy(row)
                if change=='hash':bad['model_code_sha256']='bad'
                elif change=='category':bad['labels']['authorization']=True;bad['label_observation_mask']['authorization']=True
                elif change=='mask':bad['labels']['authorization']=False
                else:bad['independent_gold']=True
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):adapter.masked_source_example(model,bad)

    def test_audit_text_and_targets_are_not_in_model_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root);adapter.prepare(*args,root/'out');model=list(adapter.jsonl_rows(root/'out/model_inputs.jsonl'))[0]
            self.assertEqual(set(model),{'record_id','model_code','model_code_sha256','input_protocol'});self.assertNotIn('SECRET',json.dumps(model))

    def test_replay_originals_groups_and_small_audit_queue_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,('1',)*5);before={str(p):adapter.sha256(p) for p in args[0].rglob('*') if p.is_file()};a=adapter.prepare(*args,root/'out');b=adapter.prepare(*args,root/'again');self.assertEqual(a,b);self.assertEqual(a['pilot_audit_rows'],2)
            for p in (root/'out').iterdir():self.assertEqual(p.read_bytes(),(root/'again'/p.name).read_bytes())
            for p,h in before.items():self.assertEqual(adapter.sha256(Path(p)),h)
            for row in adapter.jsonl_rows(root/'out/source_targets.jsonl'):self.assertTrue(row['group_constraints']['direct_verified_context_corpus_overlap']);self.assertFalse(row['evaluation_eligible'])
            with self.assertRaises(FileExistsError):adapter.prepare(*args,root/'out')
            with self.assertRaises(ValueError):adapter.prepare(*args,args[2]/'overwrite')

    def test_wrong_owner_or_span_is_rejected_before_export(self):
        for change in ['owner','span']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=fixture(root);scope=args[2];rows=list(adapter.jsonl_rows(scope/'function_scope_links.jsonl'))
                if change=='owner':rows[0]['selected_declaration']['name']='Other'
                else:rows[0]['selected_span']['end_character']+=1
                adapter.write_jsonl(scope/'function_scope_links.jsonl',rows);update_manifest(scope,'manifest.json','function_scope_links.jsonl')
                with self.assertRaisesRegex(ValueError,'owner|span'):adapter.prepare(*args,root/'out')

    def test_tampered_artifacts_and_reviewed_parent_do_not_silently_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root);view=args[3];rows=list(adapter.jsonl_rows(view/'function_model_views.jsonl'));rows[0]['human_review_status']='reviewed';adapter.write_jsonl(view/'function_model_views.jsonl',rows);update_manifest(view,'manifest.json','function_model_views.jsonl')
            context=args[1];cm=json.loads((context/'recovery_manifest.json').read_text());cm['model_view_manifest_sha256']=adapter.sha256(view/'manifest.json');(context/'recovery_manifest.json').write_text(json.dumps(cm))
            scope=args[2];sm=json.loads((scope/'manifest.json').read_text());sm['context_release_manifest_sha256']=adapter.sha256(context/'recovery_manifest.json');(scope/'manifest.json').write_text(json.dumps(sm))
            with self.assertRaisesRegex(ValueError,'Annotated/assigned'):adapter.prepare(*args,root/'out')
            (args[1]/'contexts/C.sol').write_text('tampered')
            with self.assertRaises(ValueError):adapter.prepare(*args,root/'out')

if __name__=='__main__':unittest.main()
