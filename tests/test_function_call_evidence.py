import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import extract_function_call_evidence as evidence


def analyze(body):return evidence.extract('function f() external { '+body+' }',0,'record','context')


def fixture(root):
    p=root/'pilot';(p/'public_output').mkdir(parents=True);function='function f() external { recipient.send(1); }';text='contract C { '+function+' }';start=text.index(function)
    row={'record_id':'one','source':'sujeetc/SCRUBD','function_identifier':'C.f','pattern_labels':dict.fromkeys(evidence.CATEGORIES),
         'human_review_status':'unreviewed','training_eligible':False,'split':None,'context_sha256':evidence.recovery.text_hash(text),'raw_sha256':evidence.recovery.text_hash(function),'characters':len(function)}
    evidence.recovery.write_jsonl(p/'public_output/function_candidates.jsonl',[row]);(p/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':evidence.sha256(p/'public_output/function_candidates.jsonl')}))
    c=p/'derivatives/context';(c/'contexts').mkdir(parents=True);(c/'contexts/C.sol').write_bytes(text.encode())
    (c/'recovery_manifest.json').write_text(json.dumps({'delivery_manifest_sha256':evidence.sha256(p/'DELIVERY_SHA256.json'),'artifact_sha256':{'contexts/C.sol':evidence.sha256(c/'contexts/C.sol')}}))
    s=p/'derivatives/scopes';s.mkdir();span={'start_character':start,'end_character':start+len(function),'raw_fragment_sha256':row['raw_sha256']}
    mapping={**row,'scope_status':'unique_qualified_scope_match','pinned_context_sha256':row['context_sha256'],'context_artifact':'contexts/C.sol','selected_span':span,'original_candidate_spans':[span]}
    evidence.recovery.write_jsonl(s/'function_scope_links.jsonl',[mapping]);(s/'manifest.json').write_text(json.dumps({'delivery_manifest_sha256':evidence.sha256(p/'DELIVERY_SHA256.json'),'context_release_manifest_sha256':evidence.sha256(c/'recovery_manifest.json'),'artifact_sha256':{'function_scope_links.jsonl':evidence.sha256(s/'function_scope_links.jsonl')}}))
    return p,c,s


class FunctionCallEvidenceTests(unittest.TestCase):
    def test_direct_modern_and_legacy_invocations_preserve_spans(self):
        cases=[('r.call("");','direct'),('r.call{value:1, gas:2000}("");','modern_options'),('r.call.gas(2000).value(1)("");','legacy_gas_value')]
        for code,shape in cases:
            with self.subTest(shape=shape):
                site=analyze(code)[0];self.assertEqual(site['invocation_shape'],shape);self.assertEqual(site['result_disposition_observation'],'bare_expression_statement');self.assertEqual(site['invocation_span']['comment_masked_text'],code[2:-1])

    def test_result_dispositions_are_observations_not_labels(self):
        cases={'r.send(1);':'bare_expression_statement','msg.sender.send(1);':'bare_expression_statement','payable(msg.sender).send(1);':'bare_expression_statement',
               'require(r.send(1));':'lexical_guard_expression','if (r.send(1)) {}':'lexical_guard_expression',
               '(bool ok,) = r.call("");':'assignment_context','bool ok = r.send(1);':'assignment_context','return r.send(1);':'return_expression',
               'if (ok) r.send(1);':'ambiguous_or_other','for (uint i=0;i<1;i++) r.send(1);':'ambiguous_or_other'}
        for code,expected in cases.items():
            with self.subTest(code=code):
                site=analyze(code)[0];self.assertEqual(site['result_disposition_observation'],expected);self.assertIsNone(site['category_label']);self.assertFalse(site['independently_reviewed']);self.assertEqual(site['receiver_type_status'],'unresolved_not_typechecked')

    def test_comments_and_literals_cannot_create_member_sites(self):
        self.assertEqual(analyze('string memory s="r.call(\"\");"; /* r.send(1); */ // r.call(\"\");\n'),[])
        self.assertEqual(analyze('string memory s = "{ call send }";'),[])

    def test_transfer_other_names_and_function_declarations_are_excluded(self):
        self.assertEqual(analyze('token.transfer(to,1); r.functionCall(data); uint call=1;'),[])
        self.assertEqual(evidence.extract('function call() external {}',0,'one','ctx'),[])

    def test_complex_receivers_and_trailing_expressions_are_not_bare_candidates(self):
        for code in ['recipients[i].send(1);','makeReceiver().send(1);','r.send(1) && ok;','consume(r.send(1));']:
            with self.subTest(code=code):self.assertEqual(analyze(code)[0]['result_disposition_observation'],'ambiguous_or_other')

    def test_unsupported_shapes_and_bad_delimiters_do_not_become_absence(self):
        for code in ['r.call;','r.call.unknown(1);']:
            with self.subTest(code=code):self.assertEqual(analyze(code)[0]['result_disposition_observation'],'unsupported_invocation')
        with self.assertRaises(ValueError):analyze('r.call(];')

    def test_nested_call_result_is_not_flagged_as_ignored(self):
        sites=analyze('r.call(other.send(1));');self.assertEqual(len(sites),2);self.assertEqual(sites[0]['result_disposition_observation'],'bare_expression_statement');self.assertEqual(sites[1]['result_disposition_observation'],'ambiguous_or_other')

    def test_unicode_crlf_and_absolute_hash_bound_offsets(self):
        raw='function f() external {\r\n /* λ annotation */ r.send(1);\r\n}'
        site=evidence.extract(raw,100,'one','ctx')[0]
        for key in ['member_span','invocation_span','statement_window_span']:
            span=site[key];start=span['start_character']-100;end=span['end_character']-100;self.assertEqual(hashlib.sha256(raw[start:end].encode()).hexdigest(),span['raw_span_sha256'])
        self.assertEqual(evidence.extract(raw,100,'one','ctx'),[site])

    def test_prepare_replay_preserves_parents_unknown_labels_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,c,s=fixture(root);before={str(f):evidence.sha256(f) for f in p.rglob('*') if f.is_file()}
            a=evidence.prepare(p,c,s,root/'out');b=evidence.prepare(p,c,s,root/'again');self.assertEqual(a,b);self.assertEqual(a['observed_target_member_sites'],1)
            for f in (root/'out').iterdir():self.assertEqual(f.read_bytes(),(root/'again'/f.name).read_bytes())
            for name,digest in before.items():self.assertEqual(evidence.sha256(Path(name)),digest)
            queue=list(evidence.jsonl_rows(root/'out/review_queue.jsonl'));self.assertEqual(len(queue),1);self.assertTrue(all(v is None for v in queue[0]['pattern_labels'].values()));self.assertFalse(any(queue[0]['label_observation_mask'].values()))
            with self.assertRaises(FileExistsError):evidence.prepare(p,c,s,root/'out')
            with self.assertRaises(ValueError):evidence.prepare(p,c,s,s/'overwrite')

    def test_unanalyzed_scope_has_null_counts_and_review_state_is_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,c,s=fixture(root);rows=list(evidence.jsonl_rows(s/'function_scope_links.jsonl'));rows[0]['scope_status']='unsupported_special_function_form'
            evidence.recovery.write_jsonl(s/'function_scope_links.jsonl',rows);m=json.loads((s/'manifest.json').read_text());m['artifact_sha256']['function_scope_links.jsonl']=evidence.sha256(s/'function_scope_links.jsonl');(s/'manifest.json').write_text(json.dumps(m))
            evidence.prepare(p,c,s,root/'out');row=list(evidence.jsonl_rows(root/'out/function_call_evidence.jsonl'))[0];self.assertIsNone(row['observed_site_count']);self.assertIsNone(row['bare_result_candidate_count'])
            rows[0]['human_review_status']='reviewed';evidence.recovery.write_jsonl(s/'function_scope_links.jsonl',rows);m['artifact_sha256']['function_scope_links.jsonl']=evidence.sha256(s/'function_scope_links.jsonl');(s/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'explicit migration'):evidence.prepare(p,c,s,root/'blocked')

if __name__=='__main__':unittest.main()
