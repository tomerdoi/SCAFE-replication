import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import sanitize_function_annotations as clean

BAD="function f() { uint x=1; <- @audit-issue isn't checked\n return; }"


def row(identity,code):
    return {'record_id':identity,'source':'controlled-fixture','source_revision':'pinned','source_identity':identity,'model_code':code,
            'raw_sha256':clean.text_hash(code),'pattern_labels':dict.fromkeys(['reentrancy','unchecked_external_call','authorization']),
            'human_review_status':'unreviewed','training_eligible':False,'split':None,'token_sha256':'legacy-signature-must-not-be-copied'}


class FunctionInlineAnnotationTests(unittest.TestCase):
    def test_exact_post_statement_forms_masked_with_lineage(self):
        for marker in ('<- audit-issue','<- @audit-issue','<- AUDIT-ISSUE'):
            text='function f(){ x(); '+marker+" isn't checked\n return; }";view,audit=clean.filter_view(text)
            self.assertEqual(len(audit['events']),1);event=audit['events'][0]
            self.assertEqual(view[:event['start']],text[:event['start']]);self.assertEqual(view[event['end']:],text[event['end']:])
            self.assertEqual(clean.text_hash(text[event['start']:event['end']]),event['parent_span_sha256'])
            self.assertTrue(all(c==' ' for c in view[event['start']:event['end']]))

    def test_literals_escapes_and_comment_delimiters_preserved(self):
        text='function f(){ string memory x="https://a/; <- @audit-issue \\"quoted\\""; }'
        view,audit=clean.filter_view(text);self.assertEqual(view,text);self.assertEqual(audit['markers_in_literals_retained'],1)

    def test_comments_not_misclassified_as_code_annotations(self):
        for text in ['function f(){ /* ; <- audit-issue */ return; }','function f(){ // ; <- @audit-issue\n return; }']:
            view,audit=clean.filter_view(text);self.assertEqual(view,text);self.assertEqual(audit['markers_in_comments_retained'],1);self.assertEqual(audit['events'],[])

    def test_genuine_comparison_and_unanchored_markers_not_masked(self):
        for text in ['function f(){ return a <-audit-issue; }','function f(){ <- @audit-issue note\nreturn; }']:
            view,audit=clean.filter_view(text);self.assertEqual(view,text);self.assertEqual(len(audit['unhandled_outside_markers']),1)

    def test_newlines_unicode_and_following_guards_unchanged(self):
        text="function f(){x(); <- audit-issue é isn't good\r\nrequire(ok);\rreturn;}"
        view,audit=clean.filter_view(text);self.assertEqual(len(view),len(text));self.assertEqual([i for i,c in enumerate(text) if c in '\r\n'],[i for i,c in enumerate(view) if c in '\r\n'])
        self.assertIn('require(ok);\rreturn;',view)

    def test_multiple_markers_in_one_tail_are_a_single_mask_event(self):
        view,audit=clean.filter_view('function f(){x(); <- audit-issue note <- @audit-issue more\n}')
        self.assertEqual(len(audit['events']),1);self.assertEqual(audit['events'][0]['marker_matches_masked'],2);self.assertEqual(audit['remaining_marker_matches'],0)

    def test_unterminated_literal_before_marker_is_retained_not_guessed(self):
        text='function f(){string x="unfinished; <- audit-issue note';view,audit=clean.filter_view(text)
        self.assertEqual(view,text);self.assertEqual(audit['scanner_diagnostic'],'unterminated_literal_before_eligible_marker');self.assertFalse(audit['events'])

    def test_known_apostrophe_failure_removed_without_semantic_claim(self):
        self.assertTrue(clean.signature_status(BAD)['status'].startswith('lexical_error'))
        view,_=clean.filter_view(BAD);self.assertEqual(clean.signature_status(view)['status'],'computed')

    def test_new_schema_preserves_unknowns_and_drops_stale_signatures(self):
        parent=row('a',BAD);before=copy.deepcopy(parent);views,audit,events=clean.create_views([parent]);view=views[0]
        self.assertEqual(parent,before);self.assertTrue(all(v is None for v in view['pattern_labels'].values()));self.assertTrue(all(v is False for v in view['label_observation_mask'].values()))
        self.assertFalse(view['training_eligible']);self.assertIsNone(view['split']);self.assertNotIn('token_sha256',view)
        self.assertEqual(view['parent_model_code_sha256'],clean.text_hash(BAD));self.assertNotEqual(view['parent_model_code_sha256'],view['model_code_sha256'])
        self.assertFalse(view['representation_review_required'])
        flagged=clean.create_views([row('b','function f(){ <- @audit-issue note\n return; }')])[0][0]
        self.assertTrue(flagged['representation_review_required']);self.assertIn('residual_declared_audit_marker_requires_review',flagged['representation_issue_flags'])

    def test_budget_and_input_type_fail_closed(self):
        for value in [None,1,'x'*(clean.MAX_CHARS+1)]:
            with self.subTest(kind=type(value).__name__),self.assertRaises(ValueError):clean.filter_view(value)

    def test_end_to_end_replay_preserves_originals_and_recomputes_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';(pilot/'public_output').mkdir(parents=True);rows=[row('a',BAD),row('b','function g(){return 1;}')]
            clean.write_jsonl(pilot/'public_output/function_candidates.jsonl',rows);source=pilot/'public_output/function_candidates.jsonl';h=clean.sha256(source)
            (pilot/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':h}))
            recovery=pilot/'derivatives/recovery';recovery.mkdir(parents=True);clean.write_jsonl(recovery/'candidate_context_links.jsonl',[{'record_id':r['record_id'],'context_file':None} for r in rows])
            (recovery/'recovery_manifest.json').write_text(json.dumps({'delivery_manifest_sha256':clean.sha256(pilot/'DELIVERY_SHA256.json'),'artifact_sha256':{'candidate_context_links.jsonl':clean.sha256(recovery/'candidate_context_links.jsonl')}}))
            baseline=pilot/'derivatives/baseline';clean.grouping.prepare(pilot,recovery,baseline);before={str(p):clean.sha256(p) for p in pilot.rglob('*') if p.is_file()}
            a=clean.prepare(pilot,baseline,recovery,root/'one');b=clean.prepare(pilot,baseline,recovery,root/'two');self.assertEqual(a,b)
            for p in (root/'one').rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(root/'two'/p.relative_to(root/'one')).read_bytes())
            for name,digest in before.items():self.assertEqual(clean.sha256(Path(name)),digest)
            self.assertEqual(a['changed_records'],1);self.assertEqual(a['before_signature_status_counts'].get('computed'),1);self.assertEqual(a['after_signature_status_counts'],{'computed':2})
            self.assertEqual(a['training_eligible_records'],0);self.assertEqual(a['assigned_splits'],0)
            with self.assertRaises(FileExistsError):clean.prepare(pilot,baseline,recovery,root/'one')
            with self.assertRaises(ValueError):clean.prepare(pilot,baseline,recovery,baseline/'overwrite')
            source.write_text('tampered')
            with self.assertRaises(ValueError):clean.prepare(pilot,baseline,recovery,root/'three')

if __name__=='__main__':unittest.main()
