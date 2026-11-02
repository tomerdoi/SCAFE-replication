import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_context_review as review

FRAGMENT='function f() external { string memory u="https://host/a/*literal*/"; }'
RAW='// SOURCE CLAIM\ncontract C { /* useful spec */\n'+FRAGMENT+'\n}'


def fixture(root):
    pilot=root/'pilot';(pilot/'public_output').mkdir(parents=True)
    row={'record_id':'stable-record','source':'CoinFabrik/solidity-rnd','source_revision':'pinned','source_identity':'C.sol:offset',
         'original_claim':'source-claim-only','license_status':'unresolved','project_id':None,'raw_sha256':review.text_hash(FRAGMENT),'model_code':FRAGMENT}
    review.write_jsonl(pilot/'public_output/function_candidates.jsonl',[row]);(pilot/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':review.sha256(pilot/'public_output/function_candidates.jsonl')}))
    recovery=pilot/'derivatives/recovered';(recovery/'contexts').mkdir(parents=True);digest=review.text_hash(RAW);(recovery/'contexts'/f'{digest}.sol').write_text(RAW)
    start=RAW.index('function');span={'start_character':start,'end_character':start+len(FRAGMENT),'start_line':3,'end_line':3,'raw_fragment_sha256':row['raw_sha256']}
    link={'record_id':row['record_id'],'context_file':f'contexts/{digest}.sol','pinned_context_sha256':digest,'fragment_localization':'exact_unique_fragment','function_spans':[span],'evaluation_status':'quarantine_existing_corpus_overlap'}
    review.write_jsonl(recovery/'candidate_context_links.jsonl',[link])
    (recovery/'recovery_manifest.json').write_text(json.dumps({'delivery_manifest_sha256':review.sha256(pilot/'DELIVERY_SHA256.json'),'artifact_sha256':{str(p.relative_to(recovery)):review.sha256(p) for p in recovery.rglob('*') if p.is_file()}}))
    prior=pilot/'derivatives/prior';prior.mkdir();review.write_jsonl(prior/'coordinator_key.jsonl',[{'record_id':row['record_id'],'review_id':'old-review'}]);(prior/'manifest.json').write_text(json.dumps({'delivery_manifest_sha256':review.sha256(pilot/'DELIVERY_SHA256.json'),'artifact_sha256':{'coordinator_key.jsonl':review.sha256(prior/'coordinator_key.jsonl')}}))
    return pilot,recovery,prior


def submitted(kit,name):
    rows=list(review.jsonl_rows(kit/'reviewer_A.jsonl'))
    for row in rows:
        row.update(reviewer_id=name,completed=True)
        for cell in row['context_assessments'].values():cell['reason']='External behavior and policy remain uncertain.'
    return rows


class FunctionContextReviewTests(unittest.TestCase):
    def test_masking_preserves_literals_unicode_offsets_and_line_endings(self):
        source='// claim é\r\nstring x="https://a/*still literal*/"; /* é\nmore */ uint y;'
        masked,imports,comments=review.masked_source_and_imports(source)
        self.assertEqual(len(source),len(masked));self.assertEqual(source.count('\n'),masked.count('\n'))
        self.assertIn('"https://a/*still literal*/"',masked);self.assertNotIn('claim',masked);self.assertEqual(len(comments),2);self.assertEqual(imports,[])
        self.assertIn('uint y;',masked)

    def test_cr_only_line_comment_ends_before_following_code(self):
        masked,_,_=review.masked_source_and_imports('// label\ruint x;');self.assertTrue(masked.endswith('\ruint x;'))

    def test_imports_ignore_strings_and_comments_but_accept_alias_forms(self):
        source='// import "fake.sol";\nstring x="import fake.sol;"; import "a.sol"; import {A as B} from "b.sol"; import * as X from "c.sol";'
        _,imports,_=review.masked_source_and_imports(source);self.assertEqual([x['literal'] for x in imports],['"a.sol"','"b.sol"','"c.sol"'])
        self.assertTrue(all(x['resolution']=='unresolved_not_fetched' for x in imports))

    def test_legacy_cr_difference_only_allowed_inside_comments(self):
        raw='/* comment\r\n */ uint x;';masked,_,comments=review.masked_source_and_imports(raw)
        legacy=masked.replace('\r',' ')
        compatibility=review.legacy_view_compatibility(raw,masked,0,len(raw),legacy,comments)
        self.assertEqual(compatibility['changed_character_count'],1)
        raw='string x="literal\rtext";';masked,_,comments=review.masked_source_and_imports(raw)
        with self.assertRaises(ValueError):review.legacy_view_compatibility(raw,masked,0,len(raw),masked.replace('\r',' '),comments)
        with self.assertRaises(ValueError):review.legacy_view_compatibility(raw,masked,0,len(raw),masked.replace('text','TEST'),comments)

    def test_unterminated_comment_literal_and_import_fail_closed(self):
        for source in ['/* unterminated','string x="unfinished','import "a.sol"']:
            with self.subTest(source=source),self.assertRaises(ValueError):review.masked_source_and_imports(source)

    def test_preparation_replay_preserves_input_releases_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);before={str(p):review.sha256(p) for p in pilot.rglob('*') if p.is_file()}
            a=review.prepare(pilot,recovery,prior,root/'one');b=review.prepare(pilot,recovery,prior,root/'two');self.assertEqual(a,b)
            for p in (root/'one').rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(root/'two'/p.relative_to(root/'one')).read_bytes())
            for name,digest in before.items():self.assertEqual(review.sha256(Path(name)),digest)
            self.assertEqual(a['records'],1);self.assertEqual(a['pattern_labels_promoted'],0);self.assertEqual(a['dependency_closure_verified_records'],0)
            item=next(review.jsonl_rows(root/'one/review_items.jsonl'));self.assertNotIn('source',item);self.assertNotIn('source_claim',item)
            self.assertEqual(item['function_code'],FRAGMENT);self.assertTrue(all(v is None for v in item['pattern_labels'].values()))
            doc=(root/'one'/item['context_file']).read_text();self.assertNotIn('SOURCE CLAIM',doc);self.assertNotIn('useful spec',doc)
            with self.assertRaises(FileExistsError):review.prepare(pilot,recovery,prior,root/'one')
            with self.assertRaises(ValueError):review.prepare(pilot,recovery,prior,recovery/'new')

    def test_raw_context_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);path=next((recovery/'contexts').glob('*.sol'));path.write_text(RAW+'\n')
            with self.assertRaises(ValueError):review.prepare(pilot,recovery,prior,root/'out')

    def test_artifact_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inside=root/'inside';inside.mkdir();(root/'outside').write_text('data')
            with self.assertRaises(ValueError):review.safe_file(inside,'../outside')
            large=inside/'large';large.write_bytes(b'x'*(16*1024*1024+1))
            with self.assertRaises(ValueError):review.read_review_rows(large)

    def test_unknown_context_agreement_never_promotes_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);kit=root/'kit';review.prepare(pilot,recovery,prior,kit)
            items=list(review.jsonl_rows(kit/'review_items.jsonl'));docs={x['context_document_id']:(kit/x['context_file']).read_text() for x in items}
            result=review.compare(items,submitted(kit,'A'),submitted(kit,'B'),docs)
            self.assertEqual(result['both_unknown_cells'],3);self.assertFalse(result['context_gate_passed']);self.assertEqual(result['training_eligible_records'],0)
            self.assertEqual(len(result['context_adjudication_queue']),3)

    def test_sufficient_proposals_require_evidence_dependencies_and_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);kit=root/'kit';review.prepare(pilot,recovery,prior,kit)
            items=list(review.jsonl_rows(kit/'review_items.jsonl'));docs={x['context_document_id']:(kit/x['context_file']).read_text() for x in items};rows=submitted(kit,'A')
            cell=rows[0]['context_assessments']['authorization'];cell['decision']='sufficient'
            with self.assertRaises(ValueError):review.validate_context_reviews(items,rows,docs)
            text=docs[rows[0]['context_document_id']];start=text.index('contract');cell.update(evidence_spans=[{'document_id':rows[0]['context_document_id'],'start':start,'end':start+8,'quote':'contract'}],dependencies_addressed='Proposed category-specific dependency rationale.',assumptions=['Explicit proposed policy assumption.'])
            review.validate_context_reviews(items,rows,docs)
            other=copy.deepcopy(rows);other[0]['reviewer_id']='B';result=review.compare(items,rows,other,docs)
            self.assertEqual(result['both_proposed_sufficient_cells'],1);self.assertFalse(result['context_gate_passed']);self.assertEqual(result['pattern_labels_promoted'],0)

    def test_missing_context_must_be_concrete_and_evidence_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);kit=root/'kit';review.prepare(pilot,recovery,prior,kit)
            items=list(review.jsonl_rows(kit/'review_items.jsonl'));docs={x['context_document_id']:(kit/x['context_file']).read_text() for x in items};rows=submitted(kit,'A');cell=rows[0]['context_assessments']['reentrancy'];cell['decision']='insufficient'
            with self.assertRaises(ValueError):review.validate_context_reviews(items,rows,docs)
            cell['needed_context']=['Concrete external callee implementation/version.'];review.validate_context_reviews(items,rows,docs)
            cell['evidence_spans']=[{'document_id':rows[0]['context_document_id'],'start':True,'end':8,'quote':'wrong'}]
            with self.assertRaises(ValueError):review.validate_context_reviews(items,rows,docs)

    def test_packet_identity_completion_labels_and_reviewers_are_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,recovery,prior=fixture(root);kit=root/'kit';review.prepare(pilot,recovery,prior,kit)
            items=list(review.jsonl_rows(kit/'review_items.jsonl'));docs={x['context_document_id']:(kit/x['context_file']).read_text() for x in items};base=submitted(kit,'A')
            variants=[[],copy.deepcopy(base),copy.deepcopy(base),copy.deepcopy(base)]
            variants[1][0]['completed']=False;variants[2][0]['pattern_labels']['reentrancy']=True;variants[3][0]['function_code']='tampered'
            for rows in variants:
                with self.subTest(rows=rows),self.assertRaises(ValueError):review.validate_context_reviews(items,rows,docs)
            with self.assertRaises(ValueError):review.compare(items,base,copy.deepcopy(base),docs)

if __name__=='__main__':unittest.main()
