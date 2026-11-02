import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import group_function_candidates as groups


def row(identity,code,source='one',project=None):
    return {'record_id':identity,'model_code':code,'source':source,'source_revision':'pinned','project_id':project,
            'pattern_labels':dict.fromkeys(['reentrancy','unchecked_external_call','authorization']),
            'training_eligible':False,'split':None,'human_review_status':'unreviewed','original_claim':'unverified'}


class FunctionGroupingTests(unittest.TestCase):
    def test_lexical_identity_preserves_literals_identifiers_and_operators(self):
        exact,_,_=groups.signatures('function f(){ return "a b"; }')
        same,_,_=groups.signatures('function /*comment*/ f ( ) {return "a b";}')
        self.assertEqual(exact,same)
        for code in ['function f(){return "ab";}','function g(){return "a b";}']:
            self.assertNotEqual(exact,groups.signatures(code)[0])
        self.assertNotEqual(groups.signatures('function f(){x++;}')[0],groups.signatures('function f(){x + +y;}')[0])
        self.assertNotEqual(groups.signatures('function f(){x>>=1;}')[0],groups.signatures('function f(){x> >=1;}')[0])

    def test_templates_rename_consistently_but_keep_constants_and_selectors(self):
        a=groups.signatures('function f(uint x) public returns(uint){return x+x;}')[1]
        self.assertEqual(a,groups.signatures('function g(uint y) public returns(uint){return y+y;}')[1])
        self.assertNotEqual(a,groups.signatures('function g(uint y) public returns(uint){return y+z;}')[1])
        self.assertNotEqual(groups.signatures('function f(){a.call(1);}')[1],groups.signatures('function g(){b.send(1);}')[1])
        self.assertNotEqual(groups.signatures('function f(){a.foo(1);}')[1],groups.signatures('function g(){b.bar(1);}')[1])
        self.assertNotEqual(groups.signatures('function f(){return 1;}')[1],groups.signatures('function g(){return 2;}')[1])

    def test_source_qualified_project_names_do_not_invent_cross_source_identity(self):
        rows=[row('a','function f(){return 1;}','one','same-name'),row('b','function g(){return 2;}','two','same-name')]
        _,_,_,summary=groups.build_groups(rows,set());self.assertEqual(summary['graphs']['evidence']['components'],2)

    def test_quarantine_propagation_levels_are_explicit_and_monotonic(self):
        rows=[row('a','function f(){return 1;}','one','P'),row('b','function g(){return 2;}','one','P'),
              row('c','function g(){return 2;}','two'),row('d','function h(){return 2;}','three')]
        membership,edges,components,summary=groups.build_groups(rows,{'a'})
        self.assertEqual(summary['direct_verified_context_corpus_overlap'],1)
        self.assertEqual(summary['graphs']['evidence']['conservative_quarantine_candidates'],3)
        self.assertEqual(summary['graphs']['strict_template']['conservative_quarantine_candidates'],4)
        self.assertEqual(summary['graphs']['evidence']['newly_propagated_quarantine_candidates'],2)
        self.assertTrue(all(not r['training_eligible'] and r['split'] is None for r in membership))
        self.assertTrue(any(r['reason']=='identifier_template_heuristic' and r['graph']=='strict_only' for r in edges))

    def test_input_order_and_source_claims_do_not_change_groups(self):
        rows=[row('a','function f(){return 1;}','one','P'),row('b','function g(){return 1;}','two')];before=copy.deepcopy(rows)
        result=groups.build_groups(rows,{'a'});other=list(reversed(copy.deepcopy(rows)))
        for r in other:r['original_claim']='opposite-source-claim'
        self.assertEqual(result,groups.build_groups(other,{'a'}));self.assertEqual(rows,before)

    def test_same_address_or_context_hold_together_without_filling_project_ids(self):
        rows=[row('a','function f(){return 1;}'),row('b','function g(){return 2;}')]
        for r in rows:r['contract_address']='0x'+'a'*40
        members,_,_,summary=groups.build_groups(rows,set());self.assertEqual(summary['graphs']['evidence']['components'],1)
        self.assertTrue(all(r['original_project_id'] is None for r in members))
        for r in rows:r.pop('contract_address');r['context_sha256']='b'*64
        self.assertEqual(groups.build_groups(rows,set())[3]['graphs']['evidence']['components'],1)

    def test_lexical_failure_is_retained_without_template_edge(self):
        members,_,_,summary=groups.build_groups([row('a','function f(){string x="unfinished;}')],set())
        self.assertTrue(members[0]['signature_status'].startswith('lexical_error'))
        self.assertIsNone(members[0]['template_sha256']);self.assertEqual(summary['records'],1)

    def test_invalid_ids_seeds_and_context_fail_closed(self):
        a=row('a','function f(){}')
        for rows,seeds in [([],set()),([a,a],set()),([a],{'missing'})]:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):groups.build_groups(rows,seeds)
        a['context_sha256']='invalid'
        with self.assertRaises(ValueError):groups.build_groups([a],set())

    def test_replay_original_preservation_and_no_split_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';(pilot/'public_output').mkdir(parents=True)
            rows=[row('a','function f(){return 1;}'),row('b','function g(){return 1;}')]
            raw_context='contract C { function f(){return 1;} }';digest=groups.text_hash(raw_context);rows[0]['context_sha256']=digest
            groups.write_jsonl(pilot/'public_output/function_candidates.jsonl',rows);source=pilot/'public_output/function_candidates.jsonl';h=groups.sha256(source)
            (pilot/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':h}))
            recovery=pilot/'derivatives/recovery';recovery.mkdir(parents=True);(recovery/'context.sol').write_text(raw_context);groups.write_jsonl(recovery/'candidate_context_links.jsonl',[{'record_id':'a','context_file':'context.sol','pinned_context_sha256':digest},{'record_id':'b','context_file':None}])
            (recovery/'recovery_manifest.json').write_text(json.dumps({'delivery_manifest_sha256':groups.sha256(pilot/'DELIVERY_SHA256.json'),'artifact_sha256':{'candidate_context_links.jsonl':groups.sha256(recovery/'candidate_context_links.jsonl'),'context.sol':groups.sha256(recovery/'context.sol')}}))
            a=groups.prepare(pilot,recovery,root/'one');b=groups.prepare(pilot,recovery,root/'two');self.assertEqual(a,b)
            for p in (root/'one').iterdir():self.assertEqual(p.read_bytes(),(root/'two'/p.name).read_bytes())
            self.assertEqual(groups.sha256(source),h);self.assertEqual(a['assigned_splits'],0)
            with self.assertRaises(FileExistsError):groups.prepare(pilot,recovery,root/'one')
            with self.assertRaises(ValueError):groups.prepare(pilot,recovery,recovery/'new')
            with self.assertRaises(ValueError):groups.prepare(pilot,recovery,pilot/'public_output/new')
            links=list(groups.jsonl_rows(recovery/'candidate_context_links.jsonl'));links[0]['pinned_context_sha256']='0'*64;groups.write_jsonl(recovery/'candidate_context_links.jsonl',links)
            m=json.loads((recovery/'recovery_manifest.json').read_text());m['artifact_sha256']['candidate_context_links.jsonl']=groups.sha256(recovery/'candidate_context_links.jsonl');(recovery/'recovery_manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):groups.prepare(pilot,recovery,root/'wrong-seed')

if __name__=='__main__':unittest.main()
