import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import map_function_scopes as mapping

FRAGMENT='function f() external { emit Done(); }'
CODE='contract First {\n'+FRAGMENT+'\n}\ncontract Second {\n'+FRAGMENT+'\n}\n'


def source_row(identifier='Second.f'):
    return {'record_id':'one','source':'sujeetc/SCRUBD','source_revision':'pin','function_identifier':identifier,
            'pattern_labels':dict.fromkeys(mapping.CATEGORIES),'training_eligible':False,'split':None,'human_review_status':'unreviewed',
            'characters':len(FRAGMENT),'raw_sha256':mapping.recovery.text_hash(FRAGMENT),'context_sha256':mapping.recovery.text_hash(CODE)}


def link_for(code=CODE):
    starts=[];offset=0
    while (start:=code.find(FRAGMENT,offset))>=0:
        starts.append(start);offset=start+len(FRAGMENT)
    return {'record_id':'one','context_file':'contexts/C.sol','pinned_context_sha256':mapping.recovery.text_hash(code),
            'fragment_localization':'ambiguous_identical_fragments' if len(starts)>1 else 'exact_unique_fragment',
            'function_spans':[{'start_character':s,'end_character':s+len(FRAGMENT),'raw_fragment_sha256':mapping.recovery.text_hash(FRAGMENT)} for s in starts],
            'pattern_labels':dict.fromkeys(mapping.CATEGORIES),'training_eligible':False,'split':None,
            'evaluation_status':'quarantine_existing_corpus_overlap'}


def fixture(root):
    pilot=root/'pilot';(pilot/'public_output').mkdir(parents=True);row=source_row()
    mapping.recovery.write_jsonl(pilot/'public_output/function_candidates.jsonl',[row])
    (pilot/'DELIVERY_SHA256.json').write_text(json.dumps({'public_output/function_candidates.jsonl':mapping.sha256(pilot/'public_output/function_candidates.jsonl')}))
    context=pilot/'derivatives/context';(context/'contexts').mkdir(parents=True);(context/'contexts/C.sol').write_bytes(CODE.encode())
    mapping.recovery.write_jsonl(context/'candidate_context_links.jsonl',[link_for()])
    m={'delivery_manifest_sha256':mapping.sha256(pilot/'DELIVERY_SHA256.json'),'artifact_sha256':{str(p.relative_to(context)):mapping.sha256(p) for p in context.rglob('*') if p.is_file()}}
    (context/'recovery_manifest.json').write_text(json.dumps(m));return pilot,context


class FunctionScopeMappingTests(unittest.TestCase):
    def test_qualified_owner_selects_second_identical_fragment_not_first(self):
        result=mapping.resolve(source_row(),link_for(),mapping.scope_index(CODE))
        self.assertEqual(result['scope_status'],'unique_qualified_scope_match');self.assertEqual(result['selected_span']['start_character'],CODE.rindex(FRAGMENT))
        self.assertEqual(result['selected_declaration']['name'],'Second');self.assertEqual(len(result['original_candidate_spans']),2)
        self.assertTrue(all(v is None for v in result['pattern_labels'].values()));self.assertFalse(any(result['label_observation_mask'].values()))

    def test_comments_literals_unicode_and_crlf_do_not_create_scopes(self):
        prefix='// contract Fake { }\r\n/* contract Other {} */\r\n'
        fragment='function f() external { string memory s = unicode"λ { contract Fake }"; }'
        code=prefix+'contract Real {\r\n'+fragment+'\r\n}'
        index=mapping.scope_index(code);self.assertEqual([d['name'] for d in index['declarations']],['Real'])
        span={'start_character':code.index(fragment),'end_character':code.index(fragment)+len(fragment)}
        evidence=mapping.span_scope(span,index,'f');self.assertTrue(evidence['direct_member']);self.assertTrue(evidence['body_span_matches'])

    def test_duplicate_declared_owner_and_two_same_owner_matches_are_unresolved(self):
        for code,expected in [('contract Second { '+FRAGMENT+' } contract Second {}','repeated_contract_name_requires_review'),
                              ('contract Second { '+FRAGMENT+' '+FRAGMENT+' }','multiple_qualified_scope_matches')]:
            with self.subTest(expected=expected):
                result=mapping.resolve(source_row(),link_for(code),mapping.scope_index(code));self.assertEqual(result['scope_status'],expected);self.assertIsNone(result['selected_span'])

    def test_wrong_owner_unsupported_qualifier_and_special_forms_are_not_guessed(self):
        result=mapping.resolve(source_row('Missing.f'),link_for(),mapping.scope_index(CODE));self.assertEqual(result['scope_status'],'qualified_contract_not_found')
        row=source_row();row['source']='CoinFabrik/solidity-rnd';self.assertEqual(mapping.resolve(row,link_for(),mapping.scope_index(CODE))['scope_status'],'unsupported_source_qualifier')
        for name in ['constructor','receive','fallback']:
            result=mapping.resolve(source_row('Second.'+name),link_for(),mapping.scope_index(CODE));self.assertEqual(result['scope_status'],'unsupported_special_function_form');self.assertIsNone(result['selected_span'])

    def test_nested_function_like_text_is_not_a_direct_member(self):
        code='contract Second { function outer() external { '+FRAGMENT+' } }'
        result=mapping.resolve(source_row(),link_for(code),mapping.scope_index(code));self.assertEqual(result['scope_status'],'qualified_scope_conflict_or_unsupported_span')
        self.assertFalse(result['candidate_scope_evidence'][0]['direct_member'])

    def test_unbalanced_context_and_complex_header_fail_closed(self):
        for code in [CODE+'{','}'+CODE,'contract C is B({x:1}) { '+FRAGMENT+' }','contract C;']:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):mapping.scope_index(code)
        result=mapping.resolve(source_row(),link_for(),None,'Unclosed source brace');self.assertEqual(result['scope_status'],'context_lexical_scope_unavailable');self.assertIsNone(result['selected_span'])

    def test_body_boundary_and_function_name_must_match(self):
        index=mapping.scope_index(CODE);span=link_for()['function_spans'][0]
        self.assertFalse(mapping.span_scope(span,index,'different')['function_name_matches'])
        bad={**span,'end_character':span['end_character']+2};self.assertFalse(mapping.span_scope(bad,index,'f')['body_span_matches'])

    def test_library_interface_and_abstract_contract_headers(self):
        code='library L { '+FRAGMENT+' } interface I { function x() external; } abstract contract C is B(1) { '+FRAGMENT+' }'
        index=mapping.scope_index(code);self.assertEqual([(d['kind'],d['name']) for d in index['declarations']],[('library','L'),('interface','I'),('contract','C')])

    def test_prepare_replay_preserves_all_parents_and_no_labels_or_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,c=fixture(root);before={str(f):mapping.sha256(f) for f in p.rglob('*') if f.is_file()}
            a=mapping.prepare(p,c,root/'out');b=mapping.prepare(p,c,root/'again');self.assertEqual(a,b);self.assertEqual(a['qualified_lexical_scope_matches'],1)
            for f in (root/'out').iterdir():self.assertEqual(f.read_bytes(),(root/'again'/f.name).read_bytes())
            for name,digest in before.items():self.assertEqual(mapping.sha256(Path(name)),digest)
            self.assertEqual(a['training_eligible_records'],0);self.assertEqual(a['assigned_splits'],0)
            with self.assertRaises(FileExistsError):mapping.prepare(p,c,root/'out')
            with self.assertRaises(ValueError):mapping.prepare(p,c,c/'overwrite')

    def test_tampered_parent_or_reviewed_state_requires_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,c=fixture(root);rows=list(mapping.jsonl_rows(c/'candidate_context_links.jsonl'));rows[0]['human_review_status']='reviewed'
            mapping.recovery.write_jsonl(c/'candidate_context_links.jsonl',rows);m=json.loads((c/'recovery_manifest.json').read_text());m['artifact_sha256']['candidate_context_links.jsonl']=mapping.sha256(c/'candidate_context_links.jsonl');(c/'recovery_manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'explicit migration'):mapping.prepare(p,c,root/'out')
            (c/'contexts/C.sol').write_text('tampered')
            with self.assertRaises(ValueError):mapping.prepare(p,c,root/'out')

if __name__=='__main__':unittest.main()
