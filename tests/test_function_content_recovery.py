import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import extend_function_context as extension

TEXT = 'contract C {\nfunction f() external { uint a=1; }\n}'
FRAGMENT = 'function f() external { uint a=1; }'


def fixture(root, text=TEXT, known=False):
    p=root/'pilot';(p/'public_output').mkdir(parents=True);raw=root/'raw';raw.mkdir()
    row={'record_id':'record-one','source':'CoinFabrik/solidity-rnd','source_revision':'pin','source_identity':f'P/vulnerable/Original.sol:{text.index("function")}',
         'source_path':'P/vulnerable/Original.sol','project_id':'P','license_status':'unresolved',
         'context_sha256':extension.recovery.text_hash(text),'raw_sha256':extension.recovery.text_hash(FRAGMENT),'characters':len(FRAGMENT),
         'model_code':FRAGMENT,'pattern_labels':{'reentrancy':None,'unchecked_external_call':None,'authorization':None},'split':None,
         'training_eligible':False,'human_review_status':'unreviewed','context_status':'source_file_available'}
    extension.recovery.write_jsonl(p/'public_output/function_candidates.jsonl',[row])
    (p/'source_manifest.json').write_text(json.dumps({'sources':{'CoinFabrik/solidity-rnd':{'revision':'pin'}}}))
    (p/'DELIVERY_SHA256.json').write_text(json.dumps({n:extension.sha256(p/n) for n in ['source_manifest.json','public_output/function_candidates.jsonl']}))
    if known:(raw/'Original.sol').write_bytes(text.encode('utf-8'))
    prior=p/'derivatives/prior';extension.recovery.recover(p,raw,prior)
    view=p/'derivatives/view';view.mkdir()
    vr={**row,'model_code_sha256':extension.recovery.text_hash(FRAGMENT),'parent_model_code_sha256':extension.recovery.text_hash(FRAGMENT),'label_observation_mask':{k:False for k in row['pattern_labels']}}
    extension.recovery.write_jsonl(view/'function_model_views.jsonl',[vr])
    (view/'manifest.json').write_text(json.dumps({'delivery_manifest_sha256':extension.sha256(p/'DELIVERY_SHA256.json'),'artifact_sha256':{'function_model_views.jsonl':extension.sha256(view/'function_model_views.jsonl')}}))
    return p,raw,prior,view


class FunctionContentRecoveryTests(unittest.TestCase):
    def test_renamed_source_exact_hash_bom_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'Different.sol').write_bytes(b'\xef\xbb\xbf'+TEXT.encode());(raw/'Alias.sol').write_text(TEXT)
            checked,matches,texts,summary=extension.scan(raw,{extension.recovery.text_hash(TEXT)},max_bytes=10000,max_files=10)
            self.assertEqual(len(matches[extension.recovery.text_hash(TEXT)]),2);self.assertEqual(len(texts),1)
            bom=next(r for r in checked if r['local_relative_path']=='Different.sol')
            self.assertNotEqual(bom['local_bytes_sha256'],bom['decoded_text_sha256']);self.assertTrue(summary['scope_exhaustive'])

    def test_near_match_and_invalid_bytes_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);(raw/'C.sol').write_text(TEXT+'\n');(raw/'invalid.sol').write_bytes(b'\xff')
            _,matches,texts,summary=extension.scan(raw,{extension.recovery.text_hash(TEXT)},max_bytes=10000,max_files=10)
            self.assertEqual(matches,{});self.assertEqual(texts,{});self.assertEqual(summary['skipped'],{'invalid_utf8':1})

    def test_byte_and_file_budget_are_deterministic_and_not_exhaustive(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)
            for n in ['a','b']:(raw/(n+'.sol')).write_text(TEXT)
            one=extension.scan(raw,set(),max_bytes=len(TEXT.encode()),max_files=10)
            self.assertEqual(one[3]['read_bytes'],len(TEXT.encode()));self.assertEqual(one[3]['stop_reason'],'total_byte_budget');self.assertFalse(one[3]['scope_exhaustive'])
            self.assertEqual(one,extension.scan(raw,set(),max_bytes=len(TEXT.encode()),max_files=10))
            self.assertEqual(extension.scan(raw,set(),max_bytes=10000,max_files=1)[3]['stop_reason'],'read_file_budget')
            for budget in [0,True,2*1024**3+1]:
                with self.assertRaises(ValueError):extension.scan(raw,set(),max_bytes=budget,max_files=10)

    def test_round_robin_order_does_not_exhaust_first_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)
            for name in ['A/1.sol','A/2.sol','B/1.sol','paper_benchmarks/X/1.sol','paper_benchmarks/Y/1.sol']:
                path=raw/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(TEXT)
            ordered,_=extension.ordered_paths(raw)
            self.assertEqual([str(p.relative_to(raw.resolve())) for p in ordered],['A/1.sol','B/1.sol','paper_benchmarks/X/1.sol','paper_benchmarks/Y/1.sol','A/2.sol'])

    def test_external_symlink_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();outside=root/'other.sol';outside.write_text(TEXT);(raw/'link.sol').symlink_to(outside)
            self.assertEqual(extension.scan(raw,{extension.recovery.text_hash(TEXT)},max_bytes=10000,max_files=10)[1],{})
            (raw/'real.sol').write_text(TEXT);checked=extension.scan(raw,set(),max_bytes=10000,max_files=10)[0];(raw/'real.sol').write_text(TEXT+'\n')
            with self.assertRaises(ValueError):extension.verify_checked(raw,checked)

    def test_extension_recovers_context_not_labels_and_quarantines_current_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,raw,prior,view=fixture(root);(raw/'Renamed.sol').write_text(TEXT)
            before={str(f):extension.sha256(f) for f in p.rglob('*') if f.is_file()}
            summary=extension.prepare(p,raw,prior,view,root/'out',max_bytes=10000,max_files=10)
            self.assertEqual(summary['new_covered_candidates'],1);self.assertEqual(summary['new_fragment_localization_counts'],{'exact_unique_fragment':1})
            self.assertEqual(summary['training_eligible_records'],0)
            link=list(extension.jsonl_rows(root/'out/candidate_context_links.jsonl'))[0]
            self.assertEqual(link['evaluation_status'],'quarantine_existing_corpus_overlap');self.assertTrue(all(v is None for v in link['pattern_labels'].values()))
            group=list(extension.jsonl_rows(root/'out/groups/candidate_groups.jsonl'))[0];self.assertTrue(group['direct_verified_context_corpus_overlap'])
            for path,digest in before.items():self.assertEqual(extension.sha256(Path(path)),digest)
            extension.prepare(p,raw,prior,view,root/'replay',max_bytes=10000,max_files=10)
            for path in (root/'out').rglob('*'):
                if path.is_file():self.assertEqual(path.read_bytes(),(root/'replay'/path.relative_to(root/'out')).read_bytes())
            with self.assertRaises(FileExistsError):extension.prepare(p,raw,prior,view,root/'out',max_bytes=10000,max_files=10)
            with self.assertRaises(ValueError):extension.prepare(p,raw,prior,view,raw/'output',max_bytes=10000,max_files=10)

    def test_existing_crlf_context_is_copied_without_newline_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);text=TEXT.replace('\n','\r\n');p,raw,prior,view=fixture(root,text=text,known=True)
            result=extension.prepare(p,raw,prior,view,root/'out',max_bytes=10000,max_files=10)
            self.assertEqual(result['prior_covered_candidates'],1);self.assertEqual(result['new_covered_candidates'],0)
            path=root/'out/contexts'/(extension.recovery.text_hash(text)+'.sol')
            self.assertEqual(path.read_bytes(),text.encode('utf-8'))

    def test_tampered_parent_and_human_state_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,raw,prior,view=fixture(root)
            rows=list(extension.jsonl_rows(view/'function_model_views.jsonl'));rows[0]['human_review_status']='reviewed'
            extension.recovery.write_jsonl(view/'function_model_views.jsonl',rows);m=json.loads((view/'manifest.json').read_text());m['artifact_sha256']['function_model_views.jsonl']=extension.sha256(view/'function_model_views.jsonl');(view/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'explicit migration'):extension.prepare(p,raw,prior,view,root/'out',max_bytes=10000,max_files=10)
            (view/'function_model_views.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):extension.prepare(p,raw,prior,view,root/'out',max_bytes=10000,max_files=10)

if __name__=='__main__':unittest.main()
