import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_function_context as recovery

SOURCE='CoinFabrik/solidity-rnd'
CODE='contract C {\nfunction f() external { uint a=1; }\n}'
FRAGMENT='function f() external { uint a=1; }'
START=CODE.index('function')


def setup_pilot(tmp,code=CODE):
    pilot=tmp/'pilot';(pilot/'public_output').mkdir(parents=True)
    row={'record_id':'record-one','source':SOURCE,'source_revision':'pinned','source_identity':f'Project/vulnerable/C.sol:{START}',
         'source_path':'Project/vulnerable/C.sol','project_id':'Project','license_status':'unresolved',
         'context_sha256':recovery.text_hash(code),'raw_sha256':recovery.text_hash(FRAGMENT),'characters':len(FRAGMENT),
         'pattern_labels':{'reentrancy':None,'unchecked_external_call':None,'authorization':None},'split':None,
         'training_eligible':False,'human_review_status':'unreviewed','context_status':'source_file_available'}
    recovery.write_jsonl(pilot/'public_output/function_candidates.jsonl',[row])
    (pilot/'source_manifest.json').write_text(json.dumps({'sources':{SOURCE:{'revision':'pinned'}}}))
    names=['public_output/function_candidates.jsonl','source_manifest.json']
    (pilot/'DELIVERY_SHA256.json').write_text(json.dumps({n:recovery.sha256(pilot/n) for n in names}))
    raw=tmp/'raw';raw.mkdir();return pilot,raw,row


class FunctionContextRecoveryTests(unittest.TestCase):
    def test_exact_context_and_fragment_with_no_label_promotion(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);(raw/'C.sol').write_text(CODE)
            result=recovery.recover(pilot,raw,tmp/'out')
            self.assertEqual(result['recovered_context_documents'],1);self.assertEqual(result['candidates_with_verified_context'],1)
            self.assertEqual(result['fragment_localization_counts'],{'exact_unique_fragment':1})
            link=list(recovery.jsonl_rows(tmp/'out/candidate_context_links.jsonl'))[0]
            self.assertEqual(link['pattern_labels'],row['pattern_labels']);self.assertFalse(link['training_eligible']);self.assertIsNone(link['split'])
            self.assertEqual(link['evaluation_status'],'quarantine_existing_corpus_overlap')
            self.assertEqual(link['function_spans'][0]['start_character'],START)
            self.assertEqual(link['semantic_context_sufficiency'],'not_reviewed')

    def test_utf8_bom_decoding_is_recorded_separately_from_raw_identity(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);(raw/'C.sol').write_bytes(b'\xef\xbb\xbf'+CODE.encode())
            recovery.recover(pilot,raw,tmp/'out');entry=list(recovery.jsonl_rows(tmp/'out/context_manifest.jsonl'))[0]
            self.assertNotEqual(entry['local_evidence'][0]['local_bytes_sha256'],entry['context_sha256'])
            self.assertEqual((tmp/'out'/entry['context_file']).read_bytes(),CODE.encode())

    def test_filename_match_or_whitespace_similarity_is_not_context_identity(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);(raw/'C.sol').write_text(CODE+'\n')
            result=recovery.recover(pilot,raw,tmp/'out');self.assertEqual(result['recovered_context_documents'],0)
            self.assertEqual(result['candidates_without_verified_context'],1)

    def test_source_aliases_are_retained_but_stored_once(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp)
            for name in ['one','two']:(raw/name).mkdir();(raw/name/'C.sol').write_text(CODE)
            result=recovery.recover(pilot,raw,tmp/'out');self.assertEqual(result['matching_local_files'],2);self.assertEqual(result['recovered_context_documents'],1)

    def test_external_symlink_is_not_read(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);outside=tmp/'outside.sol';outside.write_text(CODE);(raw/'C.sol').symlink_to(outside)
            result=recovery.recover(pilot,raw,tmp/'out');self.assertEqual(result['candidates_with_verified_context'],0)
            # rg omits symlinks by default; also test containment if a listing includes one.
            listing=SimpleNamespace(returncode=0,stderr=b'',stdout=str(raw/'C.sol').encode()+b'\0')
            with patch('recover_function_context.subprocess.run',return_value=listing):
                checked,inventory=recovery.scan_local(raw,{'C.sol'})
            self.assertEqual(checked,[]);self.assertEqual(inventory['skipped'],{'path_outside_root':1})

    def test_missing_pinned_context_remains_unknown(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);row.pop('context_sha256');recovery.write_jsonl(pilot/'public_output/function_candidates.jsonl',[row])
            m=json.loads((pilot/'DELIVERY_SHA256.json').read_text());m['public_output/function_candidates.jsonl']=recovery.sha256(pilot/'public_output/function_candidates.jsonl');(pilot/'DELIVERY_SHA256.json').write_text(json.dumps(m))
            (raw/'C.sol').write_text(CODE);result=recovery.recover(pilot,raw,tmp/'out');self.assertEqual(result['recovered_context_documents'],0)
            self.assertEqual(list(recovery.jsonl_rows(tmp/'out/candidate_context_links.jsonl'))[0]['context_status'],'no_pinned_context_identity')

    def test_wrong_fragment_hash_does_not_invent_location(self):
        _,status=recovery.exact_function_spans({'source':SOURCE,'source_identity':f'C.sol:{START}','characters':len(FRAGMENT),'raw_sha256':'0'*64},CODE)
        self.assertEqual(status,'fragment_not_localized')

    def test_identical_named_functions_are_explicitly_ambiguous(self):
        code=FRAGMENT+'\n'+FRAGMENT;row={'source':'sujeetc/SCRUBD','function_identifier':'C.f','characters':len(FRAGMENT),'raw_sha256':recovery.text_hash(FRAGMENT)}
        spans,status=recovery.exact_function_spans(row,code);self.assertEqual(len(spans),2);self.assertEqual(status,'ambiguous_identical_fragments')

    def test_bad_or_missing_fragment_identity_not_guessed(self):
        for size in [True,None,0,recovery.MAX_BYTES+1]:
            spans,status=recovery.exact_function_spans({'characters':size,'raw_sha256':'x'},CODE)
            self.assertEqual(spans,[]);self.assertEqual(status,'missing_fragment_identity')

    def test_replay_and_originals_preserved_and_outputs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);source=raw/'C.sol';source.write_text(CODE);before=recovery.verify_delivery(pilot);h=recovery.sha256(source)
            a=recovery.recover(pilot,raw,pilot/'derivatives/v1');b=recovery.recover(pilot,raw,pilot/'derivatives/v2');self.assertEqual(a,b)
            for path in (pilot/'derivatives/v1').rglob('*'):
                if path.is_file():self.assertEqual(path.read_bytes(),(pilot/'derivatives/v2'/path.relative_to(pilot/'derivatives/v1')).read_bytes())
            self.assertEqual(recovery.verify_delivery(pilot),before);self.assertEqual(recovery.sha256(source),h)
            with self.assertRaises(FileExistsError):recovery.recover(pilot,raw,pilot/'derivatives/v1')
            with self.assertRaises(ValueError):recovery.recover(pilot,raw,pilot/'public_output/changed')
            with self.assertRaises(ValueError):recovery.recover(pilot,raw,raw/'derivative')

    def test_tampered_delivery_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);pilot,raw,row=setup_pilot(tmp);(pilot/'source_manifest.json').write_text('{}')
            with self.assertRaises(ValueError):recovery.recover(pilot,raw,tmp/'out')

    def test_oversized_and_invalid_utf8_sources_are_skipped(self):
        with tempfile.TemporaryDirectory() as value:
            tmp=Path(value);(tmp/'A.sol').write_bytes(b'\xff');(tmp/'B.sol').write_bytes(b'x'*(recovery.MAX_BYTES+1))
            rows,inventory=recovery.scan_local(tmp,{'A.sol','B.sol'});self.assertEqual(rows,[]);self.assertEqual(inventory['skipped'],{'invalid_utf8':1,'oversized':1})

if __name__=='__main__':unittest.main()
