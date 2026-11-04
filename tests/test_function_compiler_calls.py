"""Synthetic decoder and failure controls; no vulnerability labels or EVM execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_function_compiler_calls import expanded_source_maps, instruction_prefix, mapped_calls, compile_case


class CompilerCallEvidenceTests(unittest.TestCase):
    def test_push_data_and_unmapped_tail_are_not_opcodes(self):
        instructions,tail=instruction_prefix('60f160faf1faf1',4)
        self.assertEqual(instructions,[(0,0x60),(2,0x60),(4,0xf1),(5,0xfa)])
        self.assertEqual(tail,1)

    def test_compressed_fields_inherit_by_instruction(self):
        self.assertEqual(expanded_source_maps('1:2:0:-;:3;;7:1:1:i'),[(1,2,0),(1,3,0),(1,3,0),(7,1,1)])

    def test_byte_offsets_account_for_utf8_and_exact_call(self):
        source='// π\ncontract C { function f() public { a.f(); } }'
        expr='a.f()';start=len(source[:source.index(expr)].encode());n=len(expr.encode())
        calls,counts=mapped_calls(source,'f1fa',f'{start}:{n}:0:-',0,[expr])
        self.assertEqual(calls[0]['expression_byte_start'],start)
        self.assertEqual(calls[0]['opcode'],'CALL');self.assertEqual(calls[0]['expression_start_line'],2)
        self.assertEqual(counts['unmapped_tail_bytes_not_analyzed'],1)

    def test_ambiguous_missing_or_foreign_source_mapping_rejected(self):
        for code,maps in [('f1fa','0:5:0:-;'),('f1','0:5:1:-'),('f1','0:5:-1:-'),('00','0:5:0:-')]:
            with self.subTest(code=code,maps=maps),self.assertRaises(ValueError):mapped_calls('a.f()',code,maps,0,['a.f()'])

    def test_call_search_is_bounded_to_function_and_not_comments(self):
        source='a.f();a.f()'
        calls,_=mapped_calls(source,'f1f1','0:5:0:-;6:5:0:-',0,['a.f()'],character_bounds=(6,11))
        self.assertEqual(calls[0]['pc'],1)
        with self.assertRaises(ValueError):mapped_calls(source,'f1','0:5:0:-',0,['a.f()'])
        with self.assertRaises(ValueError):mapped_calls('// a.f()','f1','3:5:0:-',0,['a.f()'])

    def test_imports_and_malformed_maps_or_bytecode_refused(self):
        with self.assertRaises(ValueError):mapped_calls('import "file.sol"; a.f()','f1','19:5:0:-',0,['a.f()'])
        for code,count in [('60',1),('00',2),('zz',1)]:
            with self.assertRaises(ValueError):instruction_prefix(code,count)
        for sm in ['','0:1:0:bad','-2:1:0:-','1:2:3:4:5:6']:
            with self.assertRaises(ValueError):expanded_source_maps(sm)

    def test_compiler_errors_are_retained_and_never_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);fake=subprocess.CompletedProcess([],0,'{"errors":[{"severity":"error","message":"not compilable"}]}','diagnostic')
            with patch('probe_function_compiler_calls.subprocess.run',return_value=fake):
                with self.assertRaises(ValueError):compile_case(Path('/unused'),'0.4.24','contract C {}','C','bad',output)
            self.assertIn('not compilable',(output/'bad_output.json').read_text())
            self.assertEqual((output/'bad_stderr.txt').read_text(),'diagnostic')


if __name__=='__main__':unittest.main()
