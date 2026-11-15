"""Parser/exclusion software fixtures only; no generated model data."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from function_reserved_overlap import named_function_fragments, fingerprints, exclusion_status

class ReservedOverlapTest(unittest.TestCase):
    def test_named_direct_implemented_only(self):
        text='interface I { function a() external; } contract C { constructor() {} function f() public { if(true) {} } fallback() external {} }'
        rows=list(named_function_fragments(text));self.assertEqual([(r['owner'],r['name']) for r in rows],[('C','f')]);self.assertEqual(rows[0]['code'],text[rows[0]['start_character']:rows[0]['end_character']])
    def test_comments_and_literals_are_not_declarations(self):
        text='contract C { /* function fake() {} */ function f() public { string memory s="function fake() { }"; } }'
        self.assertEqual([r['name'] for r in named_function_fragments(text)],['f'])
    def test_overloads_and_multiple_owners_retained(self):
        text='contract C { function f() public {} function f(uint a) public {} } library L { function g() internal {} }'
        self.assertEqual([r['name'] for r in named_function_fragments(text)],['f','f','g'])
    def test_malformed_is_not_clean(self):
        with self.assertRaises(ValueError):list(named_function_fragments('contract C { function f() public {'))
    def test_comment_whitespace_normalization_matches(self):
        a=fingerprints('function f() public { return; }');b=fingerprints('function f() /* note */ public{ return ;}')
        self.assertNotEqual(a['exact'],b['exact']);self.assertEqual(a['tokens'],b['tokens'])
    def test_template_is_heuristic_and_member_selectors_preserved(self):
        a=fingerprints('function a(uint x) public { x.send(1); }');b=fingerprints('function b(uint y) public { y.send(1); }');c=fingerprints('function b(uint y) public { y.transfer(1); }')
        self.assertEqual(a['template'],b['template']);self.assertNotEqual(b['template'],c['template']);self.assertNotEqual(a['tokens'],b['tokens'])
    def test_no_match_never_clears(self):self.assertEqual(exclusion_status([]),'unresolved_no_reserved_match_found')
    def test_template_and_related_groups_not_direct_claims(self):
        self.assertEqual(exclusion_status(['template']),'quarantine_reserved_template_match');self.assertEqual(exclusion_status(['project_family']),'quarantine_reserved_related_group');self.assertEqual(exclusion_status(['tokens','template']),'exclude_direct_reserved_content_match')

if __name__=='__main__':unittest.main()
