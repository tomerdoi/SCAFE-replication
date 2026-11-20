"""Simulated software controls only, not compiler-produced research data.

The minimal AST schema follows existing test_modifier_reference_scope.py.
The trusted receipts below are simulated test inputs, never qualified builds.
"""
from dataclasses import replace
import hashlib
import json
import unittest

from src.evidence.compiler_bound_certificate import (
    AnswerCertificate, CompilerPins, Question, question_digest,
    verify_answer_certificate,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def fixture(variant="A"):
    # Exactly the same query source and queried bytes in both build variants.
    query = ('// π: byte offsets, not character offsets\n'
             'import {Base} from "base/Base.sol";\n'
             'contract Vault is Base { function f() external guard {} }\n')
    base = f"// build variant {variant}\ncontract Base {{ modifier guard() {{ _; }} modifier other() {{ _; }} }}\n"
    target_path = f"impl{variant}/Base.sol"
    query_raw, base_raw = query.encode(), base.encode()
    start = query_raw.index(b"guard")
    target_start = base_raw.index(b"modifier guard")
    target_length = base_raw.index(b"}", target_start) + 1 - target_start
    other_start = base_raw.index(b"modifier other")
    other_length = base_raw.index(b"}", other_start) + 1 - other_start

    def node(i, kind, src, **values):
        return {"id": i, "nodeType": kind, "src": src, **values}

    source_input = {
        "language": "Solidity",
        "sources": {"query.sol": {"content": query}, target_path: {"content": base}},
        "settings": {"remappings": [f"base/=impl{variant}/"],
                     "outputSelection": {"*": {"": ["ast"]}}},
    }
    invocation = node(103, "ModifierInvocation", f"{start}:5:0",
                      modifierName=node(104, "IdentifierPath", f"{start}:5:0",
                                        name="guard", referencedDeclaration=202))
    target = node(202, "ModifierDefinition", f"{target_start}:{target_length}:1", name="guard")
    other = node(203, "ModifierDefinition", f"{other_start}:{other_length}:1", name="other")
    function = node(102, "FunctionDefinition", f"{start}:5:0", name="f", modifiers=[invocation])
    vault = node(101, "ContractDefinition", f"0:{len(query_raw)}:0", name="Vault", nodes=[function])
    query_ast = node(100, "SourceUnit", f"0:{len(query_raw)}:0", nodes=[vault])
    authority = node(201, "ContractDefinition", f"0:{len(base_raw)}:1", name="Base", nodes=[target, other])
    base_ast = node(200, "SourceUnit", f"0:{len(base_raw)}:1", nodes=[authority])
    output = {"sources": {
        "query.sol": {"id": 0, "ast": query_ast}, target_path: {"id": 1, "ast": base_ast}}}
    input_raw, output_raw = raw_json(source_input), raw_json(output)
    pins = CompilerPins("0.8.18+commit.87f61d96.test-fixture", "a" * 64,
                        sha(input_raw), sha(output_raw))
    question = Question("query.sol", start, 5, "guard", sha(b"guard"))
    certificate = AnswerCertificate(question_digest(question), pins, target_path,
        target_start, target_length, sha(base_raw[target_start:target_start + target_length]))
    return question, pins, input_raw, output_raw, certificate, (other_start, other_length), source_input, output


def context_switch_fixture(variant="A"):
    """Same query AND declaration-file bytes; another input changes binding."""
    query = ('import {Base} from "base/Base.sol";\n'
             'contract Vault is Base { function f() external guard {} }\n')
    definitions = ('contract ParentOne { modifier guard() { _; } }\n'
                   'contract ParentTwo { modifier guard() { _; } }\n')
    parent = "ParentOne" if variant == "A" else "ParentTwo"
    bridge = f'import {{{parent}}} from "../defs.sol"; contract Base is {parent} {{}}\n'
    bridge_path = f"impl{variant}/Base.sol"
    source_input = {"language": "Solidity", "sources": {
        "query.sol": {"content": query}, "defs.sol": {"content": definitions},
        bridge_path: {"content": bridge}}, "settings": {
            "remappings": [f"base/=impl{variant}/"],
            "outputSelection": {"*": {"": ["ast"]}}}}
    qstart = query.encode().index(b"guard")
    raw_definitions = definitions.encode()
    first = raw_definitions.index(b"modifier guard")
    second = raw_definitions.index(b"modifier guard", first + 1)
    length = raw_definitions.index(b"}", first) + 1 - first

    def node(i, kind, src, **values):
        return {"id": i, "nodeType": kind, "src": src, **values}

    name_node = node(104, "IdentifierPath", f"{qstart}:5:0", name="guard",
                     referencedDeclaration=202 if variant == "A" else 302)
    invocation = node(103, "ModifierInvocation", f"{qstart}:5:0", modifierName=name_node)
    function = node(102, "FunctionDefinition", f"{qstart}:5:0", name="f", modifiers=[invocation])
    vault = node(101, "ContractDefinition", f"0:{len(query.encode())}:0", name="Vault", nodes=[function])
    query_ast = node(100, "SourceUnit", f"0:{len(query.encode())}:0", nodes=[vault])
    first_modifier = node(202, "ModifierDefinition", f"{first}:{length}:1", name="guard")
    second_modifier = node(302, "ModifierDefinition", f"{second}:{length}:1", name="guard")
    first_parent = node(201, "ContractDefinition", f"0:{len(raw_definitions)}:1",
                        name="ParentOne", nodes=[first_modifier])
    second_parent = node(301, "ContractDefinition", f"0:{len(raw_definitions)}:1",
                         name="ParentTwo", nodes=[second_modifier])
    definitions_ast = node(200, "SourceUnit", f"0:{len(raw_definitions)}:1",
                           nodes=[first_parent, second_parent])
    bridge_contract = node(401, "ContractDefinition", f"0:{len(bridge.encode())}:2", name="Base", nodes=[])
    bridge_ast = node(400, "SourceUnit", f"0:{len(bridge.encode())}:2", nodes=[bridge_contract])
    output = {"sources": {
        "query.sol": {"id": 0, "ast": query_ast},
        "defs.sol": {"id": 1, "ast": definitions_ast},
        bridge_path: {"id": 2, "ast": bridge_ast}}}
    input_raw, output_raw = raw_json(source_input), raw_json(output)
    pins = CompilerPins("0.8.18+commit.87f61d96.test-fixture", "a" * 64,
                        sha(input_raw), sha(output_raw))
    question = Question("query.sol", qstart, 5, "guard", sha(b"guard"))
    start = first if variant == "A" else second
    other = second if variant == "A" else first
    certificate = AnswerCertificate(question_digest(question), pins, "defs.sol", start, length,
        sha(raw_definitions[start:start + length]))
    return question, pins, input_raw, output_raw, certificate, (other, length), source_input, output


class CompilerBoundReplay(unittest.TestCase):
    def check(self, item, certificate=None):
        question, pins, input_raw, output_raw, original, *_ = item
        return verify_answer_certificate(question, pins, input_raw, output_raw,
                                         original if certificate is None else certificate)

    def test_valid_simulated_relation(self):
        self.assertEqual(self.check(fixture()).state, "verified_relation")

    def test_unrelated_valid_declaration_span_rejected(self):
        item = fixture()
        original = item[4]
        start, length = item[5]
        content = item[6]["sources"][original.declaration_source_unit]["content"].encode()
        changed = replace(original, declaration_start=start, declaration_length=length,
                          declaration_bytes_sha256=sha(content[start:start + length]))
        self.assertEqual(self.check(item, changed).state, "rejected")
        self.assertEqual(self.check(item, changed).reason, "answer_relation_mismatch")

    def test_identical_queried_bytes_different_build_and_declaration(self):
        first, second = fixture("A"), fixture("B")
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[6]["sources"]["query.sol"], second[6]["sources"]["query.sol"])
        self.assertNotEqual(first[1].standard_input_sha256, second[1].standard_input_sha256)
        self.assertNotEqual(first[4].declaration_source_unit, second[4].declaration_source_unit)
        self.assertEqual(self.check(first).state, "verified_relation")
        self.assertEqual(self.check(second).state, "verified_relation")
        self.assertEqual(self.check(second, first[4]).reason, "compiler_build_mismatch")
        self.assertEqual(self.check(first, second[4]).reason, "compiler_build_mismatch")

    def test_rewrapping_wrong_answer_with_new_build_still_rejected(self):
        first, second = fixture("A"), fixture("B")
        changed = replace(first[4], compiler=second[1])
        self.assertEqual(self.check(second, changed).reason, "answer_relation_mismatch")

    def test_context_only_change_with_identical_query_and_target_file_bytes(self):
        first, second = context_switch_fixture("A"), context_switch_fixture("B")
        self.assertEqual(first[0], second[0])
        for source_unit in ["query.sol", "defs.sol"]:
            self.assertEqual(first[6]["sources"][source_unit], second[6]["sources"][source_unit])
        self.assertEqual(first[4].declaration_bytes_sha256, second[4].declaration_bytes_sha256)
        self.assertNotEqual(first[4].declaration_start, second[4].declaration_start)
        self.assertEqual(self.check(first).state, "verified_relation")
        self.assertEqual(self.check(second).state, "verified_relation")
        self.assertEqual(self.check(second, first[4]).reason, "compiler_build_mismatch")
        self.assertEqual(self.check(first, second[4]).reason, "compiler_build_mismatch")
        self.assertEqual(self.check(second, replace(first[4], compiler=second[1])).reason,
                         "answer_relation_mismatch")

    def test_unrelated_same_named_modifier_span_rejected(self):
        item = context_switch_fixture("A")
        start, length = item[5]
        content = item[6]["sources"]["defs.sol"]["content"].encode()
        changed = replace(item[4], declaration_start=start, declaration_length=length,
                          declaration_bytes_sha256=sha(content[start:start + length]))
        self.assertEqual(self.check(item, changed).reason, "answer_relation_mismatch")

    def test_wrong_question_rejected(self):
        item = fixture()
        changed = replace(item[4], question_sha256="b" * 64)
        self.assertEqual(self.check(item, changed).reason, "question_mismatch")

    def test_binary_pin_or_version_swap_rejected(self):
        item = fixture()
        for compiler in [replace(item[1], binary_sha256="b" * 64),
                         replace(item[1], version="different-build")]:
            self.assertEqual(self.check(item, replace(item[4], compiler=compiler)).reason,
                             "compiler_build_mismatch")

    def test_raw_input_or_output_swap_is_unavailable_reference(self):
        first, second = fixture("A"), fixture("B")
        for input_raw, output_raw in [(second[2], first[3]), (first[2], second[3])]:
            result = verify_answer_certificate(first[0], first[1], input_raw, output_raw, first[4])
            self.assertEqual(result.state, "reference_unavailable")

    def test_changed_query_byte_commitment_not_accepted(self):
        item = fixture()
        question = replace(item[0], queried_bytes_sha256="b" * 64)
        changed = replace(item[4], question_sha256=question_digest(question))
        result = verify_answer_certificate(question, item[1], item[2], item[3], changed)
        self.assertEqual(result.state, "reference_unavailable")

    def test_boolean_span_rejected(self):
        item = fixture()
        self.assertEqual(self.check(item, replace(item[4], declaration_start=True)).state, "rejected")

    def test_answer_fragment_hash_change_rejected(self):
        item = fixture()
        changed = replace(item[4], declaration_bytes_sha256="b" * 64)
        self.assertEqual(self.check(item, changed).reason, "answer_relation_mismatch")

    def test_duplicate_json_key_is_not_a_valid_reference(self):
        item = fixture()
        input_raw = b'{"language":"Solidity","language":"Solidity","sources":{}}'
        pins = replace(item[1], standard_input_sha256=sha(input_raw))
        certificate = replace(item[4], compiler=pins)
        result = verify_answer_certificate(item[0], pins, input_raw, item[3], certificate)
        self.assertEqual(result.state, "reference_unavailable")

    def test_missing_or_error_ast_is_not_negative_truth(self):
        for error in ["missing", "compiler_error", "duplicate_node", "constructor",
                      "missing_reference", "missing_root_id", "unknown_diagnostic"]:
            item = fixture()
            output = item[7]
            if error == "missing":
                del output["sources"][item[4].declaration_source_unit]["ast"]
            elif error == "compiler_error":
                output["errors"] = [{"severity": "error", "message": "software control"}]
            elif error == "unknown_diagnostic":
                output["errors"] = [{"message": "severity not observed"}]
            elif error == "missing_root_id":
                del output["sources"][item[4].declaration_source_unit]["ast"]["id"]
            elif error == "duplicate_node":
                output["sources"][item[4].declaration_source_unit]["ast"]["id"] = 100
            elif error == "missing_reference":
                invocation = output["sources"]["query.sol"]["ast"]["nodes"][0]["nodes"][0]["modifiers"][0]
                invocation["modifierName"]["referencedDeclaration"] = 999999
            else:
                target = output["sources"][item[4].declaration_source_unit]["ast"]["nodes"][0]["nodes"][0]
                target["nodeType"] = "ContractDefinition"
            output_raw = raw_json(output)
            pins = replace(item[1], standard_output_sha256=sha(output_raw))
            certificate = replace(item[4], compiler=pins)
            result = verify_answer_certificate(item[0], pins, item[2], output_raw, certificate)
            self.assertEqual(result.state, "reference_unavailable", error)

    def test_no_answer_not_verified_or_counted_correct(self):
        item = fixture()
        result = verify_answer_certificate(item[0], item[1], item[2], item[3], None)
        self.assertEqual(result.state, "no_answer")

    def test_no_implicit_question_text_or_path_normalization(self):
        item = fixture()
        self.assertNotEqual(question_digest(item[0]),
                            question_digest(replace(item[0], source_unit="alias/query.sol")))
        with self.assertRaises(ValueError):
            question_digest(replace(item[0], modifier_name=" guard "))
        with self.assertRaises(ValueError):
            question_digest(replace(item[0], start=True))


if __name__ == "__main__":
    unittest.main()
