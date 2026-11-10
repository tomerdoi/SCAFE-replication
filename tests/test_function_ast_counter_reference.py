"""Software-only AST schema fixtures, not research model training examples."""
import unittest
from src.evaluation.ast_counter_reference import source_construct_counts


def unit(*nodes): return {'nodeType': 'SourceUnit', 'nodes': list(nodes)}


class ASTCounterReferenceTest(unittest.TestCase):
    def test_only_contract_kind_contract_enters_primary_count(self):
        ast = unit(*[{'nodeType': 'ContractDefinition', 'contractKind': kind, 'nodes': []}
                     for kind in ['contract', 'library', 'interface']])
        out = source_construct_counts(ast)
        self.assertEqual(out['primary_reference']['n_contracts'], 1)
        self.assertEqual(out['declaration_counts']['libraries'], 1)
        self.assertEqual(out['declaration_counts']['interfaces'], 1)

    def test_function_unit_separates_abstract_types_and_special_forms(self):
        ast = unit({'nodeType': 'FunctionDefinition', 'kind': 'function', 'name': 'implemented', 'body': {}},
                   {'nodeType': 'FunctionDefinition', 'kind': 'function', 'name': 'abstract', 'body': None},
                   {'nodeType': 'FunctionTypeName'},
                   *[{'nodeType': 'FunctionDefinition', 'kind': kind, 'name': ''}
                     for kind in ['constructor', 'fallback', 'receive']])
        out = source_construct_counts(ast)
        self.assertEqual(out['primary_reference']['n_functions'], 3)
        self.assertEqual(out['declaration_counts']['named_functions'], 2)
        self.assertEqual(out['declaration_counts']['constructors'], 1)
        self.assertEqual(out['declaration_counts']['fallback'], 1)
        self.assertEqual(out['declaration_counts']['receive'], 1)
        self.assertFalse(out['security_labels_or_model_eligibility'])

    def test_declarations_count_but_strings_and_calls_do_not(self):
        out = source_construct_counts(unit({'nodeType': 'EventDefinition'}, {'nodeType': 'ModifierDefinition'},
            {'nodeType': 'Literal', 'value': 'event X; modifier Y; contract Z'},
            {'nodeType': 'FunctionCall', 'expression': {'nodeType': 'Identifier', 'name': 'function'}}))
        self.assertEqual(out['primary_reference'], {'n_contracts': 0, 'n_functions': 0, 'n_events': 1, 'n_modifiers': 1})

    def test_wrong_or_legacy_AST_shape_rejects(self):
        for value in [None, [], {'name': 'SourceUnit', 'children': []}, {'nodeType': 'SourceUnit', 'nodes': {}}]:
            with self.assertRaises(ValueError): source_construct_counts(value)

    def test_nested_source_units_do_not_silently_merge(self):
        with self.assertRaises(ValueError): source_construct_counts(unit(unit()))

    def test_unknown_declaration_kinds_fail_closed(self):
        for node in [{'nodeType': 'ContractDefinition', 'contractKind': 'unknown'},
                     {'nodeType': 'FunctionDefinition', 'name': 'f'},
                     {'nodeType': 'FunctionDefinition', 'kind': 'unknown', 'name': 'f'}]:
            with self.assertRaises(ValueError): source_construct_counts(unit(node))

    def test_structure_budget_and_cycles_are_bounded(self):
        ast = unit(); ast['cycle'] = ast
        with self.assertRaises(ValueError): source_construct_counts(ast, max_structures=5)
        for limit in [True, 0, 100001]:
            with self.assertRaises(ValueError): source_construct_counts(unit(), max_structures=limit)

    def test_unnamed_ordinary_function_not_guessed_as_constructor(self):
        with self.assertRaises(ValueError):
            source_construct_counts(unit({'nodeType': 'FunctionDefinition', 'kind': 'function', 'name': ''}))


if __name__ == '__main__': unittest.main()
