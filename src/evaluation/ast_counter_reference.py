"""Bounded compiler-AST construct counts; no security labels or model admission."""
from __future__ import annotations

VERSION = 'solc-modern-source-construct-counts-v1'


def source_construct_counts(ast: dict, *, max_structures: int = 100000) -> dict:
    """Count constructs in one compact SourceUnit AST, not imported source units.

    n_functions is a keyword-construct reference: named function declarations
    (including abstract ones) plus FunctionTypeName expressions. It is NOT a
    count of executable functions. Constructor/fallback/receive are separate.
    """
    if not isinstance(ast, dict) or ast.get('nodeType') != 'SourceUnit' or not isinstance(ast.get('nodes'), list):
        raise ValueError('Expected a compact SourceUnit AST with nodes')
    if type(max_structures) is not int or not 1 <= max_structures <= 100000:
        raise ValueError('Expected a bounded AST traversal budget')
    declarations = {'contracts': 0, 'libraries': 0, 'interfaces': 0,
                    'named_functions': 0, 'function_types': 0, 'events': 0,
                    'modifiers': 0, 'constructors': 0, 'fallback': 0, 'receive': 0}
    stack = [ast]; visited = 0
    while stack:
        node = stack.pop()
        if not isinstance(node, (dict, list)): continue
        visited += 1
        if visited > max_structures: raise ValueError('AST traversal budget exceeded')
        if isinstance(node, list):
            stack.extend(node)
            continue
        kind = node.get('nodeType')
        if kind == 'SourceUnit' and node is not ast:
            raise ValueError('Nested source units require explicit separate selection')
        if kind == 'ContractDefinition':
            category = {'contract': 'contracts', 'library': 'libraries', 'interface': 'interfaces'}.get(node.get('contractKind'))
            if category is None: raise ValueError('Unsupported contract kind')
            declarations[category] += 1
        elif kind == 'FunctionDefinition':
            category = {'function': 'named_functions', 'constructor': 'constructors',
                        'fallback': 'fallback', 'receive': 'receive'}.get(node.get('kind'))
            if category is None: raise ValueError('Unsupported/missing function kind')
            if category == 'named_functions' and not node.get('name'):
                raise ValueError('Unnamed ordinary function needs version-specific interpretation')
            declarations[category] += 1
        elif kind == 'FunctionTypeName': declarations['function_types'] += 1
        elif kind == 'EventDefinition': declarations['events'] += 1
        elif kind == 'ModifierDefinition': declarations['modifiers'] += 1
        stack.extend(value for value in node.values() if isinstance(value, (dict, list)))
    return {
        'version': VERSION,
        'primary_reference': {'n_contracts': declarations['contracts'],
                              'n_functions': declarations['named_functions'] + declarations['function_types'],
                              'n_events': declarations['events'], 'n_modifiers': declarations['modifiers']},
        'declaration_counts': declarations,
        'structures_visited': visited,
        'function_unit': 'named_declarations_including_abstract_plus_function_type_expressions;not_executable_function_count',
        'security_labels_or_model_eligibility': False,
    }
