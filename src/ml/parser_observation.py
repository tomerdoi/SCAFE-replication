"""Opt-in, process-local AST observations; never a legacy model feature recipe."""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import math

VERSION = "ast-observation-v1"
FIELDS = (
    "ast_n_functions", "ast_inheritance_edges", "ast_n_payable_fns",
    "ast_pure_view_ratio", "ast_max_fn_stmts", "ast_avg_fn_stmts",
    "ast_std_fn_stmts",
)
MAX_BYTES = 2 * 1024**2


def _counts(tree):
    """Retain legacy seven-field definitions, including its depth-10 proxy."""
    if not isinstance(tree, dict) or tree.get("type") != "SourceUnit":
        raise ValueError("Expected SourceUnit")
    contracts = [c for c in tree.get("children", [])
                 if c.get("type") == "ContractDefinition"]
    functions = []; sizes = []; mutability = Counter(); inheritance = 0

    def walk(node, depth=0):
        if depth > 10 or node is None:
            return []
        if isinstance(node, list):
            return [item for child in node for item in walk(child, depth + 1)]
        if isinstance(node, dict):
            return [node] + [item for key in ("statements", "body", "trueBody", "falseBody")
                             if node.get(key) for item in walk(node[key], depth + 1)]
        return []

    for contract in contracts:
        bases = contract.get("baseContracts", [])
        inheritance += len(bases) if isinstance(bases, list) else 0
        for node in contract.get("subNodes", []):
            if node.get("type") == "FunctionDefinition":
                functions.append(node)
                if node.get("stateMutability"):
                    mutability[node["stateMutability"]] += 1
                if node.get("body"):
                    sizes.append(len(walk(node["body"])))
    avg = sum(sizes) / len(sizes) if sizes else 0.0
    values = (len(functions), inheritance, mutability["payable"],
              (mutability["view"] + mutability["pure"]) / max(1, len(functions)),
              max(sizes, default=0), avg,
              math.sqrt(sum((x - avg)**2 for x in sizes) / len(sizes)) if sizes else 0.0)
    return dict(zip(FIELDS, values))


def observe_ast(source: str) -> dict:
    """Record diagnostics without source snippets; mask all fields on uncertainty.

    Use in an isolated, bounded worker: Python/ANTLR parsing is not thread-safe
    or intrinsically time-bounded. No Solidity code is executed. Error-free syntax
    and successful traversal do not establish compiler or security correctness.
    """
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Expected bounded UTF-8 source")
    from antlr4 import CommonTokenStream, InputStream
    from antlr4.error.ErrorListener import ErrorListener
    from solidity_parser import parser as implementation

    diagnostics = Counter()

    class Listener(ErrorListener):
        def __init__(self, stage):
            self.stage = stage

        def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
            # Do not retain messages/tokens: they can contain original source.
            diagnostics[self.stage] += 1

    metadata = {"version": VERSION, "parser_distribution": "solidity-parser",
                "parser_version": version("solidity-parser"),
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "source_bytes": len(source.encode("utf-8")), "exception_class": None,
                "status": "empty_source", "legacy_compatible": False,
                "model_input_approved": False}
    counts = dict.fromkeys(FIELDS)
    if source.strip():
        stage = "parse_exception"
        try:
            lexer = implementation.SolidityLexer(InputStream(source))
            lexer.removeErrorListeners(); lexer.addErrorListener(Listener("lexer"))
            parser = implementation.SolidityParser(CommonTokenStream(lexer))
            parser.removeErrorListeners(); parser.addErrorListener(Listener("parser"))
            tree = parser.sourceUnit()
            stage = "visitor_exception"
            ast = implementation.AstVisitor().visit(tree)
            if diagnostics:
                metadata["status"] = "recovered_with_diagnostics"
            else:
                stage = "count_exception"
                counts = _counts(ast)
                metadata.update(status="syntax_return_without_diagnostics", legacy_compatible=True)
        except Exception as error:
            metadata.update(status=stage, exception_class=type(error).__name__)
    metadata["diagnostic_counts"] = dict(diagnostics)
    observed = metadata["status"] == "syntax_return_without_diagnostics"
    return {"values": counts, "observation_mask": dict.fromkeys(FIELDS, observed),
            "provenance": metadata}
