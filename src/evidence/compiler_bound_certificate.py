"""Independent, evaluation-only compiler-relation replay.

This does not call the retrieval resolver, read the filesystem, run a compiler,
or establish source admission/runtime security.
The caller must obtain trusted pins from a separately qualified compiler run,
never from the submitted certificate or from self-asserted artifact hashes.
If used operationally, charge full reference preparation and replay costs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal


MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCES = 1000
MAX_AST_CONTAINERS = 1_000_000
_HEX = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*")


@dataclass(frozen=True)
class Question:
    source_unit: str
    start: int
    length: int
    modifier_name: str
    queried_bytes_sha256: str


@dataclass(frozen=True)
class CompilerPins:
    version: str
    binary_sha256: str
    standard_input_sha256: str
    standard_output_sha256: str


@dataclass(frozen=True)
class AnswerCertificate:
    question_sha256: str
    compiler: CompilerPins
    declaration_source_unit: str
    declaration_start: int
    declaration_length: int
    declaration_bytes_sha256: str


@dataclass(frozen=True)
class ReplayCheck:
    state: Literal["verified_relation", "rejected", "reference_unavailable", "no_answer"]
    reason: str


class _ReferenceUnavailable(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _digest(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _unit(value: object) -> bool:
    # Source-unit identities are compiler VFS keys, not host filesystem paths.
    # Preserve exact spelling; never collapse aliases, case, or '..' segments.
    return isinstance(value, str) and bool(value) and "\x00" not in value


def question_digest(question: Question) -> str:
    """Canonical task-specific key; offsets/lengths are UTF-8 bytes."""
    if not isinstance(question, Question) or not (
        _unit(question.source_unit)
        and _integer(question.start)
        and _integer(question.length, 1)
        and isinstance(question.modifier_name, str)
        and _NAME.fullmatch(question.modifier_name)
        and _digest(question.queried_bytes_sha256)
    ):
        raise ValueError("Invalid normalized modifier-declaration question")
    payload = {
        "task": "solidity.modifier_declaration.v1",
        "source_unit": question.source_unit,
        "start": question.start,
        "length": question.length,
        "modifier_name": question.modifier_name,
        "queried_bytes_sha256": question.queried_bytes_sha256,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8", errors="strict")
    return _sha(raw)


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _ReferenceUnavailable("Duplicate JSON key")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise _ReferenceUnavailable("Non-finite JSON number")


def _decode(raw: bytes, maximum: int) -> dict:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise _ReferenceUnavailable("Invalid reference artifact size/type")
    value = json.loads(raw.decode("utf-8", errors="strict"),
                       object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise _ReferenceUnavailable("Reference must be a JSON object")
    return value


def _span(node: dict, unit_ids: dict[int, str], content: dict[str, bytes],
          owning_unit: str) -> tuple[str, int, int]:
    value = node.get("src")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+:[0-9]+:[0-9]+", value):
        raise _ReferenceUnavailable("Unobserved or generated reference span")
    start, length, source_id = (int(part) for part in value.split(":"))
    unit = unit_ids.get(source_id)
    if unit != owning_unit or length < 1 or start + length > len(content[unit]):
        raise _ReferenceUnavailable("Unbound or out-of-range reference span")
    content[unit][:start].decode("utf-8", errors="strict")
    content[unit][start:start + length].decode("utf-8", errors="strict")
    return unit, start, length


def _reference(question: Question, trusted: CompilerPins,
               input_raw: bytes, output_raw: bytes) -> tuple[str, int, int, str]:
    if not isinstance(input_raw, bytes) or not isinstance(output_raw, bytes):
        raise _ReferenceUnavailable("Reference artifacts must be exact bytes")
    if len(input_raw) > MAX_INPUT_BYTES or len(output_raw) > MAX_OUTPUT_BYTES:
        raise _ReferenceUnavailable("Reference artifacts exceed bounds")
    if (_sha(input_raw) != trusted.standard_input_sha256
            or _sha(output_raw) != trusted.standard_output_sha256):
        raise _ReferenceUnavailable("Reference artifacts differ from trusted build")
    source_input = _decode(input_raw, MAX_INPUT_BYTES)
    output = _decode(output_raw, MAX_OUTPUT_BYTES)
    if source_input.get("language") != "Solidity":
        raise _ReferenceUnavailable("Unsupported reference language")
    errors = output.get("errors", [])
    if not isinstance(errors, list) or any(
        not isinstance(error, dict) or error.get("severity") not in ("warning", "info")
        for error in errors
    ):
        raise _ReferenceUnavailable("Compiler errors or invalid diagnostics")
    inputs, outputs = source_input.get("sources"), output.get("sources")
    if not (isinstance(inputs, dict) and isinstance(outputs, dict)
            and 0 < len(inputs) <= MAX_SOURCES and set(inputs) == set(outputs)):
        raise _ReferenceUnavailable("Incomplete compiler source mapping")
    content = {}
    for unit, item in inputs.items():
        if not (_unit(unit) and isinstance(item, dict)
                and isinstance(item.get("content"), str) and "urls" not in item):
            raise _ReferenceUnavailable("Require pinned inline source content")
        content[unit] = item["content"].encode("utf-8", errors="strict")
    if question.source_unit not in content:
        raise _ReferenceUnavailable("Query source not observed")
    query_raw = content[question.source_unit]
    if (question.start + question.length > len(query_raw)
            or _sha(query_raw[question.start:question.start + question.length])
            != question.queried_bytes_sha256):
        raise _ReferenceUnavailable("Question bytes differ from trusted input")
    unit_ids, nodes, invocations = {}, {}, []
    for unit, item in outputs.items():
        if not isinstance(item, dict) or not _integer(item.get("id")):
            raise _ReferenceUnavailable("Invalid compiler source ID")
        source_id = item["id"]
        if source_id in unit_ids:
            raise _ReferenceUnavailable("Duplicate compiler source ID")
        unit_ids[source_id] = unit
    visited = 0
    for unit, item in outputs.items():
        root = item.get("ast")
        if (not isinstance(root, dict) or root.get("nodeType") != "SourceUnit"
                or not _integer(root.get("id"))):
            raise _ReferenceUnavailable("Full AST not observed")
        pending = [root]
        while pending:
            value = pending.pop()
            visited += 1
            if visited > MAX_AST_CONTAINERS:
                raise _ReferenceUnavailable("AST exceeds bounded checker work")
            if isinstance(value, list):
                pending.extend(child for child in value if isinstance(child, (dict, list)))
            elif isinstance(value, dict):
                if "nodeType" in value and "id" in value:
                    node_id = value["id"]
                    if not _integer(node_id) or node_id in nodes:
                        raise _ReferenceUnavailable("Invalid/duplicate AST node ID")
                    nodes[node_id] = (unit, value)
                if value.get("nodeType") == "ModifierInvocation":
                    invocations.append((unit, value))
                pending.extend(child for child in value.values()
                               if isinstance(child, (dict, list)))
    candidates = []
    for unit, node in invocations:
        if unit != question.source_unit:
            continue
        location = _span(node, unit_ids, content, unit)
        if location[1:] == (question.start, question.length):
            candidates.append(node)
    if len(candidates) != 1:
        raise _ReferenceUnavailable("Question invocation not uniquely observed")
    name = candidates[0].get("modifierName")
    if not isinstance(name, dict):
        raise _ReferenceUnavailable("Missing modifier reference")
    names = [name[key] for key in ("name", "namePath") if key in name]
    if not names or any(value != question.modifier_name for value in names):
        raise _ReferenceUnavailable("Question name differs from compiler node")
    reference_id = name.get("referencedDeclaration")
    if not _integer(reference_id) or reference_id not in nodes:
        raise _ReferenceUnavailable("Declaration reference not observed")
    target_unit, target = nodes[reference_id]
    if (target.get("nodeType") != "ModifierDefinition"
            or target.get("name") != question.modifier_name.rsplit(".", 1)[-1]):
        raise _ReferenceUnavailable("Reference is not the requested modifier declaration")
    unit, start, length = _span(target, unit_ids, content, target_unit)
    return unit, start, length, _sha(content[unit][start:start + length])


def verify_answer_certificate(question: Question, trusted: CompilerPins,
                              input_raw: bytes, output_raw: bytes,
                              certificate: AnswerCertificate | None) -> ReplayCheck:
    """Check an answered relation against an independently pinned full AST.

    This is an evaluation/reference audit, not a source-only proof. A trusted
    qualified run must bind the compiler binary/version and exact input/output.
    Rehashing arbitrary supplied artifacts does not create that trust.
    """
    try:
        expected_question = question_digest(question)
        if not isinstance(trusted, CompilerPins) or not (
            isinstance(trusted.version, str) and trusted.version
            and all(_digest(value) for value in (
                trusted.binary_sha256, trusted.standard_input_sha256,
                trusted.standard_output_sha256))
        ):
            return ReplayCheck("reference_unavailable", "invalid_trusted_build")
        if certificate is None:
            return ReplayCheck("no_answer", "no_answered_certificate")
        if not isinstance(certificate, AnswerCertificate):
            return ReplayCheck("rejected", "invalid_certificate_schema")
        if certificate.compiler != trusted:
            return ReplayCheck("rejected", "compiler_build_mismatch")
        if certificate.question_sha256 != expected_question:
            return ReplayCheck("rejected", "question_mismatch")
        if not (_unit(certificate.declaration_source_unit)
                and _integer(certificate.declaration_start)
                and _integer(certificate.declaration_length, 1)
                and _digest(certificate.declaration_bytes_sha256)):
            return ReplayCheck("rejected", "invalid_answer_schema")
        relation = _reference(question, trusted, input_raw, output_raw)
        proposed = (certificate.declaration_source_unit,
                    certificate.declaration_start, certificate.declaration_length,
                    certificate.declaration_bytes_sha256)
        return (ReplayCheck("verified_relation", "compiler_reference_edge_matches")
                if proposed == relation else ReplayCheck("rejected", "answer_relation_mismatch"))
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        return ReplayCheck("reference_unavailable", "invalid_or_unobserved_reference")
