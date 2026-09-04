"""Isolated compilation-free relational features for a reentrancy PoC.

The production SCAFE extractor is intentionally not imported or modified here.
This module builds a lightweight function/call/effect summary directly from
Solidity source and local relative imports.  It is an approximation: it does
not resolve overloads, dynamic dispatch, or compiler-level storage aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


RELATIONAL_FEATURE_COLUMNS = (
    "rel_n_external_calls",
    "rel_n_value_calls",
    "rel_n_callback_calls",
    "rel_n_untrusted_target_calls",
    "rel_n_storage_target_calls",
    "rel_n_untrusted_value_calls",
    "rel_n_untrusted_callback_calls",
    "rel_n_guarded_untrusted_calls",
    "rel_n_unguarded_untrusted_calls",
    "rel_n_direct_call_then_state_write",
    "rel_n_direct_state_write_then_call",
    "rel_n_transitive_call_then_state_write",
    "rel_n_transitive_state_write_then_call",
    "rel_n_untrusted_call_then_state_write",
    "rel_n_unguarded_untrusted_call_then_state_write",
    "rel_n_unguarded_untrusted_value_then_state_write",
    "rel_n_transitive_write_then_untrusted_callback",
    "rel_n_unguarded_transitive_write_then_callback",
    "rel_n_sender_dependent_writes_after_call",
    "rel_n_public_external_entrypoints",
    "rel_n_reentrancy_guarded_entrypoints",
    "rel_n_external_calls_in_loops",
    "rel_n_multi_call_entrypoints",
    "rel_max_write_reachability_depth",
    "rel_untrusted_call_ratio",
    "rel_unguarded_untrusted_call_ratio",
    "rel_call_then_write_ratio",
    "rel_write_then_callback_ratio",
)

RELATIONAL_METADATA_COLUMNS = (
    "rel_meta_source_files",
    "rel_meta_unresolved_imports",
    "rel_meta_functions",
    "rel_meta_state_variables",
)

_FUNCTION_RE = re.compile(
    r"\bfunction\s*([A-Za-z_]\w*)?\s*\(([^)]*)\)([^;{]*)\{",
    re.DOTALL,
)
_SPECIAL_FUNCTION_RE = re.compile(
    r"\b(fallback|receive)\s*\(\s*\)([^;{]*)\{",
    re.DOTALL,
)
_IMPORT_RE = re.compile(
    r"""\bimport\s+(?:(?:\{[^;]*\}\s+from\s+)|(?:\*\s+as\s+\w+\s+from\s+))?["']([^"']+)["']\s*;""",
    re.DOTALL,
)
_STATE_DECL_RE = re.compile(
    r"\b(?:"
    r"mapping\s*\([^;{}]+\)"
    r"|(?:u?int\d*|address(?:\s+payable)?|bool|string|bytes\d*)"
    r"|[A-Z][A-Za-z0-9_]*(?:\s*\[\s*\])?"
    r")\s+"
    r"(?:(?:public|private|internal|constant|immutable|override)\s+)*"
    r"([A-Za-z_]\w*)\s*(?:=[^;]*)?;",
    re.DOTALL,
)
_ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*(?:\[[^;\n]*?\]\s*)?"
    r"(?:=|\+=|-=|\*=|/=|%=|\+\+|--)"
)
_DELETE_RE = re.compile(r"\bdelete\s+([A-Za-z_]\w*)")
_LOW_LEVEL_CALL_RE = re.compile(
    r"(?P<target>msg\.sender|tx\.origin|[A-Za-z_]\w*|"
    r"(?:address|payable)\s*\([^)]*\))"
    r"\s*\.\s*(?P<method>call|send|transfer|delegatecall|staticcall)\b"
)
_TYPED_CALL_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_]*\s*\(\s*(?P<target>[A-Za-z_]\w*|msg\.sender)"
    r"\s*\)\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*\("
)
_OBJECT_CALL_RE = re.compile(
    r"(?P<target>msg\.sender|[A-Za-z_]\w*)\s*\.\s*"
    r"(?P<method>transferFrom|safeTransferFrom|safeTransfer|transfer|sendValue|"
    r"tokenFallback|tokensReceived|onERC721Received|onERC1155Received|"
    r"callback|notify|execute|withdraw)\s*\("
)
_GENERIC_OBJECT_CALL_RE = re.compile(
    r"(?P<target>msg\.sender|tx\.origin|[A-Za-z_]\w*)\s*\.\s*"
    r"(?P<method>[A-Za-z_]\w*)\s*\("
)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_PARAMETER_RE = re.compile(
    r"(?:address(?:\s+payable)?|u?int\d*|bool|string|bytes\d*|"
    r"[A-Z][A-Za-z0-9_]*(?:\s*\[\s*\])?)\s+"
    r"(?:(?:memory|calldata|storage|payable)\s+)*([A-Za-z_]\w*)"
)
_CALLBACK_RE = re.compile(
    r"(?:callback|fallback|hook|notify|tokensReceived|onERC\d+Received)",
    re.IGNORECASE,
)
_REENTRANCY_GUARD_RE = re.compile(
    r"\b(?:nonReentrant|noReentrancy|reentrancyGuard|mutex|locked)\b",
    re.IGNORECASE,
)
_ACCESS_GUARD_RE = re.compile(
    r"\b(?:onlyOwner|onlyRole|whenNotPaused|only[A-Z]\w*)\b"
)
_LOOP_RE = re.compile(r"\b(?:for|while)\s*\(")


@dataclass(frozen=True)
class SourceBundle:
    source: str
    files: int
    unresolved_imports: int


@dataclass(frozen=True)
class ExternalCall:
    position: int
    target: str
    method: str
    untrusted: bool
    storage_target: bool
    callback: bool
    sends_value: bool
    in_loop: bool


@dataclass
class FunctionSummary:
    name: str
    parameters: frozenset[str]
    public_entrypoint: bool
    reentrancy_guarded: bool
    access_restricted: bool
    body: str
    state_writes: tuple[tuple[int, str, bool], ...]
    internal_calls: tuple[tuple[int, str], ...]
    external_calls: tuple[ExternalCall, ...]


def _mask_comments_and_strings(source: str) -> str:
    """Replace comments/string contents with spaces while preserving offsets."""

    chars = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(chars):
        current = chars[index]
        nxt = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if current == "/" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and nxt == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current in {'"', "'"}:
                quote = current
                chars[index] = " "
                index += 1
                state = "string"
                continue
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            if current == "*" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                chars[index] = " "
        elif state == "string":
            if current == "\\" and nxt:
                chars[index] = chars[index + 1] = " "
                index += 2
                continue
            if current == quote:
                chars[index] = " "
                state = "code"
            elif current != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def _remove_comments(source: str) -> str:
    """Remove comments while retaining quoted import paths."""

    chars = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(chars):
        current = chars[index]
        nxt = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if current == "/" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and nxt == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current in {'"', "'"}:
                quote = current
                state = "string"
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            if current == "*" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                chars[index] = " "
        elif state == "string":
            if current == "\\" and nxt:
                index += 2
                continue
            if current == quote:
                state = "code"
        index += 1
    return "".join(chars)


def _read_text(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


@lru_cache(maxsize=100_000)
def read_source_bundle(
    path_value: str,
    *,
    max_files: int = 64,
    max_bytes_per_file: int = 2_000_000,
) -> SourceBundle:
    """Read a source file/directory and recursively include local imports."""

    if not path_value:
        return SourceBundle("", 0, 0)
    path = Path(path_value).expanduser()
    if not path.exists():
        return SourceBundle("", 0, 0)

    queue: list[Path]
    if path.is_file():
        queue = [path.resolve()]
    elif path.is_dir():
        queue = [item.resolve() for item in sorted(path.rglob("*.sol"))[:max_files]]
    else:
        return SourceBundle("", 0, 0)

    visited: set[Path] = set()
    sources: list[str] = []
    unresolved = 0
    while queue and len(visited) < max_files:
        current = queue.pop(0)
        if current in visited or current.suffix.lower() != ".sol":
            continue
        text = _read_text(current, max_bytes_per_file)
        visited.add(current)
        if not text:
            continue
        sources.append(f"\n// ===== FILE: {current.name} =====\n{text}")
        without_comments = _remove_comments(text)
        for import_path in _IMPORT_RE.findall(without_comments):
            if not import_path.startswith("."):
                unresolved += 1
                continue
            candidate = (current.parent / import_path).resolve()
            if candidate.is_file() and candidate.suffix.lower() == ".sol":
                if candidate not in visited:
                    queue.append(candidate)
            else:
                unresolved += 1
    return SourceBundle("\n".join(sources), len(visited), unresolved)


def _balanced_body(masked: str, opening_brace: int) -> tuple[int, str] | None:
    depth = 1
    cursor = opening_brace + 1
    while cursor < len(masked) and depth:
        if masked[cursor] == "{":
            depth += 1
        elif masked[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    return cursor, masked[opening_brace + 1 : cursor - 1]


def _extract_function_parts(
    masked: str,
) -> list[tuple[str, str, str, str, int, int]]:
    parts: list[tuple[str, str, str, str, int, int]] = []
    for match in _FUNCTION_RE.finditer(masked):
        opening = match.end() - 1
        balanced = _balanced_body(masked, opening)
        if balanced is None:
            continue
        end, body = balanced
        parts.append(
            (
                match.group(1) or "__legacy_fallback__",
                match.group(2),
                match.group(3),
                body,
                match.start(),
                end,
            )
        )
    for match in _SPECIAL_FUNCTION_RE.finditer(masked):
        opening = match.end() - 1
        balanced = _balanced_body(masked, opening)
        if balanced is None:
            continue
        end, body = balanced
        parts.append(
            (
                f"__{match.group(1)}__",
                "",
                match.group(2),
                body,
                match.start(),
                end,
            )
        )
    return sorted(parts, key=lambda item: item[-2])


def _state_variables(
    masked: str,
    functions: Sequence[tuple[str, str, str, str, int, int]],
) -> set[str]:
    outside = list(masked)
    for *_, start, end in functions:
        outside[start:end] = " " * (end - start)
    return set(_STATE_DECL_RE.findall("".join(outside)))


def _state_writes(
    body: str, state_variables: set[str]
) -> tuple[tuple[int, str, bool], ...]:
    writes: list[tuple[int, str, bool]] = []
    for match in _ASSIGNMENT_RE.finditer(body):
        variable = match.group(1)
        if variable in state_variables:
            statement_end = body.find(";", match.start())
            statement = body[
                match.start() : statement_end if statement_end >= 0 else match.end()
            ]
            sender_dependent = bool(
                re.search(r"\b(?:msg\.sender|tx\.origin)\b", statement)
                or re.search(rf"\b{re.escape(variable)}\s*\[", statement)
            )
            writes.append((match.start(), variable, sender_dependent))
    for match in _DELETE_RE.finditer(body):
        if match.group(1) in state_variables:
            writes.append((match.start(), match.group(1), True))
    return tuple(sorted(set(writes)))


def _target_name(target: str) -> str:
    identifiers = re.findall(r"[A-Za-z_]\w*", target)
    return identifiers[-1] if identifiers else target


def _inside_loop(body: str, position: int) -> bool:
    """Conservatively detect whether a call is inside an open loop block."""

    prefix = body[:position]
    loop_matches = list(_LOOP_RE.finditer(prefix))
    if not loop_matches:
        return False
    latest = loop_matches[-1].start()
    opening = prefix.find("{", latest)
    if opening < 0:
        return True
    return prefix[opening:].count("{") > prefix[opening:].count("}")


def _external_calls(
    body: str, parameters: set[str], state_variables: set[str]
) -> tuple[ExternalCall, ...]:
    calls: dict[tuple[int, str, str], ExternalCall] = {}
    matches: Iterable[re.Match[str]] = (
        list(_LOW_LEVEL_CALL_RE.finditer(body))
        + list(_TYPED_CALL_RE.finditer(body))
        + list(_OBJECT_CALL_RE.finditer(body))
        + list(_GENERIC_OBJECT_CALL_RE.finditer(body))
    )
    for match in matches:
        target = match.group("target").replace(" ", "")
        method = match.group("method")
        target_name = _target_name(target)
        untrusted = (
            "msg.sender" in target
            or "tx.origin" in target
            or target_name in parameters
        )
        storage_target = target_name in state_variables
        if (
            match.re is _GENERIC_OBJECT_CALL_RE
            and not untrusted
            and not storage_target
        ):
            continue
        callback = bool(_CALLBACK_RE.search(method))
        suffix = body[match.start() : min(len(body), match.end() + 80)]
        sends_value = method in {"send", "transfer"} or bool(
            re.search(r"\bvalue\s*[:(]", suffix)
        )
        call = ExternalCall(
            position=match.start(),
            target=target,
            method=method,
            untrusted=untrusted,
            storage_target=storage_target,
            callback=callback,
            sends_value=sends_value,
            in_loop=_inside_loop(body, match.start()),
        )
        calls[(call.position, call.target, call.method)] = call
    return tuple(sorted(calls.values(), key=lambda call: call.position))


def _function_summaries(masked: str) -> tuple[list[FunctionSummary], set[str]]:
    parts = _extract_function_parts(masked)
    state_variables = _state_variables(masked, parts)
    names = {name for name, *_ in parts}
    summaries: list[FunctionSummary] = []
    for name, parameters_text, suffix, body, *_ in parts:
        parameters = set(_PARAMETER_RE.findall(parameters_text))
        internal_calls = tuple(
            (match.start(), match.group(1))
            for match in _CALL_RE.finditer(body)
            if match.group(1) in names and match.group(1) != name
        )
        summaries.append(
            FunctionSummary(
                name=name,
                parameters=frozenset(parameters),
                public_entrypoint=(
                    name.startswith("__")
                    or bool(re.search(r"\b(?:public|external)\b", suffix))
                ),
                reentrancy_guarded=bool(_REENTRANCY_GUARD_RE.search(suffix)),
                access_restricted=bool(_ACCESS_GUARD_RE.search(suffix)),
                body=body,
                state_writes=_state_writes(body, state_variables),
                internal_calls=internal_calls,
                external_calls=_external_calls(
                    body, parameters, state_variables
                ),
            )
        )
    return summaries, state_variables


def _write_reachability(
    functions: Sequence[FunctionSummary],
) -> tuple[dict[str, bool], dict[str, int]]:
    """Return whether/depth at which each function can write state."""

    graph: dict[str, set[str]] = {}
    for function in functions:
        graph.setdefault(function.name, set()).update(
            callee for _, callee in function.internal_calls
        )
    direct: dict[str, bool] = {}
    for function in functions:
        direct[function.name] = (
            direct.get(function.name, False) or bool(function.state_writes)
        )
    reachable = dict(direct)
    depth = {name: (0 if value else 10**6) for name, value in direct.items()}
    for _ in range(max(1, len(functions))):
        changed = False
        for name, callees in graph.items():
            candidate_depths = [
                depth[callee] + 1
                for callee in callees
                if callee in depth and depth[callee] < 10**6
            ]
            candidate = min(candidate_depths, default=10**6)
            if candidate < depth[name]:
                depth[name] = candidate
                reachable[name] = True
                changed = True
        if not changed:
            break
    return reachable, depth


def extract_relational_reentrancy_features(
    source: str,
    *,
    source_files: int = 1,
    unresolved_imports: int = 0,
) -> dict[str, float]:
    """Extract contract-level relational summaries from a Solidity bundle."""

    values = {column: 0.0 for column in RELATIONAL_FEATURE_COLUMNS}
    metadata = {column: 0.0 for column in RELATIONAL_METADATA_COLUMNS}
    if not source:
        return {**values, **metadata}

    masked = _mask_comments_and_strings(source)
    functions, state_variables = _function_summaries(masked)
    reachable_writes, write_depth = _write_reachability(functions)

    metadata.update(
        {
            "rel_meta_source_files": float(source_files),
            "rel_meta_unresolved_imports": float(unresolved_imports),
            "rel_meta_functions": float(len(functions)),
            "rel_meta_state_variables": float(len(state_variables)),
        }
    )
    maximum_depth = 0
    for function in functions:
        if not function.external_calls:
            continue
        if function.public_entrypoint:
            values["rel_n_public_external_entrypoints"] += 1
            if function.reentrancy_guarded:
                values["rel_n_reentrancy_guarded_entrypoints"] += 1
            if len(function.external_calls) > 1:
                values["rel_n_multi_call_entrypoints"] += 1

        direct_positions = [position for position, _, _ in function.state_writes]
        sender_positions = [
            position
            for position, _, sender_dependent in function.state_writes
            if sender_dependent
        ]
        for call in function.external_calls:
            values["rel_n_external_calls"] += 1
            values["rel_n_value_calls"] += float(call.sends_value)
            values["rel_n_callback_calls"] += float(call.callback)
            values["rel_n_untrusted_target_calls"] += float(call.untrusted)
            values["rel_n_storage_target_calls"] += float(call.storage_target)
            values["rel_n_untrusted_value_calls"] += float(
                call.untrusted and call.sends_value
            )
            values["rel_n_untrusted_callback_calls"] += float(
                call.untrusted and call.callback
            )
            values["rel_n_unguarded_untrusted_calls"] += float(
                call.untrusted
                and function.public_entrypoint
                and not function.reentrancy_guarded
            )
            values["rel_n_guarded_untrusted_calls"] += float(
                call.untrusted
                and function.public_entrypoint
                and function.reentrancy_guarded
            )
            values["rel_n_external_calls_in_loops"] += float(call.in_loop)

            direct_after = any(position > call.position for position in direct_positions)
            direct_before = any(position < call.position for position in direct_positions)
            internal_after = [
                callee
                for position, callee in function.internal_calls
                if position > call.position and reachable_writes.get(callee, False)
            ]
            internal_before = [
                callee
                for position, callee in function.internal_calls
                if position < call.position and reachable_writes.get(callee, False)
            ]
            transitive_after = bool(internal_after)
            transitive_before = bool(internal_before)

            values["rel_n_direct_call_then_state_write"] += float(direct_after)
            values["rel_n_direct_state_write_then_call"] += float(direct_before)
            values["rel_n_transitive_call_then_state_write"] += float(
                transitive_after
            )
            values["rel_n_transitive_state_write_then_call"] += float(
                transitive_before
            )
            values["rel_n_untrusted_call_then_state_write"] += float(
                call.untrusted and (direct_after or transitive_after)
            )
            values["rel_n_unguarded_untrusted_call_then_state_write"] += float(
                call.untrusted
                and function.public_entrypoint
                and not function.reentrancy_guarded
                and (direct_after or transitive_after)
            )
            values[
                "rel_n_unguarded_untrusted_value_then_state_write"
            ] += float(
                call.untrusted
                and call.sends_value
                and function.public_entrypoint
                and not function.reentrancy_guarded
                and (direct_after or transitive_after)
            )
            values["rel_n_transitive_write_then_untrusted_callback"] += float(
                call.untrusted and call.callback and transitive_before
            )
            values["rel_n_unguarded_transitive_write_then_callback"] += float(
                call.untrusted
                and call.callback
                and transitive_before
                and function.public_entrypoint
                and not function.reentrancy_guarded
            )
            values["rel_n_sender_dependent_writes_after_call"] += float(
                any(position > call.position for position in sender_positions)
            )
            for callee in internal_before + internal_after:
                candidate = write_depth.get(callee, 0)
                if candidate < 10**6:
                    maximum_depth = max(maximum_depth, candidate + 1)

    values["rel_max_write_reachability_depth"] = float(maximum_depth)
    external_calls = max(values["rel_n_external_calls"], 1.0)
    callback_calls = max(values["rel_n_callback_calls"], 1.0)
    values["rel_untrusted_call_ratio"] = (
        values["rel_n_untrusted_target_calls"] / external_calls
    )
    values["rel_unguarded_untrusted_call_ratio"] = (
        values["rel_n_unguarded_untrusted_calls"] / external_calls
    )
    values["rel_call_then_write_ratio"] = (
        values["rel_n_direct_call_then_state_write"]
        + values["rel_n_transitive_call_then_state_write"]
    ) / external_calls
    values["rel_write_then_callback_ratio"] = (
        values["rel_n_transitive_write_then_untrusted_callback"]
        / callback_calls
    )
    return {**values, **metadata}


@lru_cache(maxsize=100_000)
def extract_relational_features_for_path(path_value: str) -> Mapping[str, float]:
    bundle = read_source_bundle(path_value)
    return extract_relational_reentrancy_features(
        bundle.source,
        source_files=bundle.files,
        unresolved_imports=bundle.unresolved_imports,
    )


def add_relational_reentrancy_features(
    frame: pd.DataFrame,
    *,
    path_column: str = "local_source_path",
) -> pd.DataFrame:
    """Return a copy augmented only with isolated ``rel_*`` PoC columns."""

    result = frame.copy()
    paths = result.get(path_column, pd.Series("", index=result.index))
    records = [
        extract_relational_features_for_path(str(value or ""))
        for value in paths.fillna("")
    ]
    features = pd.DataFrame.from_records(records, index=result.index)
    for column in RELATIONAL_FEATURE_COLUMNS + RELATIONAL_METADATA_COLUMNS:
        result[column] = pd.to_numeric(
            features.get(column, 0.0), errors="coerce"
        ).fillna(0.0)
        result[column] = np.nan_to_num(
            result[column].to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return result
