"""Fail-closed, source-only resolution of Solidity modifier declarations.

This module deliberately does not consume compiler reference IDs or security
labels.  It resolves a narrow structural question from exact source text and
records every file read so the answer can be replayed and costed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Literal, Mapping


Status = Literal["answered", "unknown"]


@dataclass(frozen=True)
class Import:
    path: str
    symbols: tuple[tuple[str, str], ...] = ()
    namespace: str | None = None


@dataclass(frozen=True)
class Contract:
    name: str
    kind: str
    start: int
    body_start: int
    body_end: int
    bases: tuple[str, ...]
    modifiers: tuple["Modifier", ...]


@dataclass(frozen=True)
class Modifier:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedSource:
    imports: tuple[Import, ...]
    contracts: tuple[Contract, ...]


@dataclass(frozen=True)
class Resolution:
    status: Status
    reason: str
    source_path: str | None
    declaration_start: int | None
    declaration_end: int | None
    files_read: tuple[str, ...]
    bytes_read: int
    source_sha256: tuple[tuple[str, str], ...]


_IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
_DECL_RE = re.compile(
    rf"\b(?P<kind>abstract\s+contract|contract|interface|library)\s+"
    rf"(?P<name>{_IDENT})(?P<tail>[^{{;]*)\{{"
)
_MODIFIER_RE = re.compile(rf"\bmodifier\s+(?P<name>{_IDENT})\b")
_IMPORT_RE = re.compile(r"\bimport\b(?P<body>.*?);", re.DOTALL)


def _mask_comments_and_literals(text: str) -> str:
    """Replace comments/string contents with spaces while retaining offsets."""
    out = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == "/" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "line_comment"
                continue
            if c == "/" and nxt == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = "block_comment"
                continue
            if c in ("'", '"'):
                quote = c
                out[i] = " "
                i += 1
                state = "string"
                continue
        elif state == "line_comment":
            if c == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block_comment":
            out[i] = " "
            if c == "*" and nxt == "/":
                out[i + 1] = " "
                i += 2
                state = "code"
                continue
        else:
            out[i] = " "
            if c == "\\" and i + 1 < len(text):
                out[i + 1] = " "
                i += 2
                continue
            if c == quote:
                state = "code"
        i += 1
    return "".join(out)


def _matching_brace(masked: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _top_level(masked: str, start: int, end: int, position: int) -> bool:
    depth = 0
    for char in masked[start:position]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return depth == 0 and start <= position < end


def _parse_bases(tail: str) -> tuple[str, ...]:
    match = re.search(r"\bis\s+(.+)$", tail, re.DOTALL)
    if not match:
        return ()
    bases: list[str] = []
    for item in match.group(1).split(","):
        name = item.strip().split("(", 1)[0].strip()
        if re.fullmatch(rf"{_IDENT}(?:\.{_IDENT})?", name):
            bases.append(name)
    return tuple(bases)


def _parse_import(body: str) -> Import | None:
    path_match = re.search(r"""["']([^"']+)["']""", body)
    if not path_match:
        return None
    path = path_match.group(1)
    namespace = re.search(rf"\*\s+as\s+({_IDENT})", body)
    if namespace:
        return Import(path=path, namespace=namespace.group(1))
    aliases: list[tuple[str, str]] = []
    braces = re.search(r"\{(.*?)\}", body, re.DOTALL)
    if braces:
        for item in braces.group(1).split(","):
            parts = re.fullmatch(
                rf"\s*({_IDENT})(?:\s+as\s+({_IDENT}))?\s*", item
            )
            if not parts:
                return None
            original, local = parts.group(1), parts.group(2) or parts.group(1)
            aliases.append((local, original))
    return Import(path=path, symbols=tuple(aliases))


def parse_source(text: str) -> ParsedSource:
    """Parse only syntax needed for imports, inheritance, and modifiers."""
    masked = _mask_comments_and_literals(text)
    imports = tuple(
        parsed
        for match in _IMPORT_RE.finditer(masked)
        if (parsed := _parse_import(text[match.start("body") : match.end("body")]))
        is not None
    )
    contracts: list[Contract] = []
    for match in _DECL_RE.finditer(masked):
        opening = match.end() - 1
        closing = _matching_brace(masked, opening)
        if closing is None:
            continue
        modifiers: list[Modifier] = []
        for modifier in _MODIFIER_RE.finditer(masked, opening + 1, closing):
            if not _top_level(masked, opening + 1, closing, modifier.start()):
                continue
            body_open = masked.find("{", modifier.end(), closing)
            semicolon = masked.find(";", modifier.end(), closing)
            if semicolon >= 0 and (body_open < 0 or semicolon < body_open):
                end = semicolon + 1
            elif body_open >= 0:
                body_close = _matching_brace(masked, body_open)
                if body_close is None or body_close > closing:
                    continue
                end = body_close + 1
            else:
                continue
            modifiers.append(
                Modifier(
                    name=modifier.group("name"),
                    start=modifier.start(),
                    end=end,
                )
            )
        contracts.append(
            Contract(
                name=match.group("name"),
                kind=" ".join(match.group("kind").split()),
                start=match.start(),
                body_start=opening,
                body_end=closing + 1,
                bases=_parse_bases(match.group("tail")),
                modifiers=tuple(modifiers),
            )
        )
    return ParsedSource(imports=imports, contracts=tuple(contracts))


class SourceRepository:
    """Constrained source reader with remapping and evidence accounting."""

    def __init__(
        self,
        root: Path,
        remappings: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Source root must be a directory")
        self.remappings = tuple(sorted(remappings, key=lambda item: -len(item[0])))
        self._raw: dict[str, bytes] = {}
        self._texts: dict[str, str] = {}
        self._parsed: dict[str, ParsedSource] = {}

    def _safe_path(self, relative: str) -> Path:
        if "\\" in relative or "\x00" in relative:
            raise ValueError("Invalid source path")
        candidate = (self.root / PurePosixPath(relative)).resolve(strict=True)
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise ValueError("Source path escapes root or is not a file")
        return candidate

    def resolve_import(self, importer: str, imported: str) -> str | None:
        candidates: list[PurePosixPath] = []
        if imported.startswith("."):
            candidates.append(PurePosixPath(importer).parent / imported)
        else:
            for prefix, replacement in self.remappings:
                if imported.startswith(prefix):
                    candidates.append(
                        PurePosixPath(replacement + imported[len(prefix) :])
                    )
            candidates.append(PurePosixPath(imported))
        for candidate in candidates:
            normalized = str(PurePosixPath(candidate))
            try:
                path = self._safe_path(normalized)
            except (FileNotFoundError, ValueError):
                continue
            return path.relative_to(self.root).as_posix()
        return None

    def read(self, relative: str) -> str:
        if relative not in self._texts:
            raw = self._safe_path(relative).read_bytes()
            self._raw[relative] = raw
            self._texts[relative] = raw.decode("utf-8", errors="strict")
        return self._texts[relative]

    def byte_to_character(self, relative: str, offset: int) -> int:
        self.read(relative)
        raw = self._raw[relative]
        if offset < 0 or offset > len(raw):
            raise ValueError("Byte offset is outside source")
        return len(raw[:offset].decode("utf-8", errors="strict"))

    def character_to_byte(self, relative: str, offset: int) -> int:
        text = self.read(relative)
        if offset < 0 or offset > len(text):
            raise ValueError("Character offset is outside source")
        return len(text[:offset].encode("utf-8"))

    def parse(self, relative: str) -> ParsedSource:
        if relative not in self._parsed:
            self._parsed[relative] = parse_source(self.read(relative))
        return self._parsed[relative]

    def evidence(self) -> tuple[tuple[str, ...], int, tuple[tuple[str, str], ...]]:
        paths = tuple(sorted(self._texts))
        hashes = tuple(
            (path, hashlib.sha256(self._raw[path]).hexdigest())
            for path in paths
        )
        return paths, sum(len(self._raw[path]) for path in paths), hashes


def _unknown(repository: SourceRepository, reason: str) -> Resolution:
    paths, byte_count, hashes = repository.evidence()
    return Resolution(
        status="unknown",
        reason=reason,
        source_path=None,
        declaration_start=None,
        declaration_end=None,
        files_read=paths,
        bytes_read=byte_count,
        source_sha256=hashes,
    )


def _answered(
    repository: SourceRepository, source_path: str, modifier: Modifier
) -> Resolution:
    paths, byte_count, hashes = repository.evidence()
    return Resolution(
        status="answered",
        reason="unique_declaration_on_resolved_linearization",
        source_path=source_path,
        declaration_start=repository.character_to_byte(source_path, modifier.start),
        declaration_end=repository.character_to_byte(source_path, modifier.end),
        files_read=paths,
        bytes_read=byte_count,
        source_sha256=hashes,
    )


def _enclosing_contract(
    parsed: ParsedSource, invocation_offset: int
) -> Contract | None:
    matches = [
        contract
        for contract in parsed.contracts
        if contract.body_start < invocation_offset < contract.body_end
    ]
    return matches[0] if len(matches) == 1 else None


def _find_exported_contract(
    repository: SourceRepository,
    source_path: str,
    name: str,
    visited: frozenset[tuple[str, str]] = frozenset(),
) -> list[tuple[str, Contract]]:
    key = (source_path, name)
    if key in visited:
        return []
    visited = visited | {key}
    parsed = repository.parse(source_path)
    local = [
        (source_path, contract)
        for contract in parsed.contracts
        if contract.name == name
    ]
    if local:
        return local
    found: list[tuple[str, Contract]] = []
    for directive in parsed.imports:
        target = repository.resolve_import(source_path, directive.path)
        if target is None or directive.namespace is not None:
            continue
        if directive.symbols:
            originals = [
                original
                for local_name, original in directive.symbols
                if local_name == name
            ]
        else:
            originals = [name]
        for original in originals:
            found.extend(
                _find_exported_contract(repository, target, original, visited)
            )
    unique = {
        (path, contract.name, contract.start): (path, contract)
        for path, contract in found
    }
    return list(unique.values())


def _resolve_contract(
    repository: SourceRepository,
    source_path: str,
    reference: str,
) -> tuple[str, Contract] | None:
    """Resolve one contract reference without opening unrelated imports."""
    parsed = repository.parse(source_path)
    if "." not in reference:
        local = [contract for contract in parsed.contracts if contract.name == reference]
        if len(local) == 1:
            return source_path, local[0]
        explicit: list[tuple[str, str]] = []
        plain_imports: list[str] = []
        for directive in parsed.imports:
            target = repository.resolve_import(source_path, directive.path)
            if target is None:
                continue
            for local_name, original_name in directive.symbols:
                if local_name == reference:
                    explicit.append((target, original_name))
            if not directive.symbols and directive.namespace is None:
                plain_imports.append(target)
        if len(explicit) == 1:
            target_path, original = explicit[0]
            matches = _find_exported_contract(repository, target_path, original)
            return matches[0] if len(matches) == 1 else None
        if len(explicit) > 1:
            return None
        found: list[tuple[str, Contract]] = []
        for target_path in plain_imports:
            found.extend(_find_exported_contract(repository, target_path, reference))
        unique = {
            (path, contract.name, contract.start): (path, contract)
            for path, contract in found
        }
        return next(iter(unique.values())) if len(unique) == 1 else None
    namespace, contract_name = reference.split(".", 1)
    targets = [
        repository.resolve_import(source_path, directive.path)
        for directive in parsed.imports
        if directive.namespace == namespace
    ]
    targets = [target for target in targets if target is not None]
    if len(targets) != 1:
        return None
    matches = _find_exported_contract(repository, targets[0], contract_name)
    return matches[0] if len(matches) == 1 else None


def _linearize(
    repository: SourceRepository,
    root_path: str,
    root_contract: Contract,
) -> list[tuple[str, Contract]] | None:
    """Compute Solidity-style C3 order, derived contract first."""
    memo: dict[tuple[str, str], list[tuple[str, Contract]]] = {}
    active: set[tuple[str, str]] = set()

    def visit(path: str, contract: Contract) -> list[tuple[str, Contract]] | None:
        key = (path, contract.name)
        if key in memo:
            return memo[key]
        if key in active:
            return None
        active.add(key)
        bases: list[tuple[str, Contract]] = []
        for base in contract.bases:
            resolved = _resolve_contract(repository, path, base)
            if resolved is None:
                active.remove(key)
                return None
            bases.append(resolved)
        parent_orders: list[list[tuple[str, Contract]]] = []
        for parent_path, parent_contract in bases:
            order = visit(parent_path, parent_contract)
            if order is None:
                active.remove(key)
                return None
            parent_orders.append(order.copy())
        # Solidity lists bases from most base-like to most derived; precedence
        # is therefore right-to-left for C3 lookup.
        sequences = parent_orders + [list(reversed(bases))]
        merged: list[tuple[str, Contract]] = []
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (
                    sequence[0]
                    for sequence in sequences
                    if not any(sequence[0] in other[1:] for other in sequences)
                ),
                None,
            )
            if candidate is None:
                active.remove(key)
                return None
            merged.append(candidate)
            for sequence in sequences:
                if sequence and sequence[0] == candidate:
                    sequence.pop(0)
        result = [(path, contract), *merged]
        memo[key] = result
        active.remove(key)
        return result

    return visit(root_path, root_contract)


def _reachable_bases(
    repository: SourceRepository,
    root_path: str,
    root_contract: Contract,
) -> list[tuple[str, Contract]] | None:
    """Resolve the complete inheritance graph without assuming a C3 order."""
    resolved: list[tuple[str, Contract]] = []
    visited: set[tuple[str, str]] = set()
    active: set[tuple[str, str]] = set()

    def visit(path: str, contract: Contract) -> bool:
        key = (path, contract.name)
        if key in active:
            return False
        if key in visited:
            return True
        active.add(key)
        for base in contract.bases:
            target = _resolve_contract(repository, path, base)
            if target is None or not visit(*target):
                active.remove(key)
                return False
        active.remove(key)
        visited.add(key)
        if key != (root_path, root_contract.name):
            resolved.append((path, contract))
        return True

    return resolved if visit(root_path, root_contract) else None


def _resolve_with_repository(
    repository: SourceRepository,
    source_path: str,
    invocation_offset: int,
    modifier_name: str,
) -> Resolution:
    if not re.fullmatch(_IDENT, modifier_name):
        return _unknown(repository, "invalid_modifier_name")
    try:
        parsed = repository.parse(source_path)
        invocation_character_offset = repository.byte_to_character(
            source_path, invocation_offset
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError):
        return _unknown(repository, "unreadable_query_source")
    contract = _enclosing_contract(parsed, invocation_character_offset)
    if contract is None:
        return _unknown(repository, "query_contract_not_unique")
    own = [modifier for modifier in contract.modifiers if modifier.name == modifier_name]
    if len(own) > 1:
        return _unknown(repository, "ambiguous_modifier_declarations")
    if own:
        return _answered(repository, source_path, own[0])
    bases = _reachable_bases(repository, source_path, contract)
    if bases is None:
        return _unknown(repository, "inheritance_resolution_failed")
    declarations = [
        (path, modifier)
        for path, candidate_contract in bases
        for modifier in candidate_contract.modifiers
        if modifier.name == modifier_name
    ]
    if len(declarations) > 1:
        return _unknown(repository, "ambiguous_inherited_modifier_declarations")
    if declarations:
        return _answered(repository, declarations[0][0], declarations[0][1])
    return _unknown(repository, "modifier_not_found")


def resolve_modifier(
    root: Path,
    source_path: str,
    invocation_offset: int,
    modifier_name: str,
    remappings: Iterable[tuple[str, str]] = (),
) -> Resolution:
    """Resolve demand-first, opening only inheritance-relevant imports."""
    return _resolve_with_repository(
        SourceRepository(root, remappings),
        source_path,
        invocation_offset,
        modifier_name,
    )


def _open_import_closure(repository: SourceRepository, source_path: str) -> bool:
    pending = [source_path]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            parsed = repository.parse(current)
        except (FileNotFoundError, UnicodeDecodeError, ValueError):
            return False
        for directive in parsed.imports:
            target = repository.resolve_import(current, directive.path)
            if target is None:
                return False
            pending.append(target)
    return True


def resolve_modifier_strong_traversal(
    root: Path,
    source_path: str,
    invocation_offset: int,
    modifier_name: str,
    remappings: Iterable[tuple[str, str]] = (),
) -> Resolution:
    """Resolve after recursively reading the complete lexical import closure."""
    repository = SourceRepository(root, remappings)
    if not _open_import_closure(repository, source_path):
        return _unknown(repository, "import_closure_incomplete")
    return _resolve_with_repository(
        repository,
        source_path,
        invocation_offset,
        modifier_name,
    )


def resolve_modifier_file_only(
    root: Path,
    source_path: str,
    invocation_offset: int,
    modifier_name: str,
    *,
    same_contract_only: bool,
) -> Resolution:
    """Implement naive-name and sound same-contract file-only baselines."""
    repository = SourceRepository(root)
    if not re.fullmatch(_IDENT, modifier_name):
        return _unknown(repository, "invalid_modifier_name")
    try:
        parsed = repository.parse(source_path)
        invocation_character_offset = repository.byte_to_character(
            source_path, invocation_offset
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError):
        return _unknown(repository, "unreadable_query_source")
    contract = _enclosing_contract(parsed, invocation_character_offset)
    if contract is None:
        return _unknown(repository, "query_contract_not_unique")
    contracts = (contract,) if same_contract_only else parsed.contracts
    candidates = [
        (candidate, modifier)
        for candidate in contracts
        for modifier in candidate.modifiers
        if modifier.name == modifier_name
    ]
    if len(candidates) != 1:
        return _unknown(repository, "file_only_declaration_not_unique")
    return _answered(repository, source_path, candidates[0][1])


def replay_certificate(
    root: Path,
    resolution: Resolution,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> bool:
    """Validate evidence hashes and the answered declaration span."""
    repository = SourceRepository(root)
    if resolution.bytes_read < 0 or tuple(sorted(resolution.files_read)) != resolution.files_read:
        return False
    observed: dict[str, str] = {}
    observed_bytes = 0
    try:
        for path in resolution.files_read:
            text = repository.read(path)
            raw = text.encode("utf-8")
            observed[path] = hashlib.sha256(raw).hexdigest()
            observed_bytes += len(raw)
    except (FileNotFoundError, UnicodeDecodeError, ValueError):
        return False
    if observed_bytes != resolution.bytes_read:
        return False
    if tuple(sorted(observed.items())) != resolution.source_sha256:
        return False
    if expected_source_hashes is not None and any(
        expected_source_hashes.get(path) != digest for path, digest in observed.items()
    ):
        return False
    if resolution.status == "unknown":
        return (
            resolution.source_path is None
            and resolution.declaration_start is None
            and resolution.declaration_end is None
        )
    if (
        resolution.source_path not in observed
        or resolution.declaration_start is None
        or resolution.declaration_end is None
        or resolution.declaration_start < 0
        or resolution.declaration_end <= resolution.declaration_start
    ):
        return False
    parsed = repository.parse(resolution.source_path)
    return any(
        repository.character_to_byte(resolution.source_path, modifier.start)
        == resolution.declaration_start
        and repository.character_to_byte(resolution.source_path, modifier.end)
        == resolution.declaration_end
        for contract in parsed.contracts
        for modifier in contract.modifiers
    )
