"""Dataset-specific source sanitization applied before model-visible processing."""

from __future__ import annotations

import re
from pathlib import Path


SOURCE_SANITIZATION_VERSION = "dappscan_swc_annotations_v2"

_DAPPSCAN = "dappscan"
_SWC_COMMENT_START = re.compile(r"(?is)^[ \t\r\n]*SWC[-_ ]?\d+\b")
_SWC_MARKER = re.compile(r"(?i)\bSWC[-_ ]?\d+\b")


def is_dappscan(dataset: object) -> bool:
    """Return whether a dataset name denotes DAppSCAN."""

    return str(dataset or "").strip().casefold() == _DAPPSCAN


def source_dataset_for_path(path_value: object, dataset: object = "") -> str:
    """Resolve dataset metadata, with a conservative DAppSCAN path fallback."""

    explicit = str(dataset or "").strip()
    if explicit:
        return explicit
    try:
        parts = Path(str(path_value or "")).parts
    except (TypeError, ValueError):
        return ""
    return "DAppSCAN" if any(
        part.casefold().startswith("dappscan") for part in parts
    ) else ""


def count_swc_annotation_markers(source: str) -> int:
    """Count explicit SWC-number markers in source text."""

    return len(_SWC_MARKER.findall(source or ""))


def sanitize_source_text(source: str, dataset: object) -> str:
    """Remove DAppSCAN's injected SWC annotation comments.

    DAppSCAN's released Solidity files contain comments such as
    ``// SWC-107-Reentrancy: L105-L127``. The same report entries define the
    binary target label, so exposing these comments to a model is direct target
    leakage. Only standalone SWC annotation comments are removed; raw files are
    never modified and ordinary source comments remain available.
    """

    text = source or ""
    if not is_dappscan(dataset):
        return text

    result: list[str] = []
    index = 0
    line_start = 0
    quote = ""
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if quote:
            result.append(current)
            if current == "\\" and index + 1 < len(text):
                result.append(text[index + 1])
                index += 2
                continue
            if current == quote:
                quote = ""
            if current == "\n":
                line_start = index + 1
            index += 1
            continue
        if current in {'"', "'"}:
            quote = current
            result.append(current)
            index += 1
            continue
        if current == "/" and following == "/":
            end = text.find("\n", index + 2)
            end = len(text) if end < 0 else end
            comment = text[index + 2:end]
            if _SWC_COMMENT_START.match(comment):
                prefix = text[line_start:index]
                if not prefix.strip():
                    # Remove indentation and the physical newline as well. A
                    # blank replacement line would still leak into LOC-derived
                    # features.
                    if prefix:
                        del result[-len(prefix):]
                    index = end + 1 if end < len(text) else end
                    line_start = index
                    continue
            else:
                result.append(text[index:end])
            index = end
            continue
        if current == "/" and following == "*":
            end = text.find("*/", index + 2)
            end = len(text) - 2 if end < 0 else end
            comment = text[index + 2:end]
            if _SWC_COMMENT_START.match(comment):
                after = end + 2
                next_newline = text.find("\n", after)
                line_end = len(text) if next_newline < 0 else next_newline
                prefix = text[line_start:index]
                suffix = text[after:line_end]
                if not prefix.strip() and not suffix.strip():
                    if prefix:
                        del result[-len(prefix):]
                    index = line_end + 1 if line_end < len(text) else line_end
                    line_start = index
                    continue
                result.append("".join(
                    character for character in text[index:after]
                    if character in "\r\n"
                ))
            else:
                result.append(text[index:end + 2])
            index = end + 2
            continue
        result.append(current)
        if current == "\n":
            line_start = index + 1
        index += 1
    return "".join(result)


def read_source_record(
    path_value: object,
    dataset: object,
    *,
    errors: str = "replace",
    max_chars: int | None = None,
) -> str:
    """Read and sanitize model-visible source, then optionally truncate it."""

    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    try:
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors=errors)
        elif path.is_dir():
            source = "\n".join(
                item.read_text(encoding="utf-8", errors=errors)
                for item in sorted(path.rglob("*.sol"))
            )
        else:
            return ""
    except (OSError, TypeError):
        return ""
    sanitized = sanitize_source_text(
        source,
        source_dataset_for_path(path_value, dataset),
    )
    return sanitized[:max_chars] if max_chars is not None else sanitized
