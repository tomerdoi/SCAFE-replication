#!/usr/bin/env python3 -u
"""
Anonymity / submission-readiness scanner.

Scans the repo and the LaTeX paper for:
  - Author / affiliation identifiers that would de-anonymize a double-blind
    submission (names, email domains, GitHub handles, ORCIDs, etc.)
  - Stale `\\author{...}` blocks in the .tex
  - Hard-coded user paths that would leak the author's home directory
  - Optional 4open.science upload structure check

Exit code is 0 only when no findings.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parents[1]
_OVERLEAF = _REPO.parent / "sc_risk_score_overleaf"

# Known author personal identifiers (extend as needed). Lower-cased.
# Some of these should be replaced by the user before submission.
PERSONAL_NAMES = [
    "tomer doitshman", "tomer.doitshman", "tomerdoitshman",
    "doitshman",
]

EMAIL_LIKE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ORCID = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b")
GITHUB_HANDLE = re.compile(r"github\.com/([A-Za-z0-9-]{3,})", re.IGNORECASE)
HOMEDIR_PATTERN = re.compile(r"/Users/[A-Za-z0-9._-]+/")

EXCLUDE_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "node_modules",
    ".idea", ".vscode", "data", "models",
}
# Limit scan to source and prose
INCLUDE_EXTS = {".py", ".sh", ".md", ".tex", ".bib", ".yml", ".yaml",
                ".toml", ".cfg", ".ini", ".txt", ".cls", ".sty"}


def _scan_text(text: str, path: Path) -> List[Tuple[Path, int, str, str]]:
    findings = []
    low = text.lower()

    for name in PERSONAL_NAMES:
        idx = 0
        while True:
            i = low.find(name, idx)
            if i == -1:
                break
            line = low.count("\n", 0, i) + 1
            findings.append((path, line, "personal_name", name))
            idx = i + len(name)

    for m in EMAIL_LIKE.finditer(text):
        # Skip generic emails that aren't author-specific
        addr = m.group(0)
        if any(s in addr.lower() for s in (
            "example.com", "domain.com", "...", "anonymized", "anon@", "noreply",
        )):
            continue
        line = text.count("\n", 0, m.start()) + 1
        findings.append((path, line, "email_like", addr))

    for m in ORCID.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        findings.append((path, line, "orcid", m.group(0)))

    for m in GITHUB_HANDLE.finditer(text):
        handle = m.group(1)
        if handle.lower() in {"anonymous", "anon", "anonymized", "user",
                               "your-username", "owner"}:
            continue
        # Skip well-known orgs / framework handles
        if handle.lower() in {
            "smartdec", "consensys", "crytic", "ethereum", "openzeppelin",
            "microsoft", "huggingface", "facebook", "google", "pytorch",
            "tensorflow", "sklearn", "scikit-learn", "pandas-dev",
            "numpy", "scipy", "matplotlib", "seaborn", "jupyter",
            "ipython", "anthropics", "openai", "ollama", "qwen",
        }:
            continue
        line = text.count("\n", 0, m.start()) + 1
        findings.append((path, line, "github_handle", handle))

    for m in HOMEDIR_PATTERN.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        findings.append((path, line, "homedir_leak", m.group(0)))

    return findings


def _scan_tree(root: Path):
    findings = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in INCLUDE_EXTS:
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        findings.extend(_scan_text(text, p))
    return findings


_ANONYMOUS_MARKERS = (
    "anonymous", "anon", "omitted", "double-blind", "blind review",
    "withheld", "redacted", "removed",
)


def _is_anonymous(block: str) -> bool:
    lo = block.lower()
    return any(marker in lo for marker in _ANONYMOUS_MARKERS)


def _check_tex_author_block():
    """Detect non-anonymous \\author{...} in the main .tex file."""
    main_tex = _OVERLEAF / "sc_vulnerability_detection.tex"
    if not main_tex.exists():
        return []
    text = main_tex.read_text(errors="replace")
    findings = []
    for m in re.finditer(r"\\author\{([^}]+)\}", text):
        block = m.group(1)
        if _is_anonymous(block):
            continue
        line = text.count("\n", 0, m.start()) + 1
        findings.append((main_tex, line, "tex_author_block", block.strip()[:120]))
    for m in re.finditer(r"\\affiliation\{([^}]+)\}", text):
        block = m.group(1)
        if _is_anonymous(block):
            continue
        line = text.count("\n", 0, m.start()) + 1
        findings.append((main_tex, line, "tex_affiliation", block.strip()[:120]))
    for m in re.finditer(r"\\email\{([^}]+)\}", text):
        block = m.group(1)
        if _is_anonymous(block):
            continue
        line = text.count("\n", 0, m.start()) + 1
        findings.append((main_tex, line, "tex_email", block.strip()[:120]))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true",
                        help="Print only summary, not individual findings.")
    parser.add_argument("--max-show", type=int, default=200)
    args = parser.parse_args()

    print("=" * 80)
    print("ANONYMITY / SUBMISSION-READINESS SCAN")
    print("=" * 80)

    repo_findings = _scan_tree(_REPO)
    overleaf_findings = _scan_tree(_OVERLEAF) if _OVERLEAF.exists() else []
    tex_findings = _check_tex_author_block()

    all_findings = repo_findings + overleaf_findings + tex_findings

    by_kind = {}
    for _, _, kind, _ in all_findings:
        by_kind[kind] = by_kind.get(kind, 0) + 1

    print(f"\nScanned:")
    print(f"  Repo:     {_REPO}")
    if _OVERLEAF.exists():
        print(f"  Overleaf: {_OVERLEAF}")

    print(f"\nSummary by finding type:")
    if not by_kind:
        print("  (no findings — looks clean)")
    for k, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s}  {n:4d}")

    if not args.quiet and all_findings:
        print(f"\nDetail (first {args.max_show}):")
        # Show personal_name/email/orcid/github first
        priority = ["personal_name", "tex_author_block", "tex_affiliation",
                    "tex_email", "email_like", "orcid", "github_handle",
                    "homedir_leak"]
        all_findings.sort(key=lambda r: (priority.index(r[2]) if r[2] in priority else 99))
        for path, line, kind, value in all_findings[: args.max_show]:
            rel = path.relative_to(path.anchor) if str(path).startswith("/") else path
            try:
                rel = path.relative_to(_REPO.parent)
            except ValueError:
                rel = path
            print(f"  [{kind:14s}] {rel}:{line}: {value}")
        if len(all_findings) > args.max_show:
            print(f"  … and {len(all_findings) - args.max_show} more")

    # Non-zero exit if non-empty findings exist (so CI / orchestrator catches them)
    return 0 if not all_findings else 2


if __name__ == "__main__":
    sys.exit(main())
