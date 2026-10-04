#!/usr/bin/env python3
"""Build a reviewer-facing SCAFE artifact without internal review material."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / "sc_risk_score_overleaf"
DEFAULT_OUTPUT = Path("/tmp/scafe-anonymous-artifact")

SOURCE_TREES = ("src", "scripts", "tests", "paper")
TOP_LEVEL_PATTERNS = (
    "README*",
    "LICENSE*",
    "requirements*.txt",
    "pyproject.toml",
    "pytest.ini",
)
ARTIFACT_PATHS = (
    "data/derived/ml_results/baseline_comparison/source_only_raw_canonical",
    "data/derived/ml_results/baseline_comparison/source_only_raw_canonical_ablations",
    "data/derived/ml_results/baseline_comparison/source_only_raw_plus_lexical_canonical",
    "data/derived/ml_results/baseline_comparison/source_only_raw_canonical_label_sensitivity",
    "data/derived/ml_results/baseline_comparison/exact_paper_benchmark_metrics.json",
    "data/derived/ml_results/baseline_comparison/exact_paper_benchmark_scored.csv",
    "data/derived/ml_results/baseline_comparison/scafe_experiment_manifest.json",
    "data/derived/feature_representation_audit",
    "data/derived/shap_results/fold_consistent_primary",
    "data/derived/parser_compatibility_audit",
    "data/derived/slither_reliability/summary.json",
    "data/derived/slither_reliability/by_source_construct.csv",
    "data/derived/slither_reliability/by_solidity_version.csv",
    "data/derived/slither_correlation/correlation_results.csv",
)
WITHIN_CORPUS_FILES = (
    "scafe_within_corpus_summary.csv",
    "scafe_within_corpus_per_seed.csv",
    "scafe_within_corpus_metadata.json",
)
PAPER_FIGURES = (
    "tab_dataset_overview.tex",
    "tab_current_model_heads.tex",
    "tab_operational_triage.tex",
    "tab_lexical_extension.tex",
    "tab_label_sensitivity.tex",
    "tab_external_slices.tex",
    "tab_published_baseline_context.tex",
    "tab_parser_strata.tex",
    "tab_parser_reliability.tex",
    "fig_model_comparison.pdf",
    "fig_cross_dataset_heatmap.pdf",
    "fig_confusion_matrices.pdf",
    "fig_feature_ablation.pdf",
    "fig_ensemble_comparison.pdf",
    "fig_feature_importance.pdf",
    "fig_shap_cases.pdf",
)
TEXT_SUFFIXES = {
    ".bib", ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".sh",
    ".sty", ".tex", ".toml", ".txt", ".yaml", ".yml",
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ORCID = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b")
ARTIFACT_README = """# SCAFE anonymous reviewer artifact

This package accompanies the manuscript on compilation-free source-record risk
ranking. It contains source code, tests, the exact generated evidence cited by
the manuscript, the manuscript source/PDF, and SHA-256 checksums.

## Verify packaged evidence

From the package root, install the pinned Python dependencies in an isolated
environment and run:

```bash
python scripts/validate_experiment_integrity.py
python -m pytest
```

The first command validates publication-critical protocol metadata and generated
tables without requiring the raw corpora. The complete local
`validate_current_paper_artifacts.py` additionally checks the source feature
matrix and therefore requires the upstream datasets and derived unified table.

## Data scope

Raw corpora and Solidity sources are not redistributed because their upstream
licenses and distribution terms differ. Generated per-record predictions,
metrics, manifests, ablations, parser/Slither summaries, and source-matching
diagnostics needed to audit the reported claims are included under
`data/derived/`. Paths stored in the package are repository-relative.
"""


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pt", "*.bin", "*.code-workspace",
                "checkpoints",
            ),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def scan_text(output: Path, identifiers: list[str]) -> None:
    forbidden = ["/Users/", *identifiers]
    detector_files = {
        "build_anonymous_artifact.py",
        "check_anonymity.py",
        "validate_current_paper_artifacts.py",
    }
    findings: list[str] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in forbidden:
            if value == "/Users/" and path.name in detector_files:
                continue
            if value and value.lower() in text.lower():
                findings.append(f"{path.relative_to(output)}: {value}")
        if EMAIL.search(text):
            findings.append(f"{path.relative_to(output)}: email-like identifier")
        if ORCID.search(text):
            findings.append(f"{path.relative_to(output)}: ORCID-like identifier")
    if findings:
        raise RuntimeError("Anonymity scan failed:\n" + "\n".join(findings))


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    for tree in SOURCE_TREES:
        copy_path(REPO / tree, args.output / tree)
    for pattern in TOP_LEVEL_PATTERNS:
        for source in REPO.glob(pattern):
            copy_path(source, args.output / source.name)
    for relative in ARTIFACT_PATHS:
        copy_path(REPO / relative, args.output / relative)
    within = (
        REPO
        / "data/derived/ml_results/baseline_comparison/source_only_within_corpus"
    )
    for name in WITHIN_CORPUS_FILES:
        copy_path(
            within / name,
            args.output
            / "data/derived/ml_results/baseline_comparison/source_only_within_corpus"
            / name,
        )

    manuscript = args.output / "manuscript"
    copy_path(PAPER / "sc_vulnerability_detection.tex", manuscript / "sc_vulnerability_detection.tex")
    copy_path(PAPER / "sc_vulnerability_refs.bib", manuscript / "sc_vulnerability_refs.bib")
    copy_path(PAPER / "sc_vulnerability_detection.pdf", manuscript / "sc_vulnerability_detection.pdf")
    for name in PAPER_FIGURES:
        copy_path(PAPER / "figures" / name, manuscript / "figures" / name)

    (args.output / "README.md").write_text(ARTIFACT_README, encoding="utf-8")
    metadata = {
        "artifact": "SCAFE reviewer package",
        "raw_datasets_included": False,
        "reason": "Upstream datasets retain their original distribution terms.",
    }
    (args.output / "ARTIFACT_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    home_identifier = Path.home().name
    identifiers = [
        home_identifier,
        home_identifier.replace(".", " ").replace("_", " ").replace("-", " "),
    ] + [
        value.strip()
        for value in os.environ.get("ANONYMITY_PERSONAL_IDENTIFIERS", "").split(",")
        if value.strip()
    ]
    scan_text(args.output, identifiers)
    write_checksums(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
