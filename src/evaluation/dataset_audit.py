"""Dataset identity and label-quality diagnostics for SCAFE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import pandas as pd

from src.ml.baseline_comparison import add_canonical_vulnerability_types

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"//.*?$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def _source_hash(path_value: object) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value or ""))
    if not path.exists():
        return ""
    try:
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="ignore")
        elif path.is_dir():
            source = "\n".join(
                item.read_text(encoding="utf-8", errors="ignore")
                for item in sorted(path.rglob("*.sol"))
            )
        else:
            return ""
    except OSError:
        return ""
    source = _COMMENT_BLOCK.sub(" ", source)
    source = _COMMENT_LINE.sub(" ", source)
    source = _WHITESPACE.sub(" ", source).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass
class DatasetAudit:
    """Compute reproducible identity and label diagnostics."""

    def run(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = add_canonical_vulnerability_types(frame)
        data["_source_hash"] = data["local_source_path"].map(_source_hash)

        summaries = []
        for dataset, group in data.groupby("source_dataset", dropna=False):
            summaries.append(
                {
                    "source_dataset": dataset,
                    "rows": len(group),
                    "clean": int((group["label_status"] == "clean").sum()),
                    "vulnerable": int((group["label_status"] == "vulnerable").sum()),
                    "unknown": int((group["label_status"] == "unknown").sum()),
                    "missing_source": int((group["_source_hash"] == "").sum()),
                    "unique_sources": int(
                        group.loc[group["_source_hash"] != "", "_source_hash"].nunique()
                    ),
                    "taxonomy_unmapped_rows": int(
                        (
                            (group["label_status"] == "vulnerable")
                            & (group["taxonomy_types"] == "")
                        ).sum()
                    ),
                }
            )
        summary_frame = pd.DataFrame(summaries)

        duplicate_groups = (
            data[data["_source_hash"] != ""]
            .groupby("_source_hash", as_index=False)
            .agg(
                rows=("contract_id", "size"),
                datasets=("source_dataset", lambda values: ";".join(sorted(set(values)))),
                labels=("label_status", lambda values: ";".join(sorted(set(values)))),
                contract_ids=("contract_id", lambda values: ";".join(values.astype(str))),
            )
        )
        duplicate_groups = duplicate_groups[duplicate_groups["rows"] > 1].copy()
        duplicate_groups["cross_dataset"] = duplicate_groups["datasets"].map(
            lambda value: len(str(value).split(";")) > 1
        )
        duplicate_groups["label_conflict"] = duplicate_groups["labels"].map(
            lambda value: len(str(value).split(";")) > 1
        )
        return summary_frame, duplicate_groups
