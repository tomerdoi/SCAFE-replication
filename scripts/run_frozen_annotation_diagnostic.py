#!/usr/bin/env python3
"""Small project-separated, frozen-weight annotation sensitivity diagnostic."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_function_tokenizer_exposure import annotation_spans, file_sha256, read_raw_source
from scripts.run_function_semantic_transfer import LABELS, _normalise, extract_functions
from src.source_sanitization import SOURCE_SANITIZATION_VERSION, sanitize_source_text

RESULTS = REPO / "data/derived/sanitized_v1/ml_results"
PROTOCOL = "frozen-annotation-diagnostic-v1"


def select_project_sample(candidates: pd.DataFrame, size: int = 24) -> pd.DataFrame:
    """Pick one eligible occurrence per project without consulting model scores."""

    if not 1 <= size <= 24:
        raise ValueError("The bounded pilot permits between one and 24 projects")
    frame = candidates.copy()
    if frame["project"].isna().any() or frame["project"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing project provenance; no source-ID fallback")
    frame["selection_hash"] = frame.apply(
        lambda row: hashlib.sha256(
            f"{PROTOCOL}|{row.project}|{row.contract_id}|{row.function_ordinal}".encode()
        ).hexdigest(), axis=1,
    )
    return (
        frame.sort_values(["selection_hash", "contract_id", "function_ordinal"])
        .drop_duplicates("project").head(size).reset_index(drop=True)
    )


def annotation_token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[bool]:
    return [
        right > left and any(left < end and right > start for start, end in spans)
        for left, right in offsets
    ]


def summarize_pairs(raw: np.ndarray, changed: np.ndarray, *, seed: int = 20261101) -> dict[str, object]:
    """Report paired output shifts; no vulnerability ground truth is inferred."""

    if raw.shape != changed.shape or raw.ndim != 2 or not len(raw):
        raise ValueError("Expected nonempty paired probability matrices")
    if not np.isfinite(raw).all() or not np.isfinite(changed).all():
        raise ValueError("Nonfinite model output")
    absolute = np.abs(changed - raw)
    maximum = absolute.max(axis=1)
    rng = np.random.default_rng(seed)
    means = maximum[rng.integers(0, len(maximum), size=(2000, len(maximum)))].mean(axis=1)
    return {
        "projects": len(raw),
        "mean_max_absolute_output_change": float(maximum.mean()),
        "conditional_project_resample95": np.quantile(means, [0.025, 0.975]).tolist(),
        "median_max_absolute_output_change": float(np.median(maximum)),
        "largest_absolute_output_change": float(maximum.max()),
        "mean_absolute_output_change_all_labels": float(absolute.mean()),
        "functions_crossing_uncalibrated_0_5_boundary": int(((raw >= 0.5) != (changed >= 0.5)).any(axis=1).sum()),
        "per_label_mean_signed_change": dict(zip(LABELS, (changed - raw).mean(axis=0).tolist(), strict=True)),
        "per_label_mean_absolute_change": dict(zip(LABELS, absolute.mean(axis=0).tolist(), strict=True)),
    }


@contextmanager
def compute_lock():
    """Use the cross-paper lock and the existing SCAFE advisory lock together."""

    handles = []
    acquired = False
    owner = Path("/tmp/paper-research-compute.owner.json")
    try:
        for name in ("/tmp/paper-research-compute.lock", "/tmp/scafe-research-iteration.lock"):
            stream = open(name, "a+")
            handles.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        owner.write_text(json.dumps({"task": "SCAFE", "protocol": PROTOCOL, "threads": 1, "maximum_projects": 24}) + "\n")
        yield
    finally:
        if acquired and owner.exists():
            # Only our successfully acquired dual lock owns this metadata.
            try:
                if json.loads(owner.read_text()).get("protocol") == PROTOCOL:
                    owner.unlink()
            except (OSError, ValueError):
                pass
        for stream in reversed(handles):
            stream.close()


def run(output: Path, projects: int) -> dict[str, object]:
    import torch
    from torch import nn
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    started = time.monotonic()
    deadline = started + 240
    registry_path = REPO / "data/derived/sanitized_v1/unified_contract_eval_scored.csv"
    window_dir = RESULTS / "function_tokenizer_exposure_20261101_1600"
    previous = json.loads((window_dir / "summary.json").read_text())
    checkpoint_dir = RESULTS / "function_semantic_transfer/codebert"
    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != previous["input_hashes"]["checkpoint_file_now"]:
        raise ValueError("Checkpoint changed since its recorded audit identity")
    if file_sha256(registry_path) != previous["input_hashes"]["registry"]:
        raise ValueError("Shared registry changed since the tokenizer audit")
    tokenizer_path = checkpoint_dir / "tokenizer/tokenizer.json"
    if file_sha256(tokenizer_path) != previous["input_hashes"]["tokenizer"]:
        raise ValueError("Tokenizer changed since the window audit")
    registry = pd.read_csv(registry_path, usecols=["source_dataset", "contract_id", "source_subdataset", "local_source_path"])
    registry = registry.loc[registry.source_dataset.eq("DAppSCAN")].rename(columns={"source_subdataset": "project"})
    candidates = pd.read_csv(window_dir / "function_window_audit.csv")
    candidates = candidates.loc[candidates.in_primary_development.eq(1) & candidates.input_ids_changed.eq(1)]
    candidates = candidates.merge(registry, on="contract_id", how="left", validate="many_to_one")
    selected = select_project_sample(candidates, projects)
    if len(selected) != projects:
        raise ValueError("Not enough independently named project groups for the frozen sample")
    selected.to_csv(output / "selected_functions.csv", index=False)
    # Restricted tensor-only loading rejects arbitrary pickled Python objects.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("version") != 1 or payload.get("labels") != list(LABELS) or payload.get("max_length") != 512:
        raise ValueError("Unsupported checkpoint schema")
    if payload.get("model_name") != "microsoft/codebert-base":
        raise ValueError("Unexpected checkpoint model")
    config = AutoConfig.from_pretrained(payload["model_name"], local_files_only=True)

    class FrozenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, len(LABELS))

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            return self.classifier(hidden)

    model = FrozenModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", local_files_only=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token for the fixed-window control")
    predictions = []
    pair_rows = []
    repeat_error = 0.0
    for index, row in enumerate(selected.itertuples(index=False)):
        if time.monotonic() > deadline:
            raise TimeoutError("Bounded 240-second inference deadline exceeded")
        raw_source = read_raw_source(Path(row.local_source_path))
        raw_functions = extract_functions(raw_source)
        clean_functions = extract_functions(sanitize_source_text(raw_source, "DAppSCAN"))
        if [name for name, _ in raw_functions] != [name for name, _ in clean_functions]:
            raise ValueError("Function sequence changed")
        raw_text, clean_text = raw_functions[row.function_ordinal][1], clean_functions[row.function_ordinal][1]
        if hashlib.sha256(raw_text.encode()).hexdigest() != row.raw_function_sha256 or hashlib.sha256(clean_text.encode()).hexdigest() != row.sanitized_function_sha256:
            raise ValueError("Selected function text changed since the token audit")
        if _normalise(raw_text) != _normalise(clean_text):
            raise ValueError("Comment removal changed the approximate code identity")
        raw_encoded = tokenizer(raw_text, truncation=True, max_length=512, return_tensors="pt", return_offsets_mapping=True)
        offsets = raw_encoded.pop("offset_mapping")[0].tolist()
        mask = annotation_token_mask(offsets, annotation_spans(raw_text))
        if not any(mask):
            raise ValueError("Selected input has no annotation tokens in window")
        masked = {key: value.clone() for key, value in raw_encoded.items()}
        masked["input_ids"][0, torch.tensor(mask, dtype=torch.bool)] = tokenizer.mask_token_id
        clean_encoded = tokenizer(clean_text, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            raw_probability = torch.sigmoid(model(**raw_encoded)).numpy()[0]
            clean_probability = torch.sigmoid(model(**clean_encoded)).numpy()[0]
            mask_probability = torch.sigmoid(model(**masked)).numpy()[0]
            if index < 3:
                repeated = torch.sigmoid(model(**raw_encoded)).numpy()[0]
                repeat_error = max(repeat_error, float(np.max(np.abs(raw_probability - repeated))))
        predictions.append((raw_probability, clean_probability, mask_probability))
        for label_index, label in enumerate(LABELS):
            pair_rows.append({"project": row.project, "contract_id": row.contract_id, "function_ordinal": row.function_ordinal, "pattern": label,
                              "raw": float(raw_probability[label_index]), "sanitized": float(clean_probability[label_index]), "fixed_window_masked": float(mask_probability[label_index])})
        print(f"Completed frozen triplet {index + 1}/{len(selected)}", flush=True)
    if repeat_error > 1e-6:
        raise ValueError("Repeated-input control is not deterministic within tolerance")
    raw, clean, masked = (np.stack([entry[index] for entry in predictions]) for index in range(3))
    pd.DataFrame(pair_rows).to_csv(output / "paired_pattern_outputs.csv", index=False)
    if file_sha256(checkpoint) != checkpoint_hash or file_sha256(registry_path) != previous["input_hashes"]["registry"]:
        raise ValueError("Checkpoint or shared registry changed during inference")
    return {
        "protocol": PROTOCOL, "source_sanitization_version": SOURCE_SANITIZATION_VERSION,
        "eligible_functions": len(candidates), "eligible_projects": int(candidates.project.nunique()),
        "sampled_projects": len(selected), "selection": "fixed hash order; one function per nonempty source_subdataset project; at most 24 projects; no scores or labels used",
        "sanitized_vs_raw": summarize_pairs(raw, clean), "fixed_window_masked_vs_raw": summarize_pairs(raw, masked),
        "repeated_raw_control_max_abs_difference": repeat_error,
        "lineage": {"checkpoint_sha256": checkpoint_hash, "registry_sha256": previous["input_hashes"]["registry"], "tokenizer_sha256": file_sha256(tokenizer_path), "selection_sha256": file_sha256(output / "selected_functions.csv"), "config_sha256": hashlib.sha256(config.to_json_string().encode()).hexdigest(), "script_sha256": file_sha256(Path(__file__)), "torch": torch.__version__, "transformers": transformers.__version__, "weights_only_load": True, "offline": True, "cpu_threads": 1},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "This newly identified frozen checkpoint is not a reproduction of historical scores; no historical-score reuse or fusion tuning.",
            "No ground-truth vulnerability labels, accuracy, calibration, detector superiority, or external generalization is evaluated.",
            "The sample is conditional on annotation exposure and project metadata; it is not representative of unannotated deployments.",
            "Comment deletion can change truncation/context; fixed-window masking holds retained context but introduces mask tokens and is not a neutral natural-code control.",
            "The bootstrap is conditional on the 24 observed project groups and omits model, corpus, annotation and selection uncertainty.",
            "The original auxiliary training CSV remains missing; synthetic-pattern label and near-clone provenance limits persist.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projects", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        with compute_lock():
            summary = run(args.output_dir, args.projects)
        summary["status"] = "completed"
    except Exception as error:
        (args.output_dir / "failure.json").write_text(json.dumps({"status": "failed_or_compute_busy", "error_type": type(error).__name__, "reason": str(error)}, indent=2) + "\n")
        raise
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
