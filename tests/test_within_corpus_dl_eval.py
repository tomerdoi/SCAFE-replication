import json

import numpy as np
import pandas as pd

from src.ml.within_corpus_dl_eval import (
    CHECKPOINT_VERSION,
    _atomic_json,
    _load_completion,
    aggregate_contract_probabilities,
    canonicalize,
    group_disjoint_inner_split,
    groups_for,
    transformer_payload_windows,
)


def _frame(tmp_path):
    sources = []
    for name in ("a", "b", "c", "d"):
        path = tmp_path / f"{name}.sol"
        path.write_text(f"contract {name} {{}}", encoding="utf-8")
        sources.append(str(path))
    return pd.DataFrame({
        "source_dataset": ["DAppSCAN"] * 4,
        "source_subdataset": ["p1", "p1", "p2", "p3"],
        "label_status": ["clean", "clean", "vulnerable", "vulnerable"],
        "contract_id": ["a", "a-copy", "c", "d"],
        "local_source_path": [sources[0], sources[0], sources[2], sources[3]],
    })


def test_canonicalization_conflicts_and_groups_do_not_overlap(tmp_path):
    frame = _frame(tmp_path)
    conflict = tmp_path / "conflict.sol"
    conflict.write_text("contract Conflict {}", encoding="utf-8")
    frame = pd.concat([frame, pd.DataFrame({
        "source_dataset": ["DAppSCAN", "DAppSCAN"],
        "source_subdataset": ["p4", "p4"],
        "label_status": ["clean", "vulnerable"],
        "contract_id": ["x", "y"],
        "local_source_path": [str(conflict), str(conflict)],
    })], ignore_index=True)
    canonical, audit = canonicalize(frame)
    assert len(canonical) == 3
    assert audit["conflicting_label_rows_excluded"] == 2
    groups = groups_for(canonical, "DAppSCAN")
    assert len(np.unique(groups)) == 3


def test_inner_split_is_group_disjoint(tmp_path):
    frame = _frame(tmp_path)
    canonical, _ = canonicalize(frame)
    groups = groups_for(canonical, "DAppSCAN")
    train, validation = group_disjoint_inner_split(canonical, groups, 42, 0.34)
    assert not set(groups[train]) & set(groups[validation])
    assert not set(canonical.iloc[train]["_source_identity"]) & set(canonical.iloc[validation]["_source_identity"])


def test_full_source_chunk_aggregation_is_contract_mean():
    actual = aggregate_contract_probabilities(["a", "a", "b"], [0.2, 0.8, 0.9])
    assert actual.to_dict("records") == [
        {"contract_id": "a", "y_prob": 0.5, "n_chunks": 2},
        {"contract_id": "b", "y_prob": 0.9, "n_chunks": 1},
    ]


def test_transformer_payload_windows_preserve_every_source_token():
    token_ids = list(range(1_025))
    windows = transformer_payload_windows(token_ids, special_token_count=2)
    assert [len(window) for window in windows] == [510, 510, 5]
    assert [token for window in windows for token in window] == token_ids


def test_completion_rejects_changed_immutable_configuration(tmp_path):
    path = tmp_path / "complete.json"
    config = {"model": "textcnn", "epochs": 10}
    _atomic_json(path, {
        "version": CHECKPOINT_VERSION,
        "config": config,
        "split_membership_hash": "split",
        "predictions": [{
            "contract_id": "a",
            "source_identity_hash": "source-a",
            "y_true": 1,
        }],
    })
    expected = [{
        "contract_id": "a",
        "source_identity_hash": "source-a",
        "y_true": 1,
    }]
    assert _load_completion(
        path,
        config,
        "split",
        expected_identities=["source-a"],
        expected_labels=[1],
    ) == expected
    assert _load_completion(path, {"model": "textcnn", "epochs": 1}, "split") is None
    assert _load_completion(path, config, "different") is None
    assert _load_completion(
        path,
        config,
        "split",
        expected_identities=["source-a"],
        expected_labels=[0],
    ) is None
