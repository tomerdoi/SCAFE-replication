import pandas as pd

from scripts.build_sampled_experiment_campaign import (
    build_sample,
    fold_audit,
    proportional_allocation,
)


def test_proportional_allocation_is_exact():
    allocation = proportional_allocation(pd.Series({0: 90, 1: 10}), 51)
    assert allocation == {0: 46, 1: 5}
    assert sum(allocation.values()) == 51


def test_sample_is_deterministic_and_stratified():
    frame = pd.DataFrame({
        "source_dataset": ["DAppSCAN"] * 100,
        "contract_id": [f"id-{index}" for index in range(100)],
        "_source_identity": [f"hash-{index}" for index in range(100)],
        "y": [0] * 90 + [1] * 10,
    })
    first, allocation = build_sample(
        frame,
        dataset="DAppSCAN",
        sample_size=50,
        seed=42,
    )
    second, _ = build_sample(
        frame,
        dataset="DAppSCAN",
        sample_size=50,
        seed=42,
    )
    assert allocation == {0: 45, 1: 5}
    assert first["contract_id"].tolist() == second["contract_id"].tolist()
    assert first["_source_identity"].is_unique


def test_fold_audit_preserves_project_groups_and_both_labels():
    rows = []
    for project in range(10):
        for label in (0, 1):
            rows.append({
                "source_dataset": "DAppSCAN",
                "source_subdataset": f"project-{project}",
                "_source_identity": f"hash-{project}-{label}",
                "y": label,
            })
    audits = fold_audit(
        pd.DataFrame(rows),
        dataset="DAppSCAN",
        folds=5,
        min_minority_per_test_fold=2,
    )
    assert len(audits) == 5
    assert all(row["test_positive"] == 2 for row in audits)
    assert all(row["test_negative"] == 2 for row in audits)
