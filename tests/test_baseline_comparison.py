import numpy as np
import pandas as pd

from src.ml.baseline_comparison import (
    DatasetSpec,
    _metrics,
    canonicalize_vulnerability_type,
    canonicalize_vulnerability_types,
    exclude_test_source_duplicates,
    select_dataset,
    source_feature_columns,
    summarize_repeated_lodo,
)


def _frame():
    return pd.DataFrame(
        {
            "source_dataset": ["Toy"] * 4,
            "label_status": ["clean", "vulnerable", "clean", "vulnerable"],
            "contract_id": ["a", "b", "c", "d"],
            "source_subdataset": ["p1", "p1", "p2", "p2"],
            "structural_signal": [0.0, 1.0, 0.1, 0.9],
            "tx_count_30d": [1, 2, 3, 4],
            "has_payload": [1, 1, 1, 1],
            "has_verified_source": [1, 1, 0, 1],
            "is_proxy": [0, 0, 1, 0],
            "is_solc_pre_08": [0, 1, 0, 1],
            "comp_complexity": [0.1, 0.9, 0.2, 0.8],
            "vuln_present": [0, 1, 0, 1],
        }
    )


def test_source_features_exclude_activity_and_labels():
    columns = source_feature_columns(_frame())
    assert columns == ["structural_signal"]
    assert "has_verified_source" not in columns
    assert "has_payload" not in columns
    assert "is_proxy" not in columns
    assert "is_solc_pre_08" not in columns
    assert "comp_complexity" not in columns


def test_select_dataset_does_not_turn_unknown_into_clean():
    frame = _frame().assign(
        source_dataset=["Toy", "Toy", "Toy", "Toy"],
        label_status=["clean", "vulnerable", "unknown", "vulnerable"],
    )
    selected = select_dataset(frame, DatasetSpec("Toy"))
    assert len(selected) == 3
    assert selected["y"].tolist() == [0, 1, 1]


def test_select_dataset_deduplicates_sources_and_drops_conflicts(tmp_path):
    duplicate = tmp_path / "duplicate.sol"
    duplicate.write_text("contract Duplicate {}")
    conflict = tmp_path / "conflict.sol"
    conflict.write_text("contract Conflict {}")
    frame = pd.DataFrame(
        {
            "source_dataset": ["Toy"] * 4,
            "label_status": ["clean", "clean", "clean", "vulnerable"],
            "contract_id": ["a", "b", "c", "d"],
            "local_source_path": [
                str(duplicate), str(duplicate), str(conflict), str(conflict)
            ],
        }
    )
    selected = select_dataset(frame, DatasetSpec("Toy"))
    assert selected["contract_id"].tolist() == ["a"]


def test_positive_only_metrics_are_recall_only():
    metrics = _metrics(
        np.ones(3, dtype=int),
        np.array([0.9, 0.2, 0.8]),
        dataset="Toy",
        model="rf",
        experiment="transfer",
    )
    assert metrics["positive_only"] is True
    assert np.isnan(metrics["f1"])
    assert metrics["recall"] == 2 / 3
    assert np.isnan(metrics["roc_auc"])


def test_vulnerability_taxonomy_normalizes_swc_cwe_and_multi_labels():
    assert canonicalize_vulnerability_type("SWC-107-Reentrancy") == "reentrancy"
    assert canonicalize_vulnerability_type("CWE-284") == "access_control"
    assert canonicalize_vulnerability_types("97: time_manipulation;129: arithmetic") == (
        "arithmetic;time_manipulation"
    )


def test_duplicate_sources_are_removed_from_training(tmp_path):
    source = tmp_path / "Contract.sol"
    source.write_text("// comment\ncontract C { }\n")
    train = pd.DataFrame(
        {
            "local_source_path": [str(source), str(tmp_path / "other.sol")],
            "y": [1, 0],
        }
    )
    train.loc[1, "local_source_path"] = str(source)
    test = pd.DataFrame({"local_source_path": [str(source)]})
    filtered, removed = exclude_test_source_duplicates(train, test)
    assert removed == 2
    assert filtered.empty


def test_repeated_lodo_summary_reports_seed_confidence_interval():
    metrics = pd.DataFrame(
        {
            "dataset": ["Toy"] * 3,
            "model": ["rf"] * 3,
            "random_state": [42, 43, 44],
            "roc_auc": [0.6, 0.7, 0.8],
            "pr_auc": [0.5, 0.6, 0.7],
            "f1": [0.4, 0.5, 0.6],
            "precision": [0.4, 0.5, 0.6],
            "recall": [0.4, 0.5, 0.6],
            "coverage": [0.4, 0.5, 0.6],
        }
    )
    summary = summarize_repeated_lodo(metrics)
    assert summary.loc[0, "n_seeds"] == 3
    assert np.isclose(summary.loc[0, "roc_auc_mean"], 0.7)
    assert summary.loc[0, "roc_auc_ci95_low"] <= 0.7
    assert summary.loc[0, "roc_auc_ci95_high"] >= 0.7
