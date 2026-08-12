import pandas as pd

from src.evaluation.experiment_manifest import (
    build_dataset_manifest,
    build_feature_manifest,
    feature_group_columns,
)


def test_dataset_manifest_counts_labels_and_missing_sources(tmp_path):
    source = tmp_path / "C.sol"
    source.write_text("contract C {}")
    frame = pd.DataFrame(
        {
            "source_dataset": ["Toy", "Toy", "Toy"],
            "label_status": ["clean", "vulnerable", "unknown"],
            "contract_id": ["a", "b", "c"],
            "local_source_path": [str(source), "", ""],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    record = build_dataset_manifest(frame, datasets=["Toy"])[0]
    assert record["rows"] == 3
    assert record["clean"] == 1
    assert record["vulnerable"] == 1
    assert record["unknown"] == 1
    assert record["source_available_rows"] == 1
    assert record["source_missing_rows"] == 2


def test_feature_manifest_separates_evaluated_and_excluded_features():
    frame = pd.DataFrame(
        {
            "source_dataset": ["Toy"],
            "label_status": ["clean"],
            "contract_id": ["a"],
            "structural_feature": [1.0],
            "tx_count_30d": [2.0],
        }
    )
    manifest = build_feature_manifest(frame)
    names = {item["name"] for item in manifest["evaluated_features"]}
    excluded = {item["name"] for item in manifest["excluded_numeric_features"]}
    assert "structural_feature" in names
    assert "tx_count_30d" not in names
    assert "tx_count_30d" in excluded
    assert manifest["semantic_catalog_count"] > 0


def test_feature_groups_are_disjoint_and_cover_evaluated_features():
    frame = pd.DataFrame(
        {
            "source_dataset": ["Toy"],
            "label_status": ["clean"],
            "contract_id": ["a"],
            "structural_feature": [1.0],
            "comp_safety": [0.2],
            "ast_n_functions": [2.0],
        }
    )
    groups = feature_group_columns(frame)
    flattened = [name for names in groups.values() for name in names]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {
        "structural_feature",
        "comp_safety",
        "ast_n_functions",
    }
