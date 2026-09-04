import pandas as pd
import pytest

from src.ml.reentrancy_relational_eval import (
    POC_VERSION,
    VARIANT_COMBINED,
    VARIANT_RELATIONAL,
    VARIANT_STRUCTURAL,
    attach_cached_relational_features,
    build_relational_feature_cache,
    canonicalize_reentrancy_source_rows,
    feature_variants,
)
from src.ml.reentrancy_relational_features import (
    RELATIONAL_FEATURE_COLUMNS,
    RELATIONAL_METADATA_COLUMNS,
)


def test_feature_variants_keep_relational_metadata_out_of_models():
    variants = feature_variants(["loc", "n_functions"])

    assert variants[VARIANT_STRUCTURAL] == ["loc", "n_functions"]
    assert variants[VARIANT_RELATIONAL] == list(RELATIONAL_FEATURE_COLUMNS)
    assert variants[VARIANT_COMBINED][:2] == ["loc", "n_functions"]
    assert set(RELATIONAL_METADATA_COLUMNS).isdisjoint(
        variants[VARIANT_COMBINED]
    )


def test_feature_cache_round_trip_requires_exact_row_identity(tmp_path):
    first = tmp_path / "first.sol"
    second = tmp_path / "second.sol"
    first.write_text("contract A {}", encoding="utf-8")
    second.write_text("contract B {}", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "contract_id": ["a", "b"],
            "source_dataset": ["one", "two"],
            "local_source_path": [str(first), str(second)],
        }
    )

    cache = build_relational_feature_cache(frame)
    restored = attach_cached_relational_features(frame, cache)

    assert (cache["poc_version"] == POC_VERSION).all()
    assert set(RELATIONAL_FEATURE_COLUMNS).issubset(restored.columns)

    reordered = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="contract_id"):
        attach_cached_relational_features(reordered, cache)


def test_feature_cache_rejects_incompatible_version(tmp_path):
    source = tmp_path / "C.sol"
    source.write_text("contract C {}", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "contract_id": ["c"],
            "source_dataset": ["one"],
            "local_source_path": [str(source)],
        }
    )
    cache = build_relational_feature_cache(frame)
    cache["poc_version"] = POC_VERSION + 1

    with pytest.raises(ValueError, match="version"):
        attach_cached_relational_features(frame, cache)


def test_pre_lodo_canonicalization_removes_conflicting_exact_sources(tmp_path):
    shared = tmp_path / "shared.sol"
    shared.write_text("contract C {}", encoding="utf-8")
    distinct = tmp_path / "distinct.sol"
    distinct.write_text("contract D {}", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "contract_id": ["a", "b", "c"],
            "source_dataset": ["one", "one", "one"],
            "local_source_path": [str(shared), str(shared), str(distinct)],
            "vuln_types": ["reentrancy", "arithmetic", "reentrancy"],
            "label_status": ["vulnerable", "vulnerable", "vulnerable"],
        }
    )

    result, audit = canonicalize_reentrancy_source_rows(frame)

    assert result["contract_id"].tolist() == ["c"]
    assert audit["conflicting_rows_excluded"] == 2
