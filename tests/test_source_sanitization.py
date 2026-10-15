import pandas as pd

from src.ml.fair_baseline_eval import _read_sources, source_hash_for_path
from src.risk.contract_risk import ContractRiskUtil
from src.source_sanitization import (
    SOURCE_SANITIZATION_VERSION,
    count_swc_annotation_markers,
    sanitize_source_text,
)


def test_dappscan_swc_annotations_are_removed_without_touching_code():
    source = """pragma solidity ^0.8.0;
// SWC-107-Reentrancy: L10-L12
contract C {
    string constant TEXT = "SWC-107 is documentation";
    uint value = 1; // SWC-101-Integer Overflow: L5
    // ordinary implementation comment
}
/* SWC-104 Unchecked Call Return Value */
"""
    sanitized = sanitize_source_text(source, "DAppSCAN")
    assert SOURCE_SANITIZATION_VERSION
    assert count_swc_annotation_markers(source) == 4
    assert count_swc_annotation_markers(sanitized) == 1
    assert '"SWC-107 is documentation"' in sanitized
    assert "ordinary implementation comment" in sanitized
    assert "uint value = 1;" in sanitized
    assert "Integer Overflow" not in sanitized
    assert "Reentrancy: L10-L12" not in sanitized
    assert "Unchecked Call Return Value" not in sanitized
    assert "\n\ncontract C" not in sanitized
    assert len(sanitized.splitlines()) == len(source.splitlines()) - 2


def test_inline_annotation_keeps_code_and_newline():
    source = "contract C {\n    uint value = 1; // SWC-101-Overflow: L2\n}\n"
    sanitized = sanitize_source_text(source, "DAppSCAN")
    assert sanitized == "contract C {\n    uint value = 1; \n}\n"


def test_source_is_sanitized_before_truncation(tmp_path):
    source = tmp_path / "annotated.sol"
    source.write_text(
        "// SWC-107-Reentrancy: " + ("x" * 100) + "\ncontract C {}\n",
        encoding="utf-8",
    )
    from src.source_sanitization import read_source_record

    assert read_source_record(source, "DAppSCAN", max_chars=13) == "contract C {}"


def test_dappscan_path_fallback_sanitizes_legacy_readers(tmp_path):
    root = tmp_path / "DAppSCAN-main"
    root.mkdir()
    source = root / "annotated.sol"
    source.write_text(
        "// SWC-107-Reentrancy: L2\ncontract C {}\n",
        encoding="utf-8",
    )
    from src.source_sanitization import read_source_record

    assert read_source_record(source, "") == "contract C {}\n"


def test_other_datasets_are_not_modified():
    source = "// SWC-107-Reentrancy\ncontract C {}"
    assert sanitize_source_text(source, "DIVE") == source


def test_model_source_reader_sanitizes_by_row_dataset(tmp_path):
    source = tmp_path / "annotated.sol"
    source.write_text(
        "// SWC-107-Reentrancy: L2\ncontract C {}\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame({
        "source_dataset": ["DAppSCAN", "DIVE"],
        "local_source_path": [str(source), str(source)],
    })
    dappscan, dive = _read_sources(frame)
    assert "SWC-107" not in dappscan
    assert "SWC-107" in dive


def test_sanitization_does_not_change_comment_insensitive_source_identity(tmp_path):
    source = tmp_path / "annotated.sol"
    source.write_text(
        "// SWC-107-Reentrancy: L2\ncontract C {}\n",
        encoding="utf-8",
    )
    before = source_hash_for_path(source)
    sanitized = sanitize_source_text(source.read_text(encoding="utf-8"), "DAppSCAN")
    clean = tmp_path / "clean.sol"
    clean.write_text(sanitized, encoding="utf-8")
    assert source_hash_for_path(clean) == before


def test_handcrafted_feature_extraction_uses_sanitized_dappscan_text(tmp_path):
    source = tmp_path / "annotated.sol"
    source.write_text(
        "// SWC-107-Reentrancy: L2\ncontract C {}\n",
        encoding="utf-8",
    )
    util = ContractRiskUtil()
    sanitized = util.extract_features_from_local_file(
        "dappscan-record",
        str(source),
        source_dataset="DAppSCAN",
    )
    unchanged = util.extract_features_from_local_file(
        "dive-record",
        str(source),
        source_dataset="DIVE",
    )
    assert sanitized["comment_ratio"] == 0
    assert unchanged["comment_ratio"] > 0
    assert sanitized["loc"] == 1
    assert unchanged["loc"] == 2


def test_annotation_titles_do_not_trigger_structural_features(tmp_path):
    source = tmp_path / "annotated.sol"
    source.write_text(
        "// SWC-112-Delegatecall to Untrusted Callee: L2\n"
        "contract C { uint value = 1; // SWC-115-tx.origin Authorization: L2\n"
        "}\n",
        encoding="utf-8",
    )
    features = ContractRiskUtil().extract_features_from_local_file(
        "dappscan-record",
        str(source),
        source_dataset="DAppSCAN",
    )
    assert features["loc"] == 2
    assert features["n_delegatecall"] == 0
    assert features["uses_tx_origin"] == 0
