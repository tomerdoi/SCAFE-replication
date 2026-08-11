import pandas as pd

from src.ml.semantic_features import (
    add_semantic_features_to_frame,
    compute_context_features,
)


def test_context_features_capture_call_guard_and_ordering_signals():
    source = """
    contract C {
        uint balances;
        function withdraw() external {
            msg.sender.call{value: balances}("");
            balances = 0;
            require(msg.sender == tx.origin);
        }
    }
    """
    features = compute_context_features(source)
    assert features["context_external_call_count"] == 1
    assert features["context_guard_count"] == 1
    assert features["context_call_before_state_write_count"] == 1
    assert features["context_access_guard_count"] >= 2


def test_semantic_frame_augmentation_is_numeric(tmp_path):
    source = tmp_path / "C.sol"
    source.write_text("contract C { function safe() external {} }")
    frame = pd.DataFrame({"local_source_path": [str(source)]})
    augmented = add_semantic_features_to_frame(frame)
    semantic_columns = [c for c in augmented if c.startswith("semantic_")]
    assert semantic_columns
    assert all(pd.api.types.is_numeric_dtype(augmented[c]) for c in semantic_columns)
