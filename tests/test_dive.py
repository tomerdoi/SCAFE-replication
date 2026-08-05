import pandas as pd

from src.datasets.dive import _types, deduplicate_records


def test_dive_labels_map_to_scafe_taxonomy():
    row = pd.Series({
        "Reentrancy": 1,
        "Access Control": 0,
        "Arithmetic": 0,
        "Unchecked Return Values": 1,
        "DoS": 0,
        "Bad Randomness": 0,
        "Front Running": 0,
        "Time manipulation": 0,
    })
    assert _types(row) == ["reentrancy", "unchecked_low_calls"]


def test_dive_deduplication_preserves_clean_and_vulnerable_semantics():
    records = [
        {
            "contract_id": "1",
            "source": "contract A {}\n",
            "vulnerability_types": [],
            "label_status": "clean",
            "source_path": "1.sol",
            "label_origin": "multi_tool_consensus",
        },
        {
            "contract_id": "2",
            "source": "contract A {}\n",
            "vulnerability_types": ["reentrancy"],
            "label_status": "vulnerable",
            "source_path": "2.sol",
            "label_origin": "multi_tool_consensus",
        },
    ]
    result = deduplicate_records(records)
    assert len(result) == 1
    assert result[0]["label_status"] == "vulnerable"
    assert result[0]["contract_ids"] == ["1", "2"]
