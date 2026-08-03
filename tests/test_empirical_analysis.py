from src.datasets.empirical_analysis import (
    deduplicate_records,
    normalize_source,
    parse_annotation_value,
    parse_tag_types,
)


def test_source_normalization_is_stable():
    assert normalize_source(" contract {\r\n}\r\n") == "contract {\n}\n"


def test_annotation_parser_accepts_repository_literal_format():
    parsed = parse_annotation_value(
        "[{'category': 'reentrancy', 'line': 12, 'line_end': 14}]"
    )
    assert parsed[0]["category"] == "reentrancy"
    assert parsed[0]["line_end"] == 14


def test_tag_parser_drops_clean_marker_and_deduplicates():
    assert parse_tag_types("reentrancy; timestamp; reentrancy;") == [
        "reentrancy",
        "timestamp",
    ]
    assert parse_tag_types("no") == []


def test_deduplication_merges_provenance_and_annotations():
    records = [
        {
            "source_hash": "same",
            "source": "contract A {}\n",
            "contract_id": "a",
            "provenance": "sample",
            "source_label": "tool-a",
            "vulnerability_types": [],
            "line_annotations": [],
        },
        {
            "source_hash": "same",
            "source": "contract A {}\n",
            "contract_id": "b",
            "provenance": "curated",
            "source_label": "tool-b",
            "vulnerability_types": ["reentrancy"],
            "line_annotations": [{"line": 3}],
        },
    ]
    result = deduplicate_records(records)
    assert len(result) == 1
    assert result[0]["label_status"] == "vulnerable"
    assert result[0]["contract_ids"] == ["a", "b"]
    assert result[0]["provenance"] == ["curated", "sample"]
    assert result[0]["line_annotations"] == [{"line": 3}]
