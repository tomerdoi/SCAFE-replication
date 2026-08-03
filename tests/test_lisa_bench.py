from src.datasets.lisa_bench import _infer_types, _join_code, deduplicate_records


def test_lisa_type_inference_preserves_scafe_taxonomy():
    labels = _infer_types(
        "Reentrancy in withdraw",
        "An attacker can re-enter before the state update.",
    )
    assert "reentrancy" in labels


def test_lisa_code_join_keeps_file_boundaries():
    source, filenames = _join_code([
        {"filename": "A.sol", "content": "contract A {}"},
        {"filename": "B.sol", "content": "contract B {}"},
    ])
    assert filenames == ["A.sol", "B.sol"]
    assert "FILE: A.sol" in source
    assert "FILE: B.sol" in source


def test_lisa_deduplication_never_creates_clean_labels():
    records = [
        {
            "record_id": "1",
            "source": "contract A {}\n",
            "title": "Finding",
            "audit_url": "https://audit.example/1",
            "impact": "high",
            "vulnerability_types": ["reentrancy"],
            "filenames": ["A.sol"],
        },
        {
            "record_id": "2",
            "source": "contract A {}\n",
            "title": "Same finding",
            "audit_url": "https://audit.example/2",
            "impact": "medium",
            "vulnerability_types": ["access_control"],
            "filenames": ["A.sol"],
        },
    ]
    result = deduplicate_records(records)
    assert len(result) == 1
    assert result[0]["label_status"] == "vulnerable"
    assert result[0]["record_ids"] == ["1", "2"]
