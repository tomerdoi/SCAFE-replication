from scripts.run_function_semantic_transfer import extract_functions


def test_extract_functions_keeps_nested_block_and_following_function():
    source = """
    contract Example {
        function first(uint256 value) external {
            if (value > 0) {
                value -= 1;
            }
        }

        function second() external {}
    }
    """

    extracted = extract_functions(source)

    assert [name for name, _ in extracted] == ["first", "second"]
    assert "value -= 1;" in extracted[0][1]
    assert extracted[1][1].endswith("{}")
