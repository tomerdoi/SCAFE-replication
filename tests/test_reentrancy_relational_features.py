import pandas as pd

from src.ml.reentrancy_relational_features import (
    RELATIONAL_FEATURE_COLUMNS,
    add_relational_reentrancy_features,
    extract_relational_reentrancy_features,
    read_source_bundle,
)


def test_detects_unguarded_untrusted_call_before_dependent_write():
    source = """
    contract Bank {
        mapping(address => uint256) balances;

        function withdraw(uint256 amount) external {
            msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_external_calls"] == 1
    assert features["rel_n_value_calls"] == 1
    assert features["rel_n_untrusted_target_calls"] == 1
    assert features["rel_n_direct_call_then_state_write"] == 1
    assert features["rel_n_unguarded_untrusted_call_then_state_write"] == 1
    assert features["rel_n_sender_dependent_writes_after_call"] == 1


def test_distinguishes_guarded_checks_effects_interactions_path():
    source = """
    contract Bank {
        mapping(address => uint256) balances;

        function withdraw(uint256 amount) external nonReentrant {
            balances[msg.sender] -= amount;
            msg.sender.call{value: amount}("");
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_direct_state_write_then_call"] == 1
    assert features["rel_n_direct_call_then_state_write"] == 0
    assert features["rel_n_reentrancy_guarded_entrypoints"] == 1
    assert features["rel_n_unguarded_untrusted_calls"] == 0


def test_detects_transitive_state_effect_before_untrusted_callback():
    source = """
    interface IRecipient {
        function tokenFallback(address sender, uint256 amount) external;
    }

    contract Token {
        mapping(address => uint256) balances;

        function transfer(address recipient, uint256 amount) external {
            _transfer(msg.sender, recipient, amount);
            IRecipient(recipient).tokenFallback(msg.sender, amount);
        }

        function _transfer(address from, address to, uint256 amount) internal {
            balances[from] -= amount;
            balances[to] += amount;
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_untrusted_callback_calls"] == 1
    assert features["rel_n_transitive_state_write_then_call"] == 1
    assert features["rel_n_transitive_write_then_untrusted_callback"] == 1
    assert features["rel_n_unguarded_transitive_write_then_callback"] == 1


def test_comments_and_strings_do_not_create_calls_or_writes():
    source = """
    contract C {
        uint256 balance;
        // msg.sender.call(""); balance = 0;
        string constant example = "msg.sender.call(''); balance = 0;";
        function safe() external view returns (uint256) {
            return balance;
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_external_calls"] == 0
    assert features["rel_n_direct_call_then_state_write"] == 0


def test_source_bundle_resolves_local_imports(tmp_path):
    storage = tmp_path / "Storage.sol"
    storage.write_text(
        "contract Storage { mapping(address => uint256) balances; }",
        encoding="utf-8",
    )
    contract = tmp_path / "Token.sol"
    contract.write_text(
        'import "./Storage.sol"; contract Token is Storage {}',
        encoding="utf-8",
    )

    bundle = read_source_bundle(str(contract))

    assert bundle.files == 2
    assert bundle.unresolved_imports == 0
    assert "mapping(address => uint256) balances" in bundle.source


def test_frame_augmentation_is_isolated_and_numeric(tmp_path):
    source = tmp_path / "C.sol"
    source.write_text(
        "contract C { function safe() external {} }",
        encoding="utf-8",
    )
    original = pd.DataFrame(
        {"contract_id": ["c"], "local_source_path": [str(source)]}
    )

    augmented = add_relational_reentrancy_features(original)

    assert list(original.columns) == ["contract_id", "local_source_path"]
    assert set(RELATIONAL_FEATURE_COLUMNS).issubset(augmented.columns)
    assert all(
        pd.api.types.is_numeric_dtype(augmented[column])
        for column in RELATIONAL_FEATURE_COLUMNS
    )


def test_detects_modern_receive_entrypoint_and_payable_parameter_call():
    source = """
    contract Receiver {
        mapping(address => uint256) balances;

        receive() external payable {
            payable(msg.sender).call{value: balances[msg.sender]}("");
            balances[msg.sender] = 0;
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_public_external_entrypoints"] == 1
    assert features["rel_n_untrusted_value_calls"] == 1
    assert features["rel_n_unguarded_untrusted_value_then_state_write"] == 1


def test_distinguishes_storage_target_from_parameter_target():
    source = """
    interface Oracle { function poke() external; }
    contract Consumer {
        Oracle public oracle;

        function update() external {
            oracle.poke();
        }
    }
    """

    features = extract_relational_reentrancy_features(source)

    assert features["rel_n_external_calls"] == 1
    assert features["rel_n_storage_target_calls"] == 1
    assert features["rel_n_untrusted_target_calls"] == 0
