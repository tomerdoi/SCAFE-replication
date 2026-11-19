from pathlib import Path

import pytest

from src.evidence.solidity_modifier_resolution import (
    replay_certificate,
    SourceRepository,
    parse_source,
    resolve_modifier,
    resolve_modifier_file_only,
    resolve_modifier_strong_traversal,
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _offset(text: str, needle: str) -> int:
    return text.index(needle)


def test_local_modifier_answer_and_certificate(tmp_path):
    source = """
    contract Vault {
        modifier ownerOnly() { _; }
        function withdraw() external ownerOnly {}
    }
    """
    _write(tmp_path, "Vault.sol", source)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(source, "ownerOnly {}"), "ownerOnly"
    )
    assert result.status == "answered"
    assert result.source_path == "Vault.sol"
    assert result.declaration_start == source.index("modifier ownerOnly")
    assert result.files_read == ("Vault.sol",)
    assert result.bytes_read == len(source.encode())
    assert len(result.source_sha256[0][1]) == 64
    assert replay_certificate(tmp_path, result)


def test_explicit_alias_inherited_modifier_is_resolved_selectively(tmp_path):
    base = "contract Authority { modifier authorized() { _; } }\n"
    child = """
    import {Authority as Auth} from "./Authority.sol";
    import "./Unrelated.sol";
    contract Vault is Auth {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "Authority.sol", base)
    _write(tmp_path, "Unrelated.sol", "contract Noise {}\n")
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "authorized {}"), "authorized"
    )
    assert result.status == "answered"
    assert result.source_path == "Authority.sol"
    assert result.files_read == ("Authority.sol", "Vault.sol")
    assert "Unrelated.sol" not in result.files_read
    strong = resolve_modifier_strong_traversal(
        tmp_path, "Vault.sol", _offset(child, "authorized {}"), "authorized"
    )
    assert strong.status == "answered"
    assert strong.source_path == result.source_path
    assert strong.files_read == ("Authority.sol", "Unrelated.sol", "Vault.sol")
    assert strong.bytes_read > result.bytes_read


def test_namespace_inheritance_is_supported(tmp_path):
    base = "contract Authority { modifier authorized() { _; } }\n"
    child = """
    import * as Access from "./Authority.sol";
    contract Vault is Access.Authority {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "Authority.sol", base)
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "authorized {}"), "authorized"
    )
    assert result.status == "answered"
    assert result.source_path == "Authority.sol"


def test_explicit_import_can_resolve_through_reexport_file(tmp_path):
    _write(
        tmp_path,
        "Authority.sol",
        "contract Authority { modifier authorized() { _; } }\n",
    )
    _write(tmp_path, "Deprecated.sol", 'import "./Authority.sol";\n')
    child = """
    import {Authority} from "./Deprecated.sol";
    contract Vault is Authority {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "authorized {}"), "authorized"
    )
    assert result.status == "answered"
    assert result.source_path == "Authority.sol"


def test_duplicate_paths_to_same_reexported_base_are_not_ambiguous(tmp_path):
    _write(
        tmp_path,
        "Authority.sol",
        "contract Authority { modifier authorized() { _; } }\n",
    )
    _write(tmp_path, "Reexport.sol", 'import "./Authority.sol";\n')
    middle = """
    import "./Authority.sol";
    import "./Reexport.sol";
    contract Middle is Authority {}
    """
    _write(tmp_path, "Middle.sol", middle)
    child = """
    import "./Middle.sol";
    contract Vault is Middle {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "authorized {}"), "authorized"
    )
    assert result.status == "answered"
    assert result.source_path == "Authority.sol"


def test_missing_import_fails_closed(tmp_path):
    source = """
    import {Authority} from "./Missing.sol";
    contract Vault is Authority {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "Vault.sol", source)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(source, "authorized {}"), "authorized"
    )
    assert result.status == "unknown"
    assert result.reason == "inheritance_resolution_failed"


def test_duplicate_same_contract_modifiers_fail_closed(tmp_path):
    source = """
    contract Vault {
        modifier guard() { _; }
        modifier guard(uint256 value) { _; }
        function withdraw() external guard {}
    }
    """
    _write(tmp_path, "Vault.sol", source)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(source, "guard {}"), "guard"
    )
    assert result.status == "unknown"
    assert result.reason == "ambiguous_modifier_declarations"


def test_derived_modifier_precedes_base(tmp_path):
    base = "contract Base { modifier guard() { _; } }\n"
    child = """
    import "./Base.sol";
    contract Vault is Base {
        modifier guard() override { _; }
        function withdraw() external guard {}
    }
    """
    _write(tmp_path, "Base.sol", base)
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "guard {}"), "guard"
    )
    assert result.status == "answered"
    assert result.source_path == "Vault.sol"


def test_unique_modifier_in_diamond_is_resolved_without_guessing_order(tmp_path):
    _write(tmp_path, "Root.sol", "contract Root { modifier guard() { _; } }\n")
    _write(
        tmp_path,
        "Left.sol",
        'import "./Root.sol"; contract Left is Root {}\n',
    )
    _write(
        tmp_path,
        "Right.sol",
        'import "./Root.sol"; contract Right is Root {}\n',
    )
    child = """
    import "./Left.sol";
    import "./Right.sol";
    contract Vault is Left, Right { function f() external guard {} }
    """
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "guard {}"), "guard"
    )
    assert result.status == "answered"
    assert result.source_path == "Root.sol"


def test_multiple_inherited_declarations_fail_closed(tmp_path):
    _write(tmp_path, "Left.sol", "contract Left { modifier guard() { _; } }\n")
    _write(tmp_path, "Right.sol", "contract Right { modifier guard() { _; } }\n")
    child = """
    import "./Left.sol";
    import "./Right.sol";
    contract Vault is Left, Right { function f() external guard {} }
    """
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(child, "guard {}"), "guard"
    )
    assert result.status == "unknown"
    assert result.reason == "ambiguous_inherited_modifier_declarations"


def test_remapping_is_constrained_to_repository(tmp_path):
    base = "contract Authority { modifier authorized() { _; } }\n"
    child = """
    import {Authority} from "@pkg/Authority.sol";
    contract Vault is Authority {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "vendor/pkg/Authority.sol", base)
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path,
        "Vault.sol",
        _offset(child, "authorized {}"),
        "authorized",
        remappings=(("@pkg/", "vendor/pkg/"),),
    )
    assert result.status == "answered"
    assert result.source_path == "vendor/pkg/Authority.sol"


def test_foundry_remapping_without_trailing_slash(tmp_path):
    base = "contract Authority { modifier authorized() { _; } }\n"
    child = """
    import {Authority} from "pkg/Authority.sol";
    contract Vault is Authority {
        function withdraw() external authorized {}
    }
    """
    _write(tmp_path, "vendor/pkg/Authority.sol", base)
    _write(tmp_path, "Vault.sol", child)
    result = resolve_modifier(
        tmp_path,
        "Vault.sol",
        _offset(child, "authorized {}"),
        "authorized",
        remappings=(("pkg", "vendor/pkg"),),
    )
    assert result.status == "answered"
    assert result.source_path == "vendor/pkg/Authority.sol"


def test_path_escape_and_symlink_escape_are_rejected(tmp_path):
    outside = tmp_path.parent / "outside.sol"
    outside.write_text("contract Outside {}")
    with pytest.raises(ValueError):
        SourceRepository(tmp_path).read("../outside.sol")
    (tmp_path / "link.sol").symlink_to(outside)
    with pytest.raises(ValueError):
        SourceRepository(tmp_path).read("link.sol")


def test_comments_and_strings_do_not_create_false_declarations():
    parsed = parse_source(
        '''
        // contract Fake { modifier fake() { _; } }
        contract Real {
            string constant TEXT = "modifier fake() { _; }";
            modifier real() { _; }
        }
        '''
    )
    assert [contract.name for contract in parsed.contracts] == ["Real"]
    assert [modifier.name for modifier in parsed.contracts[0].modifiers] == ["real"]


def test_invalid_name_and_outside_contract_are_unknown(tmp_path):
    source = "contract Vault { modifier guard() { _; } }\n"
    _write(tmp_path, "Vault.sol", source)
    invalid = resolve_modifier(tmp_path, "Vault.sol", 10, "guard()")
    assert invalid.status == "unknown"
    assert invalid.reason == "invalid_modifier_name"
    outside = resolve_modifier(tmp_path, "Vault.sol", 0, "guard")
    assert outside.status == "unknown"
    assert outside.reason == "query_contract_not_unique"


def test_file_only_baselines_keep_wrong_answer_observable(tmp_path):
    source = """
    contract Noise { modifier guard() { _; } }
    contract Vault { function withdraw() external guard {} }
    """
    _write(tmp_path, "Vault.sol", source)
    offset = _offset(source, "guard {}")
    naive = resolve_modifier_file_only(
        tmp_path,
        "Vault.sol",
        offset,
        "guard",
        same_contract_only=False,
    )
    sound = resolve_modifier_file_only(
        tmp_path,
        "Vault.sol",
        offset,
        "guard",
        same_contract_only=True,
    )
    assert naive.status == "answered"
    assert naive.source_path == "Vault.sol"
    assert sound.status == "unknown"


def test_certificate_replay_detects_source_change(tmp_path):
    source = "contract Vault { modifier guard() { _; } function f() guard {} }\n"
    _write(tmp_path, "Vault.sol", source)
    result = resolve_modifier(
        tmp_path, "Vault.sol", _offset(source, "guard {}"), "guard"
    )
    assert replay_certificate(tmp_path, result)
    _write(tmp_path, "Vault.sol", source + "\n")
    assert not replay_certificate(tmp_path, result)


def test_compiler_byte_offsets_are_preserved_with_unicode_comments(tmp_path):
    source = """
    // ──────────────────────────────────────────────
    contract Vault {
        modifier guard() { _; }
        function f() external guard {}
    }
    """
    _write(tmp_path, "Vault.sol", source)
    invocation = len(source[: source.index("guard {}")].encode("utf-8"))
    declaration = len(source[: source.index("modifier guard")].encode("utf-8"))
    result = resolve_modifier(tmp_path, "Vault.sol", invocation, "guard")
    assert result.status == "answered"
    assert result.declaration_start == declaration
    assert replay_certificate(tmp_path, result)


def test_incomplete_strong_import_closure_fails_closed(tmp_path):
    source = """
    import "./Missing.sol";
    contract Vault {
        modifier guard() { _; }
        function f() external guard {}
    }
    """
    _write(tmp_path, "Vault.sol", source)
    result = resolve_modifier_strong_traversal(
        tmp_path, "Vault.sol", _offset(source, "guard {}"), "guard"
    )
    assert result.status == "unknown"
    assert result.reason == "import_closure_incomplete"
