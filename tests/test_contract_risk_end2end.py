import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _addr(hex_nibble: str) -> str:
    # 0x + 40 hex chars
    return "0x" + (hex_nibble * 40)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class TestContractRiskEndToEnd(unittest.TestCase):
    def test_contract_risk_score_end2end_over_mock_contracts(self) -> None:
        # Ensure repo root is on sys.path so `import src...` works when tests are run from elsewhere.
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from src.risk import contract_risk as cr  # local import so we can patch module globals
        from src.risk.contract_risk import ContractRiskUtil

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            data_dir = tmp / "data"
            sources_dir = data_dir / "etherscan_sources"
            activity_dir = data_dir / "etherscan_activity"
            derived_dir = data_dir / "derived"

            risky = _addr("1")
            safe = _addr("2")

            # --- registry (address universe) ---
            registry_csv = tmp / "contract_registry.csv"
            pd.DataFrame(
                [
                    {"address": risky, "sources": "unit_test"},
                    {"address": safe, "sources": "unit_test"},
                ]
            ).to_csv(registry_csv, index=False)

            # --- mock Solidity sources ---
            risky_source = """
            // intentionally risky patterns for the test
            pragma solidity ^0.8.20;

            contract Risky {
                address public owner;
                modifier onlyOwner() { require(msg.sender == owner); _; }

                function boom(address target, bytes calldata data) external payable onlyOwner {
                    // interaction + dangerous patterns
                    (bool ok,) = target.call{value: 1}(data);
                    require(ok);
                    target.delegatecall(data);
                    if (tx.origin == msg.sender) { }
                    assembly { let x := 1 }
                    unchecked { uint x = 1; x++; }
                }

                function bye() external onlyOwner {
                    selfdestruct(payable(msg.sender));
                }
            }
            """

            safe_source = """
            pragma solidity ^0.8.20;

            // "Safe-ish" patterns for scoring contrast in the test
            contract Safer {
                // signals used by the heuristic scorer
                function pause() external { }
                function unpause() external { }
                function timelock() external pure returns (string memory) { return "TimelockController"; }
                function guard() external pure returns (string memory) { return "ReentrancyGuard nonReentrant"; }
                function ownable2step() external pure returns (string memory) { return "Ownable2Step"; }
            }
            """

            # --- mock Etherscan source payloads (shape expected by ContractRiskUtil.extract_contract_features) ---
            risky_payload = {
                "etherscan": {
                    "result": [
                        {
                            "CompilerVersion": "v0.8.20+commit.a1b79de6",
                            "Proxy": "1",
                            "Implementation": _addr("a"),
                        }
                    ]
                },
                "parsed_sources": {"contracts": [risky_source, "upgradeTo(address) external {}", "initializer"]},
                # minimal bytecode with DELEGATECALL (0xF4) + SELFDESTRUCT (0xFF)
                "bytecode": {"eth_getCode": {"result": "0x60006000f4ff"}},
            }

            safe_payload = {
                "etherscan": {
                    "result": [
                        {
                            "CompilerVersion": "v0.8.20+commit.a1b79de6",
                            "Proxy": "0",
                            "Implementation": "",
                        }
                    ]
                },
                "parsed_sources": {"contracts": [safe_source]},
                "bytecode": {"eth_getCode": {"result": "0x6000"}},
            }

            _write_json(sources_dir / f"{risky.lower()}.json", risky_payload)
            _write_json(sources_dir / f"{safe.lower()}.json", safe_payload)

            # --- mock Etherscan activity payloads (shape expected by ContractRiskUtil.extract_contract_features) ---
            now = int(time.time())

            def tx_row(*, frm: str, to: str, days_ago: int, is_error: str = "0", value_wei: str = "0") -> dict:
                return {
                    "timeStamp": str(now - days_ago * 86400),
                    "from": frm,
                    "to": to,
                    "isError": is_error,
                    "value": value_wei,
                }

            risky_activity = {
                "txlist": {
                    "rows": [
                        tx_row(frm=_addr("b"), to=risky, days_ago=2, value_wei=str(int(0.2e18))),
                        tx_row(frm=_addr("c"), to=risky, days_ago=10, is_error="1", value_wei="0"),
                        tx_row(frm=risky, to=_addr("d"), days_ago=12, value_wei=str(int(0.1e18))),
                    ]
                },
                "txlistinternal": {"rows": [tx_row(frm=_addr("e"), to=risky, days_ago=15)]},
                "tokentx": {
                    "rows": [
                        {
                            "timeStamp": str(now - 3 * 86400),
                            "from": _addr("f"),
                            "to": risky,
                            "contractAddress": _addr("9"),
                        }
                    ]
                },
                "logs": {
                    "upgraded": {"response": {"result": [{"x": 1}, {"x": 2}]}},
                    "admin_changed": {"response": {"result": [{"x": 1}]}},
                    "ownership_transferred": {"response": {"result": [{"x": 1}]}},
                    "role_granted": {"response": {"result": [{"x": 1}, {"x": 2}, {"x": 3}]}},
                    "role_revoked": {"response": {"result": []}},
                    "paused": {"response": {"result": []}},
                    "unpaused": {"response": {"result": []}},
                },
            }

            safe_activity = {
                "txlist": {"rows": [tx_row(frm=_addr("b"), to=safe, days_ago=20, value_wei="0")]},
                "txlistinternal": {"rows": []},
                "tokentx": {"rows": []},
                "logs": {
                    "upgraded": {"response": {"result": []}},
                    "admin_changed": {"response": {"result": []}},
                    "ownership_transferred": {"response": {"result": []}},
                    "role_granted": {"response": {"result": []}},
                    "role_revoked": {"response": {"result": []}},
                    "paused": {"response": {"result": []}},
                    "unpaused": {"response": {"result": []}},
                },
            }

            _write_json(activity_dir / f"{risky.lower()}.json", risky_activity)
            _write_json(activity_dir / f"{safe.lower()}.json", safe_activity)

            out_csv = tmp / "contract_risk_out.csv"

            # Patch module-level directories so the scorer reads our temp payloads.
            with patch.object(cr, "ETHERSCAN_SOURCES_DIR", sources_dir), patch.object(cr, "ETHERSCAN_ACTIVITY_DIR", activity_dir), patch.object(
                cr, "DERIVED_DIR", derived_dir
            ):
                ContractRiskUtil().run(registry_csv=registry_csv, out_csv=out_csv, limit=0)

            self.assertTrue(out_csv.exists(), "Expected scorer to write output CSV")
            df = pd.read_csv(out_csv)
            self.assertGreaterEqual(len(df), 2)
            self.assertIn("risk_raw", df.columns)
            self.assertIn("risk_z", df.columns)
            self.assertIn("comp_dangerous", df.columns)
            self.assertIn("comp_activity", df.columns)

            df["address"] = df["address"].astype(str).str.lower()
            risky_row = df[df["address"] == risky.lower()].iloc[0]
            safe_row = df[df["address"] == safe.lower()].iloc[0]

            # Sanity: risky contract should score higher than safe contract in this fixture.
            self.assertGreater(float(risky_row["risk_raw"]), float(safe_row["risk_raw"]))
            self.assertGreater(float(risky_row["comp_dangerous"]), float(safe_row["comp_dangerous"]))
            self.assertGreaterEqual(float(risky_row["comp_activity"]), float(safe_row["comp_activity"]))

