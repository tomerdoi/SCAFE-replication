import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class TestUnifiedContractEvalLocalSource(unittest.TestCase):
    def test_local_source_takes_precedence_over_address_cache(self) -> None:
        from src.evaluation import unified_contract_eval_risk_score_eval as evaluator_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "sample.sol"
            source_path.write_text(
                "pragma solidity ^0.8.20;\n"
                "contract Sample { function f() external {} }\n",
                encoding="utf-8",
            )
            input_xlsx = root / "input.xlsx"
            output_csv = root / "scored.csv"
            pd.DataFrame(
                [
                    {
                        "address": "0x" + "1" * 40,
                        "contract_id": "local:sample",
                        "local_source_path": str(source_path),
                        "source_dataset": "Empirical-Analysis",
                        "chain": "eth",
                        "label_status": "vulnerable",
                        "vuln_present": 1,
                    }
                ]
            ).to_excel(input_xlsx, index=False, sheet_name="contracts")

            with patch.object(evaluator_module, "DERIVED_DIR", root / "derived"):
                evaluator = evaluator_module.UnifiedContractEvalRiskScoreEvaluation()
                evaluator.run(
                    in_xlsx=input_xlsx,
                    out_scored_csv=output_csv,
                    resume=False,
                    fetch_sources=False,
                    fetch_activity=False,
                )

            scored = pd.read_csv(output_csv)
            self.assertGreater(int(scored.loc[0, "loc"]), 0)
            self.assertGreater(int(scored.loc[0, "n_total_tokens"]), 0)
            self.assertEqual(int(scored.loc[0, "has_verified_source"]), 1)


if __name__ == "__main__":
    unittest.main()
