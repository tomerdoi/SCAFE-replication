## Contract Risk Score (pivoted project)

This repo now supports a **contract-level technical risk score** pipeline (with the existing AVS work kept as an optional case study).

### What the contract risk score is (and isn’t)
- **Is**: a reproducible, interpretable **technical risk** score for smart contracts (upgrade/admin surface, dangerous patterns, complexity proxies, etc.).
- **Is not**: a credit risk / default probability model (those require protocol-specific user/position labels).

### Quickstart (using your cached sources)
If you already have cached Etherscan payloads under `data/etherscan_sources/` (this repo does), you can run:

```bash
/Users/tomer.doitshman/PycharmProjects/sc_review/venv/bin/python -m src.pipelines.main_contract_risk_pipeline
```

Outputs:
- `data/contract_registry.csv` (address universe to score)
- `data/derived/contract_risk.csv` (one row per contract)
- `data/derived/contract_risk_top50.csv` (top 50 convenience export)
- `data/derived/contract_risk_ablation.csv` + `data/derived/contract_risk_report.json` (evaluation artifacts)
- `figures/contract_risk_distribution.png` + `figures/contract_risk_components_heatmap.png` (plots)

### Fetch/update sources from Etherscan (optional)
If you want to refresh or expand coverage, set `ETHERSCAN_API_KEY` (via `secrets.json` or env) and run:

```bash
/Users/tomer.doitshman/PycharmProjects/sc_review/venv/bin/python -m src.clients.etherscan_source --addresses-csv data/contract_registry.csv --fetch-all --skip-existing
```

### Optional: labels for AUC/AP evaluation
If you provide a small labeled subset at `data/labels/contract_labels.csv` (schema: `address,label`), the evaluation step will compute AUC/AP for baselines and ablations.

### Legacy / AVS pipeline
The older AVS risk pipeline is still available (and can reuse the contract risk output as a component), but it is no longer the “main” entrypoint:

```bash
/Users/tomer.doitshman/PycharmProjects/sc_review/venv/bin/python -m src.pipelines.main_pipeline
```


