## Contract Risk Score (pivoted project)

This repo now supports a **contract-level technical risk score** pipeline (with the existing AVS work kept as an optional case study).

### Recompute the unified dataset scores (cache-only, no refetch)
If you change the risk-score formula (e.g. in `src/risk/contract_risk.py`) and want to **recalculate all unified scores using the cached contract payloads**, do:

```bash
rm -f data/derived/unified_contract_eval_contract_risk.csv
python -m src.evaluation.unified_contract_eval_risk_score_eval
```

This will re-score using the local caches under:
- `data/etherscan_sources/` (source + bytecode payloads)
- `data/etherscan_activity/` (activity payloads, if fetched)

Important: ensure the `unified_contract_eval_risk_score_eval.py` run is configured with fetching disabled (cache-only), otherwise it may call Etherscan again.

### What the contract risk score is (and isn’t)
- **Is**: a reproducible, interpretable **technical risk** score for smart contracts (upgrade/admin surface, dangerous patterns, complexity proxies, etc.).
- **Is not**: a credit risk / default probability model (those require protocol-specific user/position labels).

### Quickstart (using your cached sources)
If you already have cached Etherscan payloads under `data/etherscan_sources/` (this repo does), you can run:

```bash
python -m src.pipelines.main_contract_risk_pipeline
```

Outputs:
- `data/contract_registry.csv` (address universe to score)
- `data/derived/contract_risk.csv` (one row per contract)
- `data/derived/contract_risk_top50.csv` (top 50 convenience export)
- `data/derived/contract_risk_ablation.csv` + `data/derived/contract_risk_report.json` (evaluation artifacts)
- `figures/contract_risk_distribution.png` + `figures/contract_risk_components_heatmap.png` (plots)

### Public incidents dataset (recent incidents from the web)
This repo includes a best-effort collector that builds a **reproducible CSV** from public sources like DeFiLlama’s hacks dataset and REKT writeups.

General “web incidents” preset:

```bash
python -m src.datasets.incidents_public_dataset --general-web --since-days 365
```

Outputs:
- `data/derived/incidents_web.csv` (filtered/hygienic general incidents dataset)
- `data/derived/incidents_web_all.csv` (raw/unfiltered union of sources)
- `data/derived/incidents_raw/*.json` (raw snapshots, if enabled)

### Fetch/update sources from Etherscan (optional)
If you want to refresh or expand coverage, set `ETHERSCAN_API_KEY` (via `secrets.json` or env) and run:

```bash
python -m src.clients.etherscan_source --addresses-csv data/contract_registry.csv --fetch-all --skip-existing
```

### Optional: labels for AUC/AP evaluation
If you provide a small labeled subset at `data/labels/contract_labels.csv` (schema: `address,label`), the evaluation step will compute AUC/AP for baselines and ablations.

### Legacy / AVS pipeline
The older AVS risk pipeline is still available (and can reuse the contract risk output as a component), but it is no longer the “main” entrypoint.

AVS case study code/data now lives under:
- `src/case_studies/avs/`
- `data/case_studies/avs/`
- `figures/case_studies/avs/`

You can also run the AVS pipeline via the new module path:

```bash
python -m src.case_studies.avs.pipelines.main_pipeline
```


