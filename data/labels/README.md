## Optional labels for evaluation

The contract risk score is primarily **unsupervised / rule-based**.
For Q1-quality evaluation, you often want *some* labeled subset to compute AUC/AP and compare variants.

### Create `data/labels/contract_labels.csv`
CSV schema:
- `address`: 0x… contract address (lower/upper case OK)
- `label`: 1 for “positive” (e.g., exploited / high-risk by expert), 0 otherwise

Example:

```csv
address,label
0x1111111111111111111111111111111111111111,1
0x2222222222222222222222222222222222222222,0
```

Then run:

```bash
python -m src.risk.contract_risk_evaluation
```

This will write:
- `data/derived/contract_risk_ablation.csv`
- `data/derived/contract_risk_report.json`
- plots under `figures/`


