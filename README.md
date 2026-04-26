# SCAFE: Compilation-Free Smart Contract Vulnerability Detection via Structural and Semantic Feature Engineering

This repository contains the code and precomputed results for the paper:

> **SCAFE: Compilation-Free Smart Contract Vulnerability Detection via Structural and Semantic Feature Engineering**

## Overview

SCAFE is a compilation-free framework for contract-level vulnerability classification that operates directly on Solidity source text. It combines two complementary feature streams:

- **Structural stream (306 features):** Code complexity, Halstead metrics, call-graph topology, data-flow indicators, vulnerability patterns, and security-practice signals — all extracted via regex-based parsing and a lightweight non-compiling AST parser.
- **Semantic stream (123 features):** Identifier-name profiling over curated risk/safety/DeFi vocabularies, plus 50 latent code topics via TF-IDF + Truncated SVD.

The full pipeline requires **no Solidity compiler, no bytecode, no GPU, and no on-chain deployment**, and applies uniformly to contracts spanning Solidity 0.4.x through 0.8.x.

## Repository Structure

```
SCAFE/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Core source code
│   ├── paths.py              # Centralized path definitions
│   ├── risk/
│   │   ├── contract_risk.py          # Feature extraction (306 structural + 123 semantic)
│   │   └── contract_risk_evaluation.py  # Risk score evaluation utilities
│   ├── ml/
│   │   ├── models.py                 # ML model definitions (RF, XGBoost, GBC, etc.)
│   │   ├── features.py              # Feature matrix preparation
│   │   ├── train_pipeline.py        # Training loop with GroupKFold CV
│   │   ├── evaluation.py            # Classification metrics (ROC-AUC, PR-AUC, F1)
│   │   ├── enhanced_pipeline.py     # Enhanced ML pipeline with full feature set
│   │   ├── dl_baselines.py          # Deep learning baselines (TextCNN, BiLSTM, CodeBERT, UniXcoder)
│   │   ├── cross_dataset_eval.py    # Cross-dataset transfer evaluation
│   │   ├── semantic_features.py     # Semantic feature engineering (identifier profiling + LSA)
│   │   └── statistical_tests.py     # Welch's t-tests, Cohen's d, DeLong tests
│   ├── baselines/
│   │   ├── slither_batch.py         # Slither static-analysis baseline runner
│   │   ├── slither_docker_batch.py  # Docker-based Slither runner
│   │   └── Dockerfile               # Docker image for reproducible Slither runs
│   └── evaluation/
│       ├── cross_dataset_benchmark.py   # Cross-dataset generalization benchmarks
│       └── unified_dataset_excel.py     # Unified evaluation dataset builder
├── scripts/                  # Experiment scripts
│   ├── run_final_experiments.py         # Main ML experiments (Tables 5-6)
│   ├── run_scafe_sem_experiments.py     # Semantic stream experiments (Table 10)
│   ├── run_unixcoder.py                # UniXcoder fine-tuning (Table 7)
│   ├── run_unixcoder_ensembles.py      # Hybrid ensemble experiments (Table 8)
│   ├── verify_paper_numbers.py         # Verify all numbers reported in the paper
│   ├── ensemble_analysis.py            # Ensemble method analysis
│   ├── two_stage_pipeline_analysis.py  # Two-stage pipeline experiments (Table 9)
│   ├── error_analysis_vs_codebert.py   # Size-stratified error analysis
│   └── small_contract_analysis.py      # Small-contract analysis
├── paper/                    # Paper figure generation
│   ├── regenerate_figures.py            # Regenerate all paper figures
│   └── generate_ensemble_figure.py     # Ensemble comparison figure
├── tests/
│   └── test_contract_risk_end2end.py   # End-to-end feature extraction tests
└── results/                  # Precomputed experiment results
    ├── ml_results/           # ML model results (JSON/CSV)
    └── dl_results/           # Deep learning baseline results
```

## Datasets

The experiments use three publicly available datasets (not included due to size; download from original sources):

| Dataset | Contracts | Source |
|---------|-----------|--------|
| **DAppSCAN** | 21,457 | [github.com/InPlusLab/DAppSCAN](https://github.com/InPlusLab/DAppSCAN) |
| **FORGE** | 4,872 | [github.com/forge-artifacts](https://github.com/AstroShen/FORGE-Artifacts) |
| **AccessControlVulnerabilities** | 94 | [github.com/AChecker](https://github.com/AChecker/AccessControlVulnerabilities) |

After downloading, place the datasets under `data/raw/`:
```
data/raw/
├── DAppSCAN-main/
├── FORGE-Artifacts-main/
└── AccessControlVulnerabilities-main/
```

## Setup

### Prerequisites
- Python 3.10+
- (Optional) GPU with PyTorch support for DL baselines

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For deep learning baselines, install PyTorch with GPU support appropriate for your hardware:
```bash
# CUDA (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon) — included in default PyTorch install
pip install torch
```

## Running Experiments

### 1. Feature Extraction

The core feature extraction is in `src/risk/contract_risk.py`. It processes raw Solidity files and produces a feature vector per contract:

```python
from src.risk.contract_risk import ContractRiskScorer

scorer = ContractRiskScorer()
features = scorer.score_contract(solidity_source_code)
```

### 2. Reproducing Paper Results

**ML experiments (Tables 5-6):** Train and evaluate tree-based classifiers with the dual-stream features.
```bash
python -m scripts.run_final_experiments
```

**Semantic stream experiments (Table 10):** Ablation of identifier profiling and LSA components.
```bash
python -m scripts.run_scafe_sem_experiments
```

**UniXcoder fine-tuning (Table 7):** Fine-tune the UniXcoder pre-trained model. Requires GPU.
```bash
python -m scripts.run_unixcoder
```

**Hybrid ensemble experiments (Table 8):** Late-fusion ensemble of SCAFE + UniXcoder.
```bash
python -m scripts.run_unixcoder_ensembles
```

**Two-stage pipeline (Table 9):** SCAFE pre-filter + UniXcoder on flagged contracts.
```bash
python -m scripts.two_stage_pipeline_analysis
```

**Verify paper numbers:** Cross-check all reported metrics against precomputed results.
```bash
python -m scripts.verify_paper_numbers
```

### 3. Regenerating Figures

```bash
python paper/regenerate_figures.py
```

### 4. Slither Baseline

The Slither baseline requires Docker:
```bash
cd src/baselines
docker build -t slither-runner .
python -m src.baselines.slither_docker_batch
```

## Precomputed Results

The `results/` directory contains precomputed experiment outputs in JSON/CSV format, so reviewers can inspect the reported numbers without re-running experiments:

- `results/ml_results/final_results.json` — Structural-stream ML results
- `results/ml_results/scafe_sem_results.json` — Dual-stream (structural + semantic) results
- `results/ml_results/unixcoder_results.json` — UniXcoder fine-tuning results
- `results/ml_results/ensemble_unixcoder_results.json` — Hybrid ensemble results
- `results/dl_results/dl_baseline_results.json` — TextCNN, BiLSTM, CodeBERT results
- `results/dl_results/statistical_tests.json` — Welch's t-tests and Cohen's d effect sizes

## Key Results Summary

On merged training data (DAppSCAN + FORGE, 5-fold GroupKFold CV):

| Method | ROC-AUC | PR-AUC | F1 | GPU |
|--------|---------|--------|-----|-----|
| SCAFE Vote Ensemble | 0.932 | 0.589 | 0.611 | No |
| TextCNN | 0.928 | 0.499 | 0.601 | Yes |
| BiLSTM | 0.928 | 0.520 | 0.599 | Yes |
| CodeBERT | 0.944 | 0.621 | 0.664 | Yes |
| UniXcoder | 0.953 | 0.645 | 0.679 | Yes |
| SCAFE + UniXcoder (hybrid) | 0.959 | 0.662 | 0.683 | Yes |

## License

This code is provided for research purposes in support of the accompanying paper.
