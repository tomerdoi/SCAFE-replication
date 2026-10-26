# SCAFE submission reproduction

Run commands from the repository root after:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/rebuild_sanitized_unified_eval.py --workers 8
```

All submission-critical full-corpus experiments use
`data/derived/sanitized_v1/unified_contract_eval_scored.csv`.

## Primary development and LODO

```bash
python3 scripts/run_scafe_within_corpus_cv.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output data/derived/sanitized_v1/ml_results/primary_development \
  --datasets DAppSCAN --models rf et hgb xgb \
  --seeds 42 43 44 45 46 --folds 5

python3 scripts/run_scafe_repeated_lodo.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/ml_results/primary_lodo \
  --datasets DAppSCAN DIVE Empirical-Analysis \
  --models et --seeds 42 43 44 45 46

python3 scripts/bootstrap_primary_lodo.py \
  --predictions data/derived/sanitized_v1/ml_results/primary_lodo/scafe_repeated_lodo_predictions.csv \
  --output data/derived/sanitized_v1/ml_results/primary_lodo/primary_record_bootstrap.csv
```

## Controls and diagnostics

```bash
python3 scripts/run_scafe_nuisance_baseline.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/ml_results/nuisance_baseline \
  --model et --seeds 42 43 44 45 46

python3 scripts/run_scafe_feature_ablations.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/ml_results/feature_ablations \
  --models et --seed 42

python3 scripts/run_scafe_repeated_lodo.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/ml_results/lexical_extension \
  --datasets DAppSCAN DIVE Empirical-Analysis \
  --models et --seeds 42 43 44 45 46 --include-semantic

python3 scripts/run_scafe_label_sensitivity.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/ml_results/label_sensitivity \
  --models et --seed 42

python3 scripts/run_fold_consistent_shap.py \
  --input data/derived/sanitized_v1/unified_contract_eval_scored.csv \
  --output-dir data/derived/sanitized_v1/shap_results/primary \
  --paper-figures ../sc_risk_score_overleaf/figures --model et
```

## Frozen sampled comparisons

The neural commands are resumable: checkpoints are accepted only when model
configuration, sanitization version, identities, labels, and split hashes
match.

```bash
python3 scripts/build_sampled_experiment_campaign.py \
  --dataset DAppSCAN --sample-size 1500 --seed 42 --folds 5

python3 scripts/run_scafe_within_corpus_cv.py \
  --input data/derived/sanitized_v1/sampled_campaign_v1/scored_sample.csv \
  --output data/derived/sanitized_v1/ml_results/sampled_campaign_v1/scafe_et \
  --datasets DAppSCAN --models et --seeds 42 --folds 5

python3 scripts/run_dl_within_corpus_cv.py \
  --input data/derived/sanitized_v1/sampled_campaign_v1/scored_sample.csv \
  --datasets DAppSCAN --models textcnn bilstm codebert unixcoder \
  --seeds 42 --folds 5 --epochs 10 --patience 3 \
  --batch-size 16 --device mps --resume

python3 scripts/build_sampled_lodo_campaign.py \
  --datasets DAppSCAN DIVE Empirical-Analysis \
  --sample-size 500 --seed 42

python3 scripts/run_scafe_repeated_lodo.py \
  --input data/derived/sanitized_v1/sampled_lodo_campaign_v1/scored_sample.csv \
  --output-dir data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/scafe_et \
  --datasets DAppSCAN DIVE Empirical-Analysis --models et --seeds 42

python3 -m src.ml.fair_baseline_eval \
  --input-csv data/derived/sanitized_v1/sampled_lodo_campaign_v1/scored_sample.csv \
  --datasets DAppSCAN,DIVE,Empirical-Analysis \
  --models textcnn,bilstm,codebert,unixcoder \
  --seeds 42 --epochs 10 --patience 3 --batch-size 16 \
  --device mps --resume \
  --output-csv data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/predictions.csv \
  --metrics-csv data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/metrics.csv \
  --metadata-json data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/protocol.json \
  --checkpoint-dir data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/checkpoints

python3 scripts/run_sampled_prediction_ensembles_v2.py develop \
  --within-scafe data/derived/sanitized_v1/ml_results/sampled_campaign_v1/scafe_et/scafe_within_corpus_predictions.csv \
  --within-dl data/derived/sanitized_v1/ml_results/sampled_campaign_v1/dl/dl_within_corpus_oof_predictions.csv \
  --output-dir data/derived/sanitized_v1/ml_results/sampled_ensemble \
  --scafe-model et

python3 scripts/run_sampled_prediction_ensembles_v2.py evaluate \
  --lock data/derived/sanitized_v1/ml_results/sampled_ensemble/ensemble_lock.json \
  --external-scafe data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/scafe_et/scafe_repeated_lodo_predictions.csv \
  --external-dl data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/predictions.csv \
  --output-dir data/derived/sanitized_v1/ml_results/sampled_ensemble
```

## Validation

```bash
python3 scripts/validate_experiment_integrity.py
python3 scripts/validate_submission_artifacts.py
python3 -m pytest -q
```
