# Sampled submission experiment plan

## Goal and reporting boundary

The corrected full feature table remains the source of truth. Computationally
cheap SCAFE experiments run on all canonical records. Expensive neural
baselines use two frozen, deterministic samples: a DAppSCAN sample for
within-corpus evaluation and an equal-size-per-corpus sample for LODO. Sampled
neural results must never be reported as full-corpus estimates.

## Frozen neural sample

- Dataset: DAppSCAN only.
- Population: the 14,791 canonical, consistently labelled source identities
  produced by the corrected sanitization-v2 pipeline.
- Sample: 1,500 records, selected without replacement and stratified in
  proportion to the population labels with seed 42.
- Expected allocation: 1,413 operational negatives and 87 positives.
- Grouping: all records from the same DAppSCAN project remain in the same outer
  fold.
- Validation gate: every one of the five `GroupKFold` test folds must contain
  both labels and at least ten positives.
- The selected identities, population/sample counts, fold counts, source-table
  digest, and sanitization version are frozen in `sample_manifest.json`.

This preserves the natural DAppSCAN prevalence and the original group-disjoint
five-fold protocol while reducing neural training records by about 90%.

## Models and protocol

Run SCAFE Random Forest, TextCNN, BiLSTM, CodeBERT, and UniXcoder on the same
sample and outer folds.

- Seed: 42.
- Outer evaluation: five project-group-disjoint folds.
- Inner validation: outer-training-only and group-disjoint.
- Threshold: fixed at 0.5.
- Input: complete sanitized source, chunked and contract-aggregated.
- Neural training: existing maximum of 10 epochs and patience 3.
- Transformer policy: freeze the lower eight layers.
- No test-label model, feature, threshold, or epoch selection.

One seed estimates performance for this fixed sampled campaign; it does not
measure seed uncertainty. Uncertainty should therefore use record/project
bootstrap intervals over out-of-fold predictions, and conclusions must remain
sample-specific.

## Frozen LODO sample

- Corpora: DAppSCAN, DIVE, and Empirical-Analysis.
- Sample: 500 records from each corpus, sampled independently without
  replacement at each corpus's natural prevalence with seed 42.
- Protocol: hold out each sampled corpus in turn; fit on the other two sampled
  corpora after removing detected source overlap with the target.
- Models: SCAFE Random Forest, TextCNN, BiLSTM, CodeBERT, and UniXcoder.
- Neural settings: seed 42, maximum 10 epochs, patience 3, batch size 16,
  complete sanitized source with contract-level chunk aggregation, Apple MPS.
- Reporting boundary: these are sampled LODO estimates. They supplement, but
  do not replace, the full-corpus SCAFE LODO results.

Equal corpus budgets prevent DIVE's much larger corpus from dominating the
sampled training pool. This changes the natural corpus mixture and must be
disclosed as an equal-budget comparison.

## Full-corpus SCAFE analyses

Because tree-based experiments complete quickly, do not sample the primary
SCAFE evidence. Regenerate on all corrected canonical records:

1. Five-seed fixed-Random-Forest LODO transfer.
2. Seed-42 fixed-head feature-group ablations.
3. Sanitized identifier/context extension.
4. Derived fixed-budget, redundancy, sensitivity, SHAP, table, and figure
   artifacts required by the manuscript.

## Artifact isolation and recovery

Sample definition:

```text
data/derived/sanitized_v1/sampled_campaign_v1/
```

Neural and matched SCAFE outputs:

```text
data/derived/sanitized_v1/ml_results/sampled_campaign_v1/
```

The DL evaluator writes:

- a training checkpoint during every fold;
- an immutable completion record and predictions for each completed unit;
- merged predictions, fold metrics, summaries, and protocol metadata after the
  campaign completes.

Re-running the exact command with `--resume` accepts only checkpoints whose
configuration, labels, identities, split membership, preprocessing version,
and sanitization version match.

## Commands

Create or verify the frozen sample:

```bash
venv/bin/python -u scripts/build_sampled_experiment_campaign.py \
  --dataset DAppSCAN --sample-size 1500 --seed 42 --folds 5
```

Matched SCAFE evaluation:

```bash
venv/bin/python -u scripts/run_scafe_within_corpus_cv.py \
  --input data/derived/sanitized_v1/sampled_campaign_v1/scored_sample.csv \
  --output data/derived/sanitized_v1/ml_results/sampled_campaign_v1/scafe \
  --datasets DAppSCAN --models rf --seeds 42 --folds 5
```

Resumable neural campaign:

```bash
venv/bin/python -u scripts/run_dl_within_corpus_cv.py \
  --input data/derived/sanitized_v1/sampled_campaign_v1/scored_sample.csv \
  --output data/derived/sanitized_v1/ml_results/sampled_campaign_v1/dl \
  --datasets DAppSCAN \
  --models textcnn bilstm codebert unixcoder \
  --seeds 42 --folds 5 --epochs 10 --patience 3 \
  --batch-size 16 --device mps --resume
```

The same DL command resumes after interruption or reboot.

Create or verify the frozen multicorpus LODO sample:

```bash
venv/bin/python -u scripts/build_sampled_lodo_campaign.py \
  --datasets DAppSCAN DIVE Empirical-Analysis \
  --sample-size 500 --seed 42
```

Matched sampled SCAFE LODO:

```bash
venv/bin/python -u scripts/run_scafe_repeated_lodo.py \
  --input data/derived/sanitized_v1/sampled_lodo_campaign_v1/scored_sample.csv \
  --output-dir data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/scafe \
  --datasets DAppSCAN DIVE Empirical-Analysis --models rf --seeds 42
```

Resumable sampled neural LODO:

```bash
venv/bin/python -u -m src.ml.fair_baseline_eval \
  --input-csv data/derived/sanitized_v1/sampled_lodo_campaign_v1/scored_sample.csv \
  --output-csv data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/predictions.csv \
  --metrics-csv data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/metrics.csv \
  --metadata-json data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/protocol.json \
  --checkpoint-dir data/derived/sanitized_v1/ml_results/sampled_lodo_campaign_v1/dl/checkpoints \
  --datasets DAppSCAN,DIVE,Empirical-Analysis \
  --models textcnn,bilstm,codebert,unixcoder \
  --seeds 42 --epochs 10 --patience 3 --batch-size 16 \
  --device mps --resume
```
