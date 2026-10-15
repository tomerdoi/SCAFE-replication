# Sampled submission experiment plan

## Goal and reporting boundary

The corrected full feature table remains the source of truth. Computationally
cheap SCAFE experiments run on all canonical records. Expensive neural
baselines run on one frozen, deterministic DAppSCAN sample. Sampled neural
results must be reported as sampled within-corpus estimates, never as
full-corpus or cross-corpus results.

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
