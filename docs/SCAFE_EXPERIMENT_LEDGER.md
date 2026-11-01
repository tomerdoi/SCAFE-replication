# SCAFE experiment ledger

## 2026-11-01 — Function-model annotation boundary audit

Status: active. Exclusive lock: `/tmp/scafe-research-iteration.lock`.
Baseline: replication `4fd7f45`, manuscript `7e4cd05`; both initially clean.
Process inventory found no matching SCAFE research jobs before lock acquisition.
No training, target selection, manuscript edits, commit, push, installation, or upload is authorized by this run.

Hypothesis fixed before evaluation: the function pipeline's raw reader may expose
DAppSCAN SWC annotation comments despite a sanitized registry path. Reusing the
canonical dataset-aware sanitizer removes this direct label channel from model
inputs while preserving intended Solidity code. This is an integrity improvement,
not a hypothesis of increased detector accuracy.

Success criteria:

1. Audit all DAppSCAN registry rows deterministically, examining extracted functions
   only when canonical sanitization changes their source. Do not fit a model or
   read external target sources/labels for selection.
2. Route scoring and overlap filtering through canonical sanitization, with
   dataset-aware caches and recorded sanitization version.
3. Reject legacy/unversioned aggregate evaluation and mismatched aggregate hashes.
4. Pass meaningful regression tests and existing integrity/submission validators.

Primary evidence will be saved to
`data/derived/sanitized_v1/ml_results/function_source_sanitization_audit_20261101/`.
Original source, checkpoints, prior scores, and manuscript remain intact.
The previously inspected DIVE/Empirical sets are not fresh confirmation data.

Validation commands planned: focused function/sanitizer tests, then applicable
broader tests if needed; `validate_experiment_integrity.py` and
`validate_submission_artifacts.py`. No manuscript compilation is needed because
this iteration does not change the manuscript.

### Completed result

The predeclared hypothesis was confirmed at the extracted-text boundary.
All 21,457 DAppSCAN registry rows were audited in 16.738 seconds; 21,455
were readable/nonempty and two were empty. Canonical sanitization changed
948 source records. Among those records, 18,069 function occurrences were
extracted before and after sanitization, with identical function-name sequences.
493 function occurrences in 325 records exposed removable annotation comments
under the legacy reader; zero did after sanitization. All 493 changed function
texts retained the same comment/whitespace-normalized identities. Of the 14,791
unique primary Extra Trees development records, 298 have affected function input.
Counts denote records/function occurrences, not independent projects.

The audit does not establish how many annotations survived the tokenizer's
512-token window or how much they affected predictions. No model was loaded by
the audit, no accuracy experiment was run, and no external target was used for
new selection. The prior function-level CodeBERT fusion remains exploratory and
requires sanitized rescoring plus repeated development-only selection before
supporting new conclusions; prior scores and checkpoints were preserved.
This finding does not invalidate the separately sanitized primary SCAFE tables.

Implemented local changes:

- The function scorer and overlap filter use the canonical dataset-aware source
  reader; caches/deduplication include dataset and path.
- Training overlap audit, resume configuration, training metadata and scoring
  metadata record the sanitization version.
- Scoring metadata records the aggregate SHA-256. Evaluation rejects missing,
  legacy, incompatible or hash-mismatched metadata before reading scores.
  Actual historical aggregates were verified to be rejected as expected.
- Added a CPU-only reproducible exposure audit and focused regression tests.

Validation completed:

- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -p 'test_function*.py' -v`: **9 passed**, including four rejection subcases,
  directory reading, dataset-specific overlap filtering/caching, and a stub-model
  integration check that observes the text sent to tokenization.
- All **9 existing source-sanitization test functions passed** through
  `unittest.FunctionTestCase`, using temporary directories for their `tmp_path`
  fixtures. No test expectations or source-sanitization implementation changed.
- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/validate_experiment_integrity.py`: passed.
- `PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/validate_submission_artifacts.py`: passed.
- AST syntax checks passed for all four changed/new Python files;
  `git diff --check` passed.
- Initial attempted pytest command failed with `No module named pytest`.
  Global and both project Python environments were checked; none had pytest.
  Nothing was installed. The full pytest suite remains **not run**.
- Manuscript was not changed; no fresh manuscript build was attempted.

Reproduce audit into a new, nonexisting directory:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/audit_function_source_sanitization.py \
  --output-dir data/derived/sanitized_v1/ml_results/function_source_sanitization_audit_NEW
```

Evidence: the dated audit directory contains `summary.json` with code/input
hashes and `record_audit.csv`. The sanitized registry SHA-256 is
`9bd0bfeb51b46df80a602eb334e7ebdcaf14d5777ce69ccce76a223e5283cfcb`.

Next bounded step: quantify actual tokenizer-window annotation exposure with the
saved local tokenizer, then plan sanitized development rescoring in a new output
directory. Verify checkpoint and score lineage before attempting incremental
reuse; do not relabel old raw scores as sanitized. Do not promote historical
external improvements as fresh confirmation. Context extraction, hard negatives,
near-clone/template grouping and provenance remain unresolved research issues.

Outcome: useful integrity improvement; **no demonstrated predictive gain**.
No commit/push approval is requested for this iteration. No commit, push, PR,
publication, training, software installation, paid compute, or private upload.
