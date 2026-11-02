# Exact local context recovery — 2 November 2026, 13:00 Israel

This iteration recovers actual source bytes for part of the function pilot without
network acquisition, new labels, or model training. It turns producer-reported
context hashes into inspectable local evidence while explicitly flagging overlap
with SCAFE's existing corpora. It does not finalize the dataset.

## Measured change

| Readiness component | Before this recovery derivative | After |
|---|---:|---:|
| Verified context source texts packaged with the pilot | 0 | 21 |
| Candidates linked to an exact pinned context text | 0 | 136 of 2,323 (5.85%) |
| Linked candidate fragment hashes/character spans checked | 0 | 136 |
| Linked candidates flagged for existing-corpus overlap | 0 explicitly in this derivative | 136 |
| Completed independent annotations | 0 | 0 |
| Training-eligible records / assigned splits | 0 / 0 | 0 / 0 |

The recovered candidates comprise 112 CoinFabrik and 24 SCRUBD records. The HF
source contributes none: these candidates have no pinned context identity to
recover by this method. The remaining 2,187 candidates have no verified context
in this derivative. The original 2,323 candidate records and all 6,969 unknown
pattern cells are unchanged. Missing project identities are not invented or filled
from audit-folder names.

Twenty-two matching local files represent 21 distinct decoded source texts. One
exact text occurs in two local audit directories; both aliases are retained.
136 candidate fragments match their original raw fragment hash at verified
character spans. For CoinFabrik the producer's pinned character offset is checked;
for SCRUBD named-function starts are searched and all matching hash/length spans
are retained. `exact_unique_fragment` means one verified candidate span under
that protocol, not global semantic uniqueness. This is not an AST, overload,
contract-scope or exploitability validation. Ambiguous or missing matches are not
guessed; the implementation preserves those states.

Four functions in the earlier 24-function review queue have recovered context
hashes. That old queue and its snippet-only rules are **unchanged**. It cannot be
silently upgraded to contextual/vulnerability judgments. A separately versioned
context-enriched review protocol and context-sufficiency assessment are needed.
A complete source file may still lack imported declarations, callers and relevant
callees, so even those four are not automatically ready for such adjudication.

## Reproduction and exact acceptance boundary

From `sc_review`, with existing Python and `rg`:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/recover_function_context.py \
  --pilot-dir /absolute/path/scafe_function_dataset_pilot \
  --raw-root /absolute/path/sc_review/data/raw \
  --output-dir /absolute/path/scafe_function_dataset_pilot/derivatives/NEW_VERSION
```

The script verifies all 18 delivered hashes and each candidate's source revision
against the pinned source manifest. It lists local Solidity paths using existing
`rg --files --null`, with default ignore rules and explicit directory exclusions,
a 40-second listing timeout and 500,000-file bound. Of 212,823 listed Solidity
paths, only the 48 files with one of the 73 candidate source basenames are read.
Each file is capped at 16 MiB, must resolve inside the authorized raw root, and
must decode as UTF-8 with optional BOM. Symlink escapes, invalid encoding and
oversized files are not accepted. Search is bounded, filename-prefiltered and
ignore-rule dependent, not an exhaustive content search of the computer.

Accept a context only if SHA-256 of its UTF-8-sig-decoded text exactly equals a
candidate's pinned context hash. Filename similarity, whitespace-normalized code,
function-name matches and source claims cannot establish context identity. Keep
both raw local-byte and decoded-text hashes; store verified context text as UTF-8
bytes named by its hash. Check all delivered originals and all 48 inventoried
local source hashes again before writing a new output directory. Existing output
directories and writes into original raw-source roots are refused. No downloaded
or contract source is executed.

The first stdlib directory walk hit its 100,000-directory budget and stopped
before creating outputs. The corrected implementation uses the bounded existing
`rg` listing instead. This resource failure and correction are recorded, rather
than counted as a completed recovery. Final recovery artifacts reproduced
byte-for-byte on a second run. Original pilot and local source hashes remained
unchanged.

## Local evidence and scientific limits

The new derivative is outside Git:

`/Users/tomer.doitshman/Desktop/D_local/func-dataset/scafe_function_dataset_pilot/derivatives/local_context_v1_20261102T1300/`

- `contexts/<sha256>.sol`: 21 verified source texts, raw evidence only.
- `context_manifest.jsonl`: source revision/path pins and every local alias/hash.
- `candidate_context_links.jsonl`: all 2,323 stable IDs, context and fragment-span
  status, original project/license fields, unknown labels and no split/promotion.
- `local_source_inventory.jsonl`: all 48 filename-matched source hash checks.
- `summary.json` and `recovery_manifest.json`: measured coverage, search protocol,
  original delivery identities and artifact hashes.

Matches come from `FORGE-Artifacts-main` and the local
`paper_benchmarks/smartbugs-wild` corpus. Every linked candidate is conservatively
marked `quarantine_existing_corpus_overlap`. Presence in a local corpus does not
prove actual historical training/test membership, but these cannot be represented
as new independent confirmation examples. Audit-directory names are provenance
hints, not verified project IDs. Context hashes provide definite file identities;
project/version/clone and SCAFE-cohort membership audits still precede frozen splits.

Raw context comments may reveal source claims or annotation cues. These files are
coordinator evidence, **not sanitized model inputs or automatically blinded review
packets**. Preserve comment/source lineage and inspect cue removal when constructing
a new review or model representation. Semantic context sufficiency remains
`not_reviewed`; source authenticity beyond the producer's pinned manifest, license
rights, complete dependencies and annotation truth are not established by matching
hashes. Original license states remain unresolved.

The next bounded step is a context-enriched review release for a small recoverable
subset, with exact source/fragment links, guarded annotation-cue handling and
explicit missing-dependency/context-sufficiency checks. Independent annotation and
adjudication, source licensing, project/near-clone grouping and frozen evaluation
cohorts still gate CodeBERT fine-tuning. Unknown remains unknown. Three categories
are not silently mapped into the legacy eleven outputs. No classifier improvement,
fresh holdout, human validation or Q1-readiness claim follows from this recovery.

## Validation

Twelve new stdlib regression tests cover exact versus similar text, BOM/byte
identity, aliases, path containment, missing identities, ambiguous fragments,
unchanged originals, replay, overwrite refusal and malformed/oversized sources.
All 62 focused tests and both project integrity validators passed. Full pytest
remains unavailable and was not installed or run. Fresh isolated full TeX/BibTeX
build, representative visual checks and backed-up PDF refresh are recorded in the task workspace `scafe_iteration_20261102_1300/`. No new source
corpus, context file or candidate data is committed or uploaded. The manuscript's
scientific claims are unchanged.
