# SCAFE-Evidence: scaled evaluation protocol

## Purpose and frozen claim

This protocol extends the passed seven-project feasibility pilot without
changing its claim.  SCAFE-Evidence resolves one compiler-checkable fact:
which declaration a named Solidity modifier invocation references.  It
retrieves exact source evidence, abstains when that evidence is insufficient,
and emits a replayable certificate.

It does **not** infer that a modifier is effective, that authorization is
correct, or that a contract is vulnerable.  Repoformer and DraCo establish
selective/dependency-guided repository retrieval; TRACE establishes
repository-level Solidity access-control analysis; GraphLedger combines
repository dependency evidence with explicit inconclusive outcomes; and
compiler/Slither name binding is established.  The broad combination is
therefore not novel.  The narrower claim under test is a provenance-bound
Solidity authorization-evidence task: compiler-checkable answers supported by
exact-source witnesses, explicit missing-evidence states, end-to-end evidence
cost, and independently replayable certificates.

## Corpus and split

Admit 20--30 immutable, non-fork, independently versioned public Solidity
projects.  Freeze all project identities, revisions, rights evidence,
compiler settings, source hashes, family decisions, and reserved-family
screens before method outcomes are read.

- At least eight project groups are untouched confirmation groups.
- Related repositories, versions, templates, forks, and shared application
  families form one group.
- Dependencies are context, never independent project groups.
- Every included source file must have an applicable license or permission.
- DAppSCAN and related reserved families remain evaluation-only; their labels,
  findings, and outcomes are not loaded.
- Exact/token/template nonmatches are exclusion evidence, not proof of
  independence.

The natural-query cohort must contain at least 300 eligible named modifier
invocations, at least 75 genuinely cross-source invocations, and at least three
cross-source queries in every confirmation project.  At least 150 queries must
be answered on the untouched split.  If these minima cannot be met without
weakening admission, stop and report corpus infeasibility.

### Batched execution

The existing oracle builder accepts at most seven projects per manifest.
Therefore freeze the complete 20--30-project registry, group assignments, and
development/confirmation split first, then execute it as multiple manifests of
at most seven projects.  Every batch must bind the same protocol hash,
qualified-compiler registry, exclusion policy, and resolver version.

Aggregate only by immutable project ID after verifying every batch manifest,
oracle summary, exclusion report, and result hash.  No project may be moved,
removed, replaced, or reclassified after any method outcome from any batch is
read.  Batch boundaries are operational and are not statistical groups.

Historical inspection-only, rights-held, family-held, or fixed-scope NO-GO
projects remain ineligible unless materially new affirmative evidence resolves
the exact recorded hold.  A planning list, public repository, permissive root
license, or content nonmatch is not admission.

## Paired methods and charged work

All methods receive the same starting file, invocation byte offset, modifier
name, remappings, and cold source repository:

1. file-name lookup;
2. sound same-contract file-only abstention;
3. a deterministic BM25/text retrieval baseline with indexing charged;
4. complete lexical import closure;
5. strong import/inheritance/alias traversal to a fixed point;
6. demand-driven SCAFE-Evidence retrieval; and
7. the compiler input closure as a full-source upper bound.

Where import closure and strong traversal read identical files, report equal
costs explicitly rather than collapsing rows.  Charge unique UTF-8 bytes and
files read, parsing/indexing, CPU time, and wall time.  Report cold per-query
and project-batch amortized costs.  Compiler reference declarations,
linearization, target paths, target spans, and method outcomes are never
resolver inputs.

## Natural-query outcomes

Report correct, wrong, and unknown counts; answered coverage; selective risk;
unique files and bytes; CPU and wall time; and certificate replay.  Report
project-level values, risk--coverage curves, exact one-sided zero-error bounds,
and paired project-bootstrap intervals.  Also report:

- confirmation separately from development;
- one-query-per-distinct-target-declaration sensitivity;
- results excluding common-library targets; and
- results excluding each project in turn.

Queries are correlated observations, not independent samples.

## Predeclared insufficient-evidence stress cohort

Construct stress cases only after natural queries and projects are frozen, by
a deterministic transformation that does not inspect method outcomes:

1. for every eligible cross-source query, make a read-only projection that
   omits its compiler-referenced declaration file;
2. retain the query file and all other paths unchanged;
3. require the method to return `unknown`, never a different declaration; and
4. retain the original full-source oracle only as the expected missing target.

Separately mutate each answered certificate in two fixed ways: replace the
first source-hash hexadecimal digit and add one byte to the declaration start.
Replay must reject both mutations.  Existing hand-written ambiguity fixtures
remain software controls and are not counted as natural project evidence.

## Decision gate

Adopt this direction for a manuscript rewrite only if the untouched
confirmation split satisfies all of the following:

- zero observed wrong natural-query answers among at least 150 answered
  confirmation queries, with the one-sided 95% binomial upper bound reported;
- natural-query coverage within two percentage points of strong traversal and
  at least 80% per project;
- at least 30% lower median bytes and 20% lower project-weighted mean bytes
  than strong traversal at matched outcomes;
- the 95% project-bootstrap lower bound for paired byte saving exceeds 10%;
- at least 70% of file-only unknowns that strong traversal can resolve are
  recovered;
- every natural answered certificate replays;
- every missing-target stress case returns `unknown`;
- every mutated certificate is rejected;
- the saving remains at least 20% after excluding common-library targets; and
- the saving remains positive in the distinct-target sensitivity analysis.

Development is used only for implementation correction and policy freezing.
Confirmation membership and thresholds are not changed after outcomes.  A
failed component is retained and causes a narrow, pivot, or stop decision.

## Resource and manuscript policy

The experiment is CPU-only and Mac-sized: no model training, hosted LLM, paid
compute, contract execution, deployment, or fuzzing.  Bound each project to
1,000 Solidity files, 100 MiB source text, and two CPU hours; stop an input
that exceeds a bound.  Compiler binaries must be official, hash-qualified,
and isolated rather than globally installed.  Include indexing and report peak
RSS; the operational gate is at most 16 GiB peak RSS and five seconds p95
per-query latency on the declared Mac.

The legacy detector manuscript is not rewritten around feasibility numbers.
Only a passed scaled confirmation gate authorizes an end-to-end manuscript
rewrite, closest-work update, figures, tables, and a publication-review
artifact.  Build or test success is not scientific validation.
