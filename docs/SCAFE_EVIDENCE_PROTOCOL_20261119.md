# SCAFE-Evidence: frozen feasibility protocol

## Research claim under test

SCAFE-Evidence is not a vulnerability detector and does not claim that a
resolved modifier makes a function secure.  It tests whether a defensive
Solidity analyzer can answer narrow, compiler-checkable repository questions
from partial source while:

1. retrieving only source evidence needed for the answer;
2. returning `unknown` when the available evidence is insufficient; and
3. emitting a replayable certificate that binds the query, answer, declaration
   span, source hashes, compiler version, and evidence cost.

The first fact is modifier-declaration resolution for named modifier
invocations.  Base-constructor invocations and unsupported syntax are separate
outcomes, not modifier answers.  A compiler-resolved full-source AST is the
reference oracle.  This fact is a prerequisite for authorization analysis, not
authorization or vulnerability ground truth.

## Novelty boundary

Selective retrieval itself is not novel: Repoformer selectively retrieves
repository context for code completion, and DraCo retrieves dependency context
using a repository graph.  Solidity name binding and compiler-AST analysis are
also established.  TRACE already analyzes repository-level Solidity access
control by locating sensitive functions, synthesizing compilable contracts, and
then analyzing generated graphs.

The candidate contribution is therefore **fail-closed evidence sufficiency for
defensive Solidity analysis**: exact-source retrieval instead of code
completion, explicit unknowns instead of confident guesses, compiler-checkable
answers, and evidence/cost certificates.  A feasibility result must beat strong
ordinary traversal; outperforming only naive file-local lookup is insufficient.

## Corpus admission

Development and confirmation projects must be real, independently versioned
multi-file Solidity repositories.  For every revision retain:

- canonical repository and immutable revision;
- original license and per-file SPDX evidence;
- exact source and compiler-input hashes;
- exact compiler version/settings and successful full-source AST output;
- project/fork/family identity and grouping;
- exclusion from DAppSCAN and every reserved evaluation family at project,
  exact-body, and conservative near-clone/template levels.

Common libraries are context, not independent projects.  Related versions,
forks, and repaired/vulnerable pairs remain one group.  Held or rights-unclear
source is not used.  DAppSCAN labels, outcomes, and examples are never used for
development, policy selection, or calibration.

Minimum feasibility corpus: at least five admitted project groups for
development and two untouched admitted groups for confirmation, with at least
30 genuine named modifier-reference queries overall and at least 10
cross-source queries.  These are feasibility minima, not a power claim.

## Paired methods

Every method receives the same query and partial starting file:

1. **file-name lookup**: answer only when one same-file declaration has the
   invoked name;
2. **sound file-only abstention**: answer only when a unique same-contract,
   same-file declaration is justified; otherwise `unknown`;
3. **import closure**: recursively read all lexical imports;
4. **strong traversal**: imports plus inheritance and alias resolution until a
   fixed point, with ambiguity causing `unknown`;
5. **SCAFE-Evidence**: demand-driven retrieval of the minimum dependency path
   needed to bind the invocation, with fail-closed checks and a certificate;
6. **full source**: compiler-oracle upper bound, not an efficiency baseline.

No method may receive compiler `referencedDeclaration`, linearization, target
paths, target spans, or oracle answers as resolver input.  Index construction,
global search, preprocessing, and cache warming are charged.  Report cold
per-query and amortized batch costs separately.

## Outcomes and metrics

Primary paired outcomes:

- correct, wrong, and unknown counts;
- answered coverage and accuracy among answered queries;
- selective risk: wrong answers divided by answered queries;
- unique source files and UTF-8 bytes read;
- wall and CPU time, including indexing/preprocessing;
- project-level results and paired bootstrap intervals over project groups.

Certificates are validated by independent replay against source hashes and the
compiler oracle.  Incorrect confident answers are never merged with unknowns.

## Frozen decision gate

Continue beyond the pilot only if all conditions hold:

- zero observed wrong SCAFE-Evidence answers;
- at least 90% answered coverage overall and no admitted project below 80%;
- at matched correctness and answered coverage, at least 20% fewer median
  per-query bytes than strong traversal;
- the 95% project-bootstrap interval for paired byte saving excludes zero;
- at least 20% of eligible queries genuinely require cross-source evidence;
- certificate replay succeeds for every answered query;
- the gain remains after excluding common-library declarations.

Stop or narrow if corpus admission fails, the gain is explained only by parser
repair/caching, or strong traversal matches the evidence cost.  A failed gate is
retained as a negative result; thresholds or corpus membership are not changed
after observing outcomes.

## Resource bound and sequencing

The pilot is CPU-only and Mac-sized: at most seven project groups, 250 source
files, 25 MiB source text, four already-qualified or separately hash-verified
compiler binaries, and two hours of CPU per project.  No model training, hosted
LLM, paid service, contract execution, deployment, or fuzzing is required.

Sequence:

1. qualify corpus and freeze development/confirmation groups;
2. generate full-source compiler oracles without inspecting policy outcomes;
3. implement methods and software tests;
4. run development once, freeze the selected SCAFE-Evidence policy, then run
   untouched confirmation once;
5. rewrite the manuscript only if the frozen gate passes.

