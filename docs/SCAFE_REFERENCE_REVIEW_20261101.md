# Reference review and testable SCAFE gaps — 1 November 2026

The eleven original PDFs in `papers_ref/` were read without modification. Page
numbers below mean **PDF pages**, including cover sheets. Published status is
not evidence that a protocol is correct, results are comparable, or a journal
is Q1. No journal-quartile claim is made: quartiles depend on index, category,
and year and were not verified here. References are methodological context,
not a leaderboard against SCAFE's heterogeneous record-label task.

## Inventory and version verification

| Local PDF | Verified edition/year and venue | Traceable evidence and verification limit |
|---|---|---|
| A Review of Learning-based Smart Contract Vulnerability Detection—A Perspective on Code Representation | ACM TOSEM 35(6), article 167, May **2026**, 46-page journal edition; accepted July 2025 | pp.1–2 and 46, DOI [10.1145/3750042](https://doi.org/10.1145/3750042). Thus acceptance year differs from issue year. ACM web endpoint returned 403; local publisher front matter is the source. |
| Automated Vulnerability Injection in Solidity Smart Contracts—A Mutation-Based Approach for Benchmark Development | **EASE 2025 conference** paper, 11 pp.; not a journal | p.1, DOI [10.1145/3756681.3756943](https://doi.org/10.1145/3756681.3756943); ACM endpoint returned 403. |
| Benchmarking LLM-Based Static Analysis for Secure Smart Contract Development—Reliability, Limitations, and Potential Hybrid Solutions | **2026 arXiv v1 extended version**, submitted 11 May; authors state shorter version accepted to IEEE COMPSAC 2026 | pp.1–2; [arXiv primary record](https://arxiv.org/abs/2605.11163) independently confirms version/date and acceptance statement. The local PDF is the extended preprint, not a proceedings version of record. |
| Contract-Graph Fusion and Cross-Graph Matching for Smart-Contract Vulnerability Detection | Applied Sciences 15(19),10844, published 9 October **2025** | p.1 and [publisher](https://www.mdpi.com/2076-3417/15/19/10844). Publisher search result confirms metadata; subsequent full-page retrieval hit 429. |
| Ensemble multi-label machine learning solidity smart contract vulnerability detection model | Cluster Computing 28,1014, published 17 October **2025** | p.1 and [publisher record](https://link.springer.com/article/10.1007/s10586-025-05725-y) independently agree. |
| Multi-Objective-Guided Static Analysis for Smart Contract Vulnerability Detection | **ICSIE 2026 conference**, 14 pp.; not a journal | p.1, DOI [10.1145/3789595.3789604](https://doi.org/10.1145/3789595.3789604); primary publisher PDF search corroborates citation; HTML endpoint 403. |
| Smart Contract Vulnerability Detection Using GNN | **IJCAI 2020**, pp.3283–3290; neither 2025 nor 2026 | p.1; [official proceedings](https://ijcai.org/Proceedings/2020/454) independently confirms year/pages and DOI 10.24963/ijcai.2020/454. |
| Static Detection of Untrusted Cross-Contract Invocations in Go Smart Contracts | **SAC 2025 conference**, 10 pp.; Go/Hyperledger rather than Solidity/EVM | p.1, DOI [10.1145/3672608.3707728](https://doi.org/10.1145/3672608.3707728); publisher PDF search corroborates citation; HTML 403. |
| Synthetic vs Real | SSRN **non-peer-reviewed preprint** on vulnerability severity; publication/posting year not independently established from the supplied undated front matter | p.1 watermark and SSRN [6902983](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6902983). SSRN retrieval 403. References to 2026 do not establish publication date. The existing bibliography key/year is not independent proof. The 71-page file repeats substantial content (e.g. pp.1–30 and 31–60); do not count repeated sections as separate evidence. |
| Vulnerability detection of smart contracts by integrating large language models and dependency analysis | Engineering Applications of Artificial Intelligence 181,115511, **2026**; local publisher edition says online 29 June 2026 | p.1, DOI [10.1016/j.engappai.2026.115511](https://doi.org/10.1016/j.engappai.2026.115511). DOI page could not be retrieved; local printed venue/date verified, current publisher status not separately retrieved. |
| brlf.pdf | **2026 accepted journal pre-proof**, Blockchain: Research and Applications; accepted 3 March 2026, explicitly not the definitive version of record | pp.1–2, DOI [10.1016/j.bcra.2026.100471](https://doi.org/10.1016/j.bcra.2026.100471). Current DOI page unavailable in this check. Do not call this local file a final typeset article. |

An exact filename/page-count/SHA-256 inventory is saved in the task workspace's
`scafe_iteration_20261101_refs/papers/inventory.json`; original PDFs are not
copied into Git or uploaded. No papers were reproduced during this review.

## Evidence to use, and practices not to inherit

- **Task and contribution definition:** the TOSEM survey classifies representation
  families and distinguishes objectives (pp.5–6,19,39–41). SCAFE needs a falsifiable
  claim about what its source representation adds beyond nuisance signals, not a
  new architecture name or larger feature count.
- **Paired robustness:** Susan et al. explain nearly matched vulnerable/negative
  variants (p.4), rename identifiers/remove cues (p.5), compare experiments under
  matched conditions (p.7), and discuss classifier error (p.10). This supports a
  paired nuisance intervention. Their automatic evaluator's reported 92% accuracy
  does not make its error random or certify our labels. Our present intervention
  removes annotation comments and does not create a verified repaired contract.
- **Mutation validity:** MuSe manually checks compilation, logs, pattern adherence
  and edits (pp.5–6), finding injection failures in 20/384 checked examples. This
  supports validating intended transformations and preserving unknown cases.
  Its generated-positive labels and single-analyzer findings must not become
  verified exploitability labels for SCAFE.
- **Baselines and ablations:** Liang et al. use held validation thresholds and
  five seeds (p.9), and separate graph/encoder/matcher ablations (p.12). However,
  low-FPR quantities are reconstructed from summary metrics under an assumed
  prevalence and ROC-hull constraint (pp.10–11); they are not directly measured
  deployment ROC points. Data are unavailable (p.13). Do not import these numbers
  as matched baselines or copy the low-FPR reconstruction.
- **Uncertainty:** Alnuman et al. report six-fold F1 variation and intervals
  (pp.16,18). Fold means are not independent projects, and narrow intervals do
  not remove label or provenance bias. Their tool-derived labels and confidence
  weighting remain a limitation (p.22).
- **Operational evaluation:** BRLF separates data preparation (pp.14–16), embedding
  and evolutionary ablations (pp.23–26), time-to-first-find and limitations
  (pp.26,29–30). Follow its separation of mechanism, outcome and resource budget;
  execution-oracle detection/coverage is not comparable to SCAFE record ranking.
- **Context/dependency mechanisms:** GoLiSA provides an explicit information-flow
  analysis and replication repository (pp.5–6); GPT4Detection supplies dependency
  context and prompt ablations (pp.12,18), with a complementary-tool limitation
  and on-request data (p.23). Surface `call`/`delegatecall` presence alone is not
  caller authorization, reachability, or exploitability.
- **Synthetic-to-real evidence:** the SSRN preprint diagnoses mapping shift and
  limits richer-feature conclusions (pp.24–26), but uses a distinct severity task,
  an author-defined composite realism score, and an input that requires author
  contact (p.27) despite a broad replication statement (p.28). Treat it as a
  hypothesis source, not validation of a realism threshold or a Q1 standard.
- **Multiple-objective analysis:** Li et al. explicitly distinguish manually
  annotated and tool-labeled evaluations and acknowledge their different strength
  (p.13). Optimizing coverage or a fitness function is not evidence of independent
  security accuracy; do not optimize using our inspected external labels.

## Prioritized testable gaps

| Priority | Testable question and controls | Success criterion and claim boundary |
|---|---|---|
| P0 — this iteration | Under one newly hash-bound frozen checkpoint, does annotation removal change function-pattern outputs? Use at most 24 preselected development projects, one function/project, all 11 labels, repeat-input control, and fixed-window annotation masking as a secondary intervention. | Quantify paired output shifts and conditional project-resample uncertainty. No accuracy improvement, repaired-code, calibrated probability, historical reproduction, or external generalization claim. Mask tokens are not a neutral natural-language control. |
| P1 | Does a context-aware layer distinguish unsafe calls from authorized wrappers, checked failures, and reachable guarded paths? | Curate traceable real-function hard negatives with modifiers/state/callers; blinded adjudication and explicit unknown labels; project/template-separated tests. Compare against structural and cheap lexical baselines before fine-tuning. |
| P1 | Does SCAFE add ranking value beyond style/provenance on an untouched cohort? | Cohort-disjoint sources with both classes, near-clone controls, frozen selection, repeated seeds, project-level uncertainty, and nuisance baselines. Previously inspected external sets remain exploratory. |
| P2 | Can the scorer yield more confirmed findings per fixed analyst budget? | Prospective review-budget study with prevalence, runtime, coverage, adjudicated findings and failure cases. High F1 under high prevalence does not establish triage value. |

Recommended paper structure: explicit scope/RQs; auditable cohort and label flow;
predefined protocol; primary outcomes and controls; mechanism ablations; resource
costs; failure examples; uncertainty; construct/internal/external/conclusion
limitations; exact artifact map. Published examples motivate these requirements;
they do not guarantee acceptance or justify copying their text or protocols.

## Local-file identity inventory

IDs follow the row order of the version table above. Hashes identify the exact
local editions reviewed, not the latest upstream editions.

| ID | PDF pages | SHA-256 |
|---|---:|---|
| paper_01 | 46 | `5ecac4585a509ee38a4e667bb0543b01cc20bbe78565687556d657efee27f7ff` |
| paper_02 | 11 | `ab7d5086f8cf1dc7ab85b6f47f6c1e9cd36711afb9165568dc05cd35543d4d9a` |
| paper_03 | 25 | `b945db62963ef720a77d8996bb4be57fc59b27720f76f23ade675640773f4d5f` |
| paper_04 | 15 | `f9a5b93717a94f9412ee6245cec768170ae4c8a35ff79899d01517e271c18de9` |
| paper_05 | 24 | `b8912776fc7018fbdeaeb1d16c90ba6b432d3d0fffcb3f0b3fa6be5a8af03e18` |
| paper_06 | 14 | `20bfdcc7a2761dca103dc58319a8ebe4116b4225a2c3dcef6c385744c3e12477` |
| paper_07 | 8 | `f8f1c5599370dbd4182a26dc935788b431007c269028dde4134d1fa94b59956d` |
| paper_08 | 10 | `3dec1ef57a55169f65afe1a77cfc1d0354839c2fa4056f49542d6ca60aeb09a8` |
| paper_09 | 71 | `94afcbc74ac1da0e96cabb008ca5949aa8ad00787c3bdf569400d1db450b789b` |
| paper_10 | 24 | `5abc11278d01a3a8c071e208ca3f471451ddaa741e9467666af601da0175510c` |
| paper_11 | 40 | `20e1dfce928c5f21ec823888fecfe2849b3c1cb197b59edf701ba83841759f87` |
