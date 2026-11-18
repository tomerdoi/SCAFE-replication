# Portable fixed-diagnostic replay, version 1

This directory packages own code and numerical expectations for the paired nuisance comparison, fixed review budgets and exact label-sensitivity study. It contains no source corpus, structured per-record predictions/labels, model weights or cache files. Those original private inputs remain outside the package and are not cleared for distribution. The aggregate expectations are a declared projection of already reported inspected-target results, not fresh confirmation.

Use the existing owning project's Python environment with numpy and scikit-learn. Do not install packages or fetch models automatically. The portable runner imports diagnostic implementations only from its adjacent scripts directory. From any working directory:

    python -I /path/to/diagnostic_replay/run_diagnostics.py --output /new/output

That runs 43 numerical software controls only. Its report explicitly says scientific replay was not attempted when no original cache root is supplied. It does not establish reproduction of measured target results.

For an authorized local operator who already has the unchanged original private caches and metadata:

    python -I /path/to/diagnostic_replay/run_diagnostics.py --output /new/replay --cache-root /path/to/original/sc_review

The root must contain the original two cached CSVs plus primary/nuisance lineage JSON at the existing data/derived/sanitized_v1/ml_results paths. The frozen budget/sensitivity protocols pin the cache and original scored-input hashes. The paired reader excludes DAppSCAN before outcome retention. No labels are changed, no source is read, no fitting or selection occurs. Input files stay read-only; output must be new and should be outside this package.

The runner compares all allowlisted scientific aggregate cells against expected_aggregates.json: exact integers/structure and finite floats within 1e-12. Runtime/timestamp and record-identity hashes are deliberately not comparison outputs. Exact frozen protocols are byte-preserved; the paired comparison uses its original default seed 20261117 and 1000 resamples. Replays check deterministic arithmetic/software portability, not independent labels, model-training reproduction, source rights, novelty, generalization, human-review utility or publication readiness.

Verify the parent package SHA256SUMS before use. Any checksum mismatch, missing dependency, input mismatch or scientific mismatch blocks a passed replay claim. The parent package retains its original manuscript attribution illustrations and is not case-free, anonymous by proof, or distribution-cleared.

The code repository does not track the three frozen JSON evidence inputs under its ignore policy. The generated local package includes them; rebuilding the package requires the explicit local diagnostic-evidence input directory. Nothing is force-added or renamed to evade an ignored-data/publication boundary.
