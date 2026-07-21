# PR-AT generated-candidate controlled evaluation

PR-AT closes the return path from PR-AS inverse design to the existing
prediction and ranking pipeline. It does not promote generated structures to
the Material Registry and does not claim experimental or computational
validation.

## Exact inputs

The runner consumes and independently replays:

- the immutable PR-AS inverse-design publication;
- the PR-ARb shortfall decision used to authorize generation;
- the exact PR-AP screening receipt and Registry shortlist;
- the PR-AO model execution directory;
- the PR-AI dataset snapshot;
- the immutable Registry snapshot;
- optional cost and remote-known-host inputs when those were used upstream.

The PR-AS publication stays descriptor-pinned while PR-AT is built. Before an
executor registers the result, PR-AT is replayed from the external upstream
anchors and the publication directory remains descriptor-pinned through the
atomic artifact-registry update.

## Candidate identity

The combined pool uses a source-neutral `candidate_id` and carries explicit
source provenance:

- `source_kind`: `registry` or `generated`;
- `source_candidate_id`;
- `source_identity_digest`;
- `source_publication_id`.

A generated structure remains a publication-scoped design. PR-AT never assigns
it a Registry `material_id` or `registry_entry_digest`.

## Prediction and global ranking

Generated SMILES are re-standardized with RDKit, checked against Registry
chemical identities, transformed with the exact PR-AO feature contract, and
predicted by the exact replayed PR-AO models. Registry candidates are also
replayed from the complete PR-AP prediction pool.

Both sources are then ranked together under the same hard constraints,
Pareto-dominance policy, and aggregate percentile calculation. This is a true
global successor; it does not append generated candidates below the old
Registry shortlist.

## Immutable publication

The versioned directory contains exactly:

- `evaluation.json`;
- `complete_predictions.jsonl`;
- `generated_candidate_exclusions.jsonl`;
- `ranked_shortlist.csv`;
- `report.md`.

Publication uses the shared no-replace directory publisher. Exact replay binds
the directory identity to PR-AS, PR-AP, PR-AO, PR-AI, Registry, configuration,
and all output bytes.

## Agent boundary

`execute_oled_generated_candidate_evaluation` is a low-risk, ungated atomic
task because it performs deterministic local inference and ranking only. Its
registered artifacts use the `oled_candidate_evaluation_*` namespace to make
clear that the publication is an evaluation artifact, not a Registry update.
It registers one immutable execution record and retries return the existing
success without dispatching the adapter again.

The next step is a narrowly scoped PR-ARb v2 consumer that accepts exactly two
source types: `registry` and `generated`. It must not introduce a universal
candidate framework or predeclare literature, external-database, simulation,
or human candidate variants. It emits the final explainable Top-N dossier.

Iterative generation, loop budgets, and stop conditions remain outside PR-AT.
