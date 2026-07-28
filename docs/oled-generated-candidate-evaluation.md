# PR-AT generated-candidate controlled evaluation

PR-AT closes the return path from PR-AS inverse design to the existing
prediction and ranking pipeline. It does not promote generated structures to
the Material Registry and does not claim experimental or computational
validation.

## Exact inputs

The legacy single-round runner consumes and independently replays:

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

## Cumulative successor

PR-AT v2 adds one optional canonical
`oled_inverse_design_generation_roster` input. The cumulative roster contains
exactly two ordered PR-AS publications, matching PR-AU's fixed generation-round
ceiling; the first round continues to use the compatible v1 single-source
path. The roster's first source must be that direct/root PR-AS publication. Its
second source must carry the complete controller request, receipt, generation
authorization, and report bundle that its PR-AS publication consumed. The
roster's last source must exactly match the ordinary
`oled_inverse_design_receipt` and latest
controller artifacts supplied to the task.

A multi-source roster also binds the exact previous PR-AT publication. The
runner independently replays that predecessor and requires every prior
candidate identity, source binding, and property prediction to remain present
and unchanged. It then exact-replays every PR-AS source, rejects duplicate
candidate IDs or repeated SMILES/InChI/InChIKey identities across generation
publications, and rebuilds the complete Registry-plus-generated prediction
pool. Constraints, Pareto dominance, percentiles, and ranks are recomputed
globally over that cumulative pool.

The v2 receipt records the roster SHA-256, ordered PR-AS publication IDs and
receipt hashes, per-publication accepted/source counts, controller
authorization IDs, and the previous evaluation ID/SHA. The latest legacy
`pr_as_publication_id` fields remain present so PR-ARb v2 can retain its narrow
latest-action binding while independently replaying the full roster.

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
success without dispatching the adapter again. A later-round child run supplies
the optional roster as `oled_inverse_design_generation_roster`; the adapter,
Executor registration verifier, PR-ARb v2, and PR-AU all replay the same roster
anchor.

The next step is a narrowly scoped PR-ARb v2 consumer that accepts exactly two
source types: `registry` and `generated`. It must not introduce a universal
candidate framework or predeclare literature, external-database, simulation,
or human candidate variants. It emits the final explainable Top-N dossier.

Loop authorization, budgets, and stop conditions remain outside PR-AT.
