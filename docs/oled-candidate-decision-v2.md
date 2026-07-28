# PR-ARb v2 final candidate decision

PR-ARb v2 is the narrow consumer of the immutable PR-AT
`oled_candidate_evaluation` publication. It produces the final explainable
Top-N recommendation artifact needed to close the first end-to-end flow.

## Scope

The candidate source roster is closed to:

- `registry`;
- `generated`.

No generic candidate base class or variants for literature, external database,
simulation, or human-submitted candidates are introduced. Generated candidates
remain publication-scoped identities and never receive Registry material IDs.

## Exact inherited request

PR-ARb v2 does not introduce another selection-request schema. It exact-replays
PR-AT and inherits the first PR-ARb request that authorized inverse design:

- target Top N;
- additional property constraints;
- optional diversity threshold;
- optional budget and currency.

Registry candidates may use costs from the exact original cost manifest. A
generated candidate has no trusted procurement cost unless a later explicit
contract supplies one, so a budgeted decision marks its cost unavailable rather
than estimating or inventing a value.

## Selection and explanation

The selector preserves the exact inherited
`rank_anchored_greedy_max_min_tanimoto.v1` policy. It chooses the highest-ranked
feasible candidate first. At each later step it chooses, from all currently
feasible candidates, the candidate whose maximum similarity to the selected
set is lowest, with PR-AT rank and candidate ID as deterministic tie-breakers.
The inherited property, cumulative-budget, and diversity-threshold constraints
remain hard feasibility checks. The publication contains:

- `candidate_decision.json` with every candidate decision and exact provenance;
- `top_candidates.csv` with the selected recommendations;
- `candidate_decision_dossier.csv` with selected and non-selected explanations;
- `report.md` with the bounded outcome.

Both CSV artifacts expose candidate source provenance and, for every modeled
property, the display name, unit, objective direction, predicted value,
screening-constraint status, and final decision-constraint status.

Final Top-N publication is fail closed. If a complete target set cannot be
formed, `top_candidates.csv` contains only its header, `selected_candidates` is
empty, and `selected_candidate_count` is zero. The dossier retains the
selection attempt as `eligible_but_incomplete_batch` evidence and marks any
provisional choices `provisional_not_final`; it never exposes a partial set as
a final recommendation.

## Claims and next step

This is a recommendation-only evaluation artifact. It performs no manual
accept/defer/reject adjudication, experiment, molecular-dynamics validation,
Registry mutation, Gold write, dataset write, or model registration.

When the requested Top N is complete, the first end-to-end candidate flow is
closed. An incomplete outcome may later be consumed by the separately bounded
PR-AU controller, subject to its iteration and generation budgets.
