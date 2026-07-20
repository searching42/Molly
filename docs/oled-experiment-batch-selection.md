# OLED candidate decision selection

PR-ARb turns one exact PR-AP Registry-screening shortlist into a local,
explainable candidate-decision dossier and Top-N output. It is a selection
boundary only: it does not order materials, plan a synthesis, operate an
instrument, record a measurement, validate a prediction, update Gold or a
dataset, mutate the Registry, or register a model.

There is no generated human `accept`/`defer`/`reject` state. A `ready` result
is the agent's bounded Top-N output. A future inverse-design branch is eligible
only when the count of candidates satisfying the explicit property constraints
is below the requested Top-N count. A `not_ready` caused by a monetary budget,
structural-diversity threshold, or another non-supply selection policy does
not request generation. When the narrow count-shortfall condition is true, new
candidates must re-enter the same PR-AP controlled prediction, filter, rank,
and dossier boundary.

## Exact inputs

The selector requires both artifacts from the same PR-AP publication:

- `screening.json` (`oled_registry_screening_receipt`)
- `ranked_shortlist.csv` (`oled_registry_screening_shortlist`)

It also requires the exact replay anchor that produced that publication:

- the PR-AO Phase-1 execution directory;
- the PR-AI dataset snapshot; and
- the immutable Registry snapshot.

It verifies the receipt's recorded SHA-256 for the shortlist before using a
row. It then exact-replays PR-AP from the three anchored inputs and requires
the reconstructed `screening.json` and `ranked_shortlist.csv` bytes to match.
This prevents a self-consistent, re-signed receipt/shortlist pair from changing
the recommended material identity, structure, name, or prediction. It also
validates the screening identity, property roster, directions, candidate
identities, ranks, predictions, and finite numeric values. The runner never
discovers a latest screening, model, dataset, or Registry.

For a monetary budget, the caller must additionally provide a local
`oled_candidate_cost_manifest` JSON artifact.  The manifest has one currency
and exact `(material_id, registry_entry_digest)` bindings with non-negative
integer `cost_minor` values.  Missing cost data makes a candidate unavailable
when a money budget is active.  Without that manifest, PR-AR supports only a
candidate-count budget and makes no procurement or availability claim.

## Selection policy

The caller supplies a positive `target_batch_size`, optional `minimums` and
`maximums` in `property=value` form, an optional `max_budget_minor`, and an
explicit `max_pairwise_tanimoto` when selecting more than one material.
Property constraints may refer only to properties already present in the
bound PR-AP receipt.

The first candidate is the highest-ranked feasible PR-AP candidate. Each later
choice is the feasible candidate with the lowest maximum Morgan (radius 2,
2048 bit, non-chiral, non-feature) Tanimoto similarity to the already selected
set; PR-AP rank and then material ID break exact similarity ties. Every
selected pair must remain within the configured threshold. RDKit is required
whenever a diversity threshold is needed; a non-chemical fallback is never
represented as structural diversity.

The selector records its exact preflight and each greedy step: candidate cost,
per-candidate and cumulative budget feasibility, prior selected set, proposed
cumulative cost, Morgan similarity, threshold, deterministic sort key, and
whether the candidate was chosen, lower priority, or infeasible. It therefore
does not infer an unselected candidate's reason from the final batch alone.
Names, units, and physical interpretations in the rendered artifacts are also
frozen into the hashed request configuration before the versioned batch ID is
derived.

If a complete batch cannot be formed, the selector publishes a `not_ready`
advisory with an empty handoff CSV.  It never quietly promotes a smaller,
partial Top-N candidate batch. Invalid/tampered inputs, invalid options, or an
unsafe output path fail before any publication.

The receipt makes that boundary machine-readable with `candidate_supply`:
`inverse_design_should_trigger=true` only for the above property-eligible
candidate count shortfall, and always records `generation_executed=false`.
It also separates provisional greedy choices from finalized Top-N choices, so
an incomplete batch can never be mistaken for a partial selected output.

## Published local artifacts

The versioned no-replace output directory contains:

- `batch_selection.json` — exact input bindings, frozen property-presentation
  contract, policy, selected and unselected decision traces, full greedy trace,
  named/unit-bearing property truth tables, diversity evidence, cost
  accounting, status, and bounded claims;
- `experiment_batch.csv` — the ready batch's material identities, screening
  ranks, predicted properties, named objectives, requested bounds, selection
  rationale, upstream provenance, and available cost data;
- `candidate_decision_dossier.csv` — every exact shortlist candidate,
  including selected and unselected rows, their property truth tables,
  deterministic decision reasons, budget/diversity evidence, and a compact
  machine-readable per-candidate greedy-step trace;
- `experiment_handoff.md` — a human-readable candidate-decision dossier; and
- an executor-owned, attempt-scoped adapter result record when run via the
  Agent RunPlan path.

The receipt always records that experimental execution, procurement, synthesis,
measurement, experimental validation, Registry mutation, Gold/dataset writes,
and model registration are false.

## Agent-managed execution

`execute_oled_experiment_batch_selection` is a medium-risk RunPlan task
protected by `gate_5_final_threshold`. Before that gate it freezes the exact
receipt, shortlist, PR-AO execution directory, PR-AI dataset snapshot,
Registry snapshot, and optional cost manifest inside the run directory. On
resume it rechecks the named sources and then sends only the frozen bytes to
the adapter. This prevents a source swap after approval from changing the
selected batch.

The Agent registers the receipt, selected Top-N CSV, full-shortlist dossier,
Markdown dossier, and immutable first-success execution record as run
artifacts. A later retry is rejected before adapter dispatch, writes only a
separate rejected-attempt record, and cannot create an unregistered batch or
overwrite the original recommendation.
