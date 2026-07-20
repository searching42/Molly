# OLED experiment batch selection

PR-AR turns one exact PR-AP Registry-screening shortlist into a local,
reviewable experimental-batch recommendation.  It is a selection and handoff
boundary only: it does not order materials, plan a synthesis, operate an
instrument, record a measurement, validate a prediction, update Gold or a
dataset, mutate the Registry, or register a model.

## Exact inputs

The selector requires both artifacts from the same PR-AP publication:

- `screening.json` (`oled_registry_screening_receipt`)
- `ranked_shortlist.csv` (`oled_registry_screening_shortlist`)

It verifies the receipt's recorded SHA-256 for the shortlist before using a
row.  It also validates the screening identity, property roster, directions,
candidate identities, ranks, predictions, and finite numeric values.  The
runner never discovers a latest screening, model, dataset, or Registry.

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

The first candidate is the highest-ranked feasible PR-AP candidate.  Each
later choice is the feasible candidate with the lowest maximum Morgan (radius
2, 2048 bit, non-chiral, non-feature) Tanimoto similarity to the already
selected set; PR-AP rank and then material ID break exact similarity ties.
Every selected pair must remain within the configured threshold.  RDKit is
required whenever a diversity threshold is needed; a non-chemical fallback is
never represented as structural diversity.

If a complete batch cannot be formed, the selector publishes a `not_ready`
advisory with an empty handoff CSV.  It never quietly promotes a smaller,
partial experimental batch.  Invalid/tampered inputs, invalid options, or an
unsafe output path fail before any publication.

## Published local artifacts

The versioned no-replace output directory contains:

- `batch_selection.json` — exact input bindings, policy, selected and
  unselected reason codes, diversity evidence, cost accounting, status, and
  bounded claims;
- `experiment_batch.csv` — the ready batch's material identities, screening
  ranks, predicted properties, selection rationale, and available cost data;
- `experiment_handoff.md` — a human-readable local handoff; and
- an executor-owned, attempt-scoped adapter result record when run via the
  Agent RunPlan path.

The receipt always records that experimental execution, procurement, synthesis,
measurement, experimental validation, Registry mutation, Gold/dataset writes,
and model registration are false.

## Agent-managed execution

`execute_oled_experiment_batch_selection` is a medium-risk RunPlan task
protected by `gate_5_final_threshold`.  Before that gate it freezes the exact
receipt, shortlist, and optional cost manifest inside the run directory.  On
resume it rechecks the named sources and then sends only the frozen bytes to
the adapter.  This prevents a source swap after approval from changing the
selected batch.

The Agent registers the receipt, CSV handoff, Markdown handoff, and immutable
first-success execution record as run artifacts.  A later failed retry writes
a separate attempt record and cannot overwrite the original recommendation.
