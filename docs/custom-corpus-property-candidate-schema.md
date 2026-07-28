# Custom Corpus Property Candidate Schema

`custom_corpus_property_candidate.v1` defines an offline artifact for
open-ended numeric scientific property candidates extracted from custom corpus
parsed documents. It sits before human review.

The schema describes what a candidate trainable property record must contain.
It does not define a fixed property whitelist. Safe labels such as
`field_name` and `canonical_property_guess` are validated structurally, but an
unfamiliar numeric property is not rejected only because it is not PLQY, HOMO,
LUMO, emission wavelength, or another known field.

## Purpose

The property candidate layer records:

- source corpus, dry-run, document, and artifact bindings
- extracted property labels and canonical property guesses
- numeric value representation
- unit information
- entity binding
- context, method, and provenance summaries
- trainability decision and reason
- confidence and review requirement

It is designed to support future deterministic, LLM/agent-generated, hybrid,
or human-seeded candidate pools. This PR does not call any model or implement
any extractor.

## Governance Chain

The full current custom corpus governance path is maintained in:

```text
docs/custom-corpus-governance-runbook.md
```

This schema sits before human review and before all admission,
package-binding, materialization-plan, execution-preflight, and candidate
quarantine materialization steps.

A valid property candidate manifest is necessary but not sufficient for human
review, admission, materialization, or training.

## Required Fields

Each `PropertyCandidateRecord` includes safe ids for corpus, dry-run,
document, source record, field, and entity binding. It also records:

- `raw_property_label`
- `canonical_property_guess`
- `property_family`
- `field_name`
- `value_kind`
- numeric value fields
- unit fields
- entity fields
- `provenance_summary`
- `extraction_source`
- `confidence`
- `trainability_decision`
- `decision_reason`
- `review_required`

## Trainability Decisions

`trainability_decision` may be:

- `candidate`
- `needs_review`
- `reject`

Candidate records require a known numeric value kind, finite machine-readable
numeric values, property and entity labels, provenance summary, decision
reason, confidence between 0 and 1, and unit handling consistent with the unit
status. Candidate records remain review-bound; `review_required` must be true.

Rejected records require `rejection_reason`. Needs-review records require
notes or a decision reason.

## Numeric Value Representation

Supported value kinds:

- `numeric_scalar`: uses `value_normalized`
- `numeric_range`: uses `value_min` and `value_max`
- `numeric_tuple`: uses `value_tuple`
- `unknown`: carries no machine-readable numeric values

All numeric values must be finite. Ranges require `value_min <= value_max`.
Tuples must contain 2 to 8 finite values.

## Unit Rules

Unit status may be:

- `explicit`
- `inferred`
- `not_applicable`
- `missing`
- `unknown`

Candidate records with `explicit` or `inferred` units require
`unit_normalized`. Candidate records with `missing` units are invalid unless
the normalized unit is explicitly `not_applicable`; otherwise they should be
`needs_review` or `reject`.

## Entity And Provenance Binding

Candidate records require:

- `entity_id`
- `entity_type`
- `source_artifact_sha256`
- `provenance_summary`

The manifest binds all records to a shared `corpus_id` and `dry_run_id`, and
records must match those ids. SHA-256 values are normalized to
`sha256:<64-lowercase-hex>`.

## Summary Output

The summary includes safe counts and labels:

- total record count
- candidate, needs-review, rejected counts
- review-required count
- unique document, entity, and field counts
- value kind counts
- property family counts
- unit status counts
- extraction source counts
- source manifest and dry-run report SHA-256 values

The summary does not include raw value summaries, provenance summaries, raw
table rows, raw article text, local paths, ParsedDocument text, MinerU bundle
paths, tokens, cookies, or private emails.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_candidate \
  --property-candidates docs/examples/custom-corpus-property-candidates.example.json \
  --output-summary /tmp/custom-corpus-property-candidates-summary.json
```

The CLI validates local JSON only, prints a safe JSON summary, optionally
writes that summary, and returns nonzero on invalid manifests.

## After Candidate Validation: Offline Planner

After a candidate manifest validates, the offline planner can produce a safe
review-planning summary:

```text
docs/custom-corpus-property-candidate-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-property-candidate-planner-evidence-template.md
```

Property candidate validation checks structure and trainability decision
fields. The planner prepares aggregate counts, safe record id lists, and review
queue planning, but still does not create human review artifacts, perform
human review, or call an LLM or agent.

Validated property candidate manifests can also feed the offline review queue
builder:

```text
docs/custom-corpus-property-candidate-review-queue.md
```

Review queue artifacts remain pre-review artifacts. They prepare material for a
future reviewer, but they do not create `custom_corpus_review.v1` decisions.

## Boundaries

- This schema does not implement property extraction.
- This schema does not call an LLM or agent.
- This schema does not evaluate extraction accuracy.
- This schema does not implement Agentic RL.
- This schema does not admit data.
- This schema does not materialize data.
- This schema does not create candidate/training CSVs.
- This schema does not run Phase 1.
- This schema does not modify `DatasetConfirmation`.
- A valid property candidate manifest is necessary but not sufficient for
  review, admission, or materialization.
