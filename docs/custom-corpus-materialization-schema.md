# Custom Corpus Materialization Plan Schema

`custom_corpus_materialization.v1` defines a future materialization plan for
package-validated custom corpus records. It records operator-confirmed intent,
source artifact hashes, candidate-only output intent, record selections,
rollback policy, and redaction policy.

This schema does not implement materialization. A valid materialization plan
does not create candidate artifacts, admit training data, run Phase 1, or set
`DatasetConfirmation.confirmed=true`. A valid materialization plan is necessary
but not sufficient for a future materializer.

## Relationship To Boundary Design

The boundary design is documented in:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

The plan schema is the first pre-materialization artifact after package
validation. It validates intent and binding only.

## Relationship To Property Candidates

Open-ended numeric scientific property candidates are documented in:

```text
docs/custom-corpus-property-candidate-schema.md
```

The upstream review-planning layer is documented in:

```text
docs/custom-corpus-property-candidate-planner.md
```

Materialization plans consume records already selected through review and
admission. The materialization plan schema does not discover scientific
properties, define a property whitelist, normalize numeric property evidence,
or decide whether a property is trainable. The property candidate schema is the
earlier pre-review layer for representing numeric property candidates, and the
property candidate planner output is review-planning evidence rather than
materialization input.

The property candidate review queue builder is also upstream of human review.
Materialization plans must not consume review queue artifacts directly;
materialization requires review, admission, and package validation.

The property review binding validator checks the queue-to-review link upstream
of admission. Materialization plans must consume records that passed human
review and admission; binding evidence alone is not a materialization input.

The property admission readiness planner is also upstream evidence. It can
summarize accepted reviewed records for future admission planning, but
materialization plans must consume records that passed explicit admission.

## Plan Contents

A materialization plan includes:

- `materialization_mode`: currently only `candidate_only`
- `materialization_decision`: `planned` or `blocked`
- source manifest, dry-run report, review manifest, admission request, and
  package validation SHA-256 values
- package validation status and admission decision
- dry-run boundary fields proving Phase 1 did not run
- explicit operator materialization confirmation
- record-level selections with either `materialize_candidate` or `exclude`
- rollback and redaction policy labels

No training mode exists.

## Confirmation Model

`CustomCorpusMaterializationConfirmation` binds the operator confirmation to
the exact source artifacts:

- manifest SHA-256
- dry-run report SHA-256
- review manifest SHA-256
- admission request SHA-256
- package validation SHA-256
- corpus id
- dry-run id
- review manifest id
- admission request id

For a `planned` materialization, confirmation must be explicit and
`confirmed_by` must be present and redacted if needed. This confirmation does
not modify `DatasetConfirmation`.

## Record Rules

`materialize_candidate` records require:

- `admission_action=admit`
- `review_decision=accept`
- normalized value summary
- provenance summary
- materialization reason

`exclude` records require an exclusion reason. Records rejected by review,
marked `needs_review`, or excluded at admission cannot use
`materialize_candidate`.

## Package And Dry-Run Boundaries

For `materialization_decision=planned`:

- `package_validation_status` must be `passed`
- `package_admission_decision` must be `eligible`
- `dry_run_phase1_status` must be `not_run`
- `dry_run_dataset_confirmation_confirmed` must be `false`
- `dry_run_training_dataset_admitted` must be `false`

If those conditions are not met, the plan must be `blocked`.

## CLI Usage

Validate a plan offline:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_materialization \
  --materialization-plan docs/examples/custom-corpus-materialization-plan.example.json \
  --output-summary /tmp/custom-corpus-materialization-plan-summary.json
```

The CLI reads local JSON only, prints a safe JSON summary, and optionally
writes that summary. It does not call MinerU, parse PDFs, run corpus workflow,
create candidate/training artifacts, or modify `DatasetConfirmation`.

## After Plan Validation: Offline Planner

After a plan validates, the offline planner can produce a safe execution
summary for a future materializer:

```text
docs/custom-corpus-materialization-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-planner-evidence-template.md
```

Plan validation checks structure and intent. The planner summarizes intended
future outputs and rollback labels, but still does not create candidate
artifacts, candidate/training CSVs, Phase 1 inputs, or materialized records.

## Summary Output

The summary records safe ids, status fields, counts, source hashes, rollback
policy, and redaction policy. It does not include raw PDF paths, article text,
local absolute paths, private home paths, ParsedDocument text, MinerU bundle
paths, tokens, Authorization headers, cookies, or private emails.

## Examples

Safe example plan:

```text
docs/examples/custom-corpus-materialization-plan.example.json
```

Future plan evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-plan-evidence-template.md
```

## Boundaries

- This schema does not implement materialization.
- A valid materialization plan does not create candidate artifacts.
- A valid materialization plan does not admit training data.
- A valid materialization plan does not run Phase 1.
- A valid materialization plan does not set `DatasetConfirmation.confirmed=true`.
- A valid materialization plan is necessary but not sufficient for a future
  materializer.
