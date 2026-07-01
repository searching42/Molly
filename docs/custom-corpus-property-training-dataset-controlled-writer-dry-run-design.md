# Custom Corpus Property Training Dataset Controlled Writer Dry-Run Design

## Purpose

This document defines the docs/test-only controlled writer dry-run design for
the property training dataset path. It specifies what a future dry-run may
validate and summarize before any implementation that simulates writer
behavior.

The design answers how a future dry-run would prove that the controlled writer
could assemble row-shaped evidence without emitting rows, raw values, or
dataset artifacts. It does not implement the dry-run, execute a dry-run,
implement a controlled writer, execute a controlled writer, create rows, create
artifacts, run Phase 1, modify `DatasetConfirmation`, run model training, run
evaluation, call LLMs or agents, call MinerU, parse documents, or run
chemistry calculations.

## Position in the Governance Chain

```text
property training dataset controlled writer value resolution dry-run
-> property training dataset controlled writer value resolution dry-run precheck
-> small public quarantine materialization evidence
-> property training dataset quarantined candidate admission boundary
-> property training dataset domain validation boundary
-> property training dataset controlled writer design plan
-> property training dataset controlled writer design plan preflight
-> property training dataset controlled writer dry-run design
-> property training dataset controlled writer dry-run
-> property training dataset controlled writer dry-run precheck
-> property training dataset controlled writer execution request design
-> property training dataset controlled writer execution request
-> future controlled writer execution request preflight
-> future explicitly confirmed controlled writer execution
```

The dry-run design is downstream of the controlled writer design plan
preflight. It is upstream of the controlled writer dry-run implementation and
the controlled writer dry-run precheck.

## Required Upstream Evidence

The dry-run design requires, at minimum:

- controlled writer design plan
- controlled writer design plan preflight
- domain validation boundary evidence
- quarantined-candidate admission boundary evidence
- small public quarantine materialization evidence
- controlled writer value resolution dry-run and precheck evidence
- controlled writer execution plan and preflight evidence
- value source manifest planner and preflight evidence
- input binding planner and preflight evidence
- writer execution request and preflight evidence
- row contract and row contract precheck evidence
- materialization dry-run and dry-run precheck evidence
- materialization plan and plan precheck evidence
- training admission readiness, request, and preflight evidence
- property candidate review and admission evidence
- quarantine candidate preflight and materialization evidence

This PR only defines the design. It does not read these artifacts at runtime.

## Dry-Run Design Scope

A future controlled writer dry-run may eventually validate whether a writer
would be able to assemble rows, while still not emitting rows or creating
artifacts. The dry-run should use only safe references, labels, counts, hashes,
status fields, redaction status, and boundary booleans.

This controlled writer dry-run design does not implement the dry-run.
This controlled writer dry-run design does not execute a dry-run.
This controlled writer dry-run design does not implement the controlled writer.
This controlled writer dry-run design does not execute the controlled writer.
This controlled writer dry-run design does not emit raw values.
This controlled writer dry-run design does not materialize values.
This controlled writer dry-run design does not serialize training rows.
This controlled writer dry-run design does not create training dataset artifacts.
This controlled writer dry-run design does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer dry-run design does not generate conformers.
This controlled writer dry-run design does not generate DPA3 structures.
This controlled writer dry-run design does not run Phase 1.
This controlled writer dry-run design does not modify DatasetConfirmation.
This controlled writer dry-run design does not run model training or evaluation.

## Future Dry-Run Input Contract

A future dry-run implementation may only consume safe references and preflight
summaries from upstream packages.

Allowed input evidence:

- safe ids
- hashes
- schema versions
- field labels
- unit labels
- source labels
- candidate counts
- accepted, needs-review, and blocked counts
- redaction status
- status labels
- boundary booleans
- coverage summaries

Forbidden input evidence in this design:

- raw property values
- exact numeric extracted values
- canonical molecular strings
- structure identifiers
- key-form structure identifiers
- table row payloads
- article body text
- paper titles
- document file names
- local paths
- output artifact paths
- row payloads
- model input tensors
- conformer data
- DPA3 structures
- credential material
- auth header material
- API key material

## Future Dry-Run Report Contract

The future report schema label is:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1
```

The future report may contain only safe, redacted fields such as:

```text
dry_run_id
dry_run_status
design_plan_preflight_id
domain_validation_boundary_id
accepted_candidate_record_count
needs_review_candidate_record_count
blocked_candidate_record_count
would_write_row_count
would_write_field_count
missing_required_field_count
would_create_training_dataset_artifact=false
would_create_csv_jsonl_parquet_lmdb=false
would_serialize_rows=false
would_materialize_values=false
redaction_status
preflight_errors
preflight_warnings
```

The report must not include actual row payloads, raw values, molecular strings,
local paths, output artifact paths, file names, or exact numeric extracted
values. It must use `would_*` booleans and aggregate counts only.

## Future Dry-Run Summary Contract

The future summary schema label is:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1
```

The future summary may include:

```text
dry_run_id
dry_run_status
dry_run_report_sha256
dry_run_report_basename
accepted_candidate_record_count
would_write_row_count
would_write_field_count
missing_required_field_count
redaction_status
controlled_writer_executed=false
training_dataset_materialized=false
dataset_artifact_created=false
serialized_rows_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

The summary must use basenames only if it references files.

## Allowed Future Dry-Run Outputs

Only in a later implementation PR, a future dry-run may write:

- dry-run report JSON
- dry-run summary JSON
- redacted dry-run evidence Markdown

The future dry-run must not write training rows, review rows, candidate rows,
CSV/JSONL/Parquet/LMDB artifacts, conformer files, DPA3 structure files, model
input tensors, Phase 1 artifacts, or DatasetConfirmation files.

## Disallowed Current Outputs

This PR must not create:

- dry-run report JSON
- dry-run summary JSON
- dry-run evidence Markdown
- training rows
- review rows
- candidate rows
- CSV artifacts
- JSONL artifacts
- Parquet artifacts
- LMDB artifacts
- conformer files
- DPA3 structure files
- model input tensors
- Phase 1 artifacts
- DatasetConfirmation files

## Side-Effect Boundary

The future dry-run implementation must:

- use a clean output directory
- fail if the output directory is dirty
- write only dry-run report, summary, and evidence files
- avoid raw values in report, summary, and evidence
- avoid serialized rows
- avoid training dataset artifact paths
- never call writer execution
- never call Phase 1
- never call model training or evaluation
- never call LLMs, agents, MinerU, document parsers, or corpus workflows
- never read documents or parsed document objects
- never run chemistry calculations

## Redaction and Non-Leakage Policy

Future dry-run report, summary, and evidence outputs must fail closed if they
include raw values, exact numeric extracted values, molecular strings, table
payloads, article body text, document file names, local paths, output paths,
row payloads, credential material, model input tensors, conformer data, or DPA3
structures.

On redaction failure, the future dry-run should write no unsafe evidence and
should return a minimal blocked summary with a safe error code.

## Dry-Run Status Semantics

`passed` means:

- all required upstream preflights passed
- design plan preflight passed
- domain validation boundary passed
- value resolution precheck passed
- accepted candidate count meets the configured minimum
- would-write counts are aggregate only
- missing required field count is zero
- no blocked candidates are present
- no needs-review candidates are present unless explicitly allowed
- no side-effect flags indicate materialization
- redaction passed

`needs_review` means:

- no hard blocker exists
- needs-review candidates are explicitly allowed
- missing optional metadata exists
- would-write counts are aggregate only
- no raw values are emitted
- no dataset artifact is created
- redaction passed

`blocked` means:

- upstream design plan preflight blocked
- domain validation blocked
- value resolution precheck blocked
- required fields are missing
- blocked or rejected candidates are present
- output directory is dirty
- raw values or molecular strings are detected
- writer execution was attempted
- rows were serialized
- dataset artifact path was created
- CSV/JSONL/Parquet/LMDB artifact was created
- Phase 1 was attempted
- DatasetConfirmation changed
- model training or evaluation was attempted
- redaction failed

## Future Dry-Run Precheck Expectations

After a future dry-run implementation exists, the next gate should be a
dry-run precheck that validates:

- dry-run report schema
- summary schema
- report and summary hash consistency
- basename-only file references
- status consistency
- would-write counts
- no raw values
- no serialized rows
- no output artifact paths
- no CSV/JSONL/Parquet/LMDB creation
- no conformer or DPA3 files
- no Phase 1
- no DatasetConfirmation mutation
- no model training or evaluation
- no forbidden imports or calls

## Implementation Blockers

Future dry-run implementation remains blocked until:

- controlled writer design plan preflight has passed
- future dry-run schema fields are reviewed
- controlled writer dry-run precheck expectations are reviewed
- side-effect policy is tested
- redaction policy is tested
- output directory policy is tested
- explicit needs-review handling is documented

## Pass Criteria

The dry-run design can be treated as passed only when:

```text
controlled_writer_design_plan_preflight_status=passed
domain_validation_boundary_status=passed
controlled_writer_value_resolution_dry_run_precheck_status=passed
accepted_candidate_record_count_meets_minimum=true
missing_required_field_count=0
redaction_status=passed
controlled_writer_executed=false
training_dataset_materialized=false
dataset_artifact_created=false
serialized_rows_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

## Needs-Review Criteria

The dry-run design can be marked needs-review when no hard blocker exists and
one or more upstream needs-review statuses are explicitly allowed by future
policy. Needs-review evidence remains non-authoritative and must not be treated
as writer-ready by default.

## Fail Criteria

The dry-run design fails if required upstream evidence is missing, design plan
preflight is blocked, domain validation is blocked, value resolution precheck
is blocked, blocked or rejected candidates appear, redaction fails, or boundary
flags indicate writer execution, value materialization, row serialization,
artifact creation, Phase 1 execution, DatasetConfirmation mutation, model
training, or evaluation.

## Residual Risks

- This document defines dry-run design only and does not prove a future
  implementation.
- Future dry-run code still requires a separate implementation review.
- Future dry-run report and summary schemas still require validation by a
  separate precheck.
- Passing this design does not authorize writer execution.

## Next Step

The next step is property training dataset controlled writer dry-run in a
separate PR. Writer execution remains out of scope for this design.
