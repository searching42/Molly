# Custom Corpus Property Training Dataset Writer Value Source Manifest Planner

The property training dataset writer value source manifest planner creates
safe value-source authorization metadata for future controlled dataset writer
work.

It sits after the property training dataset writer input binding plan
preflight and before any future value source manifest preflight or controlled
training dataset writer.

## Purpose

The writer input binding plan preflight proves that future row fields are
bound to safe source labels, source hashes, source record ids, and
derivation-rule labels. It still does not authorize a future writer to read
value-bearing source artifacts.

This planner defines which value-bearing source artifacts a future controlled
writer may read later, and which value-bearing row fields each source may
satisfy. It does not read source payloads and it does not materialize values.

## Relationship To Future Writer Work

A controlled writer cannot safely run from field bindings alone. It also needs
a manifest that binds each value-bearing field to an allowed artifact basename
and SHA-256 hash. This planner emits that manifest as authorization metadata
only. Future writer execution remains a separate explicit step.

## Value Source Records

The planner creates one value source record for each bound value-bearing field
in each writer input binding record.

Value-bearing fields:

- `property_name`
- `property_value`
- `property_unit`
- `property_value_normalized`
- `property_unit_normalized`
- `compound_id`
- `canonical_smiles`

Each value source record contains safe ids, the value field name, allowed
source artifact label, source artifact basename, source SHA-256, source record
id, derivation-rule label, and boundary flags. It must not contain source
payload values.

Allowed source artifact labels:

- `writer_execution_request`
- `materialization_dry_run_report`
- `row_contract`
- `materialization_plan`
- `training_admission_execution_ledger`
- `quarantine_candidate_records`

## Value Field Coverage

The planner reports value field coverage counts and missing value source field
counts. By default, bound required value fields must be covered. If the input
binding preflight is explicitly allowed in `needs_review`, missing value
sources can remain review evidence instead of authorizing writer execution.

## Manifest Schema

The manifest schema is:

```text
custom_corpus_property_training_dataset_writer_value_source_manifest.v1
```

Manifest statuses:

- `planned`
- `needs_review`
- `blocked`

The manifest includes safe basenames and SHA-256 hashes for input artifacts,
source ids/statuses, value source records, value field coverage, missing value
source counts, planner errors, warnings, redaction status, and a boundary
statement.

## Summary Schema

The planner summary schema is:

```text
custom_corpus_property_training_dataset_writer_value_source_manifest_planner.v1
```

It contains safe basenames only, SHA-256 hashes, aggregate counts, value
source record ids, binding ids, writer request record ids, planned candidate
ids, coverage summaries, missing-source summaries, planner errors, warnings,
and redaction status.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner \
  --training-dataset-writer-input-binding-plan-preflight /tmp/property_training_dataset_writer_input_binding_plan_preflight_summary.json \
  --training-dataset-writer-input-binding-plan /tmp/property_training_dataset_writer_input_binding_plan.json \
  --training-dataset-writer-input-binding-planner-summary /tmp/property_training_dataset_writer_input_binding_planner_summary.json \
  --training-dataset-writer-execution-request-preflight /tmp/property_training_dataset_writer_execution_request_preflight_summary.json \
  --training-dataset-writer-execution-request /tmp/property_training_dataset_writer_execution_request.json \
  --training-dataset-writer-execution-request-summary /tmp/property_training_dataset_writer_execution_request_summary.json \
  --training-dataset-materialization-dry-run-precheck /tmp/property_training_dataset_materialization_dry_run_precheck_summary.json \
  --training-dataset-materialization-dry-run-report /tmp/property_training_dataset_materialization_dry_run_report.json \
  --training-dataset-materialization-dry-run-summary /tmp/property_training_dataset_materialization_dry_run_summary.json \
  --training-dataset-row-contract-precheck /tmp/property_training_dataset_row_contract_precheck_summary.json \
  --training-dataset-row-contract /tmp/property_training_dataset_row_contract.json \
  --training-dataset-row-contract-summary /tmp/property_training_dataset_row_contract_summary.json \
  --training-dataset-materialization-plan-precheck /tmp/property_training_dataset_materialization_plan_precheck_summary.json \
  --training-dataset-materialization-plan /tmp/property_training_dataset_materialization_plan.json \
  --training-dataset-materialization-planner-summary /tmp/property_training_dataset_materialization_planner_summary.json \
  --training-admission-execution-ledger-precheck /tmp/property_training_admission_execution_ledger_precheck_summary.json \
  --training-admission-execution-ledger /tmp/property_training_admission_execution_ledger.json \
  --training-admission-execution-ledger-summary /tmp/property_training_admission_execution_ledger_summary.json \
  --training-admission-execution-dry-run-precheck /tmp/property_training_admission_execution_dry_run_precheck_summary.json \
  --training-admission-execution-dry-run-report /tmp/property_training_admission_execution_dry_run_report.json \
  --training-admission-execution-request /tmp/property_training_admission_execution_request.json \
  --training-admission-execution-request-summary /tmp/property_training_admission_execution_request_summary.json \
  --training-admission-execution-request-preflight /tmp/property_training_admission_execution_request_preflight_summary.json \
  --training-admission-request-draft /tmp/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/property_training_admission_request_draft_precheck_summary.json \
  --training-admission-request-plan /tmp/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /tmp/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /tmp/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /tmp/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /tmp/property_quarantine_candidate_records.json \
  --output-dir /tmp/property-training-dataset-writer-value-source-manifest \
  --value-source-manifest-id property-value-source-manifest-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-writer-value-source-manifest
```

Optional controls:

- `--allow-input-binding-preflight-needs-review`
- `--minimum-value-source-records <n>`
- `--no-require-all-bound-value-fields-covered`

Return codes:

- `0` for `planned` or `needs_review`
- `1` for `blocked`

## Redaction

The planner scans the manifest, summary, value source records, and Markdown
before writing. It fail-closes if forbidden material appears, including raw
property values, canonical SMILES, InChI/InChIKey values, raw table rows, raw
article text, local paths, private paths, PDF names or paths, serialized rows,
training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structure data, credentials, or full upstream payloads.

## Boundaries

- The planner creates value source authorization metadata only.
- The planner does not execute a dataset writer.
- The planner does not read source payloads.
- The planner does not materialize values.
- The planner does not create serialized training rows.
- The planner does not create training dataset artifacts.
- The planner does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not generate conformers.
- The planner does not generate DPA3 structures.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner does not run model training or evaluation.
- A planned value source manifest is necessary but not sufficient for future
  controlled dataset writing.
