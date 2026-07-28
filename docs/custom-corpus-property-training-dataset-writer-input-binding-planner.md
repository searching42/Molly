# Custom Corpus Property Training Dataset Writer Input Binding Planner

The property training dataset writer input binding planner creates a safe
field-source binding plan for future controlled dataset writer work.

It sits after the property training dataset writer execution request preflight
and before the property training dataset writer input binding plan preflight.

The planner answers which allowed source artifact and derivation rule should
populate each future row field. It does not materialize field values and it
does not serialize training rows.

## Purpose

A writer execution request proves which records may be written and which row
contract applies. It does not prove where value-bearing fields such as
`property_value`, `property_unit`, `property_value_normalized`,
`property_unit_normalized`, `compound_id`, or `canonical_smiles` will come
from.

This planner binds future row fields to safe source artifact labels, source
hashes, source record ids, and derivation-rule labels. It keeps raw values and
canonical structure strings out of the emitted plan.

## Inputs

The planner reads the writer execution request preflight, writer execution
request and summary, materialization dry-run precheck, dry-run report and
summary, row contract precheck, row contract and summary, materialization plan
precheck, materialization plan and planner summary, ledger evidence, training
admission execution evidence, request draft evidence, request plan/preflight,
training admission readiness evidence, and quarantine candidate evidence.

All inputs are local JSON artifacts. The planner does not read PDFs,
ParsedDocument output, MinerU bundles, or dataset output paths.

## Plan Schema

The planner writes:

```text
custom_corpus_property_training_dataset_writer_input_binding_plan.v1
```

Plan statuses:

- `planned`
- `needs_review`
- `blocked`

`planned` means all hard checks passed and no allowed needs-review evidence
remains. `needs_review` means the package is internally consistent but either
the writer request preflight or required field binding evidence still needs
review under an explicit allowance. `blocked` means a schema, status, hash,
id, record, field-binding, boundary, or redaction check failed.

## Summary Schema

The planner summary schema is:

```text
custom_corpus_property_training_dataset_writer_input_binding_planner.v1
```

It includes safe basenames and SHA-256 hashes for inputs, source ids/statuses,
binding record counts, writer request ids, row preview ids, planned candidate
ids, missing required/optional field counts, planner errors, warnings, and
redaction status.

## Required Field Binding Rules

The planner creates binding entries for every required field in the row
contract:

- `dataset_record_id`
- `candidate_record_id`
- `record_id`
- `document_id`
- `field_name`
- `property_name`
- `property_value`
- `property_unit`
- `property_value_normalized`
- `property_unit_normalized`
- `task_type`
- `compound_id`
- `canonical_smiles`
- `source_artifact_sha256`
- `review_artifact_sha256`
- `admission_request_sha256`
- `training_admission_execution_ledger_sha256`
- `training_dataset_materialization_plan_sha256`

Identity and provenance fields can bind to existing ID/hash artifacts.
Value-bearing fields are marked `bound` only when an allowed source artifact
declares field availability. If a value-bearing field has no safe source
declaration, the binding is marked `missing_source`.

With the default `--require-all-required-fields-bound` behavior, any missing
required source blocks the plan. With `--no-require-all-required-fields-bound`,
missing required sources produce `needs_review` when no hard consistency
error exists.

## Optional Field Binding Rules

The planner also creates optional field binding entries from the row contract.
Optional fields without safe source declarations are tracked as missing
optional fields. Missing optional fields do not block the plan by themselves.

## Value-Bearing Field Safety

The planner may include field names, source labels, source hashes, source
record ids, and derivation-rule labels. It must not include raw property
values, canonical SMILES strings, InChI/InChIKey values, raw table rows, raw
article text, serialized dataset rows, PDF names or paths, local paths, or
future dataset output paths.

Allowed source artifact labels are:

- `writer_execution_request`
- `materialization_dry_run_report`
- `row_contract`
- `materialization_plan`
- `training_admission_execution_ledger`
- `quarantine_candidate_records`

## Dedup And Split Binding Rules

Each binding record includes a dedup/split binding plan. The dedup key rule is
derived from canonical molecule identity, property name, normalized value,
normalized unit, and source artifact hash. The split group key defaults to
canonical molecule identity to reduce molecule-level leakage. Row-id based
splitting is explicitly forbidden.

The dedup key and split group key are not materialized by this planner.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner \
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
  --output-dir /tmp/property-training-dataset-writer-input-binding \
  --writer-input-binding-plan-id property-writer-input-binding-plan-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-writer-input-binding-plan
```

Return codes:

- `0` for `planned` or `needs_review`
- `1` for `blocked`

## Redaction

The planner scans the plan, summary, binding records, and Markdown evidence
before writing. It fail-closes if forbidden material appears, including local
paths, private paths, credentials, PDF names or paths, CSV/JSONL/Parquet/LMDB
paths, raw article text, raw table rows, serialized rows, conformer data, DPA3
structure data, obvious SMILES strings, or InChI strings.

## After Planning: Input Binding Plan Preflight

After a writer input binding plan is generated, the next governance layer is
the input binding plan preflight:

- `docs/custom-corpus-property-training-dataset-writer-input-binding-plan-preflight.md`
- `docs/evidence/templates/custom-corpus-property-training-dataset-writer-input-binding-plan-preflight-evidence-template.md`

The planner output is not authoritative by itself. The preflight revalidates
the binding plan against upstream writer request, row contract, dry-run,
materialization, ledger, readiness, and quarantine evidence before any future
controlled writer can use it. The preflight still does not execute a writer,
materialize values, or serialize training rows.

## Boundaries

- The planner creates input bindings only.
- The planner does not execute a dataset writer.
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
- A planned input binding package is necessary but not sufficient for future
  controlled dataset writing.
