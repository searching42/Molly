# Custom Corpus Property Training Dataset Materialization Dry-Run Precheck

The property training dataset materialization dry-run precheck validates an
existing `custom_corpus_property_training_dataset_materialization_dry_run.v1`
package before any future dataset writer or controlled materializer request.

It answers whether the dry-run report, dry-run summary, row preview summaries,
and upstream evidence are internally consistent, safe, hash-bound, and still
free of serialized training rows or dataset artifacts.

It does not write a dataset.
It does not run the dry-run.

## Relationship To Materialization Dry-Run

The upstream dry-run is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-dry-run.md
```

The dry-run creates safe row preview summaries. The precheck reads those
outputs plus the row contract precheck, row contract package, materialization
plan package, ledger evidence, execution dry-run evidence, execution request
evidence, request draft evidence, request plan/preflight evidence, training
admission readiness evidence, and quarantine candidate evidence. It validates
the full chain again and emits precheck evidence only.

Future dataset writing remains separate.

## After Precheck: Writer Execution Request

After the dry-run package precheck passes, the writer execution request
builder can create a reviewable request packet for a future dataset writer:

```text
docs/custom-corpus-property-training-dataset-writer-execution-request.md
```

Future evidence template:

```text
docs/evidence/templates/custom-corpus-property-training-dataset-writer-execution-request-evidence-template.md
```

The writer execution request remains non-executing. It does not run a dataset
writer, serialize training rows, create training CSV/JSONL/Parquet/LMDB files,
create candidate CSV/JSONL/Parquet/LMDB files, run Phase 1, or modify
`DatasetConfirmation`.

## Validation Rules

The precheck validates:

- dry-run report and summary schema versions
- dry-run status and optional needs-review gating
- dry-run report SHA-256 binding from the summary
- source SHA-256 bindings across all upstream artifacts
- safe basenames, safe ids, and schema/status fields
- materialization dry-run id, row contract id, materialization plan id,
  execution ledger id, execution request id, corpus id, and dataset name
  consistency
- row preview count and row preview ids
- row preview ids, contract reference ids, planned dataset ids, ledger ids, and
  planned candidate ids
- excluded, blocked, and needs-review candidate leakage
- field coverage summary shape
- model-family compatibility summary shape
- output-format compatibility summary shape
- post-dry-run boundary fields
- redaction boundaries

The post-dry-run boundary must remain:

- `training_admitted=true`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

## Row Preview Safety

Row previews are summaries only. They may include safe ids, safe field names,
model-family labels, output-format labels, counts, status labels, and SHA-256
hashes.

Row previews must not include raw property values, raw table rows, raw article
text, PDF names or paths, local paths, serialized training rows, concrete
output paths, conformer data, DPA3 structures, full candidate payloads, or full
materialized payloads.

## Summary Checks

The field coverage summary must preserve required and optional row field
labels from the row contract and must remain aggregate-only.

The model-family compatibility summary must include:

- `generic_property_predictor`
- `unimol`
- `dpa3`

It must keep:

- `conformers_generated=false`
- `dpa3_structures_generated=false`

The output-format compatibility summary must include:

- `jsonl`
- `parquet`
- `lmdb`
- `csv`

It must keep:

- `jsonl_created=false`
- `parquet_created=false`
- `lmdb_created=false`
- `csv_created=false`

## Summary Schema

The precheck emits:

```text
custom_corpus_property_training_dataset_materialization_dry_run_precheck.v1
```

Status values:

- `passed`
- `needs_review`
- `blocked`

`passed` means all checks passed and no needs-review evidence remains.
`needs_review` means no hard error was found, but allowed needs-review evidence
is present. `blocked` means a schema, status, hash, id, record, row preview,
boundary, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run_precheck \
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
  --output-summary /tmp/property_training_dataset_materialization_dry_run_precheck_summary.json \
  --output-markdown /tmp/property_training_dataset_materialization_dry_run_precheck_summary.md
```

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The summary and Markdown evidence may include field names, schema/status
fields, safe ids, SHA-256 hashes, aggregate counts, allowed quality flags,
model-family labels, output-format labels, row preview ids, and safe error
codes.

They must not include raw property values, raw table rows, raw article text,
local absolute paths, private paths, PDF names or paths, serialized dataset
rows, training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structures, or full upstream payloads.

## Boundaries

- The precheck validates dry-run evidence only.
- Row previews are summaries only.
- The precheck does not create training dataset artifacts.
- The precheck does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not generate conformers.
- The precheck does not generate DPA3 structures.
- The precheck does not run Phase 1.
- The precheck does not modify `DatasetConfirmation`.
- The precheck does not run model training or evaluation.
- A passed precheck is necessary but not sufficient for future dataset writing.
