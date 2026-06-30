# Custom Corpus Property Training Dataset Writer Execution Request Preflight

The property training dataset writer execution request preflight validates an
existing writer execution request package before any future controlled dataset
writer is considered.

It reads the writer execution request, writer execution request summary,
materialization dry-run precheck, materialization dry-run report and summary,
row contract package, materialization plan package, ledger evidence, training
admission execution evidence, request draft evidence, request plan/preflight
evidence, training admission readiness evidence, and quarantine candidate
evidence.

It does not execute a writer.
It does not create a dataset.

## Purpose

The preflight answers whether the writer execution request is internally
consistent, hash-bound to validated upstream evidence, and still free of
serialized rows, output paths, and dataset artifacts.

The preflight output is evidence for a future controlled writer. It is not a
writer run and it is not a dataset artifact.

## Summary Schema

The preflight emits:

```text
custom_corpus_property_training_dataset_writer_execution_request_preflight.v1
```

Status values:

- `passed`
- `needs_review`
- `blocked`

`passed` means all checks passed and no needs-review evidence remains.
`needs_review` means no hard error was found, but allowed needs-review evidence
is present. `blocked` means schema, status, hash, id, record, boundary, or
redaction checks failed.

## Validation Rules

The preflight validates:

- writer execution request schema, status, mode, and boundary fields
- writer execution request summary schema and request SHA binding
- materialization dry-run precheck status and source hashes
- dry-run report and summary consistency
- row contract, materialization plan, ledger, request, readiness, and
  quarantine source hashes
- corpus, writer request, materialization dry-run, materialization plan, row
  contract, execution ledger, execution request, and dataset-name consistency
- writer request record counts and ids
- row preview ids, planned dataset record ids, ledger ids, and candidate ids
- excluded, blocked, and needs-review candidate leakage
- requested writer mode and output-format labels
- redaction boundaries

`needs_review` writer request evidence blocks by default. It can produce a
`needs_review` preflight only when `--allow-writer-request-needs-review` is
explicitly set and no hard consistency error exists.

## Request Record Safety

Writer request records are safe ID/hash/label summaries only. They may include
field names, request ids, row preview ids, planned dataset ids, candidate ids,
ledger ids, SHA-256 hashes, output-format labels, and model-family labels.

They must not include raw property values, raw table rows, raw article text,
PDF names or paths, local paths, serialized training rows, concrete output
file paths, CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3 structures,
full candidate payloads, full materialized payloads, or full upstream
payloads.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight \
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
  --output-summary /tmp/property_training_dataset_writer_execution_request_preflight_summary.json \
  --output-markdown /tmp/property_training_dataset_writer_execution_request_preflight_summary.md
```

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Boundaries

- The preflight validates writer request evidence only.
- The preflight does not execute a dataset writer.
- The preflight does not create training dataset artifacts.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not generate conformers.
- The preflight does not generate DPA3 structures.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- A passed preflight is necessary but not sufficient for future controlled
  dataset writing.
