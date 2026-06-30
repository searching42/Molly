# Custom Corpus Property Training Dataset Writer Execution Request

The property training dataset writer execution request builder creates a
reviewable request packet for a future dataset writer.

It sits after the property training dataset materialization dry-run precheck.
It reads the dry-run precheck, dry-run report and summary, row contract
package, materialization plan package, ledger evidence, training admission
execution evidence, request draft evidence, request plan/preflight evidence,
training admission readiness evidence, and quarantine candidate evidence.

It does not execute a writer.
It does not create a dataset.

## Purpose

The builder answers whether dry-run row preview summaries can be safely turned
into a future writer execution request. The request records are ID/hash-only
instructions for a later writer layer. They are not serialized training rows
and they do not contain output paths.

## Output Artifacts

The builder writes a clean run directory:

```text
<output-dir>/<writer-execution-request-id>/
  property_training_dataset_writer_execution_request.json
  property_training_dataset_writer_execution_request_summary.json
  redacted_property_training_dataset_writer_execution_request_evidence.md
```

The request schema is:

```text
custom_corpus_property_training_dataset_writer_execution_request.v1
```

The summary schema is:

```text
custom_corpus_property_training_dataset_writer_execution_request_builder.v1
```

## Request Records

Each request record is derived from one dry-run row preview. It contains safe
ids, status labels, requested output-format labels, model-family labels, and
SHA-256 bindings. It must not include raw property values, raw table rows,
article text, PDF names or paths, local paths, serialized rows, future output
paths, conformer data, or DPA3 structures.

The requested action is a label only:

```text
write_training_dataset_row_later
```

## Validation Rules

The builder validates:

- dry-run precheck status, schema, and redaction boundary
- dry-run report and summary schema/status/hash consistency
- row contract, materialization plan, ledger, request, readiness, and
  quarantine source hashes
- corpus, materialization plan, row contract, materialization dry-run,
  execution ledger, execution request, and dataset-name consistency
- row preview counts, ids, planned dataset records, ledger ids, and candidate
  ids
- excluded, blocked, and needs-review candidate leakage
- requested output-format labels
- clean output directory and explicit confirmation flag

`needs_review` dry-run precheck evidence blocks by default. It can produce a
`needs_review` request only when `--allow-dry-run-precheck-needs-review` is
explicitly set and no hard consistency error exists.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request \
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
  --output-dir /tmp/property-training-dataset-writer-execution-request \
  --writer-execution-request-id property-training-dataset-writer-request-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-writer-execution-request
```

Return codes:

- `0` for `written` or `needs_review`
- `1` for `blocked`

## Boundaries

- The builder creates a training dataset writer execution request only.
- The builder does not execute a dataset writer.
- The builder does not create training dataset artifacts.
- The builder does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The builder does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The builder does not generate conformers.
- The builder does not generate DPA3 structures.
- The builder does not run Phase 1.
- The builder does not modify `DatasetConfirmation`.
- The builder does not run model training or evaluation.
- A writer execution request is necessary but not sufficient for future
  dataset writing.
