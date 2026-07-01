# Custom Corpus Property Training Dataset Controlled Writer Value Resolution Dry-Run

The property training dataset controlled writer value resolution dry-run checks
whether a future controlled writer could resolve required row fields from only
the authorized value-bearing source payloads.

It sits after the controlled writer execution plan preflight and before any
future value-resolution precheck or controlled training dataset writer.

## Purpose

The controlled writer execution plan preflight proves the invocation package is
safe, hash-bound, and label-only. It does not prove that required row fields can
be resolved from authorized source payloads. This dry-run fills that gap.

The dry-run may read explicitly authorized local JSON source payloads, but it
must not emit raw values, serialized rows, output paths, or dataset artifacts.

## Inputs

The dry-run reads local JSON artifacts only:

- controlled writer execution plan preflight
- controlled writer execution plan and planner summary
- value source manifest preflight
- value source manifest and planner summary
- writer input binding plan preflight
- writer input binding plan and planner summary
- writer execution request preflight
- writer execution request and summary
- materialization dry-run precheck, report, and summary
- row contract package
- materialization plan package
- ledger evidence
- training admission evidence
- quarantine candidate evidence

## Schemas

Report:

```text
custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1
```

Summary:

```text
custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1
```

Statuses:

- `passed`: every required field resolves and no needs-review evidence remains.
- `needs_review`: no hard error exists, but explicitly allowed needs-review
  evidence or missing required field coverage remains.
- `blocked`: schema, status, SHA, id, source authorization, field-resolution,
  boundary, output-label, or redaction checks failed.

## Value Resolution

For each writer request or input binding record, the dry-run emits a safe
resolution record with ids, hashes, source labels, field names, and derivation
rule labels only.

Resolution records may include:

- value resolution record id
- writer request record id
- writer input binding record id
- row preview id
- planned dataset record id
- candidate record id
- record id
- document id
- field name
- resolved and missing required field names
- resolved and missing optional field names
- value source record ids
- source artifact labels and SHA-256 hashes
- derivation rule labels

Resolution records must not include raw property values, canonical SMILES,
InChI/InChIKey values, raw table rows, raw article text, PDF names or paths,
local paths, serialized rows, output paths, conformer data, DPA3 structures, or
full source payloads.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run \
  --training-dataset-controlled-writer-execution-plan-preflight /tmp/property_training_dataset_controlled_writer_execution_plan_preflight_summary.json \
  --training-dataset-controlled-writer-execution-plan /tmp/property_training_dataset_controlled_writer_execution_plan.json \
  --training-dataset-controlled-writer-execution-planner-summary /tmp/property_training_dataset_controlled_writer_execution_planner_summary.json \
  --training-dataset-writer-value-source-manifest-preflight /tmp/property_training_dataset_writer_value_source_manifest_preflight_summary.json \
  --training-dataset-writer-value-source-manifest /tmp/property_training_dataset_writer_value_source_manifest.json \
  --training-dataset-writer-value-source-manifest-planner-summary /tmp/property_training_dataset_writer_value_source_manifest_planner_summary.json \
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
  --output-dir /tmp/property-training-dataset-value-resolution-dry-run \
  --value-resolution-dry-run-id property-value-resolution-dry-run-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-controlled-writer-value-resolution-dry-run
```

Optional controls:

- `--allow-controlled-writer-execution-plan-preflight-needs-review`
- `--minimum-resolution-records <n>`
- `--no-require-all-required-fields-resolved`

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The dry-run scans the report, summary, resolution records, and Markdown before
writing. It fail-closes if forbidden material appears, including raw property
values, canonical SMILES, InChI/InChIKey values, raw table rows, raw article
text, local paths, private paths, PDF names or paths, serialized rows,
training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structure data, full upstream payloads, full source payloads, output paths, or
credentials.

## Boundaries

- The dry-run is controlled writer value resolution only.
- The controlled writer is not executed.
- Authorized source payloads may be read.
- Values may be resolved internally but are not emitted.
- Values are not materialized into rows.
- The dry-run does not create serialized training rows.
- The dry-run does not materialize a training dataset.
- The dry-run does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The dry-run does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The dry-run does not generate conformers.
- The dry-run does not generate DPA3 structures.
- The dry-run does not create Uni-Mol or DPA3 input artifacts.
- The dry-run does not run Phase 1.
- The dry-run does not modify `DatasetConfirmation`.
- The dry-run does not run model training or evaluation.
- The dry-run does not call LLMs, MinerU, PDF parsers, or corpus workflows.
- A passed value-resolution dry-run is necessary but not sufficient for future
  controlled writer execution.
