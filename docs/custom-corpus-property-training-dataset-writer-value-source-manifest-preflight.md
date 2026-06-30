# Custom Corpus Property Training Dataset Writer Value Source Manifest Preflight

The property training dataset writer value source manifest preflight validates
the value-source manifest package produced by the planner before any future
controlled training dataset writer can use it.

It sits after the property training dataset writer value source manifest
planner and before any future controlled training dataset writer.

## Purpose

The value source manifest planner records which safe source artifact labels,
basenames, hashes, source record ids, and derivation-rule labels may be used
later to populate value-bearing row fields. The preflight checks that package
against the planner summary and upstream evidence chain.

This preflight does not read source payloads and does not materialize values.
It only decides whether the manifest package is internally consistent enough
to be considered by a future controlled writer.

## Inputs

The preflight reads local JSON artifacts only:

- value source manifest
- value source manifest planner summary
- writer input binding plan preflight
- writer input binding plan
- writer input binding planner summary
- writer execution request and preflight evidence
- training dataset dry-run, row contract, materialization plan, ledger,
  training admission, and quarantine candidate evidence

## Statuses

The summary schema is:

```text
custom_corpus_property_training_dataset_writer_value_source_manifest_preflight.v1
```

Statuses:

- `passed`: the value source manifest package and upstream evidence are
  consistent.
- `needs_review`: only explicitly allowed needs-review evidence remains.
- `blocked`: schema, status, SHA, id, count, coverage, safety, or redaction
  checks failed.

## Validation Rules

The preflight validates:

- value source manifest schema and planner summary schema
- all upstream schemas and statuses
- actual SHA-256 values against declared SHA fields
- corpus, dataset, row contract, materialization, ledger, writer request, and
  input binding ids
- value source record count and ids
- binding record ids and writer request record ids
- value field coverage and missing source counts
- excluded, blocked, or needs-review candidates do not leak into the manifest
- source artifact labels are allowed
- source artifact basenames are safe basenames only
- source payload read flags remain false
- value materialization flags remain false
- writer execution flags remain false

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_preflight \
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
  --output-summary /tmp/property_training_dataset_writer_value_source_manifest_preflight_summary.json \
  --output-markdown /tmp/property_training_dataset_writer_value_source_manifest_preflight_summary.md
```

Optional controls:

- `--allow-value-source-manifest-needs-review`
- `--minimum-value-source-records <n>`
- `--no-require-all-value-fields-covered`

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The preflight serializes the summary and Markdown evidence before writing and
fail-closes if forbidden material appears, including local paths, private
paths, PDF names or paths, raw article text, raw table rows, raw property
values, canonical SMILES, InChI/InChIKey values, serialized rows,
training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structure data, credentials, or signed URL markers.

## Boundaries

- The preflight validates value source manifest packages only.
- The preflight does not execute a dataset writer.
- The preflight does not read source payloads.
- The preflight does not materialize values.
- The preflight does not create serialized training rows.
- The preflight does not create training dataset artifacts.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not generate conformers.
- The preflight does not generate DPA3 structures.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- A passed value source manifest preflight is necessary but not sufficient for
  future controlled dataset writing.
