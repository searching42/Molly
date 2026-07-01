# Custom Corpus Property Training Dataset Controlled Writer Execution Plan Preflight

The property training dataset controlled writer execution plan preflight
validates a controlled writer execution plan package before any future writer
can be implemented or invoked.

It sits after the controlled writer execution plan and before any future
controlled training dataset writer.

## Purpose

The controlled writer execution plan defines how a future writer may be
invoked. This preflight independently checks that the plan is safe,
hash-bound, internally consistent, and still free of source payload reads,
materialized values, serialized rows, output paths, and dataset artifacts.

It does not execute a writer and does not create a dataset.

## Inputs

The preflight reads local JSON artifacts only:

- controlled writer execution plan
- controlled writer execution planner summary
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

## Schema

The preflight summary schema is:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight.v1
```

Statuses:

- `passed`: the controlled writer execution plan and upstream evidence are
  consistent and no needs-review evidence remains.
- `needs_review`: no hard error exists, but explicitly allowed needs-review
  or partial evidence remains.
- `blocked`: schema, status, SHA, id, record, output-label, boundary, or
  redaction checks failed.

## Checks

The preflight validates:

- plan schema and planner summary schema
- `writer_execution_mode=controlled_writer_execution_plan_only`
- controlled writer plan status
- upstream preflight, planner, request, dry-run, row contract, materialization
  plan, ledger, training admission, and quarantine statuses
- SHA-256 bindings across the full chain
- corpus, dataset, row contract, materialization plan, writer request, input
  binding plan, and value source manifest ids
- requested output formats as labels only
- planned output artifact labels as labels only, not paths
- allowed source artifact basenames and SHA-256 hashes
- allowed value field names against value source manifest coverage
- row count expectations against value source records, writer request records,
  and binding records
- boundary flags showing no writer execution, no source payload reads, no
  value materialization, no dataset artifacts, no Phase 1, no
  `DatasetConfirmation` change, no model training, and no evaluation

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight \
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
  --output-summary /tmp/property_training_dataset_controlled_writer_execution_plan_preflight_summary.json \
  --output-markdown /tmp/property_training_dataset_controlled_writer_execution_plan_preflight_summary.md
```

Optional controls:

- `--allow-controlled-writer-execution-plan-needs-review`
- `--no-require-controlled-writer-execution-plan-planned`
- `--minimum-value-source-records <n>`

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The preflight scans the summary and Markdown evidence before writing. It
fail-closes if forbidden material appears, including raw property values,
canonical SMILES, InChI/InChIKey values, raw table rows, raw article text,
local paths, private paths, PDF names or paths, serialized rows,
training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structure data, credentials, full upstream payloads, or output paths.

## Boundaries

- The preflight validates a controlled writer execution plan only.
- The preflight does not execute a writer.
- The preflight does not read source payloads.
- The preflight does not materialize values.
- The preflight does not create serialized training rows.
- The preflight does not materialize a training dataset.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not generate conformers.
- The preflight does not generate DPA3 structures.
- The preflight does not create Uni-Mol or DPA3 input artifacts.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- The preflight does not call LLMs, MinerU, PDF parsers, or corpus workflows.
- A passed controlled writer execution plan preflight is necessary but not
  sufficient for future controlled writer execution.
