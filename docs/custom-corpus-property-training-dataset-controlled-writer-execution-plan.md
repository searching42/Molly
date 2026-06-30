# Custom Corpus Property Training Dataset Controlled Writer Execution Plan

The property training dataset controlled writer execution plan defines how a
future controlled writer may be invoked without running that writer.

It sits after the value source manifest preflight and before any future
controlled writer execution plan preflight or controlled training dataset
writer.

## Purpose

The value source manifest preflight validates which value-bearing source
artifacts a future writer may read. It still does not define writer invocation
policy, allowed outputs, naming labels, provenance preservation, or redaction
rules. This planner fills that gap by creating a safe execution plan.

The plan answers what a future controlled writer would be allowed to read,
write, and preserve. It does not execute the writer and does not create a
dataset.

## Inputs

The planner reads local JSON artifacts only:

- value source manifest preflight
- value source manifest
- value source manifest planner summary
- writer input binding plan preflight
- writer input binding plan
- writer input binding planner summary
- writer execution request preflight
- writer execution request and summary
- materialization dry-run precheck, report, and summary
- row contract package
- materialization plan package
- ledger evidence
- training admission evidence
- quarantine candidate evidence

## Plan Schema

The controlled writer execution plan schema is:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_plan.v1
```

The planner summary schema is:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1
```

Statuses:

- `planned`: the plan and upstream evidence are consistent.
- `needs_review`: explicitly allowed needs-review evidence remains.
- `blocked`: schema, status, SHA, id, output-label, safety, boundary, or
  redaction checks failed.

## Plan Content

The plan includes:

- safe ids and SHA-256 bindings for all inputs
- `writer_execution_mode=controlled_writer_execution_plan_only`
- requested output formats as labels only
- allowed source artifact basenames and SHA-256 hashes
- allowed value field names
- row contract id and SHA-256
- value source manifest id and SHA-256
- writer input binding plan id and SHA-256
- planned output artifact labels only, not paths
- output directory policy labels only
- file naming policy labels only
- row count expectations as aggregate counts
- provenance preservation requirements
- redaction policy
- boundary statement

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan \
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
  --output-dir /tmp/property-training-dataset-controlled-writer-plan \
  --controlled-writer-execution-plan-id property-controlled-writer-plan-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-controlled-writer-execution-plan
```

Optional controls:

- `--allow-value-source-manifest-preflight-needs-review`
- `--minimum-value-source-records <n>`

Return codes:

- `0` for `planned` or `needs_review`
- `1` for `blocked`

## Redaction

The planner scans the plan, summary, and Markdown evidence before writing. It
fail-closes if forbidden material appears, including raw property values,
canonical SMILES, InChI/InChIKey values, raw table rows, raw article text,
local paths, private paths, PDF names or paths, serialized rows,
training/candidate CSV/JSONL/Parquet/LMDB paths, conformer data, DPA3
structure data, credentials, or full upstream payloads.

## Boundaries

- The planner creates a controlled writer execution plan only.
- The planner does not execute a writer.
- The planner does not read source payloads.
- The planner does not materialize values.
- The planner does not create serialized training rows.
- The planner does not materialize a training dataset.
- The planner does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not generate conformers.
- The planner does not generate DPA3 structures.
- The planner does not create Uni-Mol or DPA3 input artifacts.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner does not run model training or evaluation.
- The planner does not call LLMs, MinerU, PDF parsers, or corpus workflows.
- A controlled writer execution plan is necessary but not sufficient for
  future controlled writer execution.
