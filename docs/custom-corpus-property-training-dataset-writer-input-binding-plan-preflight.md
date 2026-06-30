# Custom Corpus Property Training Dataset Writer Input Binding Plan Preflight

The property training dataset writer input binding plan preflight validates a
writer input binding plan before any future controlled dataset writer can use
it.

It sits after the property training dataset writer input binding planner and
before any future writer input binding execution or controlled dataset writer.

The preflight checks the binding plan, planner summary, writer request
preflight package, row contract, dry-run evidence, materialization plan,
ledger evidence, training admission evidence, and quarantine candidate
evidence. It emits safe JSON and optional Markdown evidence only.

## Purpose

The writer input binding planner defines where future row fields may come
from. This preflight verifies that the plan still matches all upstream hashes,
ids, record counts, source labels, derivation rules, dedup/split rules, and
boundary flags before it can be handed to a future writer input binding step.

It does not materialize values and it does not serialize training rows.

## Inputs

Required inputs:

- training dataset writer input binding plan
- training dataset writer input binding planner summary
- training dataset writer execution request preflight
- training dataset writer execution request and summary
- training dataset materialization dry-run precheck, report, and summary
- training dataset row contract precheck, contract, and summary
- training dataset materialization plan precheck, plan, and planner summary
- training admission execution ledger precheck, ledger, and ledger summary
- training admission execution dry-run precheck and report
- training admission execution request, summary, and preflight
- training admission request draft, draft summary, and draft precheck
- training admission request plan and preflight
- training admission readiness summary
- quarantine candidate preflight summary and candidate records

The preflight reads local JSON artifacts only. It does not read PDFs,
ParsedDocument output, MinerU bundles, dataset output paths, or raw article
text.

## Preflight Schema

The preflight emits:

```text
custom_corpus_property_training_dataset_writer_input_binding_plan_preflight.v1
```

Preflight statuses:

- `passed`
- `needs_review`
- `blocked`

`passed` means all hard checks passed and no needs-review evidence remains.
`needs_review` means the package is internally consistent but an explicitly
allowed needs-review binding plan or missing required field source remains.
`blocked` means schema, status, SHA, id, record, field-binding, boundary, or
redaction checks failed.

## Required Checks

The preflight validates:

- binding plan and planner summary schema versions
- writer request package schema and status fields
- all upstream source SHA-256 values
- corpus, dry-run, materialization plan, row contract, writer request, and
  candidate ids where present
- binding record count and binding record ids
- writer request record ids and row preview ids
- planned dataset candidate ids
- excluded, blocked, and needs-review candidate leakage
- required and optional field binding statuses
- allowed source artifact labels
- safe derivation rule labels
- `values_materialized=false`
- dedup key and split group key rules are labels only
- row-id based splitting remains forbidden
- `writer_executed=false`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

## Field Binding Safety

Binding records may include safe ids, safe field names, source artifact
labels, source hashes, source record ids, and derivation-rule labels.

Binding records must not include raw property values, canonical SMILES,
InChI/InChIKey values, raw table rows, article text, PDF names or paths, local
paths, future dataset output paths, serialized training rows, conformer data,
or DPA3 structure data.

Allowed source artifact labels are:

- `writer_execution_request`
- `materialization_dry_run_report`
- `row_contract`
- `materialization_plan`
- `training_admission_execution_ledger`
- `quarantine_candidate_records`

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight \
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
  --output-summary /tmp/property_training_dataset_writer_input_binding_plan_preflight_summary.json \
  --output-markdown /tmp/property_training_dataset_writer_input_binding_plan_preflight_summary.md
```

Optional controls:

- `--allow-binding-plan-needs-review`
- `--no-require-binding-plan-planned`
- `--minimum-binding-records <n>`
- `--no-require-all-required-fields-bound`

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The preflight scans the JSON summary and Markdown evidence before writing. It
fail-closes if forbidden material appears, including local paths, private
paths, credentials, PDF names or paths, CSV/JSONL/Parquet/LMDB paths, raw
article text, raw table rows, serialized rows, conformer data, DPA3 structure
data, obvious SMILES strings, or InChI strings.

## Boundaries

- The preflight validates writer input binding plans only.
- The preflight does not execute a dataset writer.
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
- A passed preflight is necessary but not sufficient for future controlled
  dataset writing.
