# Custom Corpus Property Training Dataset Materialization Plan Precheck

The property training dataset materialization plan precheck validates a
`custom_corpus_property_training_dataset_materialization_plan.v1` package
before any future training dataset row contract or writer can consider it.

The precheck answers only whether the plan is internally consistent,
hash-bound to upstream evidence, and safe to hand to a future dataset writer
design. It does not decide to write a dataset.

## Relationship To The Planner

The upstream planner is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-planner.md
```

The planner creates a plan only. The precheck reads that plan, the planner
summary, the execution ledger precheck, the execution ledger and ledger
summary, dry-run evidence, execution request evidence, draft evidence, request
planning evidence, readiness evidence, and quarantine candidate evidence. It
then emits safe precheck evidence.

## Inputs

The precheck requires:

- `custom_corpus_property_training_dataset_materialization_plan.v1`
- `custom_corpus_property_training_dataset_materialization_planner.v1`
- `custom_corpus_property_training_admission_execution_ledger_precheck.v1`
- `custom_corpus_property_training_admission_execution_ledger.v1`
- `custom_corpus_property_training_admission_execution_ledger_summary.v1`
- `custom_corpus_property_training_admission_execution_dry_run_precheck.v1`
- `custom_corpus_property_training_admission_execution_dry_run.v1`
- `custom_corpus_property_training_admission_execution_request.v1`
- `custom_corpus_property_training_admission_execution_request_builder.v1`
- `custom_corpus_property_training_admission_execution_request_preflight.v1`
- `custom_corpus_property_training_admission_request_draft.v1`
- `custom_corpus_property_training_admission_request_draft_builder.v1`
- `custom_corpus_property_training_admission_request_draft_precheck.v1`
- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`

## Precheck Rules

The precheck validates schema versions, source SHA-256 bindings, safe IDs,
status fields, candidate eligibility, planned output format labels, target
model family labels, planned record counts, ledger record binding, and
redaction boundaries.

It requires:

- `training_admitted=true`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

Planned dataset records must remain ID/hash/label summaries. They must not
contain serialized training rows, raw table rows, raw article text, PDF names
or paths, local paths, or training output paths.

## Status Meanings

`passed` means the plan package and upstream evidence are consistent.

`needs_review` means no hard consistency error was found, but the plan or
upstream evidence carries an explicitly allowed needs-review or partial status.

`blocked` means schema, status, SHA, ID, record eligibility, boundary, or
redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck \
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
  --output-summary /tmp/property_training_dataset_materialization_plan_precheck_summary.json \
  --output-markdown /tmp/property_training_dataset_materialization_plan_precheck_summary.md
```

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Redaction

The summary and Markdown evidence contain safe IDs, SHA-256 hashes, aggregate
counts, status fields, safe error codes, output format labels, and target model
family labels only. They must not contain raw rows, raw article text, local
paths, PDF names, private paths, tokens, serialized dataset rows, or dataset
output paths.

## Boundaries

- The precheck validates a materialization plan only.
- The precheck does not create training dataset artifacts.
- The precheck does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not run Phase 1.
- The precheck does not modify `DatasetConfirmation`.
- The precheck does not run model training or evaluation.
- A passed precheck is necessary but not sufficient for future training dataset
  writing.
