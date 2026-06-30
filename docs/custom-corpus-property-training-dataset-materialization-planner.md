# Custom Corpus Property Training Dataset Materialization Planner

The property training dataset materialization planner reads a
ledger-precheck-passed property training admission package and produces a safe
plan for future training dataset writing.

The planner creates a plan only. It does not create training dataset
artifacts, does not create training CSV/JSONL/Parquet/LMDB artifacts, does not
create candidate CSV/JSONL/Parquet/LMDB artifacts, does not run Phase 1, does
not modify `DatasetConfirmation`, and does not run model training or
evaluation.

## Relationship To Ledger Precheck

The upstream ledger precheck is documented in:

```text
docs/custom-corpus-property-training-admission-execution-ledger-precheck.md
```

The ledger precheck validates the execution ledger and upstream evidence. The
training dataset materialization planner consumes that evidence and emits a
safe plan for future dataset writing. A planned materialization package is
necessary but not sufficient for future training dataset writing.

## Inputs

The planner requires:

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

It reads local JSON artifacts only. It does not read PDFs, ParsedDocument
outputs, MinerU bundles, raw extracted text, candidate/training dataset files,
or training outputs.

## Planning Rules

The planner validates:

- schema versions for every input
- ledger precheck status and optional needs-review allowance
- execution ledger status and source boundary fields
- source SHA-256 bindings across the full upstream chain
- corpus, dry-run, review, admission, materialization, execution, quarantine,
  review queue, and property candidate ids
- ledger, dry-run, execution request, draft, and planned candidate record counts
- candidate eligibility and leakage checks
- planned output format labels
- target model family labels

For each ledger record, the planner emits one safe planned dataset record
summary. Planned dataset records are ID/hash/label summaries only. They are not
serialized training rows.

## Schemas

Plan artifact:

```text
custom_corpus_property_training_dataset_materialization_plan.v1
```

Planner summary:

```text
custom_corpus_property_training_dataset_materialization_planner.v1
```

## Status Meanings

`planned` means the ledger precheck passed, upstream evidence is consistent,
and a safe materialization plan was written.

`needs_review` means no hard error was found, but ledger precheck or upstream
evidence carries an explicitly allowed needs-review or partial status.

`blocked` means confirmation was missing or schema, status, SHA, id, record
eligibility, output directory, source boundary, or redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_planner \
  --training-admission-execution-ledger-precheck /tmp/custom-corpus-property-training-admission-execution-ledger-precheck-summary.json \
  --training-admission-execution-ledger /tmp/property-training-admission-execution-ledger/property-training-admission-execution-ledger-example-001/property_training_admission_execution_ledger.json \
  --training-admission-execution-ledger-summary /tmp/property-training-admission-execution-ledger/property-training-admission-execution-ledger-example-001/property_training_admission_execution_ledger_summary.json \
  --training-admission-execution-dry-run-precheck /tmp/custom-corpus-property-training-admission-execution-dry-run-precheck-summary.json \
  --training-admission-execution-dry-run-report /tmp/custom-corpus-property-training-admission-execution-dry-run/property-training-admission-execution-dry-run-example-001/property_training_admission_execution_dry_run_report.json \
  --training-admission-execution-request /tmp/property-training-admission-execution-request/property-training-admission-execution-request-example-001/property_training_admission_execution_request.json \
  --training-admission-execution-request-summary /tmp/property-training-admission-execution-request/property-training-admission-execution-request-example-001/property_training_admission_execution_request_summary.json \
  --training-admission-execution-request-preflight /tmp/custom-corpus-property-training-admission-execution-request-preflight-summary.json \
  --training-admission-request-draft /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/custom-corpus-property-training-admission-request-draft-precheck-summary.json \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-request-preflight /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --quarantine-candidate-records /tmp/property_quarantine_candidate_records.json \
  --output-dir /tmp/custom-corpus-property-training-dataset-materialization-plan \
  --materialization-plan-id property-training-dataset-materialization-plan-example-001 \
  --created-by operator-redacted \
  --dataset-name property-training-dataset \
  --planned-output-format jsonl \
  --target-model-family generic_property_predictor \
  --confirm-training-dataset-materialization-plan
```

Return codes:

- `0` when plan status is `planned` or `needs_review`
- `1` when plan status is `blocked`

## Output Artifacts

The planner creates a run-scoped clean directory:

```text
<output-dir>/<materialization-plan-id>/
  property_training_dataset_materialization_plan.json
  property_training_dataset_materialization_planner_summary.json
  redacted_property_training_dataset_materialization_plan_evidence.md
```

It does not write training CSV, training JSONL, training Parquet, training
LMDB, candidate CSV, candidate JSONL, candidate Parquet, candidate LMDB,
Phase 1 artifacts, `DatasetConfirmation` artifacts, model training artifacts,
or evaluation artifacts.

## After Materialization Planner: Plan Precheck

The next pre-dataset-writing layer is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-plan-precheck.md
```

Future plan precheck evidence template:

```text
docs/evidence/templates/custom-corpus-property-training-dataset-materialization-plan-precheck-evidence-template.md
```

The planner creates a plan only. The plan precheck validates that plan package
against the full upstream ledger, dry-run, execution request, request draft,
request planning, readiness, and quarantine evidence chain. The plan precheck
still does not create training dataset artifacts, training
CSV/JSONL/Parquet/LMDB artifacts, candidate CSV/JSONL/Parquet/LMDB artifacts,
Phase 1 inputs, or `DatasetConfirmation` changes. The future dataset
writer/materializer remains separate.

## Boundaries

- The planner creates a plan only.
- The planner does not create training dataset artifacts.
- The planner does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner does not run model training or evaluation.
- The planner does not call an LLM or agent.
- The planner does not call MinerU.
- The planner does not parse PDFs.
- The planner does not read ParsedDocument content.
- A planned materialization package is necessary but not sufficient for future
  training dataset writing.
