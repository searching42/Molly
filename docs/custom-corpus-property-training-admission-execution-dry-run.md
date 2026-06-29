# Custom Corpus Property Training Admission Execution Dry-Run

The property training admission execution dry-run validates an
execution-request-preflighted package and simulates which records a future
training admission execution layer would admit. It emits safe ID/hash-only
dry-run record summaries and redacted evidence.

The dry-run simulates execution only. It does not execute training admission,
admit training data, create training CSV/JSONL/Parquet/LMDB artifacts, create
candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, modify
`DatasetConfirmation`, or run model training or evaluation.

## Relationship To Execution Request Preflight

The upstream execution request preflight is documented in:

```text
docs/custom-corpus-property-training-admission-execution-request-preflight.md
```

The preflight validates package consistency for an existing execution request.
This dry-run then simulates future training admission execution as labels only.
Actual training admission execution remains separate and unimplemented.

## After Dry-Run: Dry-Run Precheck

The downstream dry-run precheck is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run-precheck.md
```

Future evidence template:

```text
docs/evidence/templates/custom-corpus-property-training-admission-execution-dry-run-precheck-evidence-template.md
```

The precheck reads an existing dry-run report and validates that it remains
bound to the execution request, execution request preflight, draft package,
request plan, readiness summary, and quarantine candidate records. It still
does not execute training admission or create training artifacts.

## Inputs

The dry-run requires:

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
outputs, MinerU bundles, raw extracted text, candidate/training
CSV/JSONL/Parquet/LMDB files, or training outputs.

## Dry-Run Rules

The dry-run checks schema versions, execution-preflight status, source
boundary fields, SHA-256 bindings, safe ids, execution record counts, planned
candidate ids, excluded/blocked/needs-review leakage, and redaction safety.

For each execution request record, it creates one dry-run record with:

- source execution and draft record ids
- candidate/materialization/admission/review/document/field ids
- `would_execute_action=would_admit_training_candidate`
- `dry_run_record_status=would_admit`
- source SHA-256 bindings
- `training_admitted=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

Dry-run records do not contain raw values, raw table rows, article text, PDF
names or paths, local paths, or training output paths.

## Report Schema

The dry-run report uses:

```text
custom_corpus_property_training_admission_execution_dry_run.v1
```

Reports have `dry_run_mode=execution_simulation_only`,
`training_admitted=false`, `phase1_status=not_run`, and
`dataset_confirmation_changed=false`.

## Status Meanings

`passed` means the execution request preflight passed, all upstream evidence is
in the expected ready/passed/planned state, and no dry-run consistency or
redaction check failed.

`needs_review` means no hard error was found, but execution preflight or
upstream evidence carries an explicitly allowed needs-review or partial status.

`blocked` means confirmation was missing or schema, status, SHA, id, record
eligibility, output directory, source boundary, or redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_dry_run \
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
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-dir /tmp/custom-corpus-property-training-admission-execution-dry-run \
  --dry-run-id property-training-admission-execution-dry-run-example-001 \
  --created-by operator-redacted \
  --confirm-training-admission-execution-dry-run
```

Return codes:

- `0` when dry-run status is `passed` or `needs_review`
- `1` when dry-run status is `blocked`

## Output Artifacts

The dry-run writes a run-scoped clean directory:

```text
<output-dir>/<dry-run-id>/
  property_training_admission_execution_dry_run_report.json
  redacted_property_training_admission_execution_dry_run_evidence.md
```

It does not write training CSV, training JSONL, training Parquet, training
LMDB, candidate CSV, candidate JSONL, candidate Parquet, candidate LMDB,
Phase 1 artifacts, `DatasetConfirmation` artifacts, model training artifacts,
or evaluation artifacts.

## Redaction Behavior

Before writing JSON or Markdown, the dry-run scans report and evidence content
for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed URL, and
raw-text markers. If unsafe material is detected, it fails closed with:

```text
property_training_admission_execution_dry_run_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The dry-run simulates execution only.
- The dry-run does not execute training admission.
- The dry-run does not admit training data.
- The dry-run does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The dry-run does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The dry-run does not run Phase 1.
- The dry-run does not modify `DatasetConfirmation`.
- The dry-run does not run model training or evaluation.
- The dry-run does not call an LLM or agent.
- The dry-run does not call MinerU.
- The dry-run does not parse PDFs.
- The dry-run does not read ParsedDocument content.
- A passed dry-run is necessary but not sufficient for future training
  admission execution.
