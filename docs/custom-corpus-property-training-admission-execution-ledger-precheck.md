# Custom Corpus Property Training Admission Execution Ledger Precheck

The property training admission execution ledger precheck validates a committed
property training admission execution ledger before any future training dataset
materialization step can consider it.

It reads the execution ledger, ledger summary, execution dry-run/precheck,
execution request/preflight, training admission draft/precheck/plan/readiness,
and quarantine candidate evidence. It emits safe JSON and Markdown evidence
only.

The precheck is not training dataset materialization. It does not create a
training admission request, does not admit additional training data, does not
create CSV/JSONL/Parquet/LMDB artifacts, does not run Phase 1, does not modify
`DatasetConfirmation`, and does not run model training or evaluation.

## Relationship To Execution Ledger

The upstream execution ledger is documented in:

```text
docs/custom-corpus-property-training-admission-execution-ledger.md
```

The ledger records safe ID/hash-only training admission ledger entries. The
precheck validates that ledger and its upstream evidence before a future
training dataset materialization layer. A passed precheck is necessary but not
sufficient for future training dataset materialization.

## Inputs

The precheck requires:

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

## Precheck Rules

The precheck validates:

- schema versions for every input
- ledger status and optional needs-review allowance
- source SHA-256 bindings across the full upstream chain
- corpus, dry-run, review, admission, materialization, execution, quarantine,
  review queue, and property candidate ids
- ledger, dry-run, execution request, draft, and planned candidate record counts
- candidate eligibility and leakage checks
- source boundary fields:
  - `phase1_status=not_run`
  - `dataset_confirmation_changed=false`
  - no pre-ledger training admission
  - no dataset artifact materialization

The precheck blocks if excluded, blocked, or needs-review candidates appear in
ledger records.

## Schema

```text
custom_corpus_property_training_admission_execution_ledger_precheck.v1
```

## Status Meanings

`passed` means the ledger is committed and all checked upstream evidence is
consistent.

`needs_review` means no hard error was found, but a needs-review ledger or
upstream status was explicitly allowed.

`blocked` means schema, status, SHA, id, record eligibility, boundary, or
redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck \
  --training-admission-execution-ledger /tmp/custom-corpus-property-training-admission-execution-ledger/property-training-admission-execution-ledger-example-001/property_training_admission_execution_ledger.json \
  --training-admission-execution-ledger-summary /tmp/custom-corpus-property-training-admission-execution-ledger/property-training-admission-execution-ledger-example-001/property_training_admission_execution_ledger_summary.json \
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
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-summary /tmp/custom-corpus-property-training-admission-execution-ledger-precheck-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-execution-ledger-precheck-summary.md
```

Return codes:

- `0` when precheck status is `passed` or `needs_review`
- `1` when precheck status is `blocked`

## Boundaries

- The precheck validates ledger evidence only.
- The precheck does not run future training dataset materialization.
- The precheck does not create training data.
- The precheck does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not run Phase 1.
- The precheck does not modify `DatasetConfirmation`.
- The precheck does not run model training or evaluation.
- The precheck does not call an LLM or agent.
- The precheck does not call MinerU.
- The precheck does not parse PDFs.
- The precheck does not read ParsedDocument content.
- A passed precheck is necessary but not sufficient for future training
  dataset materialization.
