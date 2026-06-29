# Custom Corpus Property Training Admission Execution Ledger

The property training admission execution ledger is the first controlled layer
allowed to mark property candidates as admitted to a training-admission ledger.
It reads a dry-run-precheck-passed package, validates the full upstream chain
again, and writes safe ID/hash-only ledger records.

The ledger is not a serialized training dataset. It does not create training
CSV/JSONL/Parquet/LMDB artifacts, does not create candidate
CSV/JSONL/Parquet/LMDB artifacts, does not run Phase 1, does not modify
`DatasetConfirmation`, and does not run model training or evaluation.

## Relationship To Dry-Run Precheck

The upstream dry-run precheck is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run-precheck.md
```

The dry-run precheck validates simulation evidence. The execution ledger
commits safe admission decisions into an append-only-style ledger artifact.
Training dataset materialization remains separate and unimplemented.

## Inputs

The ledger writer requires:

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

## Execution Rules

The ledger writer requires explicit confirmation and a clean run-scoped output
directory. It validates schema versions, dry-run precheck status, dry-run
status, execution request status, source SHA-256 bindings, source ids, record
counts, candidate eligibility, redaction safety, and source boundary fields.

For each dry-run record, it writes one ledger record with:

- source dry-run, execution, draft, candidate, materialization, admission,
  review, document, and field ids
- `admission_action=admit_training_candidate`
- `ledger_record_status=admitted_to_training_ledger`
- `training_admitted=true`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`
- source SHA-256 bindings through dry-run and dry-run precheck

Ledger records do not contain raw values, raw table rows, article text, PDF
names or paths, local paths, training output paths, or serialized dataset rows.

## Schemas

Ledger artifact:

```text
custom_corpus_property_training_admission_execution_ledger.v1
```

Summary artifact:

```text
custom_corpus_property_training_admission_execution_ledger_summary.v1
```

## Status Meanings

`committed` means the dry-run precheck passed, all upstream evidence is in the
expected ready/passed/planned state, and ledger writing completed safely.

`needs_review` means no hard error was found, but dry-run precheck or upstream
evidence carries an explicitly allowed needs-review or partial status.

`blocked` means confirmation was missing or schema, status, SHA, id, record
eligibility, output directory, source boundary, or redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_ledger \
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
  --output-dir /tmp/custom-corpus-property-training-admission-execution-ledger \
  --execution-ledger-id property-training-admission-execution-ledger-example-001 \
  --created-by operator-redacted \
  --confirm-training-admission-ledger-write
```

Return codes:

- `0` when execution status is `committed` or `needs_review`
- `1` when execution status is `blocked`

## Output Artifacts

The ledger writer creates a run-scoped clean directory:

```text
<output-dir>/<execution-ledger-id>/
  property_training_admission_execution_ledger.json
  property_training_admission_execution_ledger_summary.json
  redacted_property_training_admission_execution_ledger_evidence.md
```

It does not write training CSV, training JSONL, training Parquet, training
LMDB, candidate CSV, candidate JSONL, candidate Parquet, candidate LMDB,
Phase 1 artifacts, `DatasetConfirmation` artifacts, model training artifacts,
or evaluation artifacts.

## Boundaries

- The ledger commits training admission decisions into a safe ledger only.
- The ledger does not create training dataset artifacts.
- The ledger does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The ledger does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The ledger does not run Phase 1.
- The ledger does not modify `DatasetConfirmation`.
- The ledger does not run model training or evaluation.
- The ledger does not call an LLM or agent.
- The ledger does not call MinerU.
- The ledger does not parse PDFs.
- The ledger does not read ParsedDocument content.
- A committed ledger is necessary but not sufficient for future training
  dataset materialization.
