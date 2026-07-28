# Custom Corpus Property Training Admission Execution Dry-Run Precheck

The property training admission execution dry-run precheck validates an
existing `custom_corpus_property_training_admission_execution_dry_run.v1`
report together with the execution request, request preflight, draft package,
request plan, readiness summary, quarantine candidate preflight, and
quarantine candidate records.

The precheck is a verification layer only. It does not run the execution
dry-run, does not execute training admission, does not admit training data,
does not create training CSV/JSONL/Parquet/LMDB artifacts, does not create
candidate CSV/JSONL/Parquet/LMDB artifacts, does not run Phase 1, does not
modify `DatasetConfirmation`, and does not run model training or evaluation.

## Relationship To Execution Dry-Run

The upstream dry-run is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run.md
```

The dry-run simulates what a future training admission execution layer would
admit. This precheck reads the dry-run report after it exists and verifies that
the report remains bound to the full upstream package. A passed precheck is
necessary but not sufficient for future training admission execution.

## After Dry-Run Precheck: Training Admission Execution Ledger

The downstream execution ledger is documented in:

```text
docs/custom-corpus-property-training-admission-execution-ledger.md
```

Future evidence template:

```text
docs/evidence/templates/custom-corpus-property-training-admission-execution-ledger-evidence-template.md
```

The dry-run precheck validates simulation evidence. The execution ledger can
commit safe admission decisions into a ledger, but training dataset
materialization remains separate and no training CSV/JSONL/Parquet/LMDB files
are created by the ledger.

## Inputs

The precheck requires:

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
outputs, MinerU bundles, raw extracted text, or training output files.

## Precheck Rules

The precheck validates schema versions, dry-run status, execution request
preflight status, request and draft statuses, SHA-256 bindings, safe ids, dry-run
record counts, execution record counts, planned candidate ids, and quarantine
candidate eligibility.

It blocks if:

- the dry-run report is blocked or invalid
- readiness is blocked
- a SHA-256 or id mismatch is found
- excluded, blocked, or needs-review candidates appear in the dry-run plan
- `training_admitted=true`
- `phase1_status` is not `not_run`
- `dataset_confirmation_changed=true`
- dry-run records contain unsafe values
- redaction fails

It returns `needs_review` when dry-run or upstream package evidence carries an
explicitly allowed needs-review or partial status and no hard error is present.

## Summary Schema

The precheck summary uses:

```text
custom_corpus_property_training_admission_execution_dry_run_precheck.v1
```

The summary contains safe basenames, SHA-256 hashes, source ids, statuses,
aggregate counts, safe record ids, warning codes, and error codes. It does not
include dry-run record payloads, raw values, raw table rows, article text, PDF
names or paths, local paths, or training output paths.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck \
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
  --output-summary /tmp/custom-corpus-property-training-admission-execution-dry-run-precheck-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-execution-dry-run-precheck-summary.md
```

Return codes:

- `0` when preflight status is `passed` or `needs_review`
- `1` when preflight status is `blocked`

## Boundaries

- The precheck does not run the execution dry-run.
- The precheck does not execute training admission.
- The precheck does not admit training data.
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
  admission execution.
