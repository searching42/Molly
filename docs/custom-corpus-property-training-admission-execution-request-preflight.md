# Custom Corpus Property Training Admission Execution Request Preflight

The property training admission execution request preflight validates a
reviewable execution request package before any future training admission
execution layer. It reads the execution request, its builder summary, the
request draft package, request plan/preflight, readiness summary, quarantine
candidate preflight, and quarantine candidate records.

The preflight validates an execution request only. It does not execute training
admission, admit training data, create training CSV/JSONL/Parquet/LMDB
artifacts, create candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1,
modify `DatasetConfirmation`, or run model training or evaluation.

## Relationship To Execution Request Builder

The upstream execution request builder is documented in:

```text
docs/custom-corpus-property-training-admission-execution-request.md
```

The builder writes a reviewable request artifact. This preflight checks that
the request artifact and its upstream evidence still agree before a future
execution layer is considered. The preflight does not call the builder, rerun
planners, modify requests, or create execution artifacts.

Future training admission execution remains separate and unimplemented.

## Inputs

The preflight requires:

- `custom_corpus_property_training_admission_execution_request.v1`
- `custom_corpus_property_training_admission_execution_request_builder.v1`
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

## Preflight Rules

The preflight checks schema versions, statuses, SHA-256 bindings, safe ids,
execution record counts, planned candidate ids, excluded/blocked/needs-review
leakage, and redaction safety. Execution request records must remain safe
ID/hash-only records and must not include raw values, raw table rows, article
text, PDF names or paths, local paths, or training output paths.

## Status Meanings

`passed` means the execution request is written, all upstream evidence is in
the expected ready/passed/planned state, and no consistency or redaction check
failed.

`needs_review` means no hard error was found, but execution request or upstream
evidence carries an explicitly allowed needs-review or partial status.

`blocked` means schema, status, SHA, id, record eligibility, source boundary,
or redaction checks failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight \
  --training-admission-execution-request /tmp/property-training-admission-execution-request/property-training-admission-execution-request-example-001/property_training_admission_execution_request.json \
  --training-admission-execution-request-summary /tmp/property-training-admission-execution-request/property-training-admission-execution-request-example-001/property_training_admission_execution_request_summary.json \
  --training-admission-request-draft /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/custom-corpus-property-training-admission-request-draft-precheck-summary.json \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-request-preflight /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-summary /tmp/custom-corpus-property-training-admission-execution-request-preflight-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-execution-request-preflight-summary.md
```

Return codes:

- `0` when preflight status is `passed` or `needs_review`
- `1` when preflight status is `blocked`

## Redaction Behavior

Before printing or writing output, the preflight scans the summary and
Markdown for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_execution_request_preflight_redaction_failed
```

Unsafe Markdown is not written.

## After Execution Request Preflight: Execution Dry-Run

The execution dry-run is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run.md
```

Future evidence template:

```text
docs/evidence/templates/custom-corpus-property-training-admission-execution-dry-run-evidence-template.md
```

The execution request preflight validates request package consistency. The
execution dry-run simulates future admission execution as labels only. Actual
training admission execution remains separate and unimplemented.

## Boundaries

- The preflight validates an execution request only.
- The preflight does not execute training admission.
- The preflight does not admit training data.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- The preflight does not call an LLM or agent.
- The preflight does not call MinerU.
- The preflight does not parse PDFs.
- The preflight does not read ParsedDocument content.
- A passed preflight is necessary but not sufficient for future training
  admission execution.
