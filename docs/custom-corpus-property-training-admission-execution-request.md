# Custom Corpus Property Training Admission Execution Request

The property training admission execution request builder writes a reviewable
request artifact for a future training admission execution gate. It consumes a
draft-precheck-validated request package and emits safe ID/hash-only request
records.

The builder creates an execution request only. It does not execute training
admission, admit training data, create training CSV/JSONL/Parquet/LMDB
artifacts, create candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1,
modify `DatasetConfirmation`, or run model training or evaluation.

## Relationship To Draft Package Precheck

The upstream draft package precheck is documented in:

```text
docs/custom-corpus-property-training-admission-request-draft-precheck.md
```

The execution request builder reads the draft, draft summary, draft precheck,
request plan, request preflight, training admission readiness summary,
quarantine candidate preflight summary, and quarantine candidate records. It
validates the chain again before writing a request. It does not rerun any
planner, preflight, materializer, or training admission executor.

Future training admission execution remains separate and unimplemented.

## Inputs

The builder requires:

- `custom_corpus_property_training_admission_request_draft.v1`
- `custom_corpus_property_training_admission_request_draft_builder.v1`
- `custom_corpus_property_training_admission_request_draft_precheck.v1`
- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`
- output directory
- execution request id
- redacted creator label
- explicit `--confirm-training-admission-execution-request-output`

It reads local JSON artifacts only. It does not read PDFs, ParsedDocument
outputs, MinerU bundles, raw extracted text, candidate/training
CSV/JSONL/Parquet/LMDB files, or training outputs.

## Execution Request Mapping

For each draft record, the builder creates one execution request record with
safe IDs and SHA-256 bindings only:

- execution request record id
- draft record id
- candidate record id
- source record id
- materialization record id
- prior materializer execution record id
- admission record id
- review id
- document id
- field name
- requested action label
- request status label
- source/review/admission/package/materialization/quarantine/readiness/plan/
  preflight/draft/precheck SHA-256 bindings

Execution request records must not contain raw values, raw table rows, article
text, PDF names or paths, local paths, training output paths, or
CSV/JSONL/Parquet/LMDB paths.

## Schemas

The request artifact uses:

```text
custom_corpus_property_training_admission_execution_request.v1
```

The builder summary uses:

```text
custom_corpus_property_training_admission_execution_request_builder.v1
```

Request artifacts have `request_mode=execution_request_only`,
`training_admitted=false`, `phase1_status=not_run`, and
`dataset_confirmation_changed=false`.

## Status Meanings

`written` means draft precheck passed, all consistency checks passed, and a
reviewable execution request was written.

`needs_review` means draft precheck was `needs_review` and
`--allow-draft-precheck-needs-review` was explicitly set.

`blocked` means confirmation was missing or a schema, status, SHA, id,
candidate eligibility, output directory, threshold, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_request \
  --training-admission-request-draft /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/custom-corpus-property-training-admission-request-draft-precheck-summary.json \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-request-preflight /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-dir /tmp/custom-corpus-property-training-admission-execution-request \
  --execution-request-id property-training-admission-execution-request-example-001 \
  --created-by operator-redacted \
  --confirm-training-admission-execution-request-output
```

Return codes:

- `0` when request status is `written` or `needs_review`
- `1` when request status is `blocked`

## Output Artifacts

The builder writes a run-scoped clean directory:

```text
<output-dir>/<execution-request-id>/
  property_training_admission_execution_request.json
  property_training_admission_execution_request_summary.json
  redacted_property_training_admission_execution_request_evidence.md
```

It does not write training CSV, training JSONL, training Parquet, training
LMDB, candidate CSV, candidate JSONL, candidate Parquet, candidate LMDB,
Phase 1 artifacts, `DatasetConfirmation` artifacts, model training artifacts,
or evaluation artifacts.

## Redaction Behavior

Before writing any artifact, the builder scans request, summary, and Markdown
content for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_execution_request_redaction_failed
```

Unsafe request and Markdown artifacts are not written.

## Boundaries

- The builder writes a training admission execution request only.
- The builder does not execute training admission.
- The builder does not admit training data.
- The builder does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The builder does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The builder does not run Phase 1.
- The builder does not modify `DatasetConfirmation`.
- The builder does not run model training or evaluation.
- The builder does not call an LLM or agent.
- The builder does not call MinerU.
- The builder does not parse PDFs.
- The builder does not read ParsedDocument content.
- A request artifact is necessary but not sufficient for future training
  admission execution.
