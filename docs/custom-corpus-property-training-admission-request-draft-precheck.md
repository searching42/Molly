# Custom Corpus Property Training Admission Request Draft Package Precheck

The property training admission request draft package precheck validates an
existing reviewable training admission request draft package before any future
training admission execution layer is introduced.

The precheck validates a draft package only. It does not execute training
admission, admit training data, create training CSV/JSONL/Parquet/LMDB
artifacts, create candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1,
modify `DatasetConfirmation`, or run model training or evaluation. A passed
precheck is necessary but not sufficient for future training admission
execution.

## Relationship To Request Draft Builder

The upstream request draft builder is documented in:

```text
docs/custom-corpus-property-training-admission-request-draft.md
```

The builder writes a reviewable draft request and summary. This precheck reads
that draft package plus the request plan, request preflight, training
admission readiness summary, quarantine candidate preflight summary, and
quarantine candidate records. It does not rerun those tools, modify the draft,
or create execution artifacts.

Future training admission execution remains separate and unimplemented.

## Inputs

The precheck requires:

- `custom_corpus_property_training_admission_request_draft.v1`
- `custom_corpus_property_training_admission_request_draft_builder.v1`
- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`

It reads local JSON artifacts only. It does not read PDFs, ParsedDocument
outputs, MinerU bundles, raw extracted text, candidate/training
CSV/JSONL/Parquet/LMDB files, or training outputs.

## Precheck Rules

The precheck verifies:

- schemas match the expected versions
- draft status is `written`, or `needs_review` only with
  `--allow-draft-needs-review`
- request plan, request preflight, readiness, and quarantine status are not
  blocked
- SHA-256 bindings match the actual source files
- corpus, dry-run, review, admission, materialization, execution, quarantine,
  review queue, and property candidate ids remain consistent
- draft records match top-level draft record ids
- planned candidate ids match draft records and upstream request planning
  evidence
- excluded, blocked, and needs-review records do not appear in draft records
- `training_admitted=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`
- emitted JSON and Markdown pass redaction scanning

## Summary Schema

The precheck summary uses:

```text
custom_corpus_property_training_admission_request_draft_precheck.v1
```

The summary includes safe basenames, SHA-256 values, source ids, upstream
statuses, record counts, draft ids, planned candidate ids, safe error codes,
warnings, and redaction status. It does not include full draft payloads, raw
candidate payloads, raw article text, local paths, private paths, PDF names,
credential material, or candidate/training data paths.

## Status Meanings

`passed` means all schema, status, SHA, id, candidate eligibility, and
redaction checks passed without needs-review evidence.

`needs_review` means no hard consistency error exists, but draft or upstream
evidence is explicitly partial or needs-review and
`--allow-draft-needs-review` was set.

`blocked` means a hard consistency check failed, a draft/upstream artifact was
blocked, a SHA/id/record check failed, the minimum record count was not met, or
redaction failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck \
  --training-admission-request-draft /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property-training-admission-request-draft/property-training-admission-request-draft-example-001/property_training_admission_request_draft_summary.json \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-request-preflight /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-summary /tmp/custom-corpus-property-training-admission-request-draft-precheck-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-request-draft-precheck-summary.md
```

Return codes:

- `0` when precheck status is `passed` or `needs_review`
- `1` when precheck status is `blocked`

## Redaction Behavior

Before printing or writing output, the precheck scans summary and Markdown
content for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_request_draft_precheck_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The precheck validates a draft package only.
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
