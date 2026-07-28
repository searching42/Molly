# Custom Corpus Property Training Admission Request Draft

The property training admission request draft builder writes a reviewable
training admission request draft from a preflight-checked request plan. The
draft is a local governance artifact for later review.

The builder writes a training admission request draft only. It does not
execute training admission, admit training data, create training
CSV/JSONL/Parquet/LMDB artifacts, create candidate CSV/JSONL/Parquet/LMDB
artifacts, run Phase 1, modify `DatasetConfirmation`, or run model
training/evaluation. A draft is necessary but not sufficient for future
training admission execution.

## Relationship To Request Preflight

The upstream request preflight is documented in:

```text
docs/custom-corpus-property-training-admission-request-preflight.md
```

Request preflight validates request planning evidence. The draft builder
consumes that preflight, the original request plan, training admission
readiness evidence, quarantine candidate preflight evidence, and quarantine
candidate records. It does not rerun those planners or preflights and does not
modify quarantine candidate artifacts.

Future training admission execution remains separate and unimplemented.

## Inputs

The builder requires:

- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`
- output directory
- request draft id
- redacted creator label
- explicit `--confirm-training-admission-request-draft-output`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, candidate/training CSV/JSONL/Parquet/LMDB files, or training outputs.

## Draft Mapping

For each planned request record summary, the builder creates one safe draft
record containing ids and SHA-256 bindings only:

- draft record id
- candidate record id
- source record id
- materialization record id
- execution record id
- admission record id
- review id
- document id
- field name
- requested action label
- request status label
- source/review/admission/package/materialization/quarantine/readiness/plan
  SHA-256 bindings

Draft records must not contain raw values, raw table rows, article text, PDF
names or paths, local paths, training output paths, or CSV/JSONL/Parquet/LMDB
paths.

## Draft Schema

The draft artifact uses:

```text
custom_corpus_property_training_admission_request_draft.v1
```

It is written as:

```text
property_training_admission_request.draft.json
```

The draft has `request_mode=draft_only`, `training_admitted=false`,
`phase1_status=not_run`, and `dataset_confirmation_changed=false`.

## Summary Schema

The builder summary uses:

```text
custom_corpus_property_training_admission_request_draft_builder.v1
```

It is written as:

```text
property_training_admission_request_draft_summary.json
```

The summary includes safe basenames, SHA-256 values, artifact ids, upstream
statuses, record counts, draft ids, planned candidate ids, errors, warnings,
and redaction status.

## Status Meanings

`written` means request preflight passed, all consistency checks passed, and a
reviewable draft was written.

`needs_review` means request preflight was partial and
`--allow-preflight-partial` was explicitly set.

`blocked` means confirmation was missing or a schema, status, SHA, id,
candidate eligibility, output directory, threshold, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_draft \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-request-preflight /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --output-dir /tmp/custom-corpus-property-training-admission-request-draft \
  --request-draft-id property-training-admission-request-draft-example-001 \
  --created-by operator-redacted \
  --confirm-training-admission-request-draft-output
```

Return codes:

- `0` when draft status is `written` or `needs_review`
- `1` when draft status is `blocked`

## Output Artifacts

The builder writes a run-scoped clean directory:

```text
<output-dir>/<request-draft-id>/
  property_training_admission_request.draft.json
  property_training_admission_request_draft_summary.json
  redacted_property_training_admission_request_draft_evidence.md
```

It does not write training CSV, training JSONL, training Parquet, training
LMDB, candidate CSV, candidate JSONL, candidate Parquet, candidate LMDB,
Phase 1 artifacts, `DatasetConfirmation` artifacts, model training artifacts,
or evaluation artifacts.

## After Request Draft: Draft Package Precheck

After a reviewable draft is written, the draft package precheck can validate
the draft, draft summary, request plan, request preflight, training admission
readiness summary, quarantine candidate preflight summary, and quarantine
candidate records as one package.

References:

- `docs/custom-corpus-property-training-admission-request-draft-precheck.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-request-draft-precheck-evidence-template.md`

The draft precheck validates package consistency only. It does not execute
training admission, admit training data, create training or candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, modify
`DatasetConfirmation`, or run model training/evaluation. Actual training
admission execution remains separate and unimplemented.

## Redaction Behavior

Before writing any artifact, the builder scans draft, summary, and Markdown
content for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_request_draft_redaction_failed
```

Unsafe draft or Markdown artifacts are not written.

## Boundaries

- The builder writes a training admission request draft only.
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
- A draft is necessary but not sufficient for future training admission
  execution.
