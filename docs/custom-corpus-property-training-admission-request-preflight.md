# Custom Corpus Property Training Admission Request Preflight

The property training admission request preflight validates an existing
`custom_corpus_property_training_admission_request_plan.v1` summary before any
future training admission request can be generated.

The preflight is not execution. It does not create a training admission
request, create training admission actions, admit training data, create
training CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, modify
`DatasetConfirmation`, materialize datasets, or run model training/evaluation.
A passed preflight is necessary but not sufficient for future training
admission execution.

## Relationship To Request Planning

The upstream request planner is documented in:

```text
docs/custom-corpus-property-training-admission-request-planner.md
```

Request plans are planning evidence only and are non-authoritative. The
preflight checks whether the plan, training admission readiness summary, and
quarantine candidate preflight summary agree on schemas, statuses, ids,
candidate eligibility, and SHA-256 bindings. Future training admission
execution remains unimplemented.

## Inputs

The preflight requires:

- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, candidate/training CSV/JSONL/Parquet/LMDB files, or training outputs.

## Preflight Rules

The preflight checks:

- request plan schema is
  `custom_corpus_property_training_admission_request_plan.v1`
- readiness schema is `custom_corpus_property_training_admission_readiness.v1`
- quarantine candidate preflight schema is
  `custom_corpus_property_quarantine_candidate_preflight.v1`
- request plan status is `planned`, or `partial` only as partial evidence
- readiness status is `ready`, or `partial` only as partial evidence
- quarantine candidate preflight status is `passed`, or `needs_review` only
  as partial evidence
- request plan, readiness, and quarantine candidate preflight error lists are
  empty
- source SHA-256 values match across the three summaries
- request plan binds to the exact readiness and quarantine preflight files
- corpus, dry-run, review manifest, admission request, materialization plan,
  execution request, quarantine run, review queue, and property candidate
  manifest ids match where present
- planned candidate ids match quarantined candidate ids
- candidate record counts match
- excluded, blocked, and needs-review records do not appear as planned
  candidates
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_training_admission_request_preflight.v1
```

It includes safe basenames, SHA-256 values, request/readiness/quarantine
statuses, artifact ids, candidate counts and ids, preflight errors, warnings,
and redaction status.

The summary does not include raw candidate payloads, raw table rows, article
text, PDF names or paths, local paths, ParsedDocument text, MinerU bundle
paths, token/auth/cookie values, private emails, CSV/JSONL/Parquet/LMDB paths,
or training outputs.

## Status Meanings

`passed` means the request plan is `planned`, readiness is `ready`, quarantine
candidate preflight is `passed`, and all consistency checks passed.

`partial` means no hard consistency check failed, but request planning,
readiness, or quarantine preflight carries allowed partial/needs-review
evidence.

`blocked` means a schema, status, hash, id, candidate eligibility, boundary,
or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_preflight \
  --training-admission-request-plan /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --output-summary /tmp/custom-corpus-property-training-admission-request-preflight-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-request-preflight-summary.md
```

Return codes:

- `0` when preflight status is `passed` or `partial`
- `1` when preflight status is `blocked`

## Redaction Behavior

Before printing or writing output, the preflight scans serialized summary and
Markdown for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_request_preflight_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The preflight validates request plans only.
- The preflight does not execute training admission.
- The preflight does not create a training admission request.
- The preflight does not create training admission actions.
- The preflight does not admit training data.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not materialize datasets.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- The preflight does not call an LLM or agent.
- The preflight does not call MinerU.
- The preflight does not parse PDFs.
- The preflight does not read ParsedDocument content.
- A passed preflight is necessary but not sufficient for future training
  admission execution.
