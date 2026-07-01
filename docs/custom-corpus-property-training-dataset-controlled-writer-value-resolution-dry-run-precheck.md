# Custom Corpus Property Training Dataset Controlled Writer Value Resolution Dry-Run Precheck

The property training dataset controlled writer value resolution dry-run
precheck validates the safe report/summary package emitted by the controlled
writer value resolution dry-run.

It sits after the value resolution dry-run and before any future controlled
training dataset writer.

## Purpose

The value resolution dry-run may read authorized local JSON source payloads and
then emits redacted evidence about required field coverage. This precheck does
not read those source payloads again. It validates only the emitted dry-run
report and summary, confirming that the package is schema-valid, hash-bound,
internally consistent, redacted, and still inside the non-writer boundary.

This is still not controlled writer execution.

## Inputs

- `custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1`
  report
- `custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1`
  summary

The precheck reads only those two JSON artifacts.

## Schema

Precheck summary:

```text
custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck.v1
```

## Status Semantics

- `passed`: the report and summary are valid, matching, redacted, and values
  are resolved according to the selected options.
- `needs_review`: no hard error exists, but the dry-run or value-resolution
  coverage is explicitly allowed to remain needs-review.
- `blocked`: schema, status, hash, id, count, boundary, record-safety, or
  redaction checks failed.

Return codes:

- `0` for `passed` or `needs_review`
- `1` for `blocked`

## Pass Criteria

- report and summary schemas match their expected versions
- report and summary statuses match
- report SHA-256 in the summary matches the actual report file
- common ids and hashes match where both artifacts provide them
- resolution record counts match report records
- required fields are resolved when required
- boundary flags remain safe:
  - `controlled_writer_executed=false`
  - `source_payloads_read=true`
  - `values_materialized=false`
  - `serialized_rows_created=false`
  - `training_dataset_materialized=false`
  - `dataset_artifact_created=false`
  - `phase1_status=not_run`
  - `dataset_confirmation_changed=false`
  - `model_training_run=false`
  - `evaluation_run=false`
- resolution records contain only safe ids, hashes, labels, field names,
  aggregate state, and boundary booleans
- redaction checks pass

## Needs-Review Criteria

- the dry-run status is `needs_review` and
  `--allow-dry-run-needs-review` is set
- values are not fully resolved and `--no-require-values-resolved` is set
- no hard consistency or redaction error exists

## Fail Criteria

- schema mismatch
- blocked, failed, invalid, or disallowed needs-review dry-run status
- report SHA mismatch
- invalid SHA-256 format
- id mismatch
- count mismatch
- no resolution records or fewer than the required minimum
- required values unresolved while values are required
- unsafe resolution record fields
- any boundary flag indicates writer execution, value materialization, row
  serialization, dataset artifact creation, Phase 1 execution,
  `DatasetConfirmation` mutation, model training, or evaluation
- raw property values, canonical SMILES, InChI/InChIKey values, raw text, PDF
  names, private paths, output paths, serialized rows, CSV/JSONL/Parquet/LMDB
  paths, conformer data, DPA3 structures, payloads, credentials, or tokens
  appear in the dry-run package or emitted evidence

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck \
  --controlled-writer-value-resolution-dry-run-report /tmp/property_training_dataset_controlled_writer_value_resolution_dry_run_report.json \
  --controlled-writer-value-resolution-dry-run-summary /tmp/property_training_dataset_controlled_writer_value_resolution_dry_run_summary.json \
  --output-summary /tmp/property_training_dataset_controlled_writer_value_resolution_dry_run_precheck_summary.json \
  --output-markdown /tmp/redacted_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck_evidence.md
```

Optional controls:

- `--allow-dry-run-needs-review`
- `--minimum-resolution-records <n>`
- `--no-require-values-resolved`
- `--no-require-dry-run-passed`

## Redaction

Before writing summary or Markdown evidence, the precheck scans for private
paths, token/auth/cookie material, PDF names, CSV/JSONL/Parquet/LMDB paths, raw
article text, raw table text, serialized rows, conformer/DPA3 markers,
InChI/SMILES-like leaks, and raw numeric sentinel values.

If redaction fails, the precheck fail-closes with a minimal blocked summary and
does not write unsafe Markdown.

## Boundaries

- The precheck validates value-resolution dry-run outputs only.
- The controlled writer is not executed.
- Authorized source payloads are not re-read by this precheck.
- Values are not emitted.
- Values are not materialized.
- The precheck does not create serialized training rows.
- The precheck does not materialize a training dataset.
- The precheck does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The precheck does not generate conformers.
- The precheck does not generate DPA3 structures.
- The precheck does not run Phase 1.
- The precheck does not modify `DatasetConfirmation`.
- The precheck does not run model training or evaluation.
- The precheck does not call LLMs, agents, MinerU, PDF parsers, or corpus
  workflows.
- A passed precheck is necessary but not sufficient for future controlled
  writer execution.
