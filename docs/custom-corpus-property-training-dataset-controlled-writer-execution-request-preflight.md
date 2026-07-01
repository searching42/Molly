# Custom Corpus Property Training Dataset Controlled Writer Execution Request Preflight

## Purpose

The property training dataset controlled writer execution request preflight is
an offline artifact validator for controlled writer execution request packages.
It validates request JSON and request summary JSON, and may inspect redacted
request evidence Markdown. It proves the package is schema-valid, hash-bound,
internally consistent, redacted, basename-only, still before explicit
confirmation, and still not authorized for writer execution.

This preflight validates artifacts only. It does not rerun the request creator,
rerun the dry-run, run dry-run precheck, explicitly confirm execution, execute
the controlled writer, create rows, or create dataset artifacts.

## Position in the Governance Chain

```text
property training dataset controlled writer value resolution dry-run
-> property training dataset controlled writer value resolution dry-run precheck
-> small public quarantine materialization evidence
-> property training dataset quarantined candidate admission boundary
-> property training dataset domain validation boundary
-> property training dataset controlled writer design plan
-> property training dataset controlled writer design plan preflight
-> property training dataset controlled writer dry-run design
-> property training dataset controlled writer dry-run
-> property training dataset controlled writer dry-run precheck
-> property training dataset controlled writer execution request design
-> property training dataset controlled writer execution request
-> property training dataset controlled writer execution request preflight
-> future explicitly confirmed controlled writer execution
```

The preflight sits after execution request creation and before any future
explicit confirmation gate.

## Input Package

The preflight reads only:

- controlled writer execution request JSON
- controlled writer execution request summary JSON
- optional redacted request evidence Markdown

It validates existing schemas:

- `custom_corpus_property_training_dataset_controlled_writer_execution_request.v1`
- `custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1`

The preflight emits a summary using:

- `custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.v1`

The future explicit confirmation schema label is
`custom_corpus_property_training_dataset_controlled_writer_explicit_confirmation.v1`.
This preflight does not create an explicit confirmation artifact.

## Preflight Checks

The preflight checks:

- request and request summary schema versions
- request SHA-256 recomputed from exact request bytes
- request basename matches the summary basename
- request and summary ids, statuses, hashes, counts, and boundary flags
- dry-run precheck summary basename and hash consistency
- corpus id and dataset name consistency
- request status semantics
- `writer_execution_authorized=false`
- `explicit_confirmation_required=true`
- `requested_next_gate=controlled_writer_execution_request_preflight`
- basename-only file references
- redaction status
- optional evidence Markdown safety

## Status Semantics

`preflight_passed` means the request package is safe and eligible for a future
explicit confirmation gate. It does not execute the writer and does not
authorize writer execution by itself.

`preflight_needs_review` means no hard blocker exists, but explicitly allowed
needs-review request status, needs-review candidates, or missing required field
counts remain. It is not execution by itself.

`preflight_blocked` means the request package failed schema, hash, basename,
status, count, authorization, boundary, redaction, or evidence checks.

## Hash and Basename Policy

The summary recomputes `request_sha256` from the exact request file bytes. The
summary records request and request-summary basenames only. Local paths,
absolute paths, source paths, and output artifact paths are not emitted.

## Redaction Policy

Request JSON, request summary JSON, optional evidence Markdown, preflight
summary JSON, and preflight Markdown are scanned for unsafe material. Redaction
failure returns a minimal blocked summary and does not write unsafe Markdown or
a normal preflight summary that echoes sensitive material.

Allowed output is limited to safe ids, basenames, hashes, aggregate counts,
status labels, redaction status, boundary booleans, and safe error or warning
codes.

## Authorization Boundary

A controlled writer execution request preflight is not controlled writer
execution. It does not authorize writer execution by itself. A passed preflight
does not replace future explicit confirmation, CI success does not authorize
execution, and merge status does not authorize execution.

The preflight must keep `writer_execution_authorized=false`.

## Explicit Confirmation Boundary

Explicit confirmation remains a separate future gate. That future gate must be
operator-visible and must bind to the exact request id, request hash, dry-run
precheck hash, and intended execution mode. This preflight must keep
`explicit_confirmation_required=true`.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight \
  --controlled-writer-execution-request property_training_dataset_controlled_writer_execution_request.json \
  --controlled-writer-execution-request-summary property_training_dataset_controlled_writer_execution_request_summary.json \
  --controlled-writer-execution-request-evidence redacted_property_training_dataset_controlled_writer_execution_request_evidence.md \
  --output-summary property_training_dataset_controlled_writer_execution_request_preflight_summary.json \
  --output-markdown redacted_property_training_dataset_controlled_writer_execution_request_preflight_evidence.md
```

Optional flags:

- `--allow-request-needs-review`
- `--no-require-request-ready-for-preflight`
- `--no-require-explicit-confirmation-required`
- `--no-require-writer-execution-unauthorized`
- `--no-require-zero-missing-required-fields`
- `--minimum-accepted-candidate-records <N>`

## Outputs

The preflight may write:

- preflight summary JSON
- redacted preflight Markdown evidence

It must not write request artifacts, explicit confirmation artifacts, training
rows, review rows, candidate rows, dataset artifacts, file-format artifacts,
conformer files, DPA3 structure files, model inputs, Phase 1 artifacts, or
DatasetConfirmation files.

## Blocked Conditions

The preflight blocks on missing or invalid request packages, wrong schemas,
request hash mismatch, request basename mismatch, request/summary mismatch,
blocked or invalid request status, unallowed needs-review request status,
unpassed dry-run precheck status, unpassed dry-run status, insufficient accepted
candidate count, blocked candidate count, unallowed needs-review candidates,
missing required fields without allowance, invalid would-write counts, writer
execution authorization, missing explicit-confirmation requirement, writer
execution flags, materialization flags, row serialization flags, dataset
artifact flags, Phase 1 execution, DatasetConfirmation mutation, model training
or evaluation, failed redaction, unsafe evidence, absolute paths, concrete
artifact paths, or forbidden markers.

## Out of Scope

This preflight does not explicitly confirm execution, execute the controlled
writer, read authorized source payloads, emit raw values, emit exact numeric
extracted values, emit molecular strings, materialize values, serialize rows,
create training dataset artifacts, create CSV/JSONL/Parquet/LMDB artifacts,
generate conformers, generate DPA3 structures, run Phase 1, modify
DatasetConfirmation, run model training or evaluation, call LLMs, call agents,
call MinerU, parse documents, run corpus workflows, perform chemistry
calculations, set `writer_execution_authorized=true`, or set
`explicit_confirmation_required=false`.

## Next Step

The next step is future explicitly confirmed controlled writer execution, not
writer execution by this preflight.

Explicit confirmation must be a separate future gate and must still bind to
the exact request id, request hash, dry-run precheck hash, and intended
execution mode.

This controlled writer execution request preflight does not explicitly confirm execution.
This controlled writer execution request preflight does not execute the controlled writer.
This controlled writer execution request preflight does not authorize writer execution by itself.
This controlled writer execution request preflight keeps explicit confirmation required.
This controlled writer execution request preflight does not emit raw values.
This controlled writer execution request preflight does not materialize values.
This controlled writer execution request preflight does not serialize training rows.
This controlled writer execution request preflight does not create training dataset artifacts.
This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer execution request preflight does not generate conformers.
This controlled writer execution request preflight does not generate DPA3 structures.
This controlled writer execution request preflight does not run Phase 1.
This controlled writer execution request preflight does not modify DatasetConfirmation.
This controlled writer execution request preflight does not run model training or evaluation.
