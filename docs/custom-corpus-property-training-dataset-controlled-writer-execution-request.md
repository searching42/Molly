# Custom Corpus Property Training Dataset Controlled Writer Execution Request

## Purpose

The property training dataset controlled writer execution request creates a
safe, hash-bound request artifact from a passed controlled writer dry-run
precheck summary. The request is a reviewable candidate for a later request
preflight gate. It is not writer execution and it does not authorize execution
by itself.

The request creator reads only the controlled writer dry-run precheck summary.
It does not rerun the dry-run, read source payloads, create rows, materialize
values, or create training dataset artifacts.

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

The request sits after the controlled writer dry-run precheck and before the
controlled writer execution request preflight or any future explicit
confirmation gate.

## State Transition Mapping

All operations are now state transitions, not standalone validations. The
system no longer validates artifacts directly. It validates only state
transitions and provenance integrity.

The controlled writer execution request maps to `REQUEST_CREATED`. It cannot be
inferred from a request filename, a passed dry-run precheck, CI success, or
merge status. The request preflight maps to `REQUEST_PRECHECKED`, and any later
approval or execution state must be reached by explicit adjacent transitions
with provenance-hash continuity.

## Provenance Binding

No execution request artifact is valid unless it is provably bound to the
`REQUEST_CREATED` state transition. The execution provenance binding layer
hash-locks the request artifact and links it to the transition id, parent
transition hash, state before/after, artifact type, and timestamp. A request
file without that binding is an orphan artifact and is not execution-ready.

## Input Summary

The only input is a controlled writer dry-run precheck summary with schema
`custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1`.
The summary must use safe ids, safe basenames, SHA-256 bindings, aggregate
counts, status labels, redaction status, and boundary booleans.

The request creator computes the input summary SHA-256 from the exact input
bytes and stores only the input basename and hash in the request package.

## Request Creation Checks

The creator validates:

- dry-run precheck schema and status
- dry-run status
- request id, requester id, and request purpose labels
- dry-run report hash format
- dry-run report and summary basenames
- accepted, needs-review, blocked, field, and would-write counts
- missing required field count
- redaction status
- execution, materialization, row serialization, Phase 1, DatasetConfirmation,
  model training, and evaluation boundary flags
- forbidden marker absence in the input and emitted request package
- clean request output directory

## Request Status Semantics

`request_ready_for_preflight` means the dry-run precheck evidence is safe enough
to create a request artifact for later preflight. It does not authorize writer
execution.

`request_needs_review` means no hard blocker exists, but an explicitly allowed
needs-review condition remains. It is not execution-ready.

`request_blocked` means the input summary, request metadata, counts, hashes,
boundary flags, redaction state, or output directory failed a hard check.

## Output Files

When the input is safe, the creator writes only:

- `property_training_dataset_controlled_writer_execution_request.json`
- `property_training_dataset_controlled_writer_execution_request_summary.json`
- `redacted_property_training_dataset_controlled_writer_execution_request_evidence.md`

These files are written under a clean run directory named by the safe request
id. Blocking input failures return a safe in-memory summary and do not write a
normal request package.

## Hash and Basename Policy

The request summary binds to the exact request JSON bytes with
`request_sha256`. The request binds to the exact dry-run precheck summary bytes
with `dry_run_precheck_summary_sha256`. File references are basenames only. The
request package must not include local paths, absolute paths, source paths, or
future artifact paths.

## Redaction Policy

Request JSON, summary JSON, and Markdown evidence are scanned before writing.
If unsafe content is detected, the creator fails closed with a minimal blocked
summary. Unsafe Markdown, normal request JSON, and normal summary JSON are not
written on redaction failure.

Allowed content is limited to ids, labels, SHA-256 hashes, aggregate counts,
status labels, boundary booleans, redaction status, safe basenames, and safe
error or warning codes.

## Authorization Boundary

A controlled writer execution request is not controlled writer execution. It
does not authorize writer execution by itself and cannot be inferred from a
passed dry-run precheck, CI success, or merge status. The request only prepares
a package for a request preflight.

The request must always set `writer_execution_authorized=false`.

## Explicit Confirmation Boundary

Explicit confirmation remains required after a request preflight. The
request must always set `explicit_confirmation_required=true`. Confirmation
must be a separate future gate that binds to the exact request id, request hash,
dry-run precheck hash, and intended execution mode.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request \
  --controlled-writer-dry-run-precheck-summary property_training_dataset_controlled_writer_dry_run_precheck_summary.json \
  --output-dir controlled_writer_execution_request_output \
  --request-id controlled-writer-execution-request-001 \
  --requested-by safe-operator-id
```

Optional flags:

- `--request-purpose <safe-purpose-label>`
- `--allow-needs-review-candidates`
- `--no-require-dry-run-precheck-passed`
- `--no-require-dry-run-passed`
- `--no-require-zero-missing-required-fields`
- `--minimum-accepted-candidate-records <N>`

## Blocked Conditions

The request is blocked for missing or invalid input, wrong schema, unsafe
request metadata, blocked or unallowed needs-review dry-run evidence, invalid
hashes, unsafe basenames, insufficient accepted records, blocked candidate
records, unallowed needs-review candidates, missing required fields without an
allowance, nonpositive would-write field counts, dirty output directories,
failed redaction, forbidden markers, writer execution, value materialization,
row serialization, dataset artifact creation, Phase 1 execution,
DatasetConfirmation mutation, model training, or evaluation.

## Out of Scope

This step does not implement request preflight, explicitly confirm execution,
execute the controlled writer, read authorized source payloads, emit raw
values, emit exact numeric extracted values, emit molecular strings, materialize
values, serialize rows, create training dataset artifacts, create
CSV/JSONL/Parquet/LMDB artifacts, generate conformers, generate DPA3
structures, run Phase 1, modify DatasetConfirmation, run model training or
evaluation, call LLMs, call agents, call MinerU, parse documents, run corpus
workflows, or perform chemistry calculations.

## Next Step

The next step is property training dataset controlled writer execution request
preflight, not writer execution.

After request preflight, a future explicit confirmation gate is still
required before controlled writer execution.

This controlled writer execution request does not implement execution request preflight.
This controlled writer execution request does not explicitly confirm execution.
This controlled writer execution request does not execute the controlled writer.
This controlled writer execution request does not authorize writer execution by itself.
This controlled writer execution request keeps explicit confirmation required.
This controlled writer execution request does not emit raw values.
This controlled writer execution request does not materialize values.
This controlled writer execution request does not serialize training rows.
This controlled writer execution request does not create training dataset artifacts.
This controlled writer execution request does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer execution request does not generate conformers.
This controlled writer execution request does not generate DPA3 structures.
This controlled writer execution request does not run Phase 1.
This controlled writer execution request does not modify DatasetConfirmation.
This controlled writer execution request does not run model training or evaluation.
