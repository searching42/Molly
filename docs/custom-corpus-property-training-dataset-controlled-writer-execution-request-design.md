# Custom Corpus Property Training Dataset Controlled Writer Execution Request Design

## Purpose

This document defines the future design boundary for a property training dataset
controlled writer execution request. It describes what a future request may
reference, what evidence must already exist, how the request must remain
hash-bound and redacted, and why a request is still not writer execution.

This is a docs/test-only design. It does not create a request artifact,
implement request creation, implement request preflight, explicitly confirm
execution, execute the controlled writer, or materialize a training dataset.

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

The execution request design sits after the controlled writer dry-run precheck
and before the controlled writer execution request artifact creator. It remains
before any future execution request preflight, explicit confirmation gate, or
controlled writer execution.

## Required Upstream Evidence

A future controlled writer execution request may only be designed from safe,
passed upstream evidence. Required evidence includes:

- `controlled_writer_dry_run_precheck_status=passed`
- `controlled_writer_dry_run_status=passed`
- `controlled_writer_dry_run_report_sha256=<sha256>`
- `controlled_writer_dry_run_report_basename=<basename only>`
- `controlled_writer_dry_run_summary_basename=<basename only>`
- `controlled_writer_design_plan_preflight_status=passed`
- `domain_validation_boundary_status=passed`
- `controlled_writer_value_resolution_dry_run_precheck_status=passed`
- `accepted_candidate_record_count >= configured minimum`
- `needs_review_candidate_record_count = 0 unless explicitly allowed by future policy`
- `blocked_candidate_record_count = 0`
- `missing_required_field_count = 0`
- `redaction_status=passed`
- `controlled_writer_executed=false`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `serialized_rows_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`
- `model_training_run=false`
- `evaluation_run=false`

The rule is: stale, missing, mismatched, blocked, or needs-review upstream evidence is not execution-ready by default.
Needs-review evidence requires an explicit future policy and must not be
treated as writer-ready.

## Execution Request Design Scope

The future request is a safe, reviewable authorization candidate for a later
preflight gate. It may bind to a passed dry-run precheck package and summarize
aggregate intent, but it must not contain raw values, row payloads, source
payloads, local paths, output paths, or any materialized training artifact.

This design defines the shape of a future request. It does not implement or
write that request.

## Future Execution Request Input Contract

A future execution request may reference only safe evidence:

- safe ids
- schema labels
- request id
- corpus id
- dataset name
- operator id or reviewer id
- dry-run precheck id
- dry-run report sha256
- dry-run report basename
- dry-run summary basename
- accepted candidate count
- would-write row count
- would-write field count
- required, resolved, and missing field counts
- redaction status
- boundary flags
- policy version labels
- approval-policy labels
- requested next gate label

The request must be detached from raw values, rows, source payloads, local
paths, output paths, and model execution.

## Future Execution Request Schema

Future request schema label:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_request.v1
```

This PR does not create an artifact using that schema. A future implementation
must keep request fields limited to safe ids, labels, hashes, counts, booleans,
and redaction status.

## Future Execution Request Summary Schema

Future request summary schema label:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1
```

The future summary must use basenames only for file references and must not
include raw payloads, rows, local paths, output paths, source values, or model
input data.

Future preflight schema label:

```text
custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.v1
```

Future explicit confirmation schema label:

```text
custom_corpus_property_training_dataset_controlled_writer_explicit_confirmation.v1
```

## Authorization Boundary

A controlled writer execution request is not a controlled writer execution.
A controlled writer execution request is not explicit confirmation.
A controlled writer execution request does not authorize execution by itself.
A controlled writer execution request must be separately prechecked before any confirmation gate.
A controlled writer execution request must not be inferred from a passed dry-run precheck alone.
A controlled writer execution request must not be inferred from CI success alone.
A controlled writer execution request must not be inferred from merge status alone.

Request creation is only a future proposal artifact. It remains blocked from
execution until a separate request preflight passes and a separate explicit
confirmation gate binds to the exact request.

## Explicit Confirmation Boundary

Explicit confirmation must be separate from the execution request.
Explicit confirmation must be separate from the execution request preflight.
Explicit confirmation must be operator-visible.
Explicit confirmation must bind to exact request id, request hash, dry-run precheck hash, and intended execution mode.
Explicit confirmation must not contain raw values or rows.
No request design, request artifact, or request preflight may execute the writer.

The future confirmation gate must remain human/operator visible and must not be
implicit in CI success, merge status, dry-run status, or preflight status.

## Allowed Future Request Fields

Allowed future request fields are safe, aggregate, and reviewable:

- `schema_version`
- `request_id`
- `request_status`
- `corpus_id`
- `dataset_name`
- `requested_by`
- `created_at`
- `dry_run_precheck_id`
- `dry_run_report_sha256`
- `dry_run_report_basename`
- `dry_run_summary_basename`
- `accepted_candidate_record_count`
- `needs_review_candidate_record_count`
- `blocked_candidate_record_count`
- `would_write_row_count`
- `would_write_field_count`
- `required_field_count`
- `resolved_required_field_count`
- `missing_required_field_count`
- `redaction_status`
- `boundary_flags`
- `policy_version_labels`
- `approval_policy_labels`
- `requested_next_gate`
- `request_errors`
- `request_warnings`

All file references must be basenames only. All hashes must be explicit safe
checksum strings.

## Disallowed Current Outputs

This PR must not produce any current execution request output. It must not
produce request JSON, request summary JSON, request preflight output, explicit
confirmation output, training rows, review rows, candidate rows, dataset
artifacts, file-format artifacts, conformer files, DPA3 structure files, model
input tensors, Phase 1 artifacts, or DatasetConfirmation artifacts.

## Disallowed Future Request Fields

A future request must not contain:

- raw property values
- exact numeric extracted values
- molecular strings
- SMILES
- InChI
- InChIKey
- row payloads
- serialized rows
- table payloads
- article text
- paper titles
- PDF names
- source payloads
- local paths
- absolute paths
- output artifact paths
- candidate CSV/JSONL/Parquet/LMDB paths
- training CSV/JSONL/Parquet/LMDB paths
- conformer data
- DPA3 structures
- model input tensors
- credentials
- authorization headers
- API keys
- cookies

These categories may be named in policy text, but the request artifact must not
contain value-like examples or concrete payloads from those categories.

## Hash and Basename Policy

The future request must bind to exact upstream artifact checksums and basenames.
It must not include local or private paths. A request preflight must
recompute referenced request and dry-run checksums from bytes before any later
confirmation gate.

Hash-bound fields must include the execution request hash, dry-run precheck
hash, dry-run report hash, and any upstream package hashes required by the
future request schema.

## Redaction and Non-Leakage Policy

The future request, future summary, and future evidence must be scanned before
writing. Redaction failure must block output and must not echo sensitive
material. Allowed content is limited to safe ids, labels, hashes, aggregate
counts, statuses, boundary booleans, safe basenames, and safe policy labels.

The future request must not read source payloads, PDFs, ParsedDocument objects,
or writer output artifacts.

## Request Status Semantics

Future status labels:

- `request_designed`
- `request_ready_for_preflight`
- `request_needs_review`
- `request_blocked`

`request_ready_for_preflight` does not authorize execution.
`request_needs_review` must not be treated as execution-ready.
`request_blocked` must stop before any preflight or execution.

A request can only proceed toward execution after a separate future request
preflight and a separate explicit confirmation gate.

## Future Execution Request Preflight Expectations

A request preflight must validate:

- request schema version
- request summary schema version
- request and summary hash consistency
- dry-run precheck hash consistency
- dry-run report hash consistency
- basename-only file references
- status consistency
- aggregate count consistency
- boundary flags remain false or `not_run`
- no raw values or row payloads
- no local paths or output paths
- no source payloads
- no dataset artifact creation
- no writer execution
- no Phase 1, DatasetConfirmation mutation, training, or evaluation
- redaction status is passed

Preflight success still must not execute the writer.

## Implementation Blockers

Future request implementation is blocked until the dry-run precheck package is
passed, domain validation remains passed, value resolution precheck remains
passed, candidate counts are accepted by policy, missing required fields remain
zero, no blocked candidates exist, redaction is passed, and all materialization
and execution boundary flags remain false or `not_run`.

Implementation is also blocked if upstream evidence is stale, missing,
mismatched, blocked, needs-review without explicit allowance, unverified, or not
hash-bound.

## Pass Criteria

The design can be considered passed when it documents safe request contents,
required upstream evidence, authorization boundaries, explicit confirmation
boundaries, future request status labels, redaction requirements, and the next
gate without creating any request artifact.

## Needs-Review Criteria

The design needs review if a future policy may allow needs-review candidates,
partial evidence, nonzero missing optional metadata, or operator-specific
approval labels. Needs-review evidence must remain separate from execution-ready
evidence and must not authorize a writer.

## Fail Criteria

The boundary fails if it implies that a dry-run precheck, CI success, merge
status, request design, request artifact, or request preflight authorizes writer
execution; if it permits raw values, rows, paths, source payloads, or model
inputs in a request; or if it claims that a training dataset artifact exists.

## Residual Risks

This design does not validate live request artifacts because no request artifact
exists in this PR. A future implementation still needs schema validation,
hash-bound request summaries, request preflight, explicit confirmation, and
post-request execution controls.

## Next Step

The next step is property training dataset controlled writer execution request,
not writer execution.

After future request implementation, a future execution request preflight is
still required. After that, a future explicit confirmation gate is still
required before controlled writer execution.

This controlled writer execution request design does not create an execution request.
This controlled writer execution request design does not implement execution request creation.
This controlled writer execution request design does not implement execution request preflight.
This controlled writer execution request design does not explicitly confirm execution.
This controlled writer execution request design does not execute the controlled writer.
This controlled writer execution request design does not emit raw values.
This controlled writer execution request design does not materialize values.
This controlled writer execution request design does not serialize training rows.
This controlled writer execution request design does not create training dataset artifacts.
This controlled writer execution request design does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer execution request design does not generate conformers.
This controlled writer execution request design does not generate DPA3 structures.
This controlled writer execution request design does not run Phase 1.
This controlled writer execution request design does not modify DatasetConfirmation.
This controlled writer execution request design does not run model training or evaluation.
