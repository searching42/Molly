# Custom Corpus Property Training Dataset Controlled Writer Dry-Run Precheck

## Purpose

The property training dataset controlled writer dry-run precheck is an offline,
deterministic validator for dry-run report and summary packages. It confirms
that the dry-run output is schema-valid, hash-bound, internally consistent,
redacted, aggregate-only, and still before any controlled writer execution
request.

The precheck validates emitted dry-run artifacts only. It does not rerun the
dry-run, execute a controlled writer, create rows, create dataset artifacts, or
authorize writer execution.

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

The dry-run precheck is downstream of the controlled writer dry-run. The
request design documents the boundary, and the next implemented gate is
property training dataset controlled writer execution request, not writer
execution.

## Provenance Binding

No precheck summary is valid unless it is provably bound to the
`REQUEST_PRECHECKED` state transition. The execution provenance binding layer
requires the precheck artifact hash to be linked to the transition id and parent
transition hash. File presence, basename matches, and schema validity are not
sufficient without this state-linked artifact binding.

## Real Literature Read-Only Acceptance Branch

The real literature branch remains read-only:

```text
real literature local manifest
-> local parsed-output presence check
-> redacted paper-level aggregate scan
-> candidate table aggregate detection
-> property field coverage aggregate
-> failure taxonomy aggregate
-> real literature read-only acceptance evidence
-> future real candidate quarantine dry-run
```

It does not rerun this precheck, execute the writer, create execution requests,
or create dataset artifacts.

## Input Package

The precheck reads:

- controlled writer dry-run report JSON
- controlled writer dry-run summary JSON
- optional redacted dry-run evidence Markdown

Input report schema:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1
```

Input summary schema:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1
```

The precheck reads no source payloads and does not call the dry-run module.

## Precheck Checks

The precheck validates:

- report and summary schema versions
- report checksum recomputed from exact report bytes
- summary report checksum and basename
- dry-run id, status, corpus id, and dataset name consistency
- candidate count consistency
- field coverage count consistency
- would-write aggregate count consistency
- boundary flag consistency
- dry-run status semantics
- missing required field policy
- needs-review candidate policy
- blocked candidate absence
- would-create and would-materialize flags remain false
- controlled writer and materialization flags remain false
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`
- redaction status and optional evidence safety

## Status Semantics

`passed` means the report and summary are valid, hash-bound, consistent,
aggregate-only, redacted, and show no side-effect or materialization boundary
violations.

`needs_review` means no hard blocker exists, but an explicitly allowed
needs-review condition remains, such as a needs-review dry-run status,
needs-review candidate count, or missing required fields when zero missing
fields are not required.

`blocked` means schema, hash, id, count, status, boundary, redaction, or
evidence safety validation failed.

## Hash and Basename Policy

The precheck recomputes the dry-run report checksum from file bytes and
requires the summary checksum to match exactly. The summary must reference the
report by basename only.

Precheck outputs also use basenames only and never include absolute paths.

## Redaction Policy

The report, summary, optional evidence Markdown, precheck summary, and precheck
Markdown are scanned before output. Unsafe material causes fail-closed behavior
with a minimal blocked summary and no unsafe Markdown.

Allowed output is limited to safe ids, schema labels, hashes, status labels,
aggregate counts, boundary booleans, and safe error codes.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck \
  --controlled-writer-dry-run-report /tmp/property_training_dataset_controlled_writer_dry_run_report.json \
  --controlled-writer-dry-run-summary /tmp/property_training_dataset_controlled_writer_dry_run_summary.json \
  --controlled-writer-dry-run-evidence /tmp/redacted_property_training_dataset_controlled_writer_dry_run_evidence.md \
  --output-summary /tmp/property_training_dataset_controlled_writer_dry_run_precheck_summary.json \
  --output-markdown /tmp/redacted_property_training_dataset_controlled_writer_dry_run_precheck_evidence.md
```

Optional flags:

```text
--allow-dry-run-needs-review
--no-require-dry-run-passed
--no-require-zero-missing-required-fields
--minimum-would-write-row-count <N>
```

The CLI prints JSON to stdout and returns zero for `passed` or `needs_review`,
and one for `blocked`.

## Outputs

The precheck summary schema is:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1
```

Optional outputs:

- precheck summary JSON
- redacted precheck evidence Markdown

No dataset artifact, row artifact, or writer execution artifact is created.

## Blocked Conditions

The precheck blocks on missing or invalid JSON, wrong schema, report hash
mismatch, basename mismatch, report/summary id mismatch, status mismatch, count
mismatch, boundary flag mismatch, unallowed needs-review status, blocked or
failed dry-run status, blocked candidate counts, missing required fields when
not allowed, would-create flags, row serialization flags, writer execution
flags, dataset materialization flags, Phase 1 execution, DatasetConfirmation
mutation, model training/evaluation flags, redaction failure, unsafe evidence,
absolute paths, or concrete artifact paths.

## Out of Scope

This controlled writer dry-run precheck does not rerun the dry-run.
This controlled writer dry-run precheck does not execute the controlled writer.
This controlled writer dry-run precheck does not emit raw values.
This controlled writer dry-run precheck does not materialize values.
This controlled writer dry-run precheck does not serialize training rows.
This controlled writer dry-run precheck does not create training dataset artifacts.
This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer dry-run precheck does not generate conformers.
This controlled writer dry-run precheck does not generate DPA3 structures.
This controlled writer dry-run precheck does not run Phase 1.
This controlled writer dry-run precheck does not modify DatasetConfirmation.
This controlled writer dry-run precheck does not run model training or evaluation.

It also does not call LLMs, agents, MinerU, document parsers, corpus workflows,
or chemistry tools.

## Next Step

The next step is property training dataset controlled writer execution request,
not writer execution.
