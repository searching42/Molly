# Custom Corpus Property Training Dataset Controlled Writer Dry-Run

## Purpose

The property training dataset controlled writer dry-run is an offline,
deterministic check for the future controlled writer boundary. It proves that a
safe input package would allow aggregate row assembly decisions without
actually assembling rows, serializing rows, emitting raw values, or creating
dataset artifacts.

The dry-run writes only a redacted report, summary, and evidence Markdown. It
does not implement controlled writer execution and it does not implement the
controlled writer dry-run precheck.

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

The dry-run is downstream of the controlled writer dry-run design. It is
upstream of the controlled writer dry-run precheck and does not authorize writer
execution.

## State Transition Mapping

All operations are now state transitions, not standalone validations. The
system no longer validates artifacts directly. It validates only state
transitions and provenance integrity.

This dry-run maps to `MATERIALIZATION_PREPARED` in the controlled writer
execution state machine. Passing dry-run output does not imply
`REQUEST_CREATED`, `REQUEST_PRECHECKED`, `REQUEST_APPROVED`,
`EXECUTION_AUTHORIZED`, or `EXECUTED`; those states require explicit adjacent
transitions with parent-hash continuity and redaction-safe evidence.

## Provenance Binding

No dry-run report or summary is valid unless it is provably bound to the
`MATERIALIZATION_PREPARED` state transition. The execution provenance binding
layer records the transition id, state before/after, artifact hash, artifact
type, parent transition hash, and timestamp so the dry-run artifact can be
audited as part of the provenance chain rather than as a standalone file.

## Real Literature Read-Only Acceptance Branch

Real literature read-only acceptance is a separate local branch:

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

That branch does not execute the controlled writer, create execution requests,
run request preflight, materialize datasets, or create training artifacts.

## Input Package

The dry-run reads one safe JSON input package with schema:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_input.v1
```

The package may contain only safe ids, schema labels, status labels, aggregate
candidate counts, field coverage counts, would-write aggregate counts,
redaction status, and boundary booleans.

It must not contain raw values, exact numeric extracted values, molecular
strings, structure identifiers, article body text, table payloads, local paths,
output artifact paths, document file names, row payloads, model input tensors,
conformer data, DPA3 structures, or credential material.

## Dry-Run Checks

The dry-run validates:

- input schema version
- safe `dry_run_id`, `corpus_id`, `dataset_name`, and design preflight id
- controlled writer design plan preflight status
- domain validation boundary status
- controlled writer value resolution precheck status
- accepted, needs-review, and blocked candidate counts
- required, resolved, and missing field coverage counts
- would-write aggregate row and field counts
- would-create and would-materialize flags remain false
- controlled writer and artifact boundary flags remain false
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`
- redaction status and forbidden marker absence
- clean run-specific output directory

## Status Semantics

`passed` means every required upstream status has passed, accepted candidate
count meets the configured minimum, required fields are fully resolved,
would-write counts are aggregate only, no blocked or needs-review candidate is
present, no side-effect flag indicates materialization, and redaction passed.

`needs_review` means no hard blocker exists, but an explicitly allowed
needs-review condition remains, such as needs-review candidate counts or
non-required field coverage gaps. Needs-review dry-run output is still not
writer-ready.

`blocked` means a hard validation, boundary, output-directory, or redaction
failure occurred. Blocked dry-runs do not write normal report, summary, or
Markdown evidence.

## Output Files

For a passing or needs-review run, outputs are written under:

```text
<output_dir>/<dry_run_id>/
```

Allowed files:

```text
property_training_dataset_controlled_writer_dry_run_report.json
property_training_dataset_controlled_writer_dry_run_summary.json
redacted_property_training_dataset_controlled_writer_dry_run_evidence.md
```

Report schema:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1
```

Summary schema:

```text
custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1
```

The summary references the report by basename and checksum only.

## Redaction Policy

The dry-run scans the input package, generated report, generated summary, and
Markdown evidence before writing. If unsafe material is found, the dry-run
fails closed with a minimal blocked summary and writes no unsafe evidence.

The report, summary, and Markdown may include safe ids, status labels, counts,
booleans, schema labels, and safe error codes. They must not include raw values,
molecular strings, article body text, table payloads, local paths, output
paths, document file names, row payloads, conformer data, DPA3 structures,
model input tensors, or credential material.

## Side-Effect Boundary

This controlled writer dry-run does not execute the controlled writer.
This controlled writer dry-run does not emit raw values.
This controlled writer dry-run does not materialize values.
This controlled writer dry-run does not serialize training rows.
This controlled writer dry-run does not create training dataset artifacts.
This controlled writer dry-run does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer dry-run does not generate conformers.
This controlled writer dry-run does not generate DPA3 structures.
This controlled writer dry-run does not run Phase 1.
This controlled writer dry-run does not modify DatasetConfirmation.
This controlled writer dry-run does not run model training or evaluation.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run \
  --controlled-writer-dry-run-input /tmp/controlled_writer_dry_run_input.json \
  --output-dir /tmp/controlled_writer_dry_run_output
```

Optional flags:

```text
--no-require-design-plan-preflight-passed
--no-require-domain-validation-passed
--no-require-value-resolution-precheck-passed
--no-require-values-resolved
--allow-needs-review-candidates
--minimum-accepted-candidate-records <N>
```

The CLI prints JSON to stdout and returns zero for `passed` or `needs_review`,
and one for `blocked`.

## Blocked Conditions

The dry-run blocks on wrong schema, missing required fields, unsafe ids, dirty
run directory, required upstream status not passed, insufficient accepted
candidate count, blocked candidate count, unallowed needs-review candidate
count, missing required fields when values are required, any would-create or
would-materialize flag set to true, writer execution flags, dataset artifact
flags, Phase 1 status other than `not_run`, DatasetConfirmation mutation, model
training or evaluation flags, unsafe emitted material, or redaction failure.

## Out of Scope

This PR does not implement controlled writer execution, the future dry-run
precheck, source payload reading, row serialization, dataset artifact writing,
review artifact writing, conformer generation, DPA3 structure generation,
Phase 1, DatasetConfirmation changes, model training, evaluation, LLM or agent
calls, MinerU calls, document parsing, corpus workflow execution, or chemistry
calculations.

## Next Step

The next step is property training dataset controlled writer dry-run precheck. Controlled writer
execution remains a separate future gate.
