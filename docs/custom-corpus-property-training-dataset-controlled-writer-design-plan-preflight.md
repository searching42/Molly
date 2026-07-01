# Custom Corpus Property Training Dataset Controlled Writer Design Plan Preflight

## Purpose

This document describes the offline deterministic preflight for a property
training dataset controlled writer design plan package. The preflight validates
that a proposed design plan is schema-valid, hash-bound, status-consistent,
redacted, and still safely before any writer implementation or
artifact-producing operation.

This preflight is a package validator only. It does not implement a controlled
writer, execute a controlled writer, run a writer dry-run, emit raw values,
materialize values, serialize rows, create training dataset artifacts, run
Phase 1, modify `DatasetConfirmation`, run model training, run evaluation,
call LLMs or agents, call MinerU, parse documents, or run chemistry
calculations.

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
-> future controlled writer execution request preflight
-> future explicitly confirmed controlled writer execution
```

This preflight is step 2 in the staged writer path. It validates the design
plan package before the controlled writer dry-run design. It does not claim
that a writer exists or that a dataset artifact exists.

## Input Package

The input is a local JSON design plan package using schema version
`custom_corpus_property_training_dataset_controlled_writer_design_plan.v1`.

The package may contain only safe ids, labels, counts, booleans, status fields,
and hashes. It must not contain raw values, molecular strings, source payloads,
local paths, output paths, source file names, paper titles, or row payloads.

## Preflight Checks

The preflight checks:

- design plan schema version
- safe design plan id, corpus id, and dataset name
- design plan status
- upstream evidence status fields
- candidate counts
- source package reference ids
- value resolution contract status
- boundary flags
- redaction status
- forbidden marker absence

The preflight reads only the controlled writer design plan JSON package.

## Status Semantics

`passed` means the design plan is valid, all required upstream statuses are
passed, values are resolved, required field misses are zero, accepted candidate
count meets the configured minimum, no needs-review or blocked candidate count
remains, boundary flags remain false or `not_run`, and redaction passes.

`needs_review` means no hard blocker exists, but a design plan or upstream
needs-review condition was explicitly allowed. Needs-review evidence is not
writer-ready by default.

`blocked` means schema, ids, statuses, counts, source references, value
resolution contract, boundary flags, or redaction checks failed.

## Redaction Policy

The preflight fails closed if the design plan, summary, or Markdown evidence
contains private paths, output paths, source file names, raw values, exact
numeric extracted values, molecular strings, table payloads, article body
text, row payloads, credential material, conformer data, or DPA3 structures.

On redaction failure, the preflight returns a minimal blocked summary and does
not write unsafe Markdown evidence.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight \
  --controlled-writer-design-plan /tmp/property_training_dataset_controlled_writer_design_plan.json \
  --output-summary /tmp/property_training_dataset_controlled_writer_design_plan_preflight_summary.json \
  --output-markdown /tmp/redacted_property_training_dataset_controlled_writer_design_plan_preflight_evidence.md
```

Optional controls:

- `--allow-design-plan-needs-review`
- `--no-require-design-plan-passed`
- `--no-require-domain-validation-passed`
- `--no-require-values-resolved`
- `--minimum-accepted-candidate-records <N>`

The CLI exits 0 for `passed` or `needs_review`, and 1 for `blocked`.

## Outputs

The summary JSON uses schema version
`custom_corpus_property_training_dataset_controlled_writer_design_plan_preflight.v1`.
It contains safe basenames, hashes, ids, status fields, candidate counts,
boundary booleans, errors, and warnings only.

The optional Markdown evidence is redacted and repeats the boundary statement
that this is only a controlled writer design plan preflight.

## Blocked Conditions

The preflight blocks on:

- wrong or missing schema
- missing required fields
- unsafe ids
- blocked, failed, missing, or invalid required statuses
- domain validation not passed unless explicitly allowed as needs-review
- value-resolution dry-run precheck blocked or missing
- unresolved values when resolution is required
- required field misses when resolution is required
- accepted candidate count below the configured minimum
- blocked candidate count above zero
- needs-review candidate count without explicit allowance
- missing or unsafe source package references
- writer implemented, writer executed, or writer dry-run executed
- values materialized
- rows serialized
- training dataset materialized
- dataset artifact created
- Phase 1 status other than `not_run`
- `DatasetConfirmation` changed
- model training or evaluation run
- redaction failure

## Out of Scope

This controlled writer design plan preflight does not implement the controlled writer.
This controlled writer design plan preflight does not execute the controlled writer.
This controlled writer design plan preflight does not run a writer dry-run.
This controlled writer design plan preflight does not emit raw values.
This controlled writer design plan preflight does not materialize values.
This controlled writer design plan preflight does not serialize training rows.
This controlled writer design plan preflight does not create training dataset artifacts.
This controlled writer design plan preflight does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer design plan preflight does not generate conformers.
This controlled writer design plan preflight does not generate DPA3 structures.
This controlled writer design plan preflight does not run Phase 1.
This controlled writer design plan preflight does not modify DatasetConfirmation.
This controlled writer design plan preflight does not run model training or evaluation.

## After Preflight: Controlled Writer Dry-Run Design

The next step after this PR is the property training dataset controlled writer
dry-run design. See
`docs/custom-corpus-property-training-dataset-controlled-writer-dry-run-design.md`
and
`docs/evidence/templates/custom-corpus-property-training-dataset-controlled-writer-dry-run-design-evidence-template.md`.

## Next Step

The next step after this PR is controlled writer dry-run design, not writer
execution.
