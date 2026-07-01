# Custom Corpus Property Training Dataset Controlled Writer Design Plan

## Purpose

This document defines the docs/test-only design plan that must sit after the
property training dataset domain validation boundary and before any future
controlled training dataset writer implementation. It describes what a future
writer would be allowed to consume, what it may eventually emit after later
implementation and explicit confirmation gates, and which invariants remain
blocked at the current stage.

The design plan exists so the repository can review the writer contract before
any implementation that could create dataset artifacts. It does not implement
writer code, invoke writer code, read source payloads, materialize values,
create rows, run calculations, run Phase 1, modify `DatasetConfirmation`, run
model training, run evaluation, call an LLM or agent, call MinerU, or parse
documents.

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
-> future controlled writer dry-run precheck
-> future controlled writer execution request
-> future explicitly confirmed controlled writer execution
```

The controlled writer design plan is downstream of the domain validation
boundary. It is upstream of the controlled writer design plan preflight and
any future writer dry-run, writer dry-run precheck, writer execution request,
and explicitly confirmed writer execution.

## Required Upstream Evidence

The design plan requires, at minimum:

- small public quarantine materialization evidence
- quarantined-candidate admission boundary evidence
- domain validation boundary evidence
- quarantine candidate materialization evidence
- quarantine candidate preflight evidence
- property candidate review and admission evidence
- training admission readiness, request, and preflight evidence
- training dataset materialization planner evidence
- materialization plan precheck evidence
- row contract and row contract precheck evidence
- materialization dry-run and dry-run precheck evidence
- writer execution request and preflight evidence
- input binding planner and preflight evidence
- value source manifest planner and preflight evidence
- controlled writer execution plan and preflight evidence
- controlled writer value resolution dry-run and precheck evidence

Only safe ids, hashes, field labels, source labels, status labels, aggregate
counts, redaction status, and boundary booleans are acceptable as design-plan
evidence.

## Writer Design Scope

A future controlled writer may eventually be responsible for creating a
reviewable training dataset artifact only after a separate implementation PR,
explicit confirmation gates, and another review of the writer execution
boundary.

This design plan does not implement the controlled writer.
This design plan does not execute the controlled writer.
This design plan does not materialize values.
This design plan does not serialize training rows.
This design plan does not create training dataset artifacts.
This design plan does not create CSV/JSONL/Parquet/LMDB artifacts.

## Input Package Contract

A future controlled writer may only consume package references that are:

- schema-valid
- hash-bound
- status-passed or explicitly allowed needs-review
- path-safe
- redacted at the evidence layer
- tied to accepted candidate ids
- tied to row contract ids
- tied to value source manifest ids
- tied to controlled writer execution plan ids
- tied to value-resolution dry-run and precheck ids
- tied to domain validation boundary status

Allowed input evidence labels:

- ids
- hashes
- field labels
- source labels
- status labels
- aggregate counts
- redaction status
- boundary booleans

Forbidden input evidence in the design plan:

- raw property values
- exact numeric extracted values
- canonical molecular strings
- structure identifiers
- key-form structure identifiers
- table row payloads
- article body text
- paper titles
- source file names
- local paths
- output artifact paths
- row payloads
- model input tensors
- conformer data
- DPA3 structures
- credential material
- auth header material
- API key material

## Admission and Domain Validation Contract

The future writer design must require:

```text
quarantined_candidate_admission_boundary_status=passed
domain_validation_boundary_status=passed
property_unit_compatibility_status=passed
numeric_plausibility_status=passed
provenance_consistency_status=passed
compound_alias_association_status=passed
duplicate_conflict_status=passed
redaction_status=passed
```

Needs-review evidence may only be considered with an explicit future policy.
It must not be treated as writer-ready by default, and needs-review candidates
must remain separate from accepted candidates.

## Value Resolution Contract

The future writer design must require:

```text
controlled_writer_value_resolution_dry_run_status=passed
controlled_writer_value_resolution_dry_run_precheck_status=passed
values_resolved=true
missing_required_field_count=0
controlled_writer_executed=false
values_materialized=false
serialized_rows_created=false
training_dataset_materialized=false
dataset_artifact_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

The design plan may reference resolution counts, field labels, source labels,
and hashes. It must not include resolved values or row payloads.

## Output Artifact Policy

The design plan distinguishes future allowed outputs from current forbidden
outputs. Current review of this document does not authorize any writer output.

Future allowed outputs, only in a later implementation PR and only after
explicit confirmation, may include:

- controlled writer execution report
- controlled writer execution summary
- redacted writer evidence Markdown
- reviewable training dataset artifact manifest
- safe artifact checksums
- safe row counts and field coverage summaries
- redacted dataset-level quality report

Current PR forbidden outputs:

- training rows
- review rows
- candidate rows
- CSV artifacts
- JSONL artifacts
- Parquet artifacts
- LMDB artifacts
- conformer files
- DPA3 structure files
- model input tensors
- Phase 1 artifacts

## Dry-Run-First Policy

Any future writer implementation must be staged:

1. writer design plan
2. writer design plan preflight
3. controlled writer dry-run
4. controlled writer dry-run precheck
5. controlled writer execution request
6. explicitly confirmed controlled writer execution

Step 2 is the controlled writer design plan preflight. This design plan does
not implement steps 3 through 6.

## Confirmation and Operator Control

Future writer stages must require explicit confirmation and a safe operator id,
for example:

```text
confirm_controlled_writer_design_plan=true
confirm_controlled_writer_dry_run=true
confirm_controlled_writer_execution=true
confirmed_by=<safe_operator_id>
```

This PR records those confirmation concepts as static documentation only. It
does not add live confirmation logic.

## Redaction and Non-Leakage Requirements

Future writer design evidence must remain redacted and label-only. It may
include safe ids, hashes, source labels, field labels, aggregate counts, status
labels, redaction status, and boundary booleans.

It must not include raw values, exact numeric extracted values, molecular
strings, structure identifier strings, table row payloads, article body text,
paper titles, source file names, local paths, output paths, row payloads,
model input tensors, conformer data, DPA3 structures, credential material,
auth header material, or API key material.

## Allowed Future Writer Outputs

Allowed future outputs are only design targets for a later implementation PR:

- controlled writer execution report
- controlled writer execution summary
- redacted writer evidence Markdown
- training dataset artifact manifest
- artifact checksums
- aggregate row counts
- aggregate field coverage summaries
- redacted dataset-level quality report

Each future output must preserve provenance back to accepted candidate ids,
row contract ids, materialization plan ids, execution ledger ids, value source
manifest ids, value-resolution dry-run ids, and domain validation boundary
evidence.

## Disallowed Current Outputs

This PR must not create:

- training rows
- review rows
- candidate rows
- CSV artifacts
- JSONL artifacts
- Parquet artifacts
- LMDB artifacts
- conformer files
- DPA3 structure files
- model input tensors
- Phase 1 artifacts
- writer dry-run reports
- writer preflight reports
- writer execution requests
- writer execution summaries

## Implementation Blockers

Writer implementation remains blocked until:

- the controlled writer design plan has been reviewed
- a separate design-plan preflight exists
- a future controlled writer dry-run design exists
- the domain validation boundary is passed
- the value-resolution dry-run precheck is passed
- redaction rules for writer execution reports are specified
- artifact naming and output-directory policy are reviewed without concrete
  output paths
- explicit confirmation behavior is designed and tested

## Pass Criteria

The design plan can be treated as passed only when:

```text
quarantined_candidate_admission_boundary_status=passed
domain_validation_boundary_status=passed
controlled_writer_value_resolution_dry_run_precheck_status=passed
property_unit_compatibility_status=passed
numeric_plausibility_status=passed
provenance_consistency_status=passed
compound_alias_association_status=passed
duplicate_conflict_status=passed
values_resolved=true
missing_required_field_count=0
redaction_status=passed
controlled_writer_executed=false
values_materialized=false
serialized_rows_created=false
training_dataset_materialized=false
dataset_artifact_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

## Needs-Review Criteria

The design plan can be marked needs-review when no hard blocker exists and one
or more upstream needs-review statuses are explicitly allowed by future policy.
Needs-review evidence remains non-authoritative and must not be treated as
writer-ready by default.

## Fail Criteria

The design plan fails if:

- required upstream evidence is missing
- quarantined-candidate admission boundary evidence is blocked
- domain validation boundary evidence is blocked
- value-resolution dry-run precheck evidence is blocked
- required field coverage is incomplete
- rejected, blocked, or excluded candidate ids appear in accepted evidence
- redaction fails
- raw values, molecular strings, table payloads, article body text, local
  paths, output paths, row payloads, conformer data, or DPA3 structures appear
- boundary flags indicate writer execution, value materialization, row
  serialization, dataset materialization, artifact creation, Phase 1 execution,
  DatasetConfirmation mutation, model training, or evaluation

## Residual Risks

- This design plan defines the intended writer contract but does not prove the
  eventual implementation.
- Artifact output policy still requires a future implementation review.
- Needs-review handling still requires explicit future policy.
- Passing this design plan does not authorize writer execution.

## After Design Plan: Design Plan Preflight

The next step is the property training dataset controlled writer design plan
preflight. See
`docs/custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight.md`
and
`docs/evidence/templates/custom-corpus-property-training-dataset-controlled-writer-design-plan-preflight-evidence-template.md`.

The preflight validates the design plan package before any future controlled
writer dry-run design. The future controlled training dataset writer
implementation remains out of scope.

## Next Step

The next step is the controlled writer design plan preflight. Writer dry-run
design remains future work after that preflight.

This controlled writer design plan does not implement the controlled writer.
This controlled writer design plan does not execute the controlled writer.
This controlled writer design plan does not emit raw values.
This controlled writer design plan does not materialize values.
This controlled writer design plan does not serialize training rows.
This controlled writer design plan does not create training dataset artifacts.
This controlled writer design plan does not create CSV/JSONL/Parquet/LMDB artifacts.
This controlled writer design plan does not generate conformers.
This controlled writer design plan does not generate DPA3 structures.
This controlled writer design plan does not run Phase 1.
This controlled writer design plan does not modify DatasetConfirmation.
This controlled writer design plan does not run model training or evaluation.
