# Custom Corpus Property Training Dataset Quarantined Candidate Admission Boundary

## Purpose

This document defines the boundary between quarantined property candidates and
any future controlled training dataset writer. It specifies which upstream
evidence must exist, which candidate statuses are acceptable, and which
redaction and boundary flags must remain in force before writer design can
continue.

The boundary is documentation and review guidance only. It does not implement
or authorize a writer, materializer, training admission executor, dataset
artifact creator, model training run, or evaluation run.

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
-> future controlled writer execution request
-> future explicitly confirmed controlled writer execution
```

This boundary sits after the small public quarantine materialization evidence
packet and before the property training dataset domain validation boundary. A
passed quarantined-candidate boundary review is necessary but not sufficient
for writer execution.

## Required Upstream Evidence

The boundary requires the following upstream evidence before any quarantined
candidate can be considered for future writer design:

- small public quarantine materialization evidence
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

The boundary consumes safe evidence summaries only: ids, hashes, labels,
counts, status labels, redaction status, and boundary booleans.

## Eligible Quarantined Candidate Criteria

A quarantined candidate is eligible for future training admission design only
when all of the following are true:

- candidate id is safe and stable
- quarantine record id is safe and stable
- review and admission status is accepted, or explicitly needs-review with a
  recorded allowance
- candidate source is public-safe or explicitly approved
- property field labels are present
- unit and status labels are present when applicable
- row contract mapping exists
- value source manifest authorization exists
- value resolution dry-run has passed, or is explicitly allowed needs-review
- value resolution dry-run precheck has passed, or is explicitly allowed
  needs-review
- no blocked, excluded, or rejected candidate is admitted
- boundary evidence emits no raw property values
- boundary evidence emits no canonical molecular strings, structure keys,
  article body text, table row payloads, private paths, source file names,
  output locations, row payloads, conformer data, or DPA3 structure data

## Training Admission Boundary Criteria

A future writer may only be designed after this boundary if these status
conditions are all satisfied:

```text
small_public_quarantine_evidence_status=passed
quarantine_candidate_preflight_status=passed
training_admission_readiness_status=ready
training_dataset_materialization_plan_precheck_status=passed
training_dataset_row_contract_precheck_status=passed
training_dataset_materialization_dry_run_precheck_status=passed
writer_execution_request_preflight_status=passed
writer_input_binding_plan_preflight_status=passed
writer_value_source_manifest_preflight_status=passed
controlled_writer_execution_plan_preflight_status=passed
controlled_writer_value_resolution_dry_run_precheck_status=passed
```

The following boundary flags must also remain true:

```text
controlled_writer_executed=false
training_dataset_materialized=false
dataset_artifact_created=false
serialized_rows_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

## Value Resolution Boundary Criteria

The boundary can consume only safe value-resolution evidence:

Allowed evidence:

- ids
- hashes
- source labels
- field names
- counts
- status labels
- redaction status
- boundary booleans

Forbidden evidence categories:

- raw property values
- canonical molecular strings
- structure identifiers and key-form structure identifiers
- table row payloads
- article body text
- source file names
- local or private paths
- output artifact locations
- row payloads
- CSV/JSONL/Parquet/LMDB artifact locations
- conformer data
- DPA3 structures
- credential material and auth material

## Public Evidence Boundary

Public evidence must be intentionally small, redacted, and reviewable. It may
use safe source labels, candidate ids, quarantine ids, counts, statuses, hashes,
and aggregate decisions. It must not include exact extracted values, molecular
strings, source payloads, article text, table text, file names, output
locations, row payloads, or credential material.

## Disallowed Outputs

This boundary does not permit any of the following:

- controlled writer execution
- source payload reading by this boundary
- value materialization
- row serialization
- training dataset materialization
- review dataset artifact creation
- candidate or training CSV/JSONL/Parquet/LMDB creation
- conformer generation
- DPA3 structure generation
- Phase 1 execution
- DatasetConfirmation mutation
- model training
- evaluation or Agentic RL
- LLM, agent, MinerU, parser, or corpus workflow calls

## Pass Criteria

The boundary can be marked passed when every required upstream evidence item is
present, every required status is passed or ready, every required boundary flag
remains false or not_run as appropriate, candidate counts match the redacted
evidence chain, value-resolution dry-run precheck is passed, and redaction
review finds no disallowed material.

## Needs-Review Criteria

The boundary can be marked needs-review only when no hard blocker exists and
the operator explicitly records an allowance for a needs-review upstream
condition. Needs-review candidates must remain separate from accepted
candidates and must not be silently treated as writer-ready.

## Fail Criteria

The boundary fails if any required evidence is missing, any required hash or id
chain is inconsistent, readiness is blocked, a rejected or blocked candidate
appears in admitted evidence, value-resolution precheck is blocked, boundary
flags indicate writer or dataset activity, or the evidence contains disallowed
raw or sensitive material.

## Residual Risks

- This boundary does not certify scientific correctness.
- Public-safe source labels still require operator review.
- Needs-review allowances remain human-governed and must be documented.
- Domain validation remains a separate downstream boundary.
- A future controlled writer still requires a separate implementation,
  preflight, review, and approval path after domain validation.

## Next Step

The next step is the property training dataset domain validation boundary. A
future controlled writer dry-run remains out of scope until the domain
boundary, design plan, design plan preflight, and later writer-specific gates
pass.

This boundary evidence does not execute a controlled writer.
This boundary evidence does not materialize values.
This boundary evidence does not serialize training rows.
This boundary evidence does not create training dataset artifacts.
This boundary evidence does not create CSV/JSONL/Parquet/LMDB artifacts.
This boundary evidence does not generate conformers.
This boundary evidence does not generate DPA3 structures.
This boundary evidence does not run Phase 1.
This boundary evidence does not modify DatasetConfirmation.
This boundary evidence does not run model training or evaluation.
