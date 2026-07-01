# Custom Corpus Property Training Dataset Domain Validation Boundary

## Purpose

This document defines the scientific and domain-validation boundary that must
sit between quarantined-candidate admission evidence and any future controlled
training dataset writer. The governance chain proves provenance, review state,
hash binding, and redaction. It does not by itself certify that a property
record is scientifically correct or domain-ready.

This boundary is docs/test only. It does not inspect source payloads, emit raw
values, run calculations, execute a writer, materialize data, create rows, run
models, call LLMs or agents, call MinerU, or parse documents.

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
-> future controlled writer execution request
-> future controlled writer execution request preflight
-> future explicitly confirmed controlled writer execution
```

The domain validation boundary is downstream of the quarantined-candidate
admission boundary and upstream of the controlled writer design plan. A passed
domain boundary is necessary but not sufficient for writer execution.

## Required Upstream Evidence

The domain boundary requires, at minimum:

- small public quarantine materialization evidence
- quarantined-candidate admission boundary evidence
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

Only safe summaries, ids, labels, hashes, aggregate counts, and status fields
are acceptable at this boundary.

## Domain Validation Scope

Future domain validation must check the following using safe labels and
aggregate counts only:

- property name and category consistency
- property-unit compatibility
- numeric plausibility status
- provenance type consistency
- experimental versus computational source labels
- condition labels when relevant
- compound and alias association consistency
- duplicate and conflict status
- missing domain metadata status
- human review escalation status

Safe status labels include:

```text
property_unit_compatibility_status=passed
numeric_plausibility_status=passed
provenance_consistency_status=passed
condition_completeness_status=needs_review
compound_alias_association_status=passed
duplicate_conflict_status=passed
```

The boundary evidence must not include raw values, exact numeric examples,
molecular strings, table payloads, article text, source file names, or output
locations.

## Property-Unit Compatibility Boundary

Future validation should verify property and unit compatibility using labels
only. Property classes may include:

- energy-like properties
- wavelength-like properties
- ratio or yield-like properties
- dimensionless descriptors
- lifetime or rate-like properties
- transition or oscillator-related properties

Common safe field labels may include HOMO, LUMO, PLQY, TDM, SOC, emission
wavelength, absorption wavelength, and energy gap. The evidence should record
only pass, needs-review, or fail counts and field labels. It must not record
numeric values.

## Numeric Plausibility Boundary

Numeric plausibility checks should produce safe status labels only:

```text
numeric_plausibility_status=passed
numeric_plausibility_status=needs_review
numeric_plausibility_status=failed
```

This boundary must not include concrete numeric examples, exact extracted
values, molecular strings, table payloads, or source text.

## Provenance and Condition Boundary

Future validation should require clear provenance labels:

```text
source_type=experimental
source_type=computational
source_type=reported
source_type=derived
source_type=needs_review
```

Condition labels may include safe categorical metadata:

```text
condition_context=solution
condition_context=film
condition_context=neat
condition_context=doped
condition_context=unknown
```

The boundary must not emit raw condition text, source-table solvent strings,
file names, paper titles, or local paths.

## Compound and Alias Association Boundary

Future validation must ensure:

- candidate id is tied to one safe compound reference
- alias mapping is explicit and reviewable
- cross-table alias reuse is not silently accepted
- one value is not assigned to the wrong compound id
- ambiguous alias mappings become needs-review
- missing compound association becomes failed or needs-review according to
  policy
- blocked or rejected candidates cannot be promoted

The boundary evidence should use only ids, hashes, labels, and aggregate
counts.

## Duplicate and Conflict Boundary

Future validation must detect:

- duplicate candidate ids
- duplicate property records for the same compound, property, and context
- conflicting values for the same compound, property, and context
- mismatched units for the same property and context
- incompatible source labels
- conflicting accepted and needs-review status

The evidence should record only conflict counts and safe record ids. It must
not emit raw conflicting values.

## Allowed Evidence

Allowed domain validation boundary evidence:

- safe ids
- source labels
- property field labels
- unit labels
- status labels
- hash values
- aggregate counts
- redaction status
- boundary booleans
- needs-review reason labels

## Disallowed Evidence

The boundary forbids:

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
- output paths
- row payloads
- candidate or training CSV/JSONL/Parquet/LMDB artifact paths
- conformer data
- DPA3 structures
- model input tensors
- credential material
- auth header material
- API key material

## Pass Criteria

The domain validation boundary can be marked passed only when:

```text
quarantined_candidate_admission_boundary_status=passed
property_unit_compatibility_status=passed
numeric_plausibility_status=passed
provenance_consistency_status=passed
condition_completeness_status=passed_or_not_applicable
compound_alias_association_status=passed
duplicate_conflict_status=passed
redaction_status=passed
controlled_writer_executed=false
training_dataset_materialized=false
dataset_artifact_created=false
serialized_rows_created=false
phase1_status=not_run
dataset_confirmation_changed=false
model_training_run=false
evaluation_run=false
```

## Needs-Review Criteria

The boundary can be marked needs-review when no hard blocker exists and one or
more of the following applies:

- domain metadata is incomplete
- condition context is missing but not mandatory for the selected property
  class
- alias mapping is ambiguous but not contradicted
- provenance is incomplete but source labels remain safe
- numeric plausibility cannot be evaluated without emitting values
- an operator explicitly records the needs-review allowance

Needs-review candidates must remain separate from accepted candidates.
Needs-review evidence must not be treated as writer-ready.

## Fail Criteria

The boundary fails if:

- required upstream evidence is missing
- quarantined-candidate admission boundary is blocked
- rejected, blocked, or excluded candidates appear in accepted evidence
- property-unit compatibility fails
- numeric plausibility fails according to configured policy
- provenance conflicts are detected
- compound or alias association is contradictory
- duplicate conflicts remain unresolved
- redaction fails
- raw values or molecular strings appear
- boundary flags indicate writer execution, dataset materialization, row
  serialization, Phase 1 execution, DatasetConfirmation mutation, training, or
  evaluation

## Residual Risks

- This boundary defines required domain checks but does not certify scientific
  truth as final.
- Future domain validation still requires policy configuration and human review
  for ambiguous cases.
- Passing this boundary does not authorize writer execution.
- Any future controlled writer still requires a separate implementation and
  review path.

## Next Step

The next step is the property training dataset controlled writer design plan
only after the domain validation boundary has passed. The writer
implementation remains out of scope for this PR.

This domain validation boundary does not execute a controlled writer.
This domain validation boundary does not emit raw values.
This domain validation boundary does not materialize values.
This domain validation boundary does not serialize training rows.
This domain validation boundary does not create training dataset artifacts.
This domain validation boundary does not create CSV/JSONL/Parquet/LMDB artifacts.
This domain validation boundary does not generate conformers.
This domain validation boundary does not generate DPA3 structures.
This domain validation boundary does not run Phase 1.
This domain validation boundary does not modify DatasetConfirmation.
This domain validation boundary does not run model training or evaluation.
