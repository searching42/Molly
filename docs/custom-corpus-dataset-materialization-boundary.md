# Custom Corpus Dataset Materialization Boundary

## Purpose

This document defines the future boundary between package-validated custom
corpus admission records and materialized dataset artifacts.

This is a design document only. It does not implement dataset materialization,
admit training data, run Phase 1, set `DatasetConfirmation.confirmed=true`,
create candidate/training CSVs, or certify scientific correctness.

## Current Governance Chain

```text
custom corpus manifest
-> custom corpus dry-run
-> property candidate manifest
-> property candidate planner
-> property candidate review queue
-> human review artifact
-> property review binding validator
-> property admission readiness planner
-> property admission request planner
-> property admission request draft
-> property admission draft package precheck
-> property-aware package binding validation
-> property materialization plan draft
-> property materialization plan preflight
-> property-aware offline materialization planner
-> property materialization dry-run
-> materializer execution request
-> materializer execution request preflight
-> property quarantine materializer
-> property quarantine candidate preflight
-> property training admission readiness
-> property training admission request planner
-> property training admission request preflight
-> property training admission request draft
-> property training admission request draft precheck
-> property training admission execution request
-> property training admission execution request preflight
-> property training admission execution dry-run
-> property training admission execution dry-run precheck
-> property training admission execution ledger
-> property training admission execution ledger precheck
-> property training dataset materialization planner
-> property training dataset materialization plan precheck
-> property training dataset row contract
-> property training dataset row contract precheck
-> property training dataset materialization dry-run
-> property training dataset materialization dry-run precheck
-> property training dataset writer execution request
-> property training dataset writer execution request preflight
-> property training dataset writer input binding planner
-> property training dataset writer input binding plan preflight
-> property training dataset writer value source manifest planner
-> property training dataset writer value source manifest preflight
-> property training dataset controlled writer execution plan
-> property training dataset controlled writer execution plan preflight
-> property training dataset controlled writer value resolution dry-run
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

Existing artifact schemas:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_property_candidate.v1`
- `custom_corpus_property_candidate_planner.v1`
- `custom_corpus_property_candidate_review_queue.v1`
- `custom_corpus_review.v1`
- `custom_corpus_property_review_binding.v1`
- `custom_corpus_property_admission_readiness.v1`
- `custom_corpus_property_admission_request_plan.v1`
- `custom_corpus_property_admission_draft_builder.v1`
- `custom_corpus_property_admission_draft_package_precheck.v1`
- `custom_corpus_property_package_binding.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_materialization_plan_draft_builder.v1`
- `custom_corpus_property_materialization_plan_preflight.v1`
- `custom_corpus_materialization.v1`
- `custom_corpus_materialization_planner.v1`
- `custom_corpus_property_materialization_planner_runner.v1`
- `custom_corpus_property_materialization_dry_run.v1`
- `custom_corpus_property_materializer_execution_request.v1`
- `custom_corpus_property_materializer_execution_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`
- `custom_corpus_property_quarantine_materializer.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_request_draft.v1`
- `custom_corpus_property_training_admission_request_draft_builder.v1`
- `custom_corpus_property_training_admission_request_draft_precheck.v1`
- `custom_corpus_property_training_admission_execution_request.v1`
- `custom_corpus_property_training_admission_execution_request_builder.v1`
- `custom_corpus_property_training_admission_execution_request_preflight.v1`
- `custom_corpus_property_training_admission_execution_dry_run.v1`
- `custom_corpus_property_training_admission_execution_dry_run_precheck.v1`
- `custom_corpus_property_training_admission_execution_ledger.v1`
- `custom_corpus_property_training_admission_execution_ledger_summary.v1`
- `custom_corpus_property_training_admission_execution_ledger_precheck.v1`
- `custom_corpus_property_training_dataset_materialization_plan.v1`
- `custom_corpus_property_training_dataset_materialization_planner.v1`
- `custom_corpus_property_training_dataset_materialization_plan_precheck.v1`
- `custom_corpus_property_training_dataset_row_contract.v1`
- `custom_corpus_property_training_dataset_row_contract_builder.v1`
- `custom_corpus_property_training_dataset_row_contract_precheck.v1`
- `custom_corpus_property_training_dataset_materialization_dry_run.v1`
- `custom_corpus_property_training_dataset_materialization_dry_run_summary.v1`
- `custom_corpus_property_training_dataset_materialization_dry_run_precheck.v1`
- `custom_corpus_property_training_dataset_writer_value_source_manifest.v1`
- `custom_corpus_property_training_dataset_writer_value_source_manifest_planner.v1`
- `custom_corpus_property_training_dataset_writer_value_source_manifest_preflight.v1`
- `custom_corpus_property_training_dataset_controlled_writer_execution_plan.v1`
- `custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1`
- `custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight.v1`
- `custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1`
- `custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1`
- `custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck.v1`

Current steps now include a candidate-only quarantine materializer for the
property path, a controlled training admission execution ledger, a ledger
precheck, a training dataset materialization planner, a materialization plan
precheck, a row contract, a row contract precheck, a training dataset
materialization dry-run, a training dataset materialization dry-run precheck,
writer request/binding/value-source planning, controlled writer execution
planning, value-resolution dry-run, value-resolution dry-run precheck, and a
small public quarantine materialization evidence checkpoint. They still stop
before training dataset writing, training artifacts, Phase 1, and
`DatasetConfirmation` mutation.

The property candidate schema represents open-ended numeric scientific
property candidates before review. It does not define a property whitelist,
call LLMs or agents, evaluate extraction accuracy, or materialize data.

The property candidate planner also sits before human review. It creates safe
review-planning summaries only; materialization still consumes reviewed,
admitted, package-validated, and materialization-plan records rather than raw
property candidate manifests directly.

The property candidate review queue builder sits after the planner and before
human review. It creates review-preparation artifacts only. Raw property
candidates and review queue artifacts do not directly materialize; future
materialization still requires review, admission, package validation, and a
materialization plan.

The property review binding validator sits after human review and before
admission. It validates queue-to-review consistency only. Review queue and
binding evidence do not directly materialize data.

The property admission readiness planner sits after review binding and before
admission. It summarizes accepted, queue-bound human review records for future
admission planning only. Readiness evidence does not directly materialize data.

The property admission request planner sits after readiness and before the
actual admission request. It prepares request-plan evidence only. Materialization
still requires actual admission, package validation, and a materialization plan;
request-plan evidence does not directly materialize data.

The property admission draft builder can write a reviewable
`custom_corpus_admission.v1` draft, but the draft is still upstream of package
binding. It does not directly materialize data, and materialization still
requires package validation and a materialization plan.

The property admission draft package precheck sits between draft generation
and formal package binding. It can report whether the draft and upstream
property summaries are consistent enough to attempt package binding, but it
does not create `custom_corpus_admission_package_validation.v1` and does not
directly materialize data.

The property-aware package binding runner sits after precheck and calls the
formal package binding validator. It can produce
`custom_corpus_admission_package_validation.v1`, but package binding output
alone still does not directly materialize data. Materialization requires a
materialization plan and future materializer.

The property materialization plan draft builder sits after formal package
binding. It can map package-validated property admission records into a
reviewable `custom_corpus_materialization.v1` draft, but that draft does not
materialize data. Materialization still requires offline planner and future
materializer boundaries.

The property materialization plan preflight sits after draft generation and
before the offline materialization planner. It checks schema/status/hash/record
consistency for the draft, but does not run the planner, run a materializer, or
execute materialization.

The property-aware offline materialization planner runner sits after preflight
and invokes the existing offline planner with property-specific gating
evidence. It can write `custom_corpus_materialization_planner.v1` planner
output plus a property-aware wrapper summary, but it still does not run a
materializer, execute materialization, create candidate/training CSVs, admit
training data, run Phase 1, or change `DatasetConfirmation`.

The property materialization dry-run runner sits after the property-aware
planner runner. It validates the existing planner output and upstream evidence
through a no-data dry-run report. The dry-run report is not materialization,
does not create candidate/training artifacts, and does not run a real
materializer.

The property materializer execution request builder sits after the dry-run. It
can write request-only handoff artifacts for a future materializer, but it
still does not run the materializer, execute materialization, create
candidate/training artifacts, admit training data, run Phase 1, or change
`DatasetConfirmation`.

The property materializer execution request preflight sits after request
generation and before any future materializer. It validates the request and
upstream dry-run/planner evidence only. It is not materialization and produces
no candidate/training artifact.

The property quarantine materializer sits after execution preflight. It writes
candidate-only quarantine records and safe summary/evidence artifacts. It does
not create training data, training CSV/JSONL/Parquet/LMDB artifacts, Phase 1
inputs, or `DatasetConfirmation` changes.

The property quarantine candidate preflight sits after quarantine
materialization and before any future training admission request. It checks
candidate-only quarantine artifacts and upstream evidence only. It is not
training admission, creates no training artifact, does not run Phase 1, and
does not change `DatasetConfirmation`.

The property training admission readiness planner sits after quarantine
candidate preflight and before any future training admission request. It checks
quarantine-candidate-preflight-passed artifacts and emits safe readiness
evidence only. It does not admit training data, create training or candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, run model training/evaluation,
or change `DatasetConfirmation`.

The property training admission request planner sits after training admission
readiness and before any future training admission request. It emits safe
request-plan evidence only. It does not create a training admission request,
create training admission actions, admit training data, create training or
candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, run model
training/evaluation, or change `DatasetConfirmation`.

The property training admission request preflight sits after request planning
and before any future training admission execution. It validates the plan,
readiness summary, and quarantine candidate preflight summary only. It does
not create or execute a training admission request, admit training data,
materialize datasets, create training or candidate CSV/JSONL/Parquet/LMDB
artifacts, run Phase 1, run model training/evaluation, or change
`DatasetConfirmation`.

The property training dataset materialization dry-run sits after row contract
precheck and before any future dataset writer. It generates safe row preview
summaries only. It does not serialize training rows, create training dataset
artifacts, create training or candidate CSV/JSONL/Parquet/LMDB artifacts,
generate conformers or DPA3 structures, run Phase 1, run model
training/evaluation, or change `DatasetConfirmation`.

The property training dataset materialization dry-run precheck sits after the
dry-run and before any future dataset writer. It validates dry-run report,
summary, row-preview, field/model/output summary, hash, id, and upstream
evidence consistency only. It is not dataset writing, row previews remain
summaries only, no training artifact is produced, Phase 1 remains separate,
and `DatasetConfirmation` remains unchanged.

The property training dataset writer execution request sits after the dry-run
precheck and before any future dataset writer execution. It emits a reviewable
request packet with safe ID/hash-only records. It does not execute a writer,
create a dataset, create training/candidate CSV/JSONL/Parquet/LMDB artifacts,
generate conformers or DPA3 structures, run Phase 1, run model
training/evaluation, or change `DatasetConfirmation`.

The property training dataset writer execution request preflight sits after
the writer execution request and before any future controlled writer. It
validates request package consistency only. It is not dataset writing, no
training artifact is produced, Phase 1 remains separate, and
`DatasetConfirmation` remains unchanged.

The property training dataset writer input binding planner sits after writer
execution request preflight and before writer input binding plan preflight. It
binds future row fields to allowed source artifact labels, hashes, source
record ids, and derivation rules only. It does not execute a writer,
materialize values, serialize training rows, create training/candidate
CSV/JSONL/Parquet/LMDB artifacts, generate conformers or DPA3 structures, run
Phase 1, run model training/evaluation, or change `DatasetConfirmation`.

The property training dataset writer input binding plan preflight sits after
the input binding planner and before any future controlled writer. It
validates the binding plan package, source hashes, ids, record counts, field
bindings, dedup/split rules, and boundary flags only. It does not execute a
writer, materialize values, serialize training rows, create training/candidate
CSV/JSONL/Parquet/LMDB artifacts, generate conformers or DPA3 structures, run
Phase 1, run model training/evaluation, or change `DatasetConfirmation`.

The property training dataset writer value source manifest planner sits after
input binding plan preflight and before value source manifest preflight. It
emits value-source authorization metadata only: source payloads are not read,
values are not materialized, no training artifact is produced, Phase 1 remains
separate, and `DatasetConfirmation` remains unchanged.

The property training dataset writer value source manifest preflight sits
after the value source manifest planner and before any future controlled
writer. It validates the manifest package, upstream hashes, ids, record
counts, value field coverage, source labels, and boundary flags only. It does
not execute a writer, read source payloads, materialize values, serialize
training rows, create training/candidate CSV/JSONL/Parquet/LMDB artifacts,
generate conformers or DPA3 structures, run Phase 1, run model
training/evaluation, or change `DatasetConfirmation`.

The property training dataset controlled writer execution plan sits after the
value source manifest preflight and before the controlled writer execution
plan preflight. It defines writer invocation
policy, allowed source basenames and hashes, output format labels, output
artifact labels, file naming policy labels, row-count expectations, and
provenance preservation requirements only. It does not execute a writer, read
source payloads, materialize values, serialize training rows, create
training/candidate CSV/JSONL/Parquet/LMDB artifacts, generate conformers or
DPA3 structures, run Phase 1, run model training/evaluation, or change
`DatasetConfirmation`.

The property training dataset controlled writer execution plan preflight sits
after the controlled writer execution plan and before any future controlled
writer. It validates that the plan package is safe, hash-bound, internally
consistent, and still free of source payload reads, materialized values,
serialized rows, output paths, training/candidate CSV/JSONL/Parquet/LMDB
artifacts, conformers, DPA3 structures, Phase 1 execution, model
training/evaluation, and `DatasetConfirmation` changes.

The property training dataset controlled writer value resolution dry-run sits
after the controlled writer execution plan preflight and before any future
value-resolution precheck or controlled writer. It may read only explicitly
authorized local JSON source payloads, resolves required field coverage
internally, and emits safe report/summary evidence without raw values,
serialized rows, output paths, training/candidate CSV/JSONL/Parquet/LMDB
artifacts, conformers, DPA3 structures, Phase 1 execution, model
training/evaluation, or `DatasetConfirmation` changes.

The property training dataset controlled writer value resolution dry-run
precheck sits after the dry-run and before any future controlled writer. It
validates only the emitted dry-run report/summary package; it does not re-read
authorized source payloads, execute a writer, emit values, materialize rows, or
create training/candidate CSV/JSONL/Parquet/LMDB artifacts.

The small public quarantine materialization evidence packet sits after the
value-resolution dry-run precheck as a docs-only acceptance note. It records a
tiny public/synthetic-public quarantine evidence scope with redacted ids and
counts only; it does not execute a writer, read source payloads, emit raw
values, serialize rows, create training artifacts, run Phase 1, or change
`DatasetConfirmation`.

The property training dataset quarantined candidate admission boundary sits
after the small public evidence packet and before any future controlled writer.
It defines the evidence, status, redaction, and boundary conditions required
before quarantined candidates can be considered for future writer design. It
does not execute a writer, read source payloads, materialize values, serialize
rows, create training artifacts, run Phase 1, or change
`DatasetConfirmation`.

The property training dataset domain validation boundary sits after the
quarantined-candidate admission boundary and before any future controlled
writer design plan. It defines scientific/domain checks for property-unit
compatibility, numeric plausibility status, provenance labels, condition
labels, compound/alias association, and duplicate/conflict status. It does not
inspect raw values, run calculations, execute a writer, create rows, run Phase
1, or change `DatasetConfirmation`.

The property training dataset controlled writer design plan sits after the
domain validation boundary and before any future controlled training dataset
writer implementation. It defines the future writer contract, input package
requirements, dry-run-first staging, confirmation concepts, redaction
invariants, and output artifact policy without implementing or executing a
writer.

The property training dataset controlled writer design plan preflight sits
after the design plan and before the controlled writer dry-run design. It
validates the design-plan package for schema, status, ids, candidate counts,
source package refs, value resolution contract, boundary flags, and redaction
without implementing or executing a writer.

The property training dataset controlled writer dry-run design sits after the
design plan preflight and before the controlled writer dry-run. It defines the
dry-run input/report/summary contracts, side-effect boundary, redaction
behavior, status semantics, and future precheck expectations.

The property training dataset controlled writer dry-run sits after the dry-run
design and before the controlled writer dry-run precheck. It reads only a safe,
aggregate-only dry-run input package and writes redacted report, summary, and
Markdown evidence. It does not execute a controlled writer, read source
payloads, emit raw values, materialize values, serialize rows, create
training/candidate CSV/JSONL/Parquet/LMDB artifacts, generate conformers or
DPA3 structures, run Phase 1, change `DatasetConfirmation`, or run model
training/evaluation.

The property training dataset controlled writer dry-run precheck sits after the
dry-run and before the execution request design/request gates. It validates
dry-run report/summary/evidence packages for schema, checksum, basename-only
references, aggregate counts, boundary flags, and redaction. It does not rerun
the dry-run, execute a controlled writer, read source payloads, emit raw values,
materialize values, serialize rows, create training/candidate
CSV/JSONL/Parquet/LMDB artifacts, generate conformers or DPA3 structures, run
Phase 1, change `DatasetConfirmation`, or run model training/evaluation.

The property training dataset controlled writer execution request design sits
after the dry-run precheck and before the execution request artifact creator. It
defines future request contents, upstream evidence requirements, authorization
boundaries, explicit confirmation boundaries, hash/basename policy, and
redaction rules. It does not create a request, implement request preflight,
confirm or execute the writer, serialize rows, create dataset artifacts, run
Phase 1, change `DatasetConfirmation`, or run model training/evaluation.

The property training dataset controlled writer execution request sits after
the request design and before any future request preflight. It reads only the
dry-run precheck summary, creates a hash-bound request package for later
preflight, keeps writer execution unauthorized, keeps explicit confirmation
required, and does not read source payloads, emit raw values, materialize
values, serialize rows, create dataset artifacts, run Phase 1, change
`DatasetConfirmation`, or run model training/evaluation.

The property training admission request draft builder sits after request
preflight and before any future training admission execution. It writes a
reviewable draft request only. It does not execute training admission, admit
training data, materialize datasets, create training or candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, run model training/evaluation,
or change `DatasetConfirmation`.

The property training admission request draft package precheck sits after
draft generation and before any future training admission execution. It
validates the draft package and upstream request/readiness/quarantine
evidence only. It does not execute training admission, admit training data,
produce training artifacts, run Phase 1, run model training/evaluation, or
change `DatasetConfirmation`.

The property training admission execution request builder sits after draft
package precheck and before any future training admission execution. It writes
reviewable request artifacts only. It does not execute training admission,
admit training data, create training or candidate CSV/JSONL/Parquet/LMDB
artifacts, run Phase 1, run model training/evaluation, or change
`DatasetConfirmation`.

## Materialization Definition

Materialization means transforming package-validated admitted records into
durable dataset artifacts that could later be consumed by Phase 1 or
downstream dataset builders.

Examples of future materialized artifacts may include:

- candidate records JSON/JSONL
- training candidate CSV
- manifest-to-record binding file
- provenance report
- reviewer/admission binding report
- rollback manifest
- redacted evidence summary

The implemented property quarantine materializer writes only
`property_quarantine_candidate_records.json` plus safe summary/evidence
artifacts. It does not create training CSVs, training JSONL/Parquet/LMDB
artifacts, Phase 1 inputs, or training data admission.

## Materialization Plan Schema

The pre-materialization plan schema is documented in:

```text
docs/custom-corpus-materialization-schema.md
```

Safe example plan:

```text
docs/examples/custom-corpus-materialization-plan.example.json
```

Future plan evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-plan-evidence-template.md
```

This schema is still pre-materialization. It validates operator intent,
source hash binding, candidate-only mode, explicit confirmation metadata,
record selection, and dry-run/package boundaries. It does not create outputs,
candidate CSVs, training CSVs, or Phase 1 inputs.

The property materialization plan draft builder is documented in:

```text
docs/custom-corpus-property-materialization-plan-draft.md
```

It creates a reviewable plan draft only. The draft is not materialization and
does not invoke the offline materialization planner or any future materializer.

The property materialization plan preflight is documented in:

```text
docs/custom-corpus-property-materialization-plan-preflight.md
```

It checks a reviewable draft before offline planner submission. It does not
materialize data and does not invoke the planner.

The property-aware offline materialization planner runner is documented in:

```text
docs/custom-corpus-property-materialization-planner-runner.md
```

It invokes the offline planner after preflight gating and explicit operator
confirmation. It remains planner execution only; it is not materialization.

The property materialization dry-run runner is documented in:

```text
docs/custom-corpus-property-materialization-dry-run.md
```

It consumes planner output and validates future materializer-readiness without
creating materialized data.

The property materializer execution request builder is documented in:

```text
docs/custom-corpus-property-materializer-execution-request.md
```

It creates a request-only future-materializer handoff after a passed dry-run.
The request is not execution and does not create materialized data.

The property materializer execution request preflight is documented in:

```text
docs/custom-corpus-property-materializer-execution-preflight.md
```

It checks request readiness before future materializer submission, but it does
not run a materializer or produce candidate/training artifacts.

The property quarantine materializer is documented in:

```text
docs/custom-corpus-property-quarantine-materializer.md
```

It writes candidate-only quarantine artifacts after a passed execution
preflight and explicit confirmation. These artifacts are still not training
data and are necessary but not sufficient for any future training admission.

The property quarantine candidate preflight is documented in:

```text
docs/custom-corpus-property-quarantine-candidate-preflight.md
```

It checks candidate-only quarantine artifacts before future training admission
planning. It produces no training data, training CSV/JSONL/Parquet/LMDB
artifacts, Phase 1 execution, or `DatasetConfirmation` changes.

The property training admission readiness planner is documented in:

```text
docs/custom-corpus-property-training-admission-readiness.md
```

It checks quarantine-candidate-preflight-passed artifacts for future training
admission readiness. It produces no training admission request, training data,
training CSV/JSONL/Parquet/LMDB artifacts, candidate CSV/JSONL/Parquet/LMDB
artifacts, Phase 1 execution, model training/evaluation, or
`DatasetConfirmation` changes.

The property training admission request planner is documented in:

```text
docs/custom-corpus-property-training-admission-request-planner.md
```

It checks readiness-ready or explicitly allowed partial artifacts and emits a
safe request plan for a future training admission request. It does not create
the request, create training admission actions, admit training data, create
training or candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, or change
`DatasetConfirmation`.

The property training admission request preflight is documented in:

```text
docs/custom-corpus-property-training-admission-request-preflight.md
```

It checks whether request-plan evidence remains consistent with readiness and
quarantine candidate preflight evidence before any future training admission
execution request. It does not execute admission, create training data,
materialize datasets, run Phase 1, or change `DatasetConfirmation`.

The property training admission request draft builder is documented in:

```text
docs/custom-corpus-property-training-admission-request-draft.md
```

It writes a reviewable draft request from preflight-passed request planning
evidence. The draft is not training admission execution, contains no training
data, produces no training artifacts, and leaves Phase 1 and
`DatasetConfirmation` unchanged.

The property training admission request draft package precheck is documented
in:

```text
docs/custom-corpus-property-training-admission-request-draft-precheck.md
```

It checks draft package consistency before future training admission
execution. The precheck is not execution, produces no training artifact, and
leaves Phase 1 and `DatasetConfirmation` unchanged.

The property training admission execution request builder is documented in:

```text
docs/custom-corpus-property-training-admission-execution-request.md
```

It writes a reviewable execution request from draft-precheck-passed evidence.
The request is not training admission execution, contains no training data,
produces no training artifacts, and leaves Phase 1 and `DatasetConfirmation`
unchanged.

The property training admission execution request preflight is documented in:

```text
docs/custom-corpus-property-training-admission-execution-request-preflight.md
```

It checks the execution request package before any future training admission
execution. The preflight is not execution, produces no training artifact, and
leaves Phase 1 and `DatasetConfirmation` unchanged.

The property training admission execution dry-run is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run.md
```

It simulates execution-request-preflight-passed packages before future
training admission execution. The dry-run is not execution, produces no
training artifact, and leaves Phase 1 and `DatasetConfirmation` unchanged.

The property training admission execution dry-run precheck is documented in:

```text
docs/custom-corpus-property-training-admission-execution-dry-run-precheck.md
```

It validates an existing execution dry-run report against the execution
request, request preflight, draft package, request plan, readiness summary, and
quarantine candidate records before any future training admission execution.
The precheck is not execution, creates no training artifact, and leaves Phase 1
and `DatasetConfirmation` unchanged.

The property training admission execution ledger is documented in:

```text
docs/custom-corpus-property-training-admission-execution-ledger.md
```

It commits dry-run-precheck-passed training admission decisions into a safe
ledger only. Ledger admission is not dataset materialization, creates no
training artifact, does not run Phase 1, and leaves `DatasetConfirmation`
unchanged.

The property training admission execution ledger precheck is documented in:

```text
docs/custom-corpus-property-training-admission-execution-ledger-precheck.md
```

It validates the committed ledger, ledger summary, and full upstream
ID/SHA/status/record chain before any future training dataset materialization
layer. The precheck is not execution, creates no training or candidate
CSV/JSONL/Parquet/LMDB artifact, does not run Phase 1, and leaves
`DatasetConfirmation` unchanged.

The property training dataset materialization planner is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-planner.md
```

It consumes ledger-precheck-passed packages and writes a plan only. The planner
is not dataset writing, produces no training or candidate
CSV/JSONL/Parquet/LMDB artifact, does not run Phase 1, and leaves
`DatasetConfirmation` unchanged.

The property training dataset materialization plan precheck is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-plan-precheck.md
```

It validates an existing materialization plan and planner summary against the
full upstream ledger package before any future row contract or dataset writer
work. The precheck is not dataset writing, creates no training or candidate
CSV/JSONL/Parquet/LMDB artifact, does not run Phase 1, and leaves
`DatasetConfirmation` unchanged.

The property training dataset row contract is documented in:

```text
docs/custom-corpus-property-training-dataset-row-contract.md
```

It defines future training-row semantics after a plan-precheck-passed package:
required fields, optional fields, field types, provenance requirements,
quality flag labels, split/dedup key rules, model-family compatibility labels,
and output-format compatibility labels. The row contract is not dataset
writing, produces no serialized rows, no training or candidate
CSV/JSONL/Parquet/LMDB artifact, no conformers, no DPA3 structures, no Phase 1
inputs, and no `DatasetConfirmation` changes.

The property training dataset row contract precheck is documented in:

```text
docs/custom-corpus-property-training-dataset-row-contract-precheck.md
```

It validates the row contract package before future materialization dry-run or
dataset writer work. The precheck is not dataset writing, creates no row
previews, no training or candidate CSV/JSONL/Parquet/LMDB artifacts, no
conformers, no DPA3 structures, no Phase 1 inputs, and no
`DatasetConfirmation` changes.

## Offline Materialization Planner

The offline planner is documented in:

```text
docs/custom-corpus-materialization-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-planner-evidence-template.md
```

The planner reads a valid `custom_corpus_materialization.v1` plan and produces
a safe JSON or Markdown planning summary. Planner output is not candidate
data, does not imply training admission, and does not create materialized
records or candidate/training artifacts.

Property-aware planner-runner evidence template:

```text
docs/evidence/templates/custom-corpus-property-materialization-planner-evidence-template.md
```

Property materialization dry-run evidence template:

```text
docs/evidence/templates/custom-corpus-property-materialization-dry-run-evidence-template.md
```

## Required Inputs For A Future Materializer

A future materializer must require:

1. custom corpus manifest
2. custom corpus dry-run report
3. human review manifest
4. admission request
5. admission package validation summary
6. explicit operator materialization confirmation
7. materialization output directory
8. materialization run id

Required input conditions:

- package validation status must be `passed`
- admission decision must be `eligible`
- no `needs_review` admission records may be materialized
- no rejected records may be materialized
- every admitted record must trace to an accepted review record
- dry-run `DatasetConfirmation.confirmed` must remain `false`
- dry-run Phase 1 status must remain `not_run`
- dry-run training dataset admitted must remain `false`

## Explicit Operator Confirmation

A future materializer must require explicit materialization confirmation. It
must not reuse synthetic dataset confirmation.

Suggested future concept:

```text
CustomCorpusMaterializationConfirmation
```

Design-only fields:

- `confirmed: bool`
- `confirmed_by: str`
- `confirmed_at: str`
- `confirmation_source: str`
- `package_validation_sha256: str`
- `admission_request_sha256: str`
- `review_manifest_sha256: str`
- `dry_run_report_sha256: str`
- `manifest_sha256: str`
- `corpus_id: str`
- `dry_run_id: str`
- `review_manifest_id: str`
- `admission_request_id: str`
- `reason: str`

Rules:

- `confirmed=true` must be explicit.
- `confirmed_by` must be non-empty and redacted if needed.
- confirmation must bind to exact artifact SHA-256 values.
- confirmation must not override failed package validation.
- confirmation must not override records with `needs_review`.
- confirmation must not bypass review completeness checks.
- confirmation must not set `DatasetConfirmation.confirmed=true` by itself.

This document only proposes the future concept. It does not implement it.

## Review Completeness Gate

A future materializer must verify:

- every materialized record has an admission record with `action=admit`
- every admitted record has a matched review record
- matched review record decision is `accept`
- review/admission document id, record id, field name, and source artifact
  SHA-256 match
- admitted records have non-empty provenance summary
- admitted records have non-empty normalized value summary
- admitted records have non-empty admission reason
- no `needs_review` record is materialized
- no rejected record is materialized
- no duplicate materialized target exists
- materialization record count equals admit count from package validation

## Provenance Binding

Each future materialized record must include safe provenance fields:

- corpus id
- dry-run id
- document id
- record id
- field name
- review id
- admission record id
- source manifest SHA-256
- dry-run report SHA-256
- review manifest SHA-256
- admission request SHA-256
- package validation summary SHA-256
- source artifact SHA-256
- review artifact SHA-256
- normalized value summary
- provenance summary
- materialization run id

Do not include:

- raw PDF path
- raw article text
- ParsedDocument text
- MinerU bundle path
- private home path
- token/auth/cookie/signed URL
- local absolute paths

## Proposed Future Output Artifacts

Design-only output directory layout:

```text
custom_corpus_materialization_<run_id>/
  materialization_summary.json
  materialized_records.jsonl
  materialized_records.csv
  provenance_bindings.jsonl
  rollback_manifest.json
  redacted_evidence_summary.md
```

Artifact policy:

| Artifact | Purpose | Commit policy | Redaction requirements |
| --- | --- | --- | --- |
| `materialization_summary.json` | Machine-readable run summary and counts. | Stay outside git by default. | No raw text, paths, tokens, or private details. |
| `materialized_records.jsonl` | Candidate materialized record payloads. | Stay outside git by default. | Safe summaries and provenance ids only. |
| `materialized_records.csv` | Tabular candidate records for future gates. | Stay outside git by default. | No raw article text or private paths. |
| `provenance_bindings.jsonl` | Per-record source/review/admission bindings. | Stay outside git by default. | SHA-256 values and safe ids only. |
| `rollback_manifest.json` | Deletion/rollback planning for generated outputs. | Stay outside git by default. | Redacted path labels where needed. |
| `redacted_evidence_summary.md` | Human-readable evidence for a future PR. | May be committed after review. | Must pass redaction checklist. |

Raw text, PDFs, ParsedDocuments, MinerU bundles, and private paths must never
be committed.

## Candidate/Training Artifact Boundary

Future materialization must use a strict two-step boundary.

Step A: materialize package-validated records into candidate artifacts only.

Step B: a separate future gate may decide whether candidate artifacts can
become training artifacts.

Rules:

- candidate artifact creation must not imply training admission
- training CSV creation must require a separate explicit gate
- Phase 1 must not run automatically after materialization
- `DatasetConfirmation.confirmed=true` must not be set by materialization alone
- any future training admission must bind to materialization summary SHA-256

## Phase 1 Boundary

Materialization must not run Phase 1. It must not call
`corpus_to_phase1_workflow` in confirmed mode, must not reuse synthetic
confirmation flags, and must not set `DatasetConfirmation.confirmed=true`.
Phase 1 remains a separate explicit future gate. Materialization evidence must
show Phase 1 was not run.

## Deletion And Rollback Design

A future materializer must produce a rollback manifest containing:

- materialization run id
- output artifact paths or redacted path labels
- output artifact SHA-256 values
- list of materialized record ids
- source package validation SHA-256
- deletion instructions
- rollback safety notes

Rules:

- operators must be able to delete materialized candidate artifacts
- rollback must not touch source PDFs
- rollback must not delete external original corpora
- rollback must distinguish local generated artifacts from committed evidence
- committed redacted evidence should be immutable unless a follow-up correction
  PR is made

## Redaction Requirements

Future materialization summaries and evidence must not include:

- raw PDFs
- local absolute paths
- private home paths
- `/Users/`
- `/home/`
- `C:\`
- tokens
- Authorization headers
- bearer tokens
- cookies
- x-api-key
- signed URLs
- raw article text
- ParsedDocument content
- MinerU bundle content
- remote task ids unless explicitly reviewed
- private emails unless redacted

Allowed:

- safe ids
- safe basenames
- SHA-256 values
- counts
- decision/status strings
- safe provenance summaries
- safe normalized value summaries
- safe binding error codes

## Pass Criteria For A Future Materialization Run

A future materialization run may pass only if:

- all source artifacts validate
- package validation status is `passed`
- admission decision is `eligible`
- explicit materialization confirmation is present
- confirmation binds exact source artifact hashes
- all materialized records are admitted and accepted
- no rejected or `needs_review` records are materialized
- materialized record count equals admit count
- output artifacts are created under a clean output directory
- redaction scan passes
- rollback manifest is written
- Phase 1 remains not run
- `DatasetConfirmation` remains unchanged

## Fail Criteria For A Future Materialization Run

A future materialization run must fail if:

- package validation failed
- admission decision is `needs_review` or `ineligible`
- any source artifact hash mismatches
- confirmation is missing or not bound to source hashes
- any admitted record lacks accepted review binding
- any rejected record is materialized
- any `needs_review` record is materialized
- provenance summary is missing
- normalized value summary is missing
- private paths or token-like values appear in output summary
- output directory is non-empty unless explicitly allowed by future design
- Phase 1 runs
- `DatasetConfirmation` changes unexpectedly

## Evidence Requirements

A future materialization evidence PR should commit only a redacted Markdown
summary.

Evidence should include:

- materialization run id
- source artifact SHA-256 values
- package validation summary SHA-256
- materialized candidate count
- excluded count
- needs_review count
- redaction scan result
- rollback manifest SHA-256
- Phase 1 status: `not_run`
- `DatasetConfirmation` changed: `false`
- statement that full artifacts are retained outside git

Evidence must not commit:

- raw PDFs
- raw extracted text
- ParsedDocument outputs
- MinerU bundles
- full local materialization outputs
- private paths
- tokens
- private emails
- signed URLs

## Future Implementation Plan

Recommended future sequence:

1. `docs/test: add custom corpus property candidate schema`
2. `test/docs: add offline custom corpus property candidate planner`
3. `docs/test: add custom corpus materialization schema`
4. `test: add offline materialization planner`
5. `test/docs: add property admission draft package precheck`
6. `test/docs: add property-aware admission package binding runner`
7. `test/docs: add property materialization plan draft builder`
8. `test/docs: add property materialization plan preflight`
9. `test/docs: add property-aware offline materialization planner runner`
10. `test/docs: add property materialization dry-run runner`
11. `test/docs: add property materializer execution request builder`
12. `test/docs: add property quarantine materializer runner`
13. `test/docs: add property quarantine candidate preflight`
14. `test/docs: add property training admission readiness planner`
15. `test/docs: add property training admission request planner`
16. `test/docs: add property training admission request draft package precheck`
17. `test/docs: add property training admission execution request builder`
18. `test/docs: add property training admission execution request preflight`
19. `test/docs: add property training admission execution dry-run`
20. `test/docs: add property training admission execution dry-run precheck`
21. `test/docs: add property training admission execution ledger`
22. `test/docs: add property training admission execution ledger precheck`
23. `test/docs: add property training dataset materialization planner`
24. `test/docs: add property training dataset materialization plan precheck`
25. `test/docs: add property training dataset row contract`
26. `test/docs: add property training dataset row contract precheck`
27. `test/docs: add property training dataset materialization dry-run`
28. `test/docs: add property training dataset materialization dry-run precheck`
29. `test/docs: add property training dataset writer execution request`
30. `test/docs: add property training dataset writer execution request preflight`
31. `test/docs: add property training dataset writer input binding planner`
32. `test/docs: add property training dataset writer input binding plan preflight`
33. `test/docs: add property training dataset writer value source manifest planner`
34. `test/docs: add property training dataset writer value source manifest preflight`
35. `test/docs: add property training dataset controlled writer execution plan`
36. `test/docs: add property training dataset controlled writer execution plan preflight`
37. `test/docs: add property training dataset controlled writer value resolution dry-run`
38. `test/docs: add property training dataset controlled writer value resolution dry-run precheck`
39. `docs: record small public quarantine materialization evidence`
40. `docs/test: design training admission boundary from quarantined candidates`
41. `docs/test: add property training dataset domain validation boundary`
42. `test/docs: add property training dataset controlled writer design plan`
43. `test/docs: add property training dataset controlled writer design plan preflight`
44. `docs/test: add property training dataset controlled writer dry-run design`
45. only later: implement explicit training artifact builder if all previous
   gates pass

Direct implementation of training materialization should not happen in the
next PR.

## Non-Goals

- no unrestricted or training materializer
- no candidate CSV
- no training CSV
- no Phase 1 execution
- no `DatasetConfirmation` change
- no automatic training admission
- no bypass of review/admission/package validation
- no scientific correctness certification
- no private corpus certification
- no MinerU Cloud API provider
- no live CI
- no fallback, retry, queue, rollback scheduler implementation
