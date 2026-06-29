# Custom Corpus Materialization Plan Schema

`custom_corpus_materialization.v1` defines a future materialization plan for
package-validated custom corpus records. It records operator-confirmed intent,
source artifact hashes, candidate-only output intent, record selections,
rollback policy, and redaction policy.

This schema does not implement materialization. A valid materialization plan
does not create candidate artifacts, admit training data, run Phase 1, or set
`DatasetConfirmation.confirmed=true`. A valid materialization plan is necessary
but not sufficient for a future materializer.

## Relationship To Boundary Design

The boundary design is documented in:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

The plan schema is the first pre-materialization artifact after package
validation. It validates intent and binding only.

## Relationship To Property Candidates

Open-ended numeric scientific property candidates are documented in:

```text
docs/custom-corpus-property-candidate-schema.md
```

The upstream review-planning layer is documented in:

```text
docs/custom-corpus-property-candidate-planner.md
```

Materialization plans consume records already selected through review and
admission. The materialization plan schema does not discover scientific
properties, define a property whitelist, normalize numeric property evidence,
or decide whether a property is trainable. The property candidate schema is the
earlier pre-review layer for representing numeric property candidates, and the
property candidate planner output is review-planning evidence rather than
materialization input.

The property candidate review queue builder is also upstream of human review.
Materialization plans must not consume review queue artifacts directly;
materialization requires review, admission, and package validation.

The property review binding validator checks the queue-to-review link upstream
of admission. Materialization plans must consume records that passed human
review and admission; binding evidence alone is not a materialization input.

The property admission readiness planner is also upstream evidence. It can
summarize accepted reviewed records for future admission planning, but
materialization plans must consume records that passed explicit admission.

The property admission request planner is upstream evidence, not a
materialization input. Materialization plans must consume records that passed
actual admission; request-plan summaries do not replace
`custom_corpus_admission.v1`.

An admission draft alone is not sufficient materialization input.
Materialization plans must consume package-validated admission artifacts.
Property admission draft package precheck is upstream evidence only; it does
not replace formal package validation and is not a materialization input.
Property-aware package binding output is also upstream evidence: it may link a
property precheck to formal package validation, but it is not materialization
execution.

Property package-validated admission records can be mapped to reviewable
materialization plan drafts by:

```text
docs/custom-corpus-property-materialization-plan-draft.md
```

Draft artifacts must validate against this existing materialization schema.
They are not execution artifacts and do not create candidate records, CSVs, or
Phase 1 inputs.

Materialization plan drafts can be preflighted before offline planner
submission:

```text
docs/custom-corpus-property-materialization-plan-preflight.md
```

Preflight validates schema, status, hashes, ids, and record consistency. It
does not execute the plan or create materialized artifacts.

After preflight, the property-aware offline materialization planner runner can
invoke the existing offline planner with property package/preflight evidence:

```text
docs/custom-corpus-property-materialization-planner-runner.md
```

The runner writes planner output and a property wrapper summary only. It does
not run a materializer, execute materialization, create candidate/training
CSVs, admit training data, run Phase 1, or change `DatasetConfirmation`.

The property materialization dry-run consumes the materialization plan and
planner output as separate inputs:

```text
docs/custom-corpus-property-materialization-dry-run.md
```

The dry-run report is separate from this materialization plan schema. It
validates hashes, statuses, ids, and record selections without writing
candidate/training data or executing materialization.

The property materializer execution request builder can consume a passed
dry-run report and this materialization plan as separate inputs:

```text
docs/custom-corpus-property-materializer-execution-request.md
```

Execution request artifacts carry safe ids and hashes only. They do not
execute the plan, run a materializer, create materialized records, create
candidate/training artifacts, admit training data, run Phase 1, or change
`DatasetConfirmation`.

Materializer execution request preflight is downstream of materialization
plans, dry-run evidence, and execution requests:

```text
docs/custom-corpus-property-materializer-execution-preflight.md
```

The preflight summary is separate from this materialization plan schema. It
contains no materialized data, no candidate/training artifacts, and no raw
property values or provenance text.

The property quarantine materializer is downstream of execution preflight:

```text
docs/custom-corpus-property-quarantine-materializer.md
```

It can write candidate-only quarantine records from a preflight-checked
execution request. Those records are not training data, do not create
training CSV/JSONL/Parquet/LMDB artifacts, do not run Phase 1, and do not
change `DatasetConfirmation`.

The property quarantine candidate preflight is downstream of candidate-only
quarantine artifacts:

```text
docs/custom-corpus-property-quarantine-candidate-preflight.md
```

The preflight summary is separate from any future training dataset schema. It
checks schema/status/hash/id/record consistency and contains no training data.

Training admission readiness is downstream of quarantine candidate preflight:

```text
docs/custom-corpus-property-training-admission-readiness.md
```

The readiness summary is separate from any training dataset schema and contains
no training data. It reports safe candidate ids and hash bindings only; it does
not create training or candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1,
admit training data, or change `DatasetConfirmation`.

Training admission execution dry-run precheck is downstream of the execution
dry-run:

```text
docs/custom-corpus-property-training-admission-execution-dry-run-precheck.md
```

The precheck validates an existing dry-run report against the upstream request,
preflight, draft, plan, readiness, and quarantine evidence. It does not execute
training admission, create training data, create training artifacts, run Phase
1, or change `DatasetConfirmation`.

Training admission execution ledger is downstream of dry-run precheck:

```text
docs/custom-corpus-property-training-admission-execution-ledger.md
```

The ledger schema is separate from any future training dataset schema. It
records safe ID/hash-only ledger admissions and contains no serialized training
dataset rows, no training CSV/JSONL/Parquet/LMDB paths, and no candidate
CSV/JSONL/Parquet/LMDB paths.

Training admission request planning is downstream of training admission
readiness:

```text
docs/custom-corpus-property-training-admission-request-planner.md
```

The request plan summary is separate from any training admission request schema
and is not materialization input. It reports safe candidate ids and hash
bindings only; it does not create a training admission request, create training
admission actions, admit training data, create training or candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, or change
`DatasetConfirmation`.

Training admission request preflight is downstream of request planning:

```text
docs/custom-corpus-property-training-admission-request-preflight.md
```

The preflight summary validates request-plan/readiness/quarantine consistency
before any future execution request. It is not materialization input and does
not execute training admission, admit training data, create datasets, run
Phase 1, or change `DatasetConfirmation`.

Training admission request drafts are downstream of request preflight:

```text
docs/custom-corpus-property-training-admission-request-draft.md
```

The request draft schema is separate from any training dataset schema. Drafts
contain safe ids and hashes only, contain no training data, and do not execute
materialization, create datasets, run Phase 1, or change
`DatasetConfirmation`.

Training admission request draft precheck is downstream of request drafts:

```text
docs/custom-corpus-property-training-admission-request-draft-precheck.md
```

The precheck summary is separate from any training dataset schema and contains
no training data. It validates draft package schema, status, hashes, ids, and
record consistency only. It does not execute training admission, create
training or candidate CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, or change
`DatasetConfirmation`.

Training admission execution requests are downstream of draft package
precheck:

```text
docs/custom-corpus-property-training-admission-execution-request.md
```

The execution request artifact is separate from any training dataset schema.
It carries safe IDs and hashes only, contains no training data, and does not
execute training admission, create training or candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, or change
`DatasetConfirmation`.

Training admission execution request preflight is downstream of execution
request generation:

```text
docs/custom-corpus-property-training-admission-execution-request-preflight.md
```

The preflight summary is separate from any training dataset schema and contains
no training data. It validates execution-request package schema, status,
hashes, ids, and record consistency only.

Training admission execution dry-run is downstream of execution request
preflight:

```text
docs/custom-corpus-property-training-admission-execution-dry-run.md
```

The dry-run report is separate from any training dataset schema and contains no
training data. It simulates future execution as labels only and does not create
training artifacts.

## Plan Contents

A materialization plan includes:

- `materialization_mode`: currently only `candidate_only`
- `materialization_decision`: `planned` or `blocked`
- source manifest, dry-run report, review manifest, admission request, and
  package validation SHA-256 values
- package validation status and admission decision
- dry-run boundary fields proving Phase 1 did not run
- explicit operator materialization confirmation
- record-level selections with either `materialize_candidate` or `exclude`
- rollback and redaction policy labels

No training mode exists.

## Confirmation Model

`CustomCorpusMaterializationConfirmation` binds the operator confirmation to
the exact source artifacts:

- manifest SHA-256
- dry-run report SHA-256
- review manifest SHA-256
- admission request SHA-256
- package validation SHA-256
- corpus id
- dry-run id
- review manifest id
- admission request id

For a `planned` materialization, confirmation must be explicit and
`confirmed_by` must be present and redacted if needed. This confirmation does
not modify `DatasetConfirmation`.

## Record Rules

`materialize_candidate` records require:

- `admission_action=admit`
- `review_decision=accept`
- normalized value summary
- provenance summary
- materialization reason

`exclude` records require an exclusion reason. Records rejected by review,
marked `needs_review`, or excluded at admission cannot use
`materialize_candidate`.

## Package And Dry-Run Boundaries

For `materialization_decision=planned`:

- `package_validation_status` must be `passed`
- `package_admission_decision` must be `eligible`
- `dry_run_phase1_status` must be `not_run`
- `dry_run_dataset_confirmation_confirmed` must be `false`
- `dry_run_training_dataset_admitted` must be `false`

If those conditions are not met, the plan must be `blocked`.

## CLI Usage

Validate a plan offline:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_materialization \
  --materialization-plan docs/examples/custom-corpus-materialization-plan.example.json \
  --output-summary /tmp/custom-corpus-materialization-plan-summary.json
```

The CLI reads local JSON only, prints a safe JSON summary, and optionally
writes that summary. It does not call MinerU, parse PDFs, run corpus workflow,
create candidate/training artifacts, or modify `DatasetConfirmation`.

## After Plan Validation: Offline Planner

After a plan validates, the offline planner can produce a safe execution
summary for a future materializer:

```text
docs/custom-corpus-materialization-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-planner-evidence-template.md
```

Property-aware runner evidence template:

```text
docs/evidence/templates/custom-corpus-property-materialization-planner-evidence-template.md
```

Property materialization dry-run evidence template:

```text
docs/evidence/templates/custom-corpus-property-materialization-dry-run-evidence-template.md
```

Plan validation checks structure and intent. The planner summarizes intended
future outputs and rollback labels, but still does not create candidate
artifacts, candidate/training CSVs, Phase 1 inputs, or materialized records.

## Summary Output

The summary records safe ids, status fields, counts, source hashes, rollback
policy, and redaction policy. It does not include raw PDF paths, article text,
local absolute paths, private home paths, ParsedDocument text, MinerU bundle
paths, tokens, Authorization headers, cookies, or private emails.

## Examples

Safe example plan:

```text
docs/examples/custom-corpus-materialization-plan.example.json
```

Future plan evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-plan-evidence-template.md
```

## Boundaries

- This schema does not implement materialization.
- A valid materialization plan does not create candidate artifacts.
- A valid materialization plan does not admit training data.
- A valid materialization plan does not run Phase 1.
- A valid materialization plan does not set `DatasetConfirmation.confirmed=true`.
- A valid materialization plan is necessary but not sufficient for a future
  materializer.
