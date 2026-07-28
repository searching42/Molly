# Custom Corpus Property Materialization Plan Draft Builder

The property materialization plan draft builder creates a reviewable
`custom_corpus_materialization.v1` draft from formally package-validated
property admission artifacts. It reads the formal package validation summary
and the property-aware package binding wrapper summary, checks local hash/id
consistency, maps admitted records into materialization-plan draft records, and
writes safe summary and evidence artifacts.

The builder creates a reviewable materialization plan draft only. It does not
run materialization, run the offline materialization planner, run any
materializer, create dataset candidate/training CSVs, admit training data, run
Phase 1, or modify `DatasetConfirmation`. A generated materialization plan
draft is necessary but not sufficient for materialization.

## Relationship To Property-Aware Package Binding

The upstream package binding runner is documented in:

```text
docs/custom-corpus-property-package-binding.md
```

Property-aware package binding runs the formal admission package validator and
links the formal `custom_corpus_admission_package_validation.v1` summary to
property-specific precheck evidence. This draft builder is downstream of that
step. It requires the property package binding summary to be `passed` by
default and requires the formal package validation status to be `passed`.

## Relationship To Materialization Schema

The materialization plan schema is documented in:

```text
docs/custom-corpus-materialization-schema.md
```

The generated draft payload uses the existing `custom_corpus_materialization.v1`
schema and is validated through the existing materialization plan validator.
The draft nature is expressed by the output filename, the builder summary, and
the evidence file. The schema itself remains unchanged.

## Inputs

The builder reads:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_package_binding.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, or review queue artifacts.

## Explicit Confirmation Requirement

The CLI requires:

```text
--confirm-materialization-plan-draft-output
```

Without this flag, the builder returns a blocked summary and writes no draft
plan. This confirmation only allows writing the draft artifact. It does not
materialize records and does not set `DatasetConfirmation.confirmed=true`.

## Draft Mapping Rules

Admission request records with `action=admit` and `review_decision=accept`
become `materialize_candidate` records in the draft plan.

Admission request records with `action=exclude` are reported in the builder
summary only. They do not become materialization records.

Blocked record ids from the property package binding wrapper are reported in
the builder summary only. They do not become materialization records.

Records with `needs_review` do not become materialization records.

Draft records carry safe identifiers, source artifact SHA-256 values, review
artifact SHA-256 values, admission request SHA-256 values, formal package
validation SHA-256 values, normalized value summaries, and provenance
summaries. They do not carry raw table rows, raw article text, PDF names, or
local paths.

## Output Artifacts

The builder creates a clean run directory:

```text
<output-dir>/<materialization-plan-id>/
  custom_corpus_materialization.draft.json
  property_materialization_plan_draft_summary.json
  redacted_property_materialization_plan_draft_evidence.md
```

The run directory must be absent or empty. This PR does not implement
overwrite.

## After Draft Generation: Materialization Plan Preflight

The next boundary is documented in:

```text
docs/custom-corpus-property-materialization-plan-preflight.md
```

Future preflight evidence should use:

```text
docs/evidence/templates/custom-corpus-property-materialization-plan-preflight-evidence-template.md
```

The draft builder creates a reviewable materialization plan draft. The
preflight checks whether that draft is ready for offline materialization
planner submission. The offline materialization planner remains a separate
explicit step, and the future materializer remains separate from both.

## Validation Behavior

The builder checks:

- package binding summary schema and status
- formal package validation schema and status
- manifest, dry-run, review manifest, admission request, package validation,
  and wrapper summary SHA-256 bindings
- corpus id, dry-run id, review manifest id, and admission request id
- dry-run decision, Phase 1 status, training-admitted flag, and
  `DatasetConfirmation` boundary
- admission record counts and admit/exclude counts
- generated draft validation against `custom_corpus_materialization.v1`
- redaction before any draft artifact is written

`binding_status=needs_review` blocks by default. It can be allowed only with:

```text
--allow-package-binding-needs-review
```

Even with that allowance, the formal package validation summary must still
report `passed`.

## Summary Schema

The builder summary uses:

```text
custom_corpus_property_materialization_plan_draft_builder.v1
```

It includes safe basenames, SHA-256 values, corpus/dry-run/review/admission
ids, package binding status, formal package validation status, dry-run
boundary fields, admission counts, materialization draft record counts, draft
artifact labels, warnings, errors, and redaction status.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_plan_draft \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-request /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --formal-package-validation /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/property_package_binding_summary.json \
  --output-dir /tmp/custom-corpus-property-materialization-plan-draft \
  --materialization-plan-id property-materialization-plan-draft-example-001 \
  --dataset-target example-candidate-target \
  --created-by operator-redacted \
  --confirm-materialization-plan-draft-output
```

## Redaction Behavior

Before writing any artifact, the builder serializes the planned draft, wrapper
summary, and evidence and scans for private path and credential markers. If
unsafe material is detected, it fails closed with:

```text
property_materialization_plan_draft_redaction_failed
```

No materialization draft JSON is written on redaction failure.

## Boundaries

- The builder creates a reviewable materialization plan draft only.
- The builder does not run materialization.
- The builder does not run the offline materialization planner.
- The builder does not run any materializer.
- The builder does not create dataset candidate/training CSVs.
- The builder does not admit training data.
- The builder does not run Phase 1.
- The builder does not modify `DatasetConfirmation`.
- A generated materialization plan draft is necessary but not sufficient for
  materialization.
