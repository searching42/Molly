# Custom Corpus Property Materialization Plan Preflight

The property materialization plan preflight checks a reviewable
`custom_corpus_materialization.v1` draft before it is submitted to the offline
materialization planner. It reads the draft plan, the property materialization
draft summary, and upstream package/admission evidence, then emits safe JSON
and optional Markdown evidence.

The preflight does not run the offline materialization planner. It does not
run any materializer, execute materialization, create dataset candidate/training
CSVs, admit training data, run Phase 1, or modify `DatasetConfirmation`. A
passed preflight is necessary but not sufficient for materialization.

## Relationship To Draft Builder

The upstream draft builder is documented in:

```text
docs/custom-corpus-property-materialization-plan-draft.md
```

The draft builder creates a reviewable `custom_corpus_materialization.v1`
draft. This preflight checks whether that draft, its builder summary, and the
upstream package/admission evidence are internally consistent enough to submit
to the offline materialization planner.

## Relationship To Offline Materialization Planner

The offline materialization planner is documented in:

```text
docs/custom-corpus-materialization-planner.md
```

Preflight output can be used as evidence before invoking the planner. It does
not substitute for the planner, and it never invokes the planner itself.

## Inputs

The preflight reads:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_package_binding.v1`
- draft `custom_corpus_materialization.v1`
- `custom_corpus_property_materialization_plan_draft_builder.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, review queues, or corpus workflow outputs.

## Preflight Rules

The preflight checks:

- every input schema is expected
- source artifact SHA-256 values match the actual local files
- corpus id, dry-run id, review manifest id, admission request id, and
  materialization plan id are consistent
- dry-run decision is `passed`
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains false
- training dataset admitted remains false
- formal package validation status is `passed`
- property package binding status is `passed` or explicitly allowed
  `needs_review`
- materialization draft status is `written`
- materialization decision is `planned`
- package admission decision is `eligible`
- materialization records only derive from admitted, accepted records
- excluded, blocked, and needs-review records are not materialization records
- materialization record counts and ids match the draft summary
- emitted summaries remain redacted

`package_binding_status=needs_review` returns preflight status `needs_review`
by default. With `--require-package-binding-passed`, it fails.

## Status Meanings

`passed` means all hard checks passed and no allowed partial evidence remains.

`needs_review` means no hard check failed, but property package binding or
draft evidence still carries an allowed review warning.

`failed` means schema validation, source hash binding, id consistency, dry-run
boundary, record mapping, or redaction failed.

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_materialization_plan_preflight.v1
```

It includes safe basenames, SHA-256 values, corpus/dry-run/review/admission
ids, materialization plan id, package binding status, formal package validation
status, materialization draft status, materialization decision, dry-run
boundary fields, record counts, safe record ids, warnings, preflight errors,
and redaction status.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_plan_preflight \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-request /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --formal-package-validation /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/property_package_binding_summary.json \
  --materialization-plan-draft /tmp/custom-corpus-property-materialization-plan-draft/property-materialization-plan-draft-example-001/custom_corpus_materialization.draft.json \
  --materialization-plan-draft-summary /tmp/custom-corpus-property-materialization-plan-draft/property-materialization-plan-draft-example-001/property_materialization_plan_draft_summary.json \
  --output-summary /tmp/custom-corpus-property-materialization-plan-preflight-summary.json \
  --output-markdown /tmp/custom-corpus-property-materialization-plan-preflight-summary.md
```

## Markdown Output

When `--output-markdown` is supplied, the preflight writes a concise redacted
Markdown report with status, ids, counts, record ids, errors, warnings, and a
boundary statement.

## Redaction Behavior

Before printing or writing summaries, the preflight serializes output and
scans for private path and credential markers. If unsafe material is detected,
it fails closed with:

```text
property_materialization_plan_preflight_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The preflight does not run the offline materialization planner.
- The preflight does not run any materializer.
- The preflight does not execute materialization.
- The preflight does not create dataset candidate/training CSVs.
- The preflight does not admit training data.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- A passed preflight is necessary but not sufficient for materialization.
