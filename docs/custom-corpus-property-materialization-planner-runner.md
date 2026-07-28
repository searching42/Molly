# Custom Corpus Property Materialization Planner Runner

The property-aware offline materialization planner runner executes the existing
offline `custom_corpus_materialization_planner.v1` planner only after a
property materialization plan preflight has passed, or after an operator
explicitly allows a preflight `needs_review` status.

The runner reads the package-validated property admission evidence, the
reviewable `custom_corpus_materialization.v1` materialization plan draft, the
property materialization plan draft summary, and the property materialization
plan preflight summary. It writes the existing offline planner output plus a
property-aware wrapper summary and redacted evidence.

This runner does not run a materializer, execute materialization, create
dataset candidate/training CSVs, admit training data, run Phase 1, or modify
`DatasetConfirmation`. A successful runner summary is necessary but not
sufficient for materialization.

## Relationship To Preflight

The upstream preflight is documented in:

```text
docs/custom-corpus-property-materialization-plan-preflight.md
```

The preflight checks whether a materialization plan draft appears ready for
offline planner submission. This runner consumes that preflight summary and
requires `preflight_status=passed` by default.

If `preflight_status=needs_review`, the runner blocks unless
`--allow-preflight-needs-review` is supplied. `preflight_status=failed` always
blocks planner execution.

## Relationship To Offline Planner

The underlying planner is documented in:

```text
docs/custom-corpus-materialization-planner.md
```

This runner calls the existing offline planner and writes its output to:

```text
offline_materialization_planner_output.json
```

The property-aware wrapper summary links that planner output back to the
property package binding, materialization plan draft, and preflight evidence.

## After Planner: Property Materialization Dry-Run

The downstream no-data dry-run is documented in:

```text
docs/custom-corpus-property-materialization-dry-run.md
```

Future dry-run evidence should use:

```text
docs/evidence/templates/custom-corpus-property-materialization-dry-run-evidence-template.md
```

The planner runner executes the offline planner. The dry-run consumes that
planner output and validates future materializer-readiness without creating
materialized data. Real materializer execution remains separate.

## Inputs

The runner requires:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_package_binding.v1`
- `custom_corpus_materialization.v1`
- `custom_corpus_property_materialization_plan_draft_builder.v1`
- `custom_corpus_property_materialization_plan_preflight.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, review queues, or corpus workflow outputs.

## Planner Gating

The runner calls the offline planner only when:

- `--confirm-offline-materialization-planner` is present
- output run directory is clean
- preflight status is `passed`, or `needs_review` is explicitly allowed
- preflight errors are empty
- formal package validation status is `passed`
- property package binding is not failed
- materialization draft status is `written`
- materialization decision is `planned`
- dry-run decision is `passed`
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains false
- all input SHA-256 values match the preflight/draft/package evidence
- materialization records are non-empty and derive only from admitted,
  accepted records
- excluded, blocked, and needs-review records are not materialization records

## Output Artifacts

The run-specific output directory is:

```text
<output-dir>/<planner-run-id>/
  offline_materialization_planner_output.json
  property_materialization_planner_summary.json
  redacted_property_materialization_planner_evidence.md
```

The runner does not create materialized records, candidate artifacts,
candidate CSVs, training CSVs, rollback manifests, or provenance binding
files.

## Wrapper Summary Schema

The property-aware wrapper summary uses:

```text
custom_corpus_property_materialization_planner_runner.v1
```

It includes safe basenames, SHA-256 values, corpus/dry-run/review/admission
ids, materialization plan id, preflight status, package binding status, formal
package validation status, materialization draft status, offline planner
status, dry-run boundary fields, record counts, safe record ids, warnings,
planner errors, and redaction status.

## Status Meanings

`planned` means the preflight passed, the offline planner ran, and the wrapper
checks found no materialization or training-boundary violations.

`needs_review` means the preflight carried an explicitly allowed
`needs_review` status and the offline planner still ran without local wrapper
errors.

`failed` means confirmation was missing, preflight/package/draft evidence was
invalid, source hashes or ids mismatched, materialization records were unsafe,
the offline planner returned blocked/failed output, or redaction failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_planner_runner \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-request /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --formal-package-validation /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/property_package_binding_summary.json \
  --materialization-plan /tmp/custom-corpus-property-materialization-plan-draft/property-materialization-plan-draft-example-001/custom_corpus_materialization.draft.json \
  --materialization-plan-draft-summary /tmp/custom-corpus-property-materialization-plan-draft/property-materialization-plan-draft-example-001/property_materialization_plan_draft_summary.json \
  --materialization-plan-preflight-summary /tmp/custom-corpus-property-materialization-plan-preflight-summary.json \
  --output-dir /tmp/custom-corpus-property-materialization-planner \
  --planner-run-id property-materialization-planner-example-001 \
  --confirm-offline-materialization-planner
```

## Redaction Behavior

Before writing wrapper summaries or evidence, the runner scans serialized
output for private path, credential, PDF, CSV, signed URL, and raw-text
markers. If unsafe material is detected, it fails closed with:

```text
property_materialization_planner_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The runner executes the offline materialization planner only.
- The runner does not run any materializer.
- The runner does not execute materialization.
- The runner does not create materialized records.
- The runner does not create dataset candidate/training CSVs.
- The runner does not admit training data.
- The runner does not run Phase 1.
- The runner does not modify `DatasetConfirmation`.
- A successful runner summary is necessary but not sufficient for
  materialization.
