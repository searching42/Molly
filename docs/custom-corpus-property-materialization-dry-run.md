# Custom Corpus Property Materialization Dry-Run

The property materialization dry-run runner validates an existing offline
materialization planner output and its property-aware wrapper summary without
running a real materializer. It produces a safe no-data dry-run report and
redacted evidence describing what would be materialized later.

The dry-run does not run a real materializer, execute materialization, create
materialized records, create dataset candidate/training CSVs, create
candidate/training JSONL/Parquet/LMDB artifacts, admit training data, run
Phase 1, or modify `DatasetConfirmation`. A passed dry-run is necessary but
not sufficient for real materializer execution.

## Relationship To Planner Runner

The upstream property-aware offline materialization planner runner is
documented in:

```text
docs/custom-corpus-property-materialization-planner-runner.md
```

The planner runner executes the offline planner and writes:

```text
offline_materialization_planner_output.json
property_materialization_planner_summary.json
```

This dry-run consumes those artifacts and validates that the planner output,
materialization plan, preflight summary, package binding evidence, and source
artifacts still agree.

## Relationship To Future Materializer

The dry-run is a no-data readiness step. It can tell an operator which records
would be considered by a future materializer, but it does not create the
future materializer, write materialized records, write candidate artifacts, or
perform rollback/deletion work.

## After Dry-Run: Execution Request Builder

After a dry-run passes, an operator may create a request-only materializer
execution packet:

```text
docs/custom-corpus-property-materializer-execution-request.md
```

Evidence template:

```text
docs/evidence/templates/custom-corpus-property-materializer-execution-request-evidence-template.md
```

The execution request builder consumes the dry-run report and upstream
evidence, then writes safe request artifacts for a future materializer. It
still does not run a materializer, execute materialization, create
candidate/training CSVs, admit training data, run Phase 1, or modify
`DatasetConfirmation`.

## Inputs

The dry-run requires:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_package_binding.v1`
- `custom_corpus_materialization.v1`
- `custom_corpus_property_materialization_plan_preflight.v1`
- `custom_corpus_materialization_planner.v1`
- `custom_corpus_property_materialization_planner_runner.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, review queues, corpus workflow outputs, or
materialized dataset artifacts.

## Explicit Confirmation

The CLI requires:

```text
--confirm-materialization-dry-run
```

Without this flag, the dry-run returns `failed` and writes no report.

## Dry-Run Validation Rules

The dry-run checks:

- property planner summary schema is
  `custom_corpus_property_materialization_planner_runner.v1`
- offline planner output schema is `custom_corpus_materialization_planner.v1`
- planner status is `planned`, or `needs_review` is explicitly allowed
- offline planner status is `planned`
- preflight status is `passed`, or `needs_review` is explicitly allowed
- package binding did not fail
- formal package validation status is `passed`
- materialization decision is `planned`
- source dry-run decision is `passed`
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains false
- all source SHA-256 values match the local files
- corpus, dry-run, review manifest, admission request, and materialization
  plan ids match
- materialization record ids and counts match the plan, preflight summary,
  property planner summary, and offline planner output
- materialization records derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not materialization records
- dry-run output contains safe ids, hashes, counts, and status fields only

The dry-run fails if the offline planner output claims materialized records,
candidate/training artifact paths, Phase 1 execution, training admission, or a
`DatasetConfirmation` mutation.

## Output Artifacts

The run-specific output directory is:

```text
<output-dir>/<dry-run-id>/
  property_materialization_dry_run_report.json
  redacted_property_materialization_dry_run_evidence.md
```

The dry-run must not create candidate records, materialized records, CSV,
JSONL, Parquet, LMDB, training artifacts, rollback manifests for actual data,
or provenance binding files for actual data.

## Dry-Run Report Schema

The JSON report uses:

```text
custom_corpus_property_materialization_dry_run.v1
```

It includes safe basenames, SHA-256 values, corpus/source dry-run/review/
admission/materialization plan ids, planner status, offline planner status,
preflight status, package binding status, formal package validation status,
materialization decision, dry-run boundary fields, record counts, safe record
ids, ID-only per-record dry-run summaries, warnings, dry-run errors, and
redaction status.

`dry_run_record_summaries` include only safe record ids and hashes. They do
not include raw values, normalized value summaries, provenance summaries, raw
table rows, article text, PDF names/paths, local paths, or private
identifiers.

## Status Meanings

`passed` means all hard consistency checks passed and no allowed
needs-review evidence remains.

`needs_review` means no hard check failed, but upstream planner/preflight
evidence carried an explicitly allowed `needs_review` status.

`failed` means confirmation was missing, a schema/hash/id/status/record check
failed, the offline planner output claimed generated artifacts or execution,
the output directory was not clean, or redaction failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_dry_run \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-request /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --formal-package-validation /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /tmp/custom-corpus-property-package-binding/property-package-binding-example-001/property_package_binding_summary.json \
  --materialization-plan /tmp/custom-corpus-property-materialization-plan-draft/property-materialization-plan-draft-example-001/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /tmp/custom-corpus-property-materialization-plan-preflight-summary.json \
  --offline-planner-output /tmp/custom-corpus-property-materialization-planner/property-materialization-planner-example-001/offline_materialization_planner_output.json \
  --property-planner-summary /tmp/custom-corpus-property-materialization-planner/property-materialization-planner-example-001/property_materialization_planner_summary.json \
  --output-dir /tmp/custom-corpus-property-materialization-dry-run \
  --dry-run-id property-materialization-dry-run-example-001 \
  --confirm-materialization-dry-run
```

## Redaction Behavior

Before writing a report or Markdown evidence, the dry-run scans serialized
output for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_materialization_dry_run_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The dry-run does not run a real materializer.
- The dry-run does not execute materialization.
- The dry-run does not create materialized records.
- The dry-run does not create dataset candidate/training CSVs.
- The dry-run does not create candidate/training JSONL/Parquet/LMDB
  artifacts.
- The dry-run does not admit training data.
- The dry-run does not run Phase 1.
- The dry-run does not modify `DatasetConfirmation`.
- A passed dry-run is necessary but not sufficient for real materializer
  execution.
