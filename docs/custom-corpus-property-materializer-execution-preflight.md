# Custom Corpus Property Materializer Execution Preflight

The property materializer execution request preflight checks an existing
request-only materializer execution packet before any future materializer
submission. It reads the execution request, the execution request builder
summary, the materialization dry-run report, and upstream property governance
evidence, then emits safe preflight evidence.

The preflight checks a materializer execution request only. It does not run a
real materializer and does not execute materialization.

## Relationship To Execution Request Builder

The execution request builder is documented in:

```text
docs/custom-corpus-property-materializer-execution-request.md
```

The builder creates a reviewable request-only artifact. This preflight checks
whether that existing request is internally consistent, hash-bound to the
same upstream evidence, and still safe for a future materializer handoff.
Real materializer execution remains separate.

## Inputs

The preflight requires:

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
- `custom_corpus_property_materialization_dry_run.v1`
- `custom_corpus_property_materializer_execution_request.v1`
- `custom_corpus_property_materializer_execution_request_builder.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, review queues, corpus workflow outputs, or
materialized dataset artifacts.

## Preflight Rules

The preflight checks:

- execution request schema is
  `custom_corpus_property_materializer_execution_request.v1`
- execution request builder summary schema is
  `custom_corpus_property_materializer_execution_request_builder.v1`
- execution request status is `written`
- execution mode is `request_only`
- materializer status is `not_run`
- Phase 1 status is `not_run`
- training admitted is false
- `DatasetConfirmation` changed is false
- dry-run status is `passed`, or `needs_review` is allowed by non-strict mode
- dry-run errors are empty
- offline planner status is `planned`
- formal package validation status is `passed`
- materialization decision is `planned`
- all source SHA-256 values match local files
- corpus, dry-run, review manifest, admission request, materialization plan,
  and execution request ids match
- materialization record ids and counts match the plan, dry-run report,
  execution request, and execution summary
- execution records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not execution request records
- execution records contain safe ids and hashes only
- summary and Markdown redaction checks pass

## Status Meanings

`passed` means all hard checks passed and no needs-review upstream evidence
remains.

`needs_review` means no hard consistency check failed, but dry-run, planner,
or preflight evidence still carries an allowed needs-review status.

`failed` means a schema, hash, id, status, record binding, boundary, or
redaction check failed.

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_materializer_execution_preflight.v1
```

It includes safe basenames, SHA-256 values, artifact ids, upstream statuses,
materializer boundary fields, record counts, safe record ids, preflight
errors, warnings, and redaction status.

The summary and Markdown evidence must not include raw values, normalized
values, provenance text, raw table rows, article text, PDF names or paths,
local paths, candidate/training artifact paths, materialized record payloads,
tokens, cookies, private emails, ParsedDocument text, or MinerU bundle paths.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materializer_execution_preflight \
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
  --materialization-dry-run-report /tmp/custom-corpus-property-materialization-dry-run/property-materialization-dry-run-example-001/property_materialization_dry_run_report.json \
  --execution-request /tmp/custom-corpus-property-materializer-execution-request/property-materializer-execution-request-example-001/property_materializer_execution_request.json \
  --execution-request-summary /tmp/custom-corpus-property-materializer-execution-request/property-materializer-execution-request-example-001/property_materializer_execution_request_summary.json \
  --output-summary /tmp/custom-corpus-property-materializer-execution-preflight-summary.json \
  --output-markdown /tmp/custom-corpus-property-materializer-execution-preflight-summary.md
```

## Redaction Behavior

Before printing or writing output, the preflight scans serialized summary and
Markdown evidence for private path, credential, PDF, CSV, JSONL, Parquet,
LMDB, signed URL, raw-text, and materialized-output markers. If unsafe
material is detected, it fails closed with:

```text
property_materializer_execution_preflight_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The preflight checks a materializer execution request only.
- The preflight does not run a real materializer.
- The preflight does not execute materialization.
- The preflight does not create materialized records.
- The preflight does not create dataset candidate/training CSVs.
- The preflight does not create candidate/training JSONL/Parquet/LMDB
  artifacts.
- The preflight does not admit training data.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- A passed preflight is necessary but not sufficient for real materializer
  execution.
