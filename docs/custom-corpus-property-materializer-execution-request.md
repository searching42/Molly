# Custom Corpus Property Materializer Execution Request

The property materializer execution request builder reads a passed property
materialization dry-run package and writes a reviewable request-only packet for
a future materializer. It does not run a materializer and does not execute
materialization.

The request builder is downstream of the property-aware offline planner and
the property materialization dry-run. It is useful only after the dry-run has
validated that the materialization plan, planner output, package binding
evidence, and upstream admission artifacts still agree.

## Purpose

The builder creates:

```text
property_materializer_execution_request.json
property_materializer_execution_request_summary.json
redacted_property_materializer_execution_request_evidence.md
```

The execution request is a handoff artifact for a future materializer. It
carries only safe ids, SHA-256 values, counts, and request boundary fields. It
does not contain raw values, provenance text, raw table rows, article text, PDF
names or paths, local paths, candidate output paths, training output paths, or
private identifiers.

## Inputs

The builder requires:

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

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, property candidate manifests, review queues, corpus workflow outputs, or
materialized dataset artifacts.

## Explicit Confirmation

The CLI requires:

```text
--confirm-materializer-execution-request-output
```

Without this flag, the builder returns `blocked` and writes no execution
request JSON.

## Request Rules

The builder writes a request only when:

- the property materialization dry-run status is `passed`, or `needs_review`
  is explicitly allowed
- the offline planner status is `planned`
- the preflight status is `passed`, or `needs_review` is explicitly allowed
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
  planner output, property planner summary, and dry-run report
- execution records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not execution request records
- output directory is clean
- redaction checks pass

## Execution Request Schema

The request JSON uses:

```text
custom_corpus_property_materializer_execution_request.v1
```

It includes:

- execution request id, created-at timestamp, and redacted operator label
- corpus/source dry-run/review/admission/materialization ids
- source artifact SHA-256 values
- `execution_mode=request_only`
- `materializer_status=not_run`
- `phase1_status=not_run`
- `training_admitted=false`
- `dataset_confirmation_changed=false`
- execution record ids
- execution records with safe ids and hashes only
- boundary statement

Execution records use `planned_action=request_materialize_candidate`. That
field is a request label only. It is not materialization and is not training
admission.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materializer_execution_request \
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
  --output-dir /tmp/custom-corpus-property-materializer-execution-request \
  --execution-request-id property-materializer-execution-request-example-001 \
  --created-by operator-redacted \
  --confirm-materializer-execution-request-output
```

## Redaction Behavior

Before writing output, the builder scans serialized request, summary, and
Markdown evidence for private path, credential, PDF, CSV, JSONL, Parquet,
LMDB, signed URL, and raw-text markers. If unsafe material is detected, it
fails closed with:

```text
property_materializer_execution_request_redaction_failed
```

No unsafe execution request or Markdown evidence is written.

## Boundaries

- The builder creates a materializer execution request only.
- The builder does not run a real materializer.
- The builder does not execute materialization.
- The builder does not create materialized records.
- The builder does not create dataset candidate/training CSVs.
- The builder does not create candidate/training JSONL/Parquet/LMDB
  artifacts.
- The builder does not admit training data.
- The builder does not run Phase 1.
- The builder does not modify `DatasetConfirmation`.
- A generated execution request is necessary but not sufficient for real
  materializer execution.
