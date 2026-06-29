# Custom Corpus Property Quarantine Candidate Preflight

The property quarantine candidate preflight checks existing candidate-only
quarantine artifacts before any future training admission request. It reads
the quarantine candidate artifact, the quarantine materializer summary, the
execution preflight summary, and upstream property governance evidence, then
emits safe JSON and optional Markdown evidence.

This preflight is not training admission. It does not create training
artifacts, run Phase 1, modify `DatasetConfirmation`, run model training, or
run evaluation.

## Relationship To Quarantine Materializer

The property quarantine materializer is documented in:

```text
docs/custom-corpus-property-quarantine-materializer.md
```

The materializer writes candidate-only quarantine records. This preflight
checks those existing records and their upstream bindings. It does not run the
quarantine materializer and does not modify the candidate artifact.

## Relationship To Future Training Admission

A passed quarantine candidate preflight is necessary but not sufficient for
future training admission. A future training admission layer must still define
explicit operator confirmation, trainability criteria, training artifact
formats, rollback/deletion behavior, and additional review evidence before any
quarantined candidate may become training data.

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
- `custom_corpus_property_materializer_execution_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`
- `custom_corpus_property_quarantine_materializer.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, candidate/training CSV/JSONL/Parquet/LMDB files, or training outputs.

## Preflight Rules

The preflight checks:

- quarantine candidate schema is
  `custom_corpus_property_quarantine_materialization.v1`
- quarantine summary schema is
  `custom_corpus_property_quarantine_materializer.v1`
- quarantine materializer status is `written`, or `needs_review` only when
  strict no-needs-review mode is not requested
- candidate materialization mode is `candidate_quarantine`
- candidate records are present and count/id fields match
- candidate records have `candidate_status=quarantined`
- candidate records carry `candidate_only`, `not_training`, `not_phase1`, and
  `dataset_confirmation_unchanged` boundary labels
- execution preflight is `passed`, or `needs_review` only when allowed
- execution request remains `written` and `request_only`
- materializer status before quarantine remains `not_run`
- materialization decision remains `planned`
- formal package validation, planner, and dry-run statuses remain successful
- all source SHA-256 values match local input files
- corpus, dry-run, review manifest, admission request, materialization plan,
  execution request, and quarantine run ids match
- candidate records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not quarantined candidates
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_quarantine_candidate_preflight.v1
```

It includes safe basenames, SHA-256 values, artifact ids, upstream statuses,
record counts, candidate/materialization/execution/admit/exclude/blocked ids,
preflight errors, warnings, and redaction status.

The summary does not include raw candidate payloads, raw table rows, article
text, PDF names or paths, local paths, ParsedDocument text, MinerU bundle
paths, token/auth/cookie values, private emails, or CSV/JSONL/Parquet/LMDB
paths.

## Status Meanings

`passed` means all hard checks passed and no needs-review evidence remains.

`needs_review` means no hard consistency check failed, but quarantine or
upstream evidence carries an allowed needs-review status.

`failed` means a schema, status, hash, id, record binding, boundary, or
redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_quarantine_candidate_preflight \
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
  --execution-preflight-summary /tmp/custom-corpus-property-materializer-execution-preflight-summary.json \
  --quarantine-candidate-records /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_candidate_records.json \
  --quarantine-materializer-summary /tmp/custom-corpus-property-quarantine-materializer/property-quarantine-materializer-example-001/property_quarantine_materializer_summary.json \
  --output-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --output-markdown /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.md
```

Return codes:

- `0` when status is `passed` or `needs_review`
- `1` when status is `failed`

## Redaction Behavior

Before printing or writing output, the preflight scans serialized summary and
Markdown for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_quarantine_candidate_preflight_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The preflight checks candidate-only quarantine artifacts.
- The preflight does not admit training data.
- The preflight does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The preflight does not run Phase 1.
- The preflight does not modify `DatasetConfirmation`.
- The preflight does not run model training or evaluation.
- The preflight does not call an LLM or agent.
- The preflight does not call MinerU.
- The preflight does not parse PDFs.
- The preflight does not read ParsedDocument content.
- A passed quarantine candidate preflight is necessary but not sufficient for
  future training admission.
