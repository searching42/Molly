# Custom Corpus Property Training Admission Request Planner

The property training admission request planner checks existing
training-admission-readiness evidence and candidate-only quarantine artifacts.
It emits a safe JSON request plan summary and optional redacted Markdown
evidence describing how a future training admission request could be organized.

Training admission request planning is not training admission. A planned
training admission action is not an admission action, and a planned candidate
is not training data. The planner does not create a training admission request,
does not admit data, and does not create training artifacts. A request plan is
necessary but not sufficient for future training admission.

## Relationship To Training Admission Readiness

The upstream readiness planner is documented in:

```text
docs/custom-corpus-property-training-admission-readiness.md
```

Readiness checks whether candidate-only quarantine records are ready to be
considered by a future training admission request. This request planner
consumes that readiness summary as evidence. It does not rerun readiness,
does not run quarantine candidate preflight, does not run the quarantine
materializer, and does not modify candidate records.

## Inputs

The planner requires:

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
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, candidate/training CSV/JSONL/Parquet/LMDB files, or training outputs.

## Planning Rules

The planner checks:

- training admission readiness schema is
  `custom_corpus_property_training_admission_readiness.v1`
- readiness status is `ready`, or `partial` only when strict ready mode is
  not requested
- readiness errors are empty
- candidate record count is positive and meets the configured minimum
- planned training admission candidate ids are present and match quarantined
  candidate ids
- candidate, materialization, and execution record counts and ids match
- source hashes match local input files and upstream summaries
- corpus, dry-run, review manifest, admission request, materialization plan,
  execution request, quarantine run, review queue, and property candidate
  manifest ids match where present
- planned candidates derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not planned as future
  training admission candidates
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_training_admission_request_plan.v1
```

It includes safe basenames, SHA-256 values, artifact ids, upstream statuses,
record counts, candidate/materialization/execution/admit/exclude/blocked ids,
safe ID/hash-only planned request record summaries, planning errors, warnings,
and redaction status.

The summary does not include raw candidate payloads, raw table rows, article
text, PDF names or paths, local paths, ParsedDocument text, MinerU bundle
paths, token/auth/cookie values, private emails, CSV/JSONL/Parquet/LMDB paths,
or training outputs.

## Status Meanings

`planned` means all hard checks passed and readiness is `ready`.

`partial` means no hard consistency check failed, but readiness is `partial`
and strict ready mode was not requested.

`blocked` means a schema, status, hash, id, record binding, boundary,
minimum-count, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_planner \
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
  --quarantine-candidate-preflight-summary /tmp/custom-corpus-property-quarantine-candidate-preflight-summary.json \
  --training-admission-readiness-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --output-summary /tmp/custom-corpus-property-training-admission-request-plan-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-request-plan-summary.md
```

Return codes:

- `0` when planner status is `planned` or `partial`
- `1` when planner status is `blocked`

## Redaction Behavior

Before printing or writing output, the planner scans serialized summary and
Markdown for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_request_plan_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The planner performs training admission request planning only.
- The planner does not create a training admission request.
- The planner does not create training admission actions.
- The planner does not admit training data.
- The planner does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not create candidate CSV/JSONL/Parquet/LMDB artifacts.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner does not run model training or evaluation.
- The planner does not implement Agentic RL.
- The planner does not call an LLM or agent.
- The planner does not call MinerU.
- The planner does not parse PDFs.
- The planner does not read ParsedDocument content.
- A request plan is necessary but not sufficient for future training
  admission.
