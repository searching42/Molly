# Custom Corpus Property Training Admission Readiness

The property training admission readiness planner checks existing
candidate-only quarantine artifacts after quarantine candidate preflight. It
reads the quarantine candidate artifact, quarantine materializer summary,
quarantine candidate preflight summary, and upstream property governance
evidence, then emits safe JSON and optional Markdown readiness evidence for a
future training admission request.

Training admission readiness is not training admission. The readiness planner
does not create training CSV/JSONL/Parquet/LMDB artifacts, create candidate
CSV/JSONL/Parquet/LMDB artifacts, run Phase 1, modify
`DatasetConfirmation`, run model training, or run evaluation. A `ready` status
is necessary but not sufficient for future training admission.

## Relationship To Quarantine Candidate Preflight

The upstream preflight is documented in:

```text
docs/custom-corpus-property-quarantine-candidate-preflight.md
```

Quarantine candidate preflight checks that candidate-only artifacts remain
internally consistent and safe. The readiness planner consumes that preflight
summary as evidence. It does not run the preflight, does not run the
quarantine materializer, and does not modify candidate records.

## Relationship To Future Training Admission

The readiness planner identifies quarantined candidate record ids that may be
considered by a future explicit training admission request. It does not create
that request, does not create admission actions, and does not admit training
data. A future training admission layer must still define operator
confirmation, trainability policy, output format, rollback/deletion behavior,
and additional evidence requirements.

## After Readiness: Training Admission Request Planning

The request planner is documented in:

```text
docs/custom-corpus-property-training-admission-request-planner.md
```

Future evidence should use:

```text
docs/evidence/templates/custom-corpus-property-training-admission-request-plan-evidence-template.md
```

Readiness evidence identifies candidate-only quarantine records that may be
eligible for future training admission planning. The request planner converts
that evidence into a safe planning summary only. It still does not create a
training admission request, create training admission actions, admit training
data, or create training CSV/JSONL/Parquet/LMDB artifacts.

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

It does not read PDFs, ParsedDocument outputs, MinerU bundles, raw extracted
text, candidate/training CSV/JSONL/Parquet/LMDB files, or training outputs.

## Readiness Rules

The planner checks:

- quarantine candidate preflight schema is
  `custom_corpus_property_quarantine_candidate_preflight.v1`
- quarantine candidate preflight status is `passed`, or `needs_review` only
  when strict passed-preflight mode is not requested
- quarantine candidate artifact schema is
  `custom_corpus_property_quarantine_materialization.v1`
- quarantine materializer status is `written`, or `needs_review` only when
  strict passed-preflight mode is not requested
- candidate materialization mode is `candidate_quarantine`
- candidate records are present and meet the configured minimum count
- candidate, materialization, and execution record counts and ids match
- candidate records carry `candidate_only`, `not_training`, `not_phase1`, and
  `dataset_confirmation_unchanged` boundary labels
- all source SHA-256 values match local input files and upstream summaries
- corpus, dry-run, review manifest, admission request, materialization plan,
  execution request, quarantine run, review queue, and property candidate
  manifest ids match where present
- candidate records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not training-readiness
  candidates
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

## Summary Schema

The JSON summary uses:

```text
custom_corpus_property_training_admission_readiness.v1
```

It includes safe basenames, SHA-256 values, artifact ids, upstream statuses,
record counts, candidate/materialization/execution/admit/exclude/blocked ids,
safe ID/hash-only readiness record summaries, readiness errors, warnings, and
redaction status.

The summary does not include raw candidate payloads, raw table rows, article
text, PDF names or paths, local paths, ParsedDocument text, MinerU bundle
paths, token/auth/cookie values, private emails, CSV/JSONL/Parquet/LMDB paths,
or training outputs.

## Status Meanings

`ready` means all hard checks passed and no needs-review evidence remains.

`partial` means no hard consistency check failed, but quarantine candidate
preflight or upstream quarantine evidence carries an allowed needs-review
status.

`blocked` means a schema, status, hash, id, record binding, boundary,
minimum-count, or redaction check failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_readiness \
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
  --output-summary /tmp/custom-corpus-property-training-admission-readiness-summary.json \
  --output-markdown /tmp/custom-corpus-property-training-admission-readiness-summary.md
```

Return codes:

- `0` when status is `ready` or `partial`
- `1` when status is `blocked`

## Redaction Behavior

Before printing or writing output, the planner scans serialized summary and
Markdown for private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed
URL, and raw-text markers. If unsafe material is detected, it fails closed
with:

```text
property_training_admission_readiness_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The readiness planner checks candidate-only quarantine artifacts.
- The readiness planner does not admit training data.
- The readiness planner does not create training CSV/JSONL/Parquet/LMDB
  artifacts.
- The readiness planner does not create candidate CSV/JSONL/Parquet/LMDB
  artifacts.
- The readiness planner does not create a training admission request.
- The readiness planner does not run Phase 1.
- The readiness planner does not modify `DatasetConfirmation`.
- The readiness planner does not run model training or evaluation.
- The readiness planner does not call an LLM or agent.
- The readiness planner does not call MinerU.
- The readiness planner does not parse PDFs.
- The readiness planner does not read ParsedDocument content.
- A `ready` status is necessary but not sufficient for future training
  admission.
