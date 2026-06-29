# Custom Corpus Property Quarantine Materializer

The property quarantine materializer writes candidate-only quarantine
artifacts after a passed property materializer execution preflight. It is the
first property-path step that may write materialized candidate records, but the
records remain quarantined and are not training data.

This runner consumes only local JSON governance artifacts. It does not call an
LLM or agent, read PDFs, read ParsedDocument content, call MinerU, run corpus
workflow, run Phase 1, or modify `DatasetConfirmation`.

## Relationship To Execution Preflight

The upstream execution request preflight is documented in:

```text
docs/custom-corpus-property-materializer-execution-preflight.md
```

The preflight checks whether a request-only materializer handoff is safe to
submit. The quarantine materializer consumes that checked request and writes
candidate-only quarantine records. A passed preflight is required by default,
but the materializer still requires explicit operator confirmation.

## Inputs

The runner requires:

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

The runner does not read property candidate manifests, review queue packets,
PDFs, ParsedDocument outputs, MinerU bundles, raw extracted text, or dataset
training artifacts.

## Output Artifacts

The output directory is run-scoped:

```text
<output-dir>/<quarantine-run-id>/
  property_quarantine_candidate_records.json
  property_quarantine_materializer_summary.json
  redacted_property_quarantine_materializer_evidence.md
```

`property_quarantine_candidate_records.json` uses:

```text
custom_corpus_property_quarantine_materialization.v1
```

It contains quarantined candidate records only. Each candidate record is bound
to the execution request, materialization plan record, admission record, review
record, source artifact hash, review artifact hash, admission request hash,
formal package validation hash, materialization plan hash, offline planner
hash, dry-run hash, execution request hash, and execution preflight hash.

`property_quarantine_materializer_summary.json` uses:

```text
custom_corpus_property_quarantine_materializer.v1
```

It records safe basenames, SHA-256 values, status fields, record counts,
candidate/materialization/execution ids, admit/exclude/blocked ids, errors,
warnings, and redaction status.

`redacted_property_quarantine_materializer_evidence.md` is a human-readable
evidence summary for review.

## After Quarantine Materialization: Quarantine Candidate Preflight

The quarantine candidate preflight is documented in:

```text
docs/custom-corpus-property-quarantine-candidate-preflight.md
```

Evidence template:

```text
docs/evidence/templates/custom-corpus-property-quarantine-candidate-preflight-evidence-template.md
```

The quarantine materializer writes candidate-only artifacts. The preflight
checks those artifacts before any future training admission request. Training
admission remains separate, and the preflight produces no training artifact.

## Candidate Record Rules

The runner writes candidate quarantine records only for execution records that
derive from materialization records whose source admission records were
admitted and accepted.

Candidate records must not derive from:

- excluded admission records
- blocked property records
- review or admission records marked `needs_review`
- rejected review records
- execution records that do not bind to materialization records

Generated records use `candidate_status=quarantined` and carry boundary labels
showing `candidate_only`, `not_training`, `not_phase1`, and
`dataset_confirmation_unchanged`.

## Status Meanings

`written` means the candidate-only quarantine artifact, summary, and evidence
were safely written.

`needs_review` means the artifacts were written only because needs-review
execution preflight evidence was explicitly allowed. The records remain
quarantined.

`failed` means confirmation, schema, hash, id, status, record binding, output
directory, or redaction checks failed. Candidate artifacts are not written on
failure.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_quarantine_materializer \
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
  --output-dir /tmp/custom-corpus-property-quarantine-materializer \
  --quarantine-run-id property-quarantine-materializer-example-001 \
  --created-by operator-redacted \
  --confirm-quarantine-materialization
```

Return codes:

- `0` when status is `written` or `needs_review`
- `1` when status is `failed`

## Redaction Behavior

Before writing candidate records, summary, or evidence, the runner scans for
private path, credential, PDF, CSV, JSONL, Parquet, LMDB, signed URL, and
raw-text markers. On redaction failure it fails closed with:

```text
property_quarantine_materializer_redaction_failed
```

Unsafe candidate or evidence artifacts are not written.

## Boundaries

- The runner writes candidate-only quarantine artifacts.
- The runner does not create training data.
- The runner does not create training CSV/JSONL/Parquet/LMDB artifacts.
- The runner does not create Phase 1 inputs.
- The runner does not run Phase 1.
- The runner does not modify `DatasetConfirmation`.
- The runner does not run an LLM or agent.
- The runner does not call MinerU.
- The runner does not parse PDFs.
- The runner does not read ParsedDocument content.
- Quarantined candidates are necessary but not sufficient for training
  admission.
