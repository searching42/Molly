# Custom Corpus Admission Package Binding

The admission package binding validator checks whether a custom corpus
admission package is internally consistent across four local artifacts:

```text
custom_corpus_manifest.v1
+ custom_corpus_dry_run.v1 report
+ custom_corpus_review.v1
+ custom_corpus_admission.v1
-> custom_corpus_admission_package_validation.v1 summary
```

It only verifies artifact binding and governance consistency. It does not
admit training data, create datasets, parse PDFs, call MinerU, run Phase 1, or
set `DatasetConfirmation.confirmed=true`.

For the complete custom corpus governance path, see
`docs/custom-corpus-governance-runbook.md`. The #155-#160 stage summary is in
`docs/custom-corpus-governance-stage-summary-20260628.md`.

## Required Inputs

The CLI requires:

- a custom corpus manifest
- a custom corpus dry-run report
- a human review manifest
- an admission request

Example:

```bash
python -m ai4s_agent.custom_corpus_admission_package \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/review_manifest.json \
  --admission-request /path/outside/git/admission_request.json \
  --output-summary /tmp/custom-corpus-admission-package-summary.json
```

The validator reads local JSON files only and prints a safe JSON summary.

## Binding Rules

Hash binding:

- `admission_request.source_manifest_sha256` must match the manifest file.
- `admission_request.source_dry_run_report_sha256` must match the dry-run
  report file.
- `admission_request.source_review_manifest_sha256` must match the review
  manifest file.
- The package summary records computed SHA-256 values for all four inputs.

ID binding:

- manifest, dry-run report, review manifest, and admission request must share
  the same `corpus_id`.
- dry-run report `run_id` must match review/admission `dry_run_id`.
- review manifest id must match admission request `review_manifest_id`.

Dry-run boundary checks:

- dry-run decision must be `passed`.
- `confirmation_boundary.dataset_confirmation_confirmed` must be `false`.
- `confirmation_boundary.phase1_status` must be `not_run`.
- `confirmation_boundary.training_dataset_admitted` must be `false`.

Review/admission binding:

- every admission record must match exactly one review record by `review_id`
- document id, record id, field name, and review decision must match
- admission `review_artifact_sha256` must equal the review manifest SHA-256
- admission `source_artifact_sha256` must match the review record source hash
- rejected or needs-review records must not be admitted
- admitted records must include safe provenance, normalized value summary, and
  admission reason

## Property Candidate Review Binding Evidence

Property candidate review binding is upstream evidence for property-candidate
review flows:

```text
docs/custom-corpus-property-review-binding.md
```

It verifies that a human review manifest corresponds to a property candidate
review queue. Package binding still validates the manifest, dry-run report,
review manifest, and admission request package. Do not substitute review queue
artifacts for human review artifacts.

Property admission readiness is also upstream evidence:

```text
docs/custom-corpus-property-admission-readiness.md
```

Readiness summaries do not substitute for `custom_corpus_admission.v1` request
artifacts. Package binding still requires the explicit admission request.

Property admission request plans are upstream evidence too:

```text
docs/custom-corpus-property-admission-request-planner.md
```

Request plans can describe what a future admission request should contain, but
they do not substitute for `custom_corpus_admission.v1`. Package binding still
validates real manifest, dry-run, review, and admission package artifacts.

Property admission drafts are closer to package binding, but still require
operator review before use:

```text
docs/custom-corpus-property-admission-draft-builder.md
```

An admission draft can become an input to future package binding only after
explicit operator review. Package binding must consume the actual draft or
admission request artifact plus the manifest, dry-run report, and review
manifest. The request plan alone is insufficient for package binding.

## Property Admission Draft Package Precheck

Property admission draft package precheck is upstream evidence before formal
package binding:

```text
docs/custom-corpus-property-admission-draft-package-precheck.md
```

The precheck compares a reviewable admission draft against the manifest,
dry-run report, review manifest, draft summary, request plan summary,
readiness summary, and review binding summary. Its output may help an operator
decide whether to run formal package binding, but it is not a substitute for
`custom_corpus_admission_package_validation.v1`. Formal package binding still
consumes the actual manifest, dry-run report, review manifest, and admission
request artifacts.

## Property-Aware Package Binding Runner

The property-aware package binding runner calls this existing formal package
binding validator after gating on property precheck evidence:

```text
docs/custom-corpus-property-package-binding.md
```

Its formal package validation output remains
`custom_corpus_admission_package_validation.v1`. The additional
property-aware wrapper summary links the formal output to the property
precheck. Materialization still requires a separate materialization plan and a
future materializer.

## Decision Semantics

The summary has two independent decision fields:

| Field | Meaning |
| --- | --- |
| `validation_status` | `passed` when all package bindings are valid and no binding errors exist; otherwise `failed`. |
| `admission_decision` | The admission request decision: `eligible`, `needs_review`, or `ineligible`. |

A package can have `validation_status=passed` and
`admission_decision=needs_review` when the package is internally consistent
but still contains records needing review.

## After Package Validation: Materialization Boundary

Package validation is necessary but not sufficient for candidate or training
artifact creation. It verifies governance binding only; it does not produce
candidate records, training CSVs, or Phase 1 inputs.

The future materialization boundary is designed in:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

Any future materializer must add explicit operator confirmation, review
completeness checks, provenance binding, rollback/deletion design, and
redacted evidence before package-validated records can become candidate
artifacts.

## Redaction Policy

The summary includes only safe basenames, safe ids, SHA-256 values, counts,
safe status strings, and stable binding error codes. It must not include:

- raw PDF paths
- manifest `pdf_path`
- private home paths
- tokens or Authorization headers
- bearer tokens, cookies, or x-api-key values
- signed URLs
- raw article text
- ParsedDocument text
- MinerU bundle paths

Before printing or writing a summary, the validator checks for forbidden
private path or credential markers. If detected, it fails closed with
`package_summary_redaction_failed`.

## Boundaries

- This validator does not admit training data.
- This validator does not create candidate/training CSVs.
- This validator does not set `DatasetConfirmation.confirmed=true`.
- This validator does not run Phase 1.
- This validator does not parse PDFs.
- This validator does not call MinerU.
- This validator does not materialize datasets.
- This validator only verifies artifact binding and governance consistency.
