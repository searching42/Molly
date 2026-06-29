# Custom Corpus Property Admission Draft Package Precheck

The property admission draft package precheck reads a reviewable
`custom_corpus_admission.v1` draft and its upstream property governance
summaries, then emits safe JSON and optional Markdown evidence describing
whether the draft appears ready for later formal admission package binding.

The precheck does not run formal package binding. It does not create
`custom_corpus_admission_package_validation.v1`, does not materialize data,
does not create dataset candidate/training CSVs, does not run Phase 1, and does
not modify `DatasetConfirmation`. A passed precheck is necessary but not
sufficient for formal package binding.

## Relationship To Draft Generation

The previous boundary is documented in:

```text
docs/custom-corpus-property-admission-draft-builder.md
```

The draft builder can write a reviewable admission draft. The package precheck
checks that draft against the manifest, dry-run report, human review manifest,
draft summary, request plan summary, readiness summary, and review binding
summary before an operator submits the package to the formal package binding
validator.

## Inputs

The precheck reads local JSON artifacts only:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- a draft `custom_corpus_admission.v1`
- `custom_corpus_property_admission_draft_builder.v1`
- `custom_corpus_property_admission_request_plan.v1`
- `custom_corpus_property_admission_readiness.v1`
- `custom_corpus_property_review_binding.v1`

It does not read PDFs, ParsedDocument outputs, MinerU bundles, property
candidate manifests, review queue JSON, materialization plans, or training
artifacts.

## Precheck Rules

The precheck validates that:

- all source artifacts have the expected schema versions
- the custom corpus dry-run decision is `passed`
- `DatasetConfirmation.confirmed=false`
- Phase 1 status is `not_run`
- training dataset admitted is `false`
- corpus ids, dry-run ids, review manifest ids, review queue ids, and property
  candidate manifest ids match across property summaries
- non-empty source artifact SHA-256 fields match the actual input files
- the admission draft validates as `custom_corpus_admission.v1`
- admission draft records match the draft summary admit/exclude ids
- request plan admit/exclude ids match the draft summary
- readiness admit/exclude ids match the request plan
- review binding reviewed ids include every admission draft record
- reviewed blocked ids and unknown review ids do not appear in the admission
  draft
- needs-review records are not admitted

The precheck reports safe error codes such as `corpus_id_mismatch`,
`source_manifest_sha256_mismatch`, `draft_summary_record_ids_mismatch`,
`review_binding_missing_admission_records`, and `dry_run_phase1_ran`.

## Status Meanings

`passed` means no hard consistency errors were found and no selected strict
flags blocked partial upstream evidence.

`needs_review` means no hard consistency errors were found, but upstream
property evidence is partial, such as a partial request plan or partial
readiness summary.

`failed` means at least one hard consistency error was found, the dry-run
boundary was violated, a required upstream artifact was blocked or failed, or
redaction failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_draft_package_precheck \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-draft /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --draft-summary /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/property_admission_draft_summary.json \
  --request-plan-summary /tmp/custom-corpus-property-admission-request-plan-summary.json \
  --readiness-summary /tmp/custom-corpus-property-admission-readiness-summary.json \
  --review-binding-summary /tmp/custom-corpus-property-review-binding-summary.json \
  --output-summary /tmp/custom-corpus-property-admission-draft-package-precheck-summary.json \
  --output-markdown /tmp/custom-corpus-property-admission-draft-package-precheck-summary.md
```

Use `--require-planned-request` to fail partial request-plan evidence. Use
`--require-ready-readiness` to fail partial readiness evidence.

## Summary Output

The precheck emits `custom_corpus_property_admission_draft_package_precheck.v1`
with safe basenames, SHA-256 values, ids, statuses, counts, admit/exclude
record ids, blocked ids, warnings, and precheck error codes.

The summary does not include raw table rows, raw article text, local absolute
paths, private paths, ParsedDocument text, MinerU bundle paths, tokens,
Authorization headers, cookies, private emails, raw PDF names, or raw PDF
paths.

## Redaction Behavior

Before printing or writing JSON/Markdown, the precheck serializes the output
and scans for private path and credential markers. If unsafe material is
detected, it fails closed with:

```text
property_admission_draft_package_precheck_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The precheck does not run formal package binding.
- The precheck does not create `custom_corpus_admission_package_validation.v1`.
- The precheck does not materialize data.
- The precheck does not create dataset candidate/training CSVs.
- The precheck does not run Phase 1.
- The precheck does not modify `DatasetConfirmation`.
- A passed precheck is necessary but not sufficient for formal package binding.
