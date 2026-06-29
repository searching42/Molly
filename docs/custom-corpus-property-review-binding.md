# Custom Corpus Property Review Binding

The property review binding validator checks whether a manually-created
`custom_corpus_review.v1` human review manifest is properly bound to a
`custom_corpus_property_candidate_review_queue.v1` review queue.

A review queue is not a review decision. A binding validator is not a reviewer.
A valid binding is necessary but not sufficient for admission.

## Input Artifacts

The validator reads two local JSON artifacts:

- `property_candidate_review_queue.json`
- `custom_corpus_review.v1` manifest

It reads local JSON only. It does not read PDFs, ParsedDocument content, MinerU
bundles, corpus workflow outputs, admission requests, materialization plans, or
training artifacts.

## Binding Rules

Manifest-level checks:

- review manifest `corpus_id` must match queue `corpus_id`
- review manifest `dry_run_id` must match queue `dry_run_id`
- review manifest `source_dry_run_report_sha256` must match the queue
- non-empty review manifest `source_manifest_sha256` must match the queue

Record-level checks:

- review record `record_id` must equal queue record `property_candidate_id`
- document id, field name, corpus id, dry-run id, and source artifact SHA-256
  must match
- review scope must be `record` or `field`
- no review record may target a blocked queue record
- no review record may target an unknown queue record
- accepted records require short extracted value, normalized value, and
  provenance summaries

Existing `custom_corpus_review.v1` validation still enforces review decision
rules for `accept`, `reject`, and `needs_review`.

## Completeness Behavior

By default, the review manifest may cover a subset of the queue. If no binding
errors exist but some queue records are unreviewed, the validator returns:

```text
needs_review
```

With `--require-complete-queue`, every queue record must have exactly one
review record. Missing reviews fail validation.

## Status Meanings

| Status | Meaning |
| --- | --- |
| `passed` | All queue records are reviewed and no binding errors exist. |
| `needs_review` | No binding errors exist, but queue completeness is not required and some queue records are unreviewed. |
| `failed` | Binding errors exist, blocked or unknown records were reviewed, hashes mismatch, or completeness is required but missing reviews exist. |

The CLI returns `0` for `passed` and `needs_review`, and `1` for `failed`.

## Summary Output

The summary schema is:

```text
custom_corpus_property_review_binding.v1
```

The summary includes safe basenames, SHA-256 values, queue/review ids, counts,
reviewed/unreviewed record ids, binding error codes, warnings, completeness
mode, and redaction status. It does not include raw value summaries,
provenance details, raw table rows, raw article text, local paths,
ParsedDocument text, MinerU bundle paths, tokens, cookies, Authorization
headers, or private emails.

## Redaction Behavior

Before printing or writing a summary, the validator scans serialized summary
content for private path and credential markers. If unsafe material is
detected, it fails closed with:

```text
property_review_binding_summary_redaction_failed
```

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_review_binding \
  --review-queue /tmp/custom-corpus-property-review-queue/property-review-queue-example-001/property_candidate_review_queue.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --output-summary /tmp/custom-corpus-property-review-binding-summary.json \
  --require-complete-queue
```

## After Binding: Admission Readiness Planner

Binding validation checks queue-to-review consistency. The next optional
planning boundary identifies reviewed records that are eligible for future
admission planning:

```text
docs/custom-corpus-property-admission-readiness.md
```

Future readiness evidence template:

```text
docs/evidence/templates/custom-corpus-property-admission-readiness-evidence-template.md
```

Readiness planning still does not create admission requests or admission
actions.

## Boundaries

- The binding validator does not perform human review.
- The binding validator does not create review decisions.
- The binding validator does not create a `custom_corpus_review.v1` manifest.
- The binding validator does not create admission requests.
- The binding validator does not materialize data.
- The binding validator does not create dataset candidate/training CSVs.
- The binding validator does not run Phase 1.
- The binding validator does not modify `DatasetConfirmation`.
- A valid binding is necessary but not sufficient for admission.
