# Custom Corpus Human Review Artifacts

Custom corpus dry-runs can produce extracted, rejected, or candidate records,
but dry-run output is never trusted automatically. Human review artifacts add a
reviewable contract between dry-run evidence and any future dataset admission
gate.

This layer validates review metadata only. It does not admit training data,
set `DatasetConfirmation.confirmed=true`, create a training dataset, or run
Phase 1.

For the complete custom corpus governance path, see
`docs/custom-corpus-governance-runbook.md`. The #155-#160 stage summary is in
`docs/custom-corpus-governance-stage-summary-20260628.md`.

## Workflow Boundary

```text
custom corpus dry-run
-> extracted/rejected/candidate records
-> custom_corpus_review.v1 artifact
-> future dataset admission gate
```

The review artifact is intentionally separate from admission. A future
admission gate must still verify manifest binding, source artifact hashes,
review completeness, redaction safety, and project policy before any record can
become training data.

## Schema

Review manifests use:

```text
custom_corpus_review.v1
```

Top-level fields:

| Field | Description |
| --- | --- |
| `review_manifest_id` | Safe stable review artifact id. |
| `corpus_id` | Corpus id from the custom corpus manifest. |
| `dry_run_id` | Dry-run id that produced the reviewed artifacts. |
| `source_dry_run_report_sha256` | SHA-256 of the dry-run report being reviewed. |
| `source_manifest_sha256` | Optional SHA-256 of the source corpus manifest. |
| `review_policy` | Short safe label for the review policy. |
| `review_records` | Non-empty list of reviewed records. |

Each review record identifies a review target with:

- `document_id`
- `record_id`
- optional `field_name`
- `review_scope`: `record`, `field`, `document`, or `corpus`

Review decisions are:

- `accept`
- `reject`
- `needs_review`

Rules:

- `reject` requires `rejection_reason`.
- `accept` must not include `rejection_reason`.
- `needs_review` requires `notes` or `confidence_note`.
- Duplicate `review_id` values are rejected.
- Duplicate review targets are rejected.
- IDs must use only letters, numbers, dot, dash, and underscore.
- SHA-256 values are normalized to `sha256:<64 lowercase hex>`.

## Validation

Validate a review manifest offline:

```bash
python -m ai4s_agent.custom_corpus_review \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --output-summary /tmp/custom-corpus-review-summary.json
```

The validator:

- reads local JSON only
- prints a safe JSON summary
- optionally writes the summary JSON
- does not call MinerU
- does not parse PDFs
- does not run corpus workflow
- does not modify datasets

## Redaction Requirements

Committed review manifests and summaries must not include:

- raw PDFs
- raw `ParsedDocument` outputs
- MinerU bundles
- pdfplumber baselines
- full extracted raw text
- private paths
- private home directories
- tokens
- Authorization headers
- cookies
- signed URLs
- private emails unless redacted

Free-text fields are length-limited and reject obvious credential-like and
private-path-like values. `extracted_value_summary` and
`normalized_value_summary` must be short summaries, not copied article text.

## Examples

Safe example manifest:

```text
docs/examples/custom-corpus-review-manifest.example.json
```

Future evidence PR template:

```text
docs/evidence/templates/custom-corpus-human-review-evidence-template.md
```

## After Human Review: Admission Gate Contract

A review artifact alone does not admit data. The next boundary is an admission
request that binds reviewed records to the source dry-run, source manifest,
review manifest, and artifact hashes.

The admission gate contract is documented in:

```text
docs/custom-corpus-dataset-admission-gate.md
```

That gate validates structural eligibility only. A future implementation must
still materialize data separately, and must explicitly bind admission records
to reviewed records and source artifacts before any training data can be
created.

The package binding validator then checks review/admission consistency against
the source manifest and dry-run report before any future materialization step:

```text
docs/custom-corpus-admission-package-binding.md
```

## Boundaries

- Human review artifacts do not automatically admit training data.
- Human review does not set `DatasetConfirmation.confirmed=true`.
- This PR does not create a training dataset.
- This PR does not run Phase 1.
- This PR does not implement production dataset admission.
- This PR does not certify scientific correctness.
- This PR does not permit raw PDFs, ParsedDocuments, MinerU bundles, or private
  paths in committed review manifests.
