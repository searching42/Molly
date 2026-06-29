# Custom Corpus Dataset Admission Gate Contract

The dataset admission gate contract defines the next boundary after custom
corpus dry-runs and human review. It validates whether a reviewed custom corpus
package is structurally eligible for a future dataset admission implementation.

This contract does not admit training data. It does not create candidate or
training CSVs, does not set `DatasetConfirmation.confirmed=true`, and does not
run Phase 1.

For the complete custom corpus governance path, see
`docs/custom-corpus-governance-runbook.md`. The #155-#160 stage summary is in
`docs/custom-corpus-governance-stage-summary-20260628.md`.

## Workflow Boundary

```text
custom corpus dry-run
-> human review artifact
-> dataset admission request
-> offline admission gate validation
-> future explicit dataset builder/admission implementation
```

The gate validates binding and intent only. A future implementation must still
explicitly bind admission records to reviewed records and source artifacts
before any training data can be created.

## Admission Request Schema

Admission requests use:

```text
custom_corpus_admission.v1
```

Top-level fields include:

| Field | Description |
| --- | --- |
| `admission_request_id` | Safe stable request id. |
| `corpus_id` | Custom corpus id. |
| `dry_run_id` | Dry-run id that produced source artifacts. |
| `review_manifest_id` | Human review artifact id. |
| `source_manifest_sha256` | SHA-256 binding to the source corpus manifest. |
| `source_dry_run_report_sha256` | SHA-256 binding to the dry-run report. |
| `source_review_manifest_sha256` | SHA-256 binding to the review manifest. |
| `admission_policy` | Safe policy label. |
| `dataset_target` | Safe target label, not a file path. |
| `admission_records` | Non-empty list of proposed record actions. |

Each admission record binds:

- document id
- record id
- optional field name
- review id
- source artifact SHA-256
- review artifact SHA-256
- review decision
- proposed admission action

## Action Rules

Review decisions are:

- `accept`
- `reject`
- `needs_review`

Admission actions are:

- `admit`
- `exclude`
- `needs_review`

Rules:

- `review_decision=accept` may use `action=admit` or `action=exclude`.
- `review_decision=reject` must use `action=exclude`.
- `review_decision=needs_review` must use `action=needs_review` or
  `action=exclude`.
- `action=admit` requires `review_decision=accept`.
- `action=admit` requires `admission_reason`.
- `action=exclude` requires `exclusion_reason`.
- `action=needs_review` requires `notes`.

## Summary Decision Rules

The validator returns a safe summary decision:

| Decision | Meaning |
| --- | --- |
| `eligible` | At least one record is marked `admit`, and no records are marked `needs_review`. |
| `needs_review` | At least one record is marked `needs_review`. |
| `ineligible` | No records are admitted, or structural validation failed before summary generation. |

Structural validation errors still fail the CLI with exit code `1`.

## Package Binding Validator

Admission request validation alone is not enough for future dataset
materialization. A request can be structurally valid while still pointing at
the wrong dry-run report, review manifest, or source manifest.

The package binding validator checks the request against the full artifact set:

```text
custom corpus manifest
-> dry-run report
-> human review manifest
-> admission request
```

Use:

```bash
python -m ai4s_agent.custom_corpus_admission_package \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/review_manifest.json \
  --admission-request /path/outside/git/admission_request.json
```

It verifies SHA-256 bindings, id consistency, dry-run confirmation boundaries,
and review/admission record consistency. It still does not create datasets or
admit records.

Detailed package validation behavior is documented in:

```text
docs/custom-corpus-admission-package-binding.md
```

## Property Candidate Review Binding

For property candidate reviews, future admission requests should be based on
queue-bound human review manifests. The property review binding validator
checks that a manually-created `custom_corpus_review.v1` manifest corresponds
to the property candidate review queue:

```text
docs/custom-corpus-property-review-binding.md
```

The binding summary can be used as pre-admission evidence, but admission
remains a separate request and validation boundary.

Property admission readiness summaries are the next pre-admission planning
artifact:

```text
docs/custom-corpus-property-admission-readiness.md
```

Future admission requests for property candidates should use readiness
summaries as pre-admission evidence. Readiness does not replace admission; the
admission request remains separate and explicit.

Property admission request planning is another upstream evidence step:

```text
docs/custom-corpus-property-admission-request-planner.md
```

Future admission requests can be drafted from request-plan evidence, but a
request plan does not replace an explicit `custom_corpus_admission.v1`
artifact. Admission remains separate and explicit.

The property admission draft builder can create reviewable admission request
drafts:

```text
docs/custom-corpus-property-admission-draft-builder.md
```

The draft artifact does not admit training data. Explicit package binding and
later materialization remain separate downstream gates.

Property admission draft package precheck can confirm draft consistency before
formal package binding:

```text
docs/custom-corpus-property-admission-draft-package-precheck.md
```

The precheck is not admission package validation. It does not create
`custom_corpus_admission_package_validation.v1`; formal package binding remains
a separate explicit step.

The future materialization boundary is designed in:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

## Validation

Validate an admission request offline:

```bash
python -m ai4s_agent.custom_corpus_admission \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --output-summary /tmp/custom-corpus-admission-summary.json
```

The validator:

- reads local JSON only
- prints a safe JSON summary
- optionally writes the summary JSON
- does not call MinerU
- does not parse PDFs
- does not run corpus workflow
- does not create datasets
- does not modify `DatasetConfirmation`

## Redaction Requirements

Admission requests and summaries must not include:

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

Free-text fields are short summaries only. They reject credential-like values,
private-path-like values, and URL query strings.

## Examples

Safe example request:

```text
docs/examples/custom-corpus-admission-request.example.json
```

Future evidence PR template:

```text
docs/evidence/templates/custom-corpus-admission-gate-evidence-template.md
```

## Boundaries

- This PR does not admit training data.
- This PR does not create candidate/training CSVs.
- This PR does not set `DatasetConfirmation.confirmed=true`.
- This PR does not run Phase 1.
- This PR does not modify dataset builder behavior.
- This PR does not certify scientific correctness.
- This PR does not allow reviewed records to bypass manifest binding or review
  completeness checks.
- A future implementation must explicitly bind admission records to reviewed
  records and source artifacts before any training data can be created.
