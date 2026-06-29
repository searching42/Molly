# Custom Corpus Materialization Plan Evidence - <DATE>

This template records validation evidence for a materialization plan only. It
is not evidence that candidate or training artifacts were created.

## Plan Summary

| Field | Value |
| --- | --- |
| Materialization plan id | `<materialization-plan-id>` |
| Materialization run id | `<materialization-run-id>` |
| Corpus id | `<corpus-id>` |
| Dry-run id | `<dry-run-id>` |
| Review manifest id | `<review-manifest-id>` |
| Admission request id | `<admission-request-id>` |
| Materialization mode | `candidate_only` |
| Materialization decision | `<planned-or-blocked>` |

## Source Binding

| Field | Value |
| --- | --- |
| Manifest SHA-256 | `sha256:<64-hex-digest>` |
| Dry-run report SHA-256 | `sha256:<64-hex-digest>` |
| Review manifest SHA-256 | `sha256:<64-hex-digest>` |
| Admission request SHA-256 | `sha256:<64-hex-digest>` |
| Package validation SHA-256 | `sha256:<64-hex-digest>` |

## Confirmation Summary

| Field | Value |
| --- | --- |
| Confirmation present | `<true-or-false>` |
| Confirmation source | `<confirmation-source>` |
| Confirmation operator label | `<redacted-operator-label>` |

## Boundary Summary

| Field | Value |
| --- | --- |
| Package validation status | `<passed-or-failed>` |
| Package admission decision | `<eligible-or-needs_review-or-ineligible>` |
| Phase 1 status | `not_run` |
| DatasetConfirmation confirmed | `false` |
| Training dataset admitted | `false` |

## Record Summary

| Field | Value |
| --- | --- |
| Total records | `<integer>` |
| Candidate records | `<integer>` |
| Excluded records | `<integer>` |

## Redaction Statement

This evidence does not include raw PDFs, ParsedDocument outputs, MinerU
bundles, full raw text, private paths, tokens, Authorization headers, cookies,
or private emails.

## Non-Admission Statement

- No artifacts created.
- No candidate CSV created.
- No training CSV created.
- No Phase 1 execution.
- No `DatasetConfirmation` change.
