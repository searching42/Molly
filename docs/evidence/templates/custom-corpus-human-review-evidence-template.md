# Custom Corpus Human Review Evidence - <DATE>

This document records redacted evidence for a custom corpus human review
artifact. It is review evidence only, not dataset admission evidence.

## Review Summary

| Field | Value |
| --- | --- |
| Review manifest id | `<review-manifest-id>` |
| Corpus id | `<corpus-id>` |
| Dry-run id | `<dry-run-id>` |
| Review record count | `<integer>` |
| Accepted count | `<integer>` |
| Rejected count | `<integer>` |
| Needs review count | `<integer>` |
| Reviewed document count | `<integer>` |
| Source dry-run report SHA-256 | `sha256:<64-hex-digest>` |
| Source manifest SHA-256 | `sha256:<64-hex-digest-or-empty>` |

## Validation Summary

| Field | Value |
| --- | --- |
| Schema version | `custom_corpus_review.v1` |
| Validator decision | `<passed-or-failed>` |
| Validation command | `<command>` |

## Redaction Statement

This evidence does not include:

- raw PDFs
- ParsedDocument outputs
- MinerU bundles
- full extracted raw text
- private paths
- tokens
- Authorization headers
- cookies

## Boundary Statement

- No `DatasetConfirmation` change.
- No Phase 1 execution.
- No training dataset admission.
- Review evidence only.
