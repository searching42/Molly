# Custom Corpus Materialization Evidence - <DATE>

This template is for future custom corpus materialization evidence. It is not
evidence of training admission or Phase 1 execution.

## Materialization Summary

| Field | Value |
| --- | --- |
| Materialization run id | `<materialization-run-id>` |
| Corpus id | `<corpus-id>` |
| Dry-run id | `<dry-run-id>` |
| Review manifest id | `<review-manifest-id>` |
| Admission request id | `<admission-request-id>` |
| Package validation summary SHA-256 | `sha256:<64-hex-digest>` |
| Materialization decision | `<passed-or-failed>` |

## Source Artifact Binding

| Field | Value |
| --- | --- |
| Manifest SHA-256 | `sha256:<64-hex-digest>` |
| Dry-run report SHA-256 | `sha256:<64-hex-digest>` |
| Review manifest SHA-256 | `sha256:<64-hex-digest>` |
| Admission request SHA-256 | `sha256:<64-hex-digest>` |
| Package validation summary SHA-256 | `sha256:<64-hex-digest>` |

## Confirmation Boundary

| Field | Value |
| --- | --- |
| Explicit materialization confirmation present | `<true-or-false>` |
| Confirmation operator label | `<redacted-operator-label>` |
| Confirmation source | `<confirmation-source>` |
| DatasetConfirmation changed | `false` |
| Phase 1 status | `not_run` |

## Materialized Candidate Summary

| Field | Value |
| --- | --- |
| Candidate record count | `<integer>` |
| Admitted source record count | `<integer>` |
| Excluded count | `<integer>` |
| Needs review count | `<integer>` |
| Rejected materialized count | `0` |
| Needs review materialized count | `0` |

## Provenance Summary

| Field | Value |
| --- | --- |
| Provenance bindings written | `<true-or-false>` |
| Rollback manifest written | `<true-or-false>` |
| Rollback manifest SHA-256 | `sha256:<64-hex-digest>` |

## Redaction Statement

This evidence does not include:

- raw PDFs
- ParsedDocument outputs
- MinerU bundles
- full raw text
- private paths
- tokens
- Authorization headers
- cookies
- private emails

## Boundary Statement

- Candidate materialization only.
- No training data admission.
- No Phase 1 execution.
- No `DatasetConfirmation` change.
