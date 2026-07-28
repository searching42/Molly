# Custom Corpus Dry-Run Evidence - <DATE>

This document records a redacted evidence summary for a future custom corpus
dry-run. It is not training data admission evidence.

## Run Summary

| Field | Value |
| --- | --- |
| Corpus id | `<corpus-id>` |
| Corpus class | `<public_literature-or-private_literature-or-synthetic_fixture-or-unknown_or_mixed>` |
| Document count | `<integer>` |
| Endpoint profile | `<profile-name-or-empty>` |
| Routing policy | `<routing-policy-name-or-empty>` |
| Redacted API origin | `<scheme://host:port-or-empty>` |
| Preflight run id | `<preflight-run-id-or-empty>` |
| Preflight decision | `<passed-or-failed-or-not_run>` |
| Dry-run decision | `<passed-or-failed>` |

## Manifest Summary

| Field | Value |
| --- | --- |
| Manifest schema version | `<custom_corpus_manifest.v1>` |
| Manifest artifact SHA-256 | `sha256:<64-hex-digest>` |
| Document ids | `<redacted-or-safe-document-ids>` |
| PDF hash coverage | `<count-with-hash>/<document-count>` |
| Source policy | `<source-policy>` |
| Redaction policy | `<summary-of-redaction-policy>` |

## Parse Summary

| Field | Value |
| --- | --- |
| Parse attempted count | `<integer>` |
| Parse success count | `<integer>` |
| Parse failure count | `<integer>` |
| ParsedDocument produced count | `<integer>` |
| MinerU protocol version summary | `<versions-or-not_applicable>` |
| Parser warnings count | `<integer>` |

## Corpus Audit Summary

| Field | Value |
| --- | --- |
| Extracted record count | `<integer>` |
| Accepted record count | `<integer>` |
| Rejected record count | `<integer>` |
| Duplicate count | `<integer>` |
| Conflict count | `<integer>` |
| Unresolved conflict count | `<integer>` |

## Confirmation Boundary

| Field | Value |
| --- | --- |
| `DatasetConfirmation.confirmed` | `false` |
| Phase 1 status | `not_run` |
| Training dataset admitted | `false` |

## Artifact Evidence

| Field | Value |
| --- | --- |
| Dry-run artifact bundle retained outside git | `<external-artifact-location>` |
| Dry-run artifact SHA-256 | `sha256:<64-hex-digest>` |
| Full artifacts committed | `false` |

## Redaction Statement

Full artifacts are retained outside git and are not committed here.

This evidence summary does not commit:

- raw PDFs
- MinerU bundles
- `ParsedDocument` outputs
- full reports
- private paths
- tokens
- Authorization headers
- signed URLs
- licensed text
- remote task ids unless explicitly reviewed

## Boundaries

- Dry-run only.
- Not production scientific accuracy evidence.
- Not training data admission.
- Not private-document handling certification.
- Not Phase 1 confirmation.
- Not MinerU Cloud API provider validation.
