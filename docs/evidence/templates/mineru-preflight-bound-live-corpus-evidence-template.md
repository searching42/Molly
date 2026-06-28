# MinerU Preflight-Bound Live Corpus Acceptance Evidence - <DATE>

This document records a redacted evidence summary for a manual
preflight-bound MinerU live corpus acceptance run.

## Run Summary

| Field | Value |
| --- | --- |
| Operator/host label | `<redacted-or-local-label>` |
| Date | `<YYYY-MM-DD>` |
| Endpoint profile | `<profile-name>` |
| Routing policy | `<routing-policy-name-or-empty>` |
| Redacted API origin | `<scheme://host:port>` |
| Endpoint kind | `<mineru_api-or-mineru_router>` |

## Preflight Evidence

| Field | Value |
| --- | --- |
| Preflight run id | `<preflight-run-id>` |
| Decision | `<passed-or-failed>` |
| Health status | `<healthy-or-ok>` |
| MinerU version | `<version-or-empty>` |
| Protocol version | `<protocol-version>` |
| Torch version | `<torch-version-or-empty>` |
| Torch CUDA version | `<torch-cuda-version-or-empty>` |
| GPU name | `<gpu-name-or-empty>` |
| GPU capability | `<gpu-capability-or-empty>` |
| Warnings | `<warning-codes-or-empty>` |
| Preflight artifact bundle location outside git | `<external-artifact-location>` |
| Preflight artifact SHA-256 | `sha256:<64-hex-digest>` |

## Bound Corpus Acceptance Evidence

| Field | Value |
| --- | --- |
| Corpus run id | `<corpus-run-id>` |
| Decision | `<passed-or-failed-or-awaiting_confirmation>` |
| `preflight_binding.matched` | `<true-or-false>` |
| `preflight_binding.mismatches` | `<[]-or-mismatch-codes>` |
| Phase 1 status | `<success-or-not_run-or-failed>` |
| Document count | `<integer>` |
| Parse success count | `<integer>/<integer>` |
| Accepted record count | `<integer>` |
| Candidate record count | `<integer>` |
| Training record count | `<integer>` |
| Rejected record count | `<integer>` |
| Conflict count | `<integer>` |
| Unresolved conflict count | `<integer>` |
| Top ranked candidate count | `<integer>` |
| Corpus artifact bundle location outside git | `<external-artifact-location>` |
| Corpus artifact SHA-256 | `sha256:<64-hex-digest>` |

## Redaction Statement

Full artifacts are retained outside git and are not committed here.

This evidence summary does not commit:

- tokens
- auth headers
- private paths
- full sensitive URLs
- full `preflight_report.json`
- full `acceptance_report.json`
- generated PDFs
- MinerU bundles
- `ParsedDocument` outputs
- pdfplumber baselines
- remote task ids
- complete artifact tarballs

## Boundaries

- Synthetic corpus only.
- Not production scientific accuracy evidence.
- Not a throughput benchmark.
- Not private-document handling validation.
- Not automatic deployment.
- Not MinerU installation or environment repair.
- Not MinerU Cloud API provider validation.
- Not fallback, retry, rollback, queueing, scheduling, or canary routing.
