# Custom Corpus Dry-Run Evidence - 2026-06-28

This document records a redacted evidence summary for a small public custom
corpus dry-run. It is not training data admission evidence.

## Run Summary

| Field | Value |
| --- | --- |
| Run id | `custom-corpus-dry-run-public-20260628-184354` |
| Date | `2026-06-28` |
| Corpus id | `public-small-corpus-20260628` |
| Corpus class | `public_literature` |
| Document count | `2` |
| Endpoint profile | `node45-loopback` |
| Routing policy | `manual-primary` |
| Redacted API origin | `http://127.0.0.1:18000` |
| Preflight run id | `mineru-preflight-node45-20260628-184223` |
| Preflight decision | `passed` |
| Preflight health status | `healthy` |
| Preflight protocol version | `2` |
| Preflight binding matched | `true` |
| Dry-run decision | `passed` |

## Manifest Summary

| Field | Value |
| --- | --- |
| Manifest schema version | `custom_corpus_manifest.v1` |
| Manifest artifact SHA-256 | `sha256:6576b4c5304c321787118a9741ee5750eee2dc63824f6b685c1428aae5f613d0` |
| Document ids | `attention_is_all_you_need_2017`, `dovepress_5fu_repurposing_2022` |
| PDF hash coverage | `2/2` |
| Source policy | `public_literature_local_pdf_dry_run` |
| Redaction policy | raw PDFs: `false`; ParsedDocuments: `false`; MinerU bundles: `false`; full reports: `false` |

The manifest was generated and retained outside git. This committed evidence
does not include the manifest body because it contains local operator paths.

## Parse Summary

| Field | Value |
| --- | --- |
| Parse attempted count | `2` |
| Parse success count | `2` |
| Parse failure count | `0` |
| ParsedDocument produced count | `2` |
| MinerU protocol version summary | `2` |
| Parser warnings count | `0` |

## Corpus Audit Summary

| Field | Value |
| --- | --- |
| Extracted record count | `0` |
| Accepted record count | `0` |
| Rejected record count | `0` |
| Duplicate count | `0` |
| Conflict count | `0` |
| Unresolved conflict count | `0` |

The selected public papers were used to validate custom corpus intake, parsing,
redaction, and confirmation-boundary behavior. They were not selected as an
OLED scientific extraction benchmark, and no extracted scientific records were
admitted.

## Confirmation Boundary

| Field | Value |
| --- | --- |
| `DatasetConfirmation.confirmed` | `false` |
| Phase 1 status | `not_run` |
| Training dataset admitted | `false` |
| Dataset manifest status | `awaiting_confirmation` |
| Training record count | `0` |

This is the most important boundary for this evidence. The custom corpus
dry-run did not enter Phase 1 and did not admit training data.

## Artifact Evidence

| Field | Value |
| --- | --- |
| External dry-run artifact location | local tmp dry-run evidence bundle, retained outside git |
| Dry-run artifact SHA-256 | `sha256:c7af99fa3790eba2d6c8a23d1dfbd49fa7e2d1cc12cb8c214477cc4f178e2107` |
| Preflight artifact SHA-256 | `sha256:94361024dbea62e657f09cb4c552159725e59c961e32e769f6affe7d9d4af41e` |
| Full artifacts committed | `false` |

The external bundle contains redacted run evidence and workflow summary files
only. The bundle itself is not committed.

## Redaction Statement

Full artifacts are retained outside git and are not committed here.

This evidence summary does not commit:

- raw PDFs
- full manifest with private paths
- MinerU bundles
- `ParsedDocument` outputs
- pdfplumber baselines
- full dry-run reports
- full corpus reports if they contain private paths or raw text
- private file paths
- private home directories
- tokens
- Authorization headers
- cookies
- signed URLs
- licensed full text
- remote task IDs unless explicitly reviewed

## Boundaries

- Docs-only evidence.
- Dry-run only.
- Public literature only.
- Not production scientific accuracy evidence.
- Not training data admission.
- Not private-document handling certification.
- Not Phase 1 confirmation.
- Not human review.
- Not dataset admission.
- Not MinerU Cloud API provider validation.
