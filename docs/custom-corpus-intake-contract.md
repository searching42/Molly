# Custom Corpus Intake Contract

This document defines the intake contract for user-supplied PDF corpora.
It describes how custom, private, public, or mixed real-PDF corpora may be
declared before they enter the controlled Molly dry-run path.

The dry-run implementation is documented in
`docs/custom-corpus-dry-run.md`. The implementation follows this contract but
does not call live services in CI, admit production datasets, or modify
`DatasetConfirmation` behavior.

## Purpose

The contract defines how future custom/private/public real-PDF corpora may be
described and admitted into a controlled Molly dry-run path.

It exists to keep the custom corpus boundary explicit:

- this is not production dataset admission
- this does not permit auto-confirmation
- this does not validate scientific extraction quality
- this does not weaken the existing synthetic `DatasetConfirmation` gate

The contract is intentionally conservative. A custom corpus dry-run may prove
that Molly can parse and summarize user-supplied PDFs safely, but it must not
turn extracted records into training data without a later human review and
dataset admission process.

## Relationship To Existing MinerU Manual Gate

The self-hosted MinerU manual gate validates infrastructure using
Molly-generated synthetic PDFs:

```text
endpoint profile
-> endpoint preflight diagnostics
-> preflight-bound synthetic live corpus acceptance
-> corpus audit
-> explicit synthetic DatasetConfirmation
-> Phase 1
```

The custom corpus dry-run is the next boundary:

- synthetic live acceptance proves parser infrastructure
- custom corpus dry-run tests controlled parsing and corpus audit without
  training admission
- production dataset admission requires human review and explicit confirmation

No custom/private corpus should use `--confirm-synthetic-dataset`. That flag is
reserved for Molly-owned generated synthetic fixtures.

## Supported Corpus Classes

| Corpus class | Description |
| --- | --- |
| `public_literature` | Public PDFs, such as open-access papers or locally available PDFs that may be referenced but not committed by default. |
| `private_literature` | PDFs with restricted access, unpublished data, licensed content, or internal/private documents. |
| `synthetic_fixture` | Generated fixtures owned by Molly. This is the only class that may use synthetic confirmation in existing live acceptance. |
| `unknown_or_mixed` | Mixed or unclear provenance. Treat conservatively as private. |

## Class Policy Matrix

| Corpus class | Commit PDFs | Commit ParsedDocument outputs | Commit raw MinerU bundles | Phase 1 automatic run | Human review required |
| --- | --- | --- | --- | --- | --- |
| `synthetic_fixture` | Allowed when intentionally part of tests/fixtures | Allowed when intentionally part of tests/fixtures | Allowed only when intentionally minimized and reviewed | Allowed only through explicit synthetic confirmation | Required before treating outputs as real scientific training data |
| `public_literature` | Not by default | Not by default | Not by default | Never | Required |
| `private_literature` | Never | Never | Never | Never | Required |
| `unknown_or_mixed` | Never | Never | Never | Never | Required |

Policy rules:

- generated synthetic fixtures may be committed when intentionally part of
  tests or fixtures
- public and private real PDFs should not be committed by default
- private and unknown/mixed corpora must not commit raw artifacts
- no real/custom corpus may auto-confirm Phase 1
- human review is required before any real/custom extracted records can become
  a training dataset

## Custom Corpus Manifest Contract

Dry-runs read a manifest that describes the corpus without
placing private document content or sensitive locations into committed
evidence.

`src/ai4s_agent/custom_corpus_manifest.py` validates this schema before any
parse request is submitted.

Top-level fields:

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | yes | Schema identifier, for example `custom_corpus_manifest.v1`. |
| `corpus_id` | yes | Safe stable corpus identifier. |
| `corpus_class` | yes | One of `public_literature`, `private_literature`, `synthetic_fixture`, or `unknown_or_mixed`. |
| `created_at` | yes | ISO-8601 timestamp or date supplied by the operator. |
| `created_by` | yes | Operator label. Use a redacted label when needed. |
| `description` | yes | Short corpus description suitable for redacted review. |
| `source_policy` | yes | Policy label describing source/provenance constraints. |
| `default_redaction_policy` | yes | Default commit/redaction behavior for artifacts. |
| `documents` | yes | List of document entries. |

Document fields:

| Field | Required | Description |
| --- | --- | --- |
| `document_id` | yes | Safe stable document identifier. |
| `pdf_path` | yes | Local path to the PDF for the operator-run dry-run. |
| `pdf_sha256` | recommended | PDF hash when available. |
| `title` | optional | Redacted or public-safe title. |
| `doi` | optional | DOI when safe to record. |
| `source_url` | optional | Source URL without tokens, signed query strings, or credentials. |
| `license_or_access` | yes | Access category or license note. |
| `provenance_note` | yes | Short provenance note safe for review. |
| `allow_raw_pdf_commit` | yes | Whether raw PDF may be committed. Defaults to `false` for real documents. |
| `allow_parsed_document_commit` | yes | Whether raw `ParsedDocument` output may be committed. Defaults to `false` for real documents. |
| `allow_mineru_bundle_commit` | yes | Whether raw MinerU bundle may be committed. Defaults to `false` for real documents. |
| `redaction_required` | yes | Whether dry-run evidence must redact document-level details. |
| `expected_document_type` | yes | Expected type, for example `scientific_paper`. |
| `notes` | optional | Safe operator note. |

Rules:

- `document_id` must be a safe stable identifier.
- `pdf_path` is local and must not be written into committed evidence except
  as a safe basename or redacted label.
- `pdf_sha256` should be recorded when possible.
- `allow_raw_pdf_commit`, `allow_parsed_document_commit`, and
  `allow_mineru_bundle_commit` default to `false` for real documents.
- `corpus_class=private_literature` implies all raw artifact commit flags must
  be `false`.
- `corpus_class=unknown_or_mixed` implies all raw artifact commit flags must be
  `false`.
- Custom corpus dry-runs must not set `DatasetConfirmation.confirmed=true`.
- Manifest values that may be committed must be safe after redaction review.

A placeholder manifest is available at:

```text
docs/examples/custom-corpus-manifest.example.json
```

The example uses `/path/outside/git/...` as a placeholder path. It is not a
real private path and must be replaced by an operator-local path outside git
for actual dry-runs.

## Redaction Rules

Committed PRs must not include:

- private file paths
- private home directories
- access tokens
- Authorization headers
- cookies
- signed URLs
- full PDFs
- raw MinerU bundles
- raw `ParsedDocument` outputs from private/custom corpora
- full acceptance reports if they include private paths or raw extracted
  content
- remote task IDs unless explicitly reviewed
- licensed article text
- private notes or unpublished data
- source URLs that include tokens or query credentials

Committed redacted evidence may include:

- corpus id
- corpus class
- document count
- redacted document ids
- safe basenames when not sensitive
- PDF SHA-256 values when allowed
- parse success/failure counts
- high-level rejection counts
- preflight binding summary
- redacted API origin
- artifact bundle SHA-256 values
- reviewed summary metrics

## Dry-Run Behavior

The custom corpus dry-run runner:

- read a manifest
- verify manifest shape and PDF hashes when supplied
- parse PDFs through configured MinerU/pdfplumber providers
- write local artifacts outside git
- run corpus audit if `ParsedDocument` outputs are produced
- always keep `DatasetConfirmation.confirmed=false`
- never run Phase 1 for custom corpus by default
- produce a dry-run report with redacted metadata
- require a separate human review workflow before training admission

The runner stops before Phase 1. If Phase 1 runs, the dry-run report is marked
failed with `phase1_ran_for_custom_corpus`.

Implementation details are in:

```text
docs/custom-corpus-dry-run.md
```

## Human Review Boundary

Human review artifacts are defined in:

```text
docs/custom-corpus-human-review.md
```

The review schema validates redacted manual review metadata for extracted
custom corpus records. Custom dry-run output alone is not enough for training
admission. Reviewed records still require a separate future admission gate.

Future dataset admission must verify:

- source manifest binding
- source dry-run report hash binding
- review completeness
- rejection/needs-review handling
- redaction safety
- project admission policy

Review record fields may include:

- extracted record id
- document id
- field/value
- review decision: `accept`, `reject`, or `needs_review`
- rejection reason
- reviewer id or redacted reviewer label
- review timestamp
- source artifact hash
- notes

Human review does not automatically mean training admission. A separate dataset
admission gate should verify manifest binding, review completeness, and
redaction safety before any real/custom records can become training data.

The full intended boundary chain is:

```text
custom corpus manifest
-> custom corpus dry-run report
-> human review manifest
-> admission request
-> package binding validator
-> future dataset builder
```

The admission request contract is documented in
`docs/custom-corpus-dataset-admission-gate.md`. It validates structure and
artifact binding only; it does not create datasets or admit records.

The package binding validator is documented in
`docs/custom-corpus-admission-package-binding.md`. It checks the request
against the manifest, dry-run report, and review manifest before any future
dataset materialization.

## Pass/Fail Criteria For Custom Corpus Dry-Run

Pass criteria:

- manifest is valid
- all referenced PDFs exist locally
- PDF hashes match when provided
- endpoint preflight passed if live MinerU is used
- no raw private artifacts are committed
- dry-run report is redacted
- `DatasetConfirmation.confirmed` remains `false`
- Phase 1 is not run
- parse and audit outputs are retained outside git with SHA-256 values

Fail criteria:

- manifest has private paths in commit-bound fields
- unknown/mixed corpus tries to allow raw artifact commits
- token-like values appear in manifest or evidence
- PDF hash mismatch
- local PDFs are missing
- custom corpus attempts to auto-confirm Phase 1
- committed evidence includes raw private extracted content
- preflight binding mismatch when required

## Boundaries

- Dry-run runner only.
- No live service calls in tests or CI.
- No live CI.
- No MinerU Cloud API provider.
- No automatic fallback, retry, queue, rollback, or scheduler.
- No `DatasetConfirmation` weakening.
- No Phase 1 admission for custom corpora.
- No production dataset admission.
- No real PDFs committed.
