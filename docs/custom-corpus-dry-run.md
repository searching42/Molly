# Custom Corpus Dry-Run Runner

`python -m ai4s_agent.custom_corpus_dry_run` is a manual dry-run path for
operator-supplied local PDFs. It validates a `custom_corpus_manifest.v1`, parses
the referenced PDFs through the existing document parsing providers, and runs
corpus extraction/audit up to the unconfirmed workflow boundary.

This runner does not admit training data. It always uses
`DatasetConfirmation.confirmed=false`, verifies Phase 1 remains `not_run`, and
marks the run failed if Phase 1 executes.

## What It Does

```text
custom corpus manifest
-> manifest validation
-> optional preflight binding
-> local PDF parsing through DocumentParseService
-> corpus extraction/audit workflow
-> DatasetConfirmation=false boundary check
-> redacted dry-run report
```

It writes local artifacts under:

```text
<output>/<run-id>/
```

Expected top-level evidence:

- `dry_run_report.json`
- `dry_run_summary.md`
- `parsed_documents/`
- `mineru_bundles/`
- `pdfplumber_baselines/` when `--compare-pdfplumber` is used
- `corpus_workflow/`

Only redacted evidence should be committed. Raw PDFs, MinerU bundles,
`ParsedDocument` outputs, and full private reports should remain outside git.

## Manifest

The runner reads `custom_corpus_manifest.v1`. A contract example is available at:

```text
docs/examples/custom-corpus-manifest.example.json
```

Validation is fail-closed for the safety boundary:

- `corpus_id` and `document_id` must be safe stable identifiers.
- `documents` must be non-empty.
- duplicate `document_id` values are rejected.
- `source_url` must not contain userinfo, query strings, fragments, tokens, or
  credential-like values.
- `pdf_sha256` is optional, but when supplied it must be a SHA-256 digest and
  must match the local PDF.
- `private_literature` and `unknown_or_mixed` force all raw artifact commit
  flags to `false`.

## Example

Direct endpoint:

```bash
python -m ai4s_agent.custom_corpus_dry_run \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --api-url http://127.0.0.1:18000 \
  --endpoint-kind mineru-api \
  --output /tmp/molly-custom-corpus-dry-run \
  --run-id "custom-corpus-dry-run-$(date +%Y%m%d-%H%M%S)" \
  --backend hybrid-engine \
  --effort medium \
  --allow-remote-upload
```

With endpoint profile and required preflight match:

```bash
python -m ai4s_agent.custom_corpus_dry_run \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-custom-corpus-dry-run \
  --run-id "custom-corpus-dry-run-$(date +%Y%m%d-%H%M%S)" \
  --preflight-report /tmp/molly-mineru-preflight/<run-id>/preflight_report.json \
  --preflight-artifact-sha256 sha256:<digest> \
  --require-preflight-match
```

There is intentionally no `--confirm-synthetic-dataset` or `--confirmed-by`
option. Custom/private corpora cannot enter Phase 1 through this runner.

## Report Redaction

The dry-run report records:

- run id
- corpus id and corpus class
- manifest basename and manifest SHA-256
- document count and PDF hash coverage
- redacted API origin
- optional preflight binding summary
- parse counts and MinerU protocol versions
- corpus audit counts
- confirmation boundary showing `confirmed=false` and `phase1_status=not_run`
- relative output paths under the run directory

The report must not include:

- absolute raw PDF paths
- private home paths
- token-like values
- Authorization headers, cookies, bearer tokens, or API keys
- raw `ParsedDocument` text
- raw MinerU bundle content

## After Dry-Run: Human Review

Dry-run outputs may inform a later human review artifact, but they still do not
admit training data. Use `custom_corpus_review.v1` review manifests to record
short, redacted review decisions for extracted records or fields.

Do not commit raw `ParsedDocument` outputs, full reports, raw extracted text,
MinerU bundles, pdfplumber baselines, private paths, or local PDF paths as part
of review evidence.

Human review artifacts are documented in:

```text
docs/custom-corpus-human-review.md
```

Even after review, a separate future dataset admission gate is required before
any custom corpus record can enter training.

## Dry-Run To Review To Admission

The current custom corpus chain is:

```text
dry-run source artifacts
-> human review summaries
-> admission gate validation
-> future dataset builder/admission implementation
```

Dry-runs produce source artifacts. Human review summarizes selected records.
The admission gate validates a reviewed admission request. None of these steps
currently run Phase 1, set `DatasetConfirmation.confirmed=true`, or create
training datasets.

The dry-run report participates in package-level hash binding. Future
admission package validation verifies that a review manifest and admission
request refer to the exact dry-run report artifact that produced the reviewed
records.

## Troubleshooting

- `invalid_manifest`: fix schema version, ids, URL safety, SHA-256 format, or
  redaction flags.
- `missing_pdf`: a manifest PDF path does not exist on the operator machine.
- `pdf_sha256_mismatch`: regenerate or correct the manifest hash.
- `preflight_match_failed`: the provided preflight report does not match the
  resolved endpoint and `--require-preflight-match` was used.
- `phase1_ran_for_custom_corpus`: the dry-run boundary was violated and the
  evidence must not be used.

## Boundaries

- Manual/offline-testable runner only.
- No live service calls in tests or CI.
- No MinerU Cloud API provider.
- No LLM extraction or external APIs.
- No fallback, retry, queue, rollback, or scheduler.
- No `DatasetConfirmation` weakening.
- No Phase 1 execution for custom corpora.
- No production dataset admission.
- No human review/admission implementation yet.
