# MinerU Live API Acceptance

This runbook describes the opt-in live acceptance lane for Molly's document
parsing provider layer. It validates the MinerU task API protocol and Molly's
normalization path against a synthetic PDF, then compares the result with the
local `pdfplumber` baseline.

It is not part of ordinary CI. It does not run automatically on pull requests,
pushes, or schedules.

## What It Proves

A passing run proves:

- Molly can submit one synthetic PDF to an explicitly configured MinerU
  API-compatible endpoint.
- The endpoint follows the task API contract used by `MinerUApiClient`.
- The returned ZIP bundle can be downloaded, safely extracted, discovered, and
  normalized into `ParsedDocument`.
- The same synthetic PDF can be parsed with `pdfplumber` as a local baseline.
- A redacted machine-readable evidence package can be attached to a milestone
  review.

It does not prove:

- performance on real papers
- large-batch throughput
- scientific extraction quality on production literature
- authorization to upload private documents
- Phase 3 extraction, conflict handling, dataset confirmation, or Phase 1
  training readiness

## Local mineru-api

Start a local service:

```bash
mineru-api \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-vlm-preload true
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Run acceptance:

```bash
python -m ai4s_agent.document_parse_live_acceptance \
  --api-url http://127.0.0.1:8000 \
  --endpoint-kind mineru-api \
  --output /tmp/molly-mineru-acceptance \
  --run-id mineru-live-smoke \
  --backend hybrid-engine \
  --effort medium \
  --compare-pdfplumber
```

## Local mineru-router

Start a local router:

```bash
mineru-router \
  --host 127.0.0.1 \
  --port 8002 \
  --local-gpus auto \
  --enable-vlm-preload true
```

Run acceptance:

```bash
python -m ai4s_agent.document_parse_live_acceptance \
  --api-url http://127.0.0.1:8002 \
  --endpoint-kind mineru-router \
  --output /tmp/molly-mineru-router-acceptance \
  --run-id mineru-router-live-smoke \
  --backend hybrid-engine \
  --effort medium \
  --compare-pdfplumber
```

## Dependencies And Startup Cost

The runner generates a small synthetic PDF with ReportLab. MinerU itself is not
installed or started by Molly.

MinerU model downloads and VLM preload can make the first run slow. Treat first
startup separately from steady-state task API behavior.

Recommended defaults:

- `backend=hybrid-engine`
- `effort=medium`
- `parse_method=auto`
- table parsing enabled
- image analysis disabled

Use high effort only for explicit higher-cost manual runs.

## Credentials And Remote Upload

The CLI never accepts API tokens as command-line arguments. Tokens are read only
from:

- `MINERU_API_TOKEN`
- `AI4S_MINERU_API_TOKEN`

Tokens and authorization headers must not appear in stdout, stderr,
`acceptance_report.json`, or `acceptance_summary.md`.

For non-loopback endpoints, pass `--allow-remote-upload` only after confirming
the document is safe to upload. The generated fixture is synthetic, but this
flag is still explicit so the operator flow matches real document policy.

## Output Layout

Each run writes a fresh run-specific directory:

```text
<output>/<run-id>/
  synthetic_source.pdf
  mineru/
    <run-id>-mineru_parsed_document.json
    <run-id>-mineru_parsed_document.md
    <run-id>-mineru_parser_audit.json
    mineru_bundle/
      ...
  pdfplumber/
    <run-id>-pdfplumber_parsed_document.json
    <run-id>-pdfplumber_parser_audit.json
  acceptance_report.json
  acceptance_summary.md
```

Existing non-empty run directories are rejected. Use a fresh `--run-id` for
reruns so old evidence cannot be silently reused.

## Report Decisions

`passed` means the live MinerU protocol and normalization smoke passed the
configured thresholds.

`needs_review` means parsing succeeded, but one or more comparison or
provenance thresholds missed.

`failed` means the service contract failed, the task failed or timed out, the
bundle could not be safely extracted, normalization failed, source hash or audit
evidence was absent, or the expected table was completely missing.

The report compares MinerU and `pdfplumber` across individual fields:

- provider success states
- parser backends
- elapsed times
- page and table counts
- normalized text-token recall
- header match rate
- row-count match
- simple cell exact-match rate
- provenance completeness
- warning counts

It intentionally avoids a single aggregate accuracy score.

## Evidence To Inspect

Attach or inspect:

- `acceptance_report.json`
- `acceptance_summary.md`
- MinerU Markdown
- MinerU `content_list.json`
- MinerU `content_list_v2.json`
- MinerU `middle.json`
- MinerU `ParsedDocument`
- MinerU parser audit
- `pdfplumber` baseline parsed document and audit

Use this evidence to distinguish:

- service/protocol acceptance
- parser-quality comparison
- real scientific extraction acceptance

Only the first two are covered by this runner.

## Boundaries

This runner does not:

- run in ordinary CI
- perform live paper acquisition
- call an LLM
- add an MCP server
- confirm datasets
- trigger Phase 1 training
- change public Flask routes
- change queued-canary behavior
- enable remote workers
- migrate storage to SQLite
