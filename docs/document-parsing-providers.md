# Document Parsing Providers

This document defines the current provider-layer direction for document
parsing. It does not change public routes, queued-canary behavior, or the
existing Phase 3 workflow defaults by itself.

## Architecture

```text
Planner / Agent / future MCP wrapper
            |
            v
   DocumentParseService
      |            |
      v            v
MinerU API     pdfplumber
primary        baseline
```

Current policy:

- `mineru_api` is the preferred primary parser when it is explicitly configured
  and remote upload is authorized.
- `pdfplumber` is the deterministic local baseline parser.
- `auto` may select MinerU only when a MinerU API-compatible base URL is
  configured and upload policy permits it.
- No silent provider fallback occurs after a MinerU parsing failure.

## Provider Contract

The provider layer exposes a stable request/result contract through
`src/ai4s_agent/document_parse_provider.py`.

Request scope:

- PDF input only
- explicit `output_dir`
- explicit provider selection
- conservative parsing defaults:
  - `parse_method="auto"`
  - `backend="hybrid-engine"`
  - `effort="medium"`
  - formula/table parsing enabled
  - image analysis disabled
  - `allow_remote_upload=false`

The request schema is JSON-safe, rejects unknown fields, and does not allow
shell commands or argv injection.

The result schema is also stable and JSON-safe. It records:

- provider identity
- parser backend
- parsed `ParsedDocument`
- output artifact references
- remote task id when applicable
- warnings
- structured error metadata
- redacted audit fields

Authorization values, tokens, and raw request headers are intentionally kept
out of the stable result.

## MinerU API Provider

`src/ai4s_agent/mineru_api_client.py` implements a direct MinerU FastAPI-style
task client using the asynchronous task endpoints:

- `GET /health`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/result`

This same client contract can target:

- a configured API-compatible managed endpoint
- a local `mineru-api`
- a local or remote `mineru-router`

Current safety rules:

- base URL is always explicit configuration
- no hardcoded hosted endpoint
- no automatic network use without an API URL
- loopback HTTP is allowed
- non-loopback upload requires `allow_remote_upload=true`
- non-loopback plain HTTP requires an explicit insecure development override
- no automatic resubmission on failure or timeout
- bounded polling only

GPU scaling, router orchestration, and multi-GPU deployment remain outside
Molly. They belong to MinerU or `mineru-router`.

## Output Normalization Policy

`src/ai4s_agent/mineru_output_normalizer.py` treats official-style MinerU
outputs as follows:

1. `content_list.json` is the primary structured reading-order source.
2. `middle.json` is retained for page/layout/backend/version provenance.
3. Markdown is preserved for human and future LLM interpretation.
4. `content_list_v2.json` is optional and secondary.
5. Markdown-only fallback is allowed only when structured content is absent,
   not malformed.

Normalization preserves:

- one-based Molly page numbers
- bbox data and its MinerU 0-1000 coordinate system
- deterministic element ids
- text/title/list/code/equation blocks
- tables with caption, footnote, raw HTML retention, page, bbox, and reading
  order
- source PDF hash
- MinerU backend and version metadata

Markdown is preserved as an artifact, but it is not the sole machine-readable
source of truth.

An LLM is not used to parse table HTML or validate numeric values in this
provider layer.

## pdfplumber Baseline

`src/ai4s_agent/document_parse_pdfplumber.py` wraps the existing
`parse_document_pdfplumber_adapter` to provide a deterministic local baseline.

It is intended for:

- local debugging
- reproducible baseline comparison
- control-plane verification when MinerU is unavailable

It does not call MinerU and does not use the network.

## Manual CLI

`src/ai4s_agent/document_parse_cli.py` provides a manual operator entrypoint.

Example local baseline:

```bash
python -m ai4s_agent.document_parse_cli \
  --provider pdfplumber \
  --input /path/to/paper.pdf \
  --output /path/to/output \
  --run-id manual-pdfplumber-baseline
```

Example MinerU API usage:

```bash
python -m ai4s_agent.document_parse_cli \
  --provider mineru-api \
  --input /path/to/paper.pdf \
  --output /path/to/output \
  --run-id manual-mineru-smoke \
  --api-url http://127.0.0.1:8000 \
  --backend hybrid-engine \
  --effort medium
```

For non-loopback uploads, `--allow-remote-upload` is required.

The CLI:

- emits JSON to stdout
- writes human diagnostics to stderr
- exits 0 only on successful parse
- reads API tokens from environment variables
- does not use queue execution
- does not mutate project routes
- does not call an LLM

## Baseline Benchmark

`src/ai4s_agent/document_parse_benchmark.py` provides a deterministic benchmark
report shape.

Current benchmark interpretation:

- pdfplumber metrics are a real local baseline on a generated synthetic PDF
- MinerU fixture metrics validate bundle discovery and normalization protocol
  only
- the fixed MinerU fixture is not evidence of live MinerU parsing accuracy

The synthetic fixture lives under `tests/fixtures/document_parse_provider/`.

## Deployment Policy

- Small explicit API jobs:
  - configured MinerU API-compatible endpoint
- Private/local documents:
  - local `mineru-api`
- Batch or multi-GPU:
  - `mineru-router`
- Baseline/debug:
  - `pdfplumber`

No automatic external upload and no silent fallback are part of the current
design.

## Not Yet Included

This provider-layer PR does not:

- call a live MinerU service in CI
- install MinerU models or GPU dependencies in Molly
- introduce an MCP server
- perform web search or live paper acquisition
- connect parsing directly to automatic dataset confirmation
- connect parsing directly to Phase 1 training
- change existing public Flask routes
- change `/api/run-plan/execute`
- change `/api/run-plan/resume`

## Next Step

The next parsing milestone should be:

1. a manual live MinerU acceptance run against an explicitly configured
   endpoint, then
2. a narrow Phase 3 parsed-document -> confirmed-dataset -> Phase 1 bridge that
   uses this provider layer without changing route defaults.
