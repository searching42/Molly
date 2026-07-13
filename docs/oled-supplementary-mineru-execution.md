# OLED Supplementary MinerU Execution

## Purpose

This stage follows the human-confirmed supplementary parser preflight. It is
the first stage in the supplementary recovery chain that may contact MinerU
and parse PDF content.

The runner consumes a content-bound parser-preflight artifact, a separately
confirmed execution manifest, a named endpoint profile, and a passed endpoint
preflight report. It parses the full approved supplementary source and writes
a redacted execution audit. It does not resolve the table/figure locator or
regenerate data candidates.

## Execution Manifest

The operator provides a local manifest like:

```json
{
  "schema_version": "oled_supplementary_mineru_execution_manifest.v1",
  "run_id": "paper016-si-mineru-001",
  "paper_id": "paper016",
  "preflight_plan_digest": "sha256:<parser-preflight-plan-digest>",
  "execution_confirmed": true,
  "reviewed_by": "reviewer-03",
  "reviewed_at": "2026-07-13T10:00:00Z",
  "endpoint_profile_name": "node45-loopback",
  "endpoint_preflight_sha256": "sha256:<endpoint-preflight-report-hash>",
  "sources": [
    {
      "source_id": "paper016-si-v1",
      "local_pdf_path": "<operator-local-path>/paper016_si.pdf"
    }
  ]
}
```

The source mappings must exactly cover the sources in the parser preflight.
The manifest is operator-local input and must not be committed. It cannot add
a source, omit a source, change the paper, or substitute a different preflight
digest.

## Endpoint Binding

The endpoint preflight report must:

- have a passed decision and healthy status;
- match the configured profile name, redacted origin, endpoint kind, backend,
  effort, parse method, upload policy, timeouts, protocol version, and health
  path; and
- hash exactly to `endpoint_preflight_sha256` in the execution manifest.

The runner resolves the named profile directly. It does not use a routing
fallback. The execution profile must use an origin-only API URL so the
redacted endpoint identity fully binds the destination. Immediately before
parsing the runner performs another live health check and requires the
configured protocol version.

## Source and Upload Binding

Before network access, every approved PDF is copied through one
`O_NOFOLLOW` file descriptor into a new run-scoped snapshot. The copy must
retain the preflight byte size, PDF envelope, and SHA-256. Source symlinks are
rejected.

`DocumentParseRequest.expected_source_pdf_sha256` binds the MinerU client to
the approved hash. The client computes SHA-256 from the same bytes placed in
the multipart upload and rejects a mismatch before sending a request. The
successful parser audit must report that exact upload hash.

## Fixed Parse Settings

The supplementary runner always uses:

- provider: `mineru_api` (never `auto`);
- the backend, effort, and parse method from the bound endpoint profile;
- full-source parsing with no inferred page range;
- formula parsing enabled;
- table parsing enabled; and
- image analysis disabled.

A provider fallback, changed source hash, protocol mismatch, output reference
outside the isolated source directory, symlinked output, or missing required
normalized outputs changes the execution result to `failed`.

## CLI

Run an endpoint preflight first, then calculate the report SHA-256 and prepare
the execution manifest. Execute with:

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_supplementary_mineru_execution \
  --preflight-artifact runs/<run_id>/review/oled_supplementary_parser_preflight.json \
  --execution-manifest <operator-local-path>/supplementary_mineru_execution_manifest.json \
  --endpoint-profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --endpoint-preflight-report <operator-local-path>/preflight_report.json \
  --output-root <operator-local-output-root>
```

API tokens are read only from `MINERU_API_TOKEN` or
`AI4S_MINERU_API_TOKEN`. They are not accepted as CLI arguments.

## Output Layout

Each execution requires a fresh run directory:

```text
<output-root>/<run-id>/
  supplementary_mineru_execution.json
  sources/
    <source-id>/
      approved_source.pdf
      mineru/
        <run-id>-<source-id>_parsed_document.json
        <run-id>-<source-id>_parsed_document.md
        <run-id>-<source-id>_parser_audit.json
        mineru_bundle/
          ...
```

The audit artifact contains only source IDs, hashes, page/byte counts, approved
target locators, parser/profile metadata, status, warning codes, and hashes of
known output files. It contains no local source path, output path, PDF bytes,
raw parsed text, table cells, credentials, or remote task ID.

If one source fails, later sources are marked `skipped` and are not submitted.
An existing run directory is never reused or overwritten.

## Boundary After Execution

A successful execution means the approved supplementary PDF was parsed by the
bound MinerU endpoint. It does not mean `Supplementary Table S1` was found.
The next stage must consume the execution audit and local parser outputs,
resolve each explicit locator conservatively, and produce a human-review
packet. No result from this stage is eligible for automatic candidate merge,
reviewed evidence staging, gold conversion, device-only admission, or dataset
writing.
