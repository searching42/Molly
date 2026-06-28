# MinerU Endpoint Preflight

This runbook describes Molly's manual MinerU endpoint preflight check. It is a
small health and environment diagnostic command intended to run before the
single-document or corpus live acceptance runners.

It is not part of ordinary CI. Tests use mocked HTTP and monkeypatched
environment diagnostics.

## What It Checks

The preflight command resolves an endpoint through the MinerU endpoint profile
configuration from `src/ai4s_agent/mineru_endpoint_profiles.py`, then checks the
resolved health endpoint:

- HTTP reachability
- HTTP status code
- health response is a JSON object
- `status` is present
- MinerU version fields such as `version`, `version_name`, or `_version_name`
- `protocol_version`
- protocol version matches the expected value, normally `2`

It writes:

- `preflight_report.json`
- `preflight_summary.md`

The report records only the redacted API origin. It does not record API tokens,
Authorization headers, userinfo, query strings, URL fragments, or complete
sensitive URLs.

## Environment Diagnostics

The report also records node45-oriented environment hints:

- `VLLM_USE_FLASHINFER_SAMPLER`
- `CUDA_HOME`
- `LD_LIBRARY_PATH` ordering
- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- CUDA device name and capability when available
- optional `nvidia-smi` driver and reported CUDA version

The node45 live acceptance evidence showed two important launch safeguards:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

and ensuring the active `mineru34` environment's NVIDIA runtime libraries appear
before stale system CUDA/cuDNN libraries in `LD_LIBRARY_PATH`.

## Example

```bash
python -m ai4s_agent.mineru_endpoint_preflight \
  --profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --policy-name manual-primary \
  --output /tmp/molly-mineru-preflight \
  --run-id "mineru-preflight-$(date +%Y%m%d-%H%M%S)"
```

Direct CLI mode is also supported:

```bash
python -m ai4s_agent.mineru_endpoint_preflight \
  --api-url http://127.0.0.1:18000 \
  --endpoint-kind mineru-api \
  --expected-protocol-version 2 \
  --output /tmp/molly-mineru-preflight \
  --run-id "mineru-preflight-$(date +%Y%m%d-%H%M%S)"
```

## Output Layout

```text
<output>/<run-id>/
  preflight_report.json
  preflight_summary.md
```

Existing non-empty run directories are rejected. Use a fresh `--run-id` for
each manual check.

## Decisions

`passed` means the health endpoint was reachable, returned HTTP 200, produced a
valid schema, and reported the expected protocol version.

`failed` means the profile was invalid, health was unreachable, HTTP status was
not 200, schema fields were missing, or protocol version did not match.

Environment issues are recorded as warnings unless they prevent running the
command itself. They are diagnostic hints, not automatic routing or fallback
logic.

## Boundaries

This preflight does not:

- add a MinerU Cloud API provider
- parse PDFs
- download MinerU result ZIPs
- run corpus acceptance
- invoke Phase 3 extraction
- invoke Phase 1
- modify `DatasetConfirmation`
- perform automatic fallback
- enqueue work
- trigger rollback
- change queued-canary behavior
- call live services in tests
- use LLMs or external APIs
