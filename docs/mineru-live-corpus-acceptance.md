# MinerU Live Corpus Acceptance

This runbook describes the manual, opt-in live acceptance bridge from a real
self-hosted MinerU endpoint into Molly's corpus-level scientific workflow.

It is not part of ordinary CI. It does not run automatically on pull requests,
pushes, schedules, or local test runs.

## Purpose

The single-document live acceptance runner validates one synthetic PDF through
the MinerU task API, bundle download, extraction, normalization, and optional
`pdfplumber` baseline comparison.

The corpus live acceptance runner validates the next integration boundary:

```text
real self-hosted MinerU parsing
-> multiple ParsedDocument outputs
-> corpus-level conflict audit
-> explicit DatasetConfirmation gate
-> Phase 1 full pipeline when confirmed
-> corpus reproducibility report
```

It uses synthetic PDFs only by default. It does not validate production
scientific accuracy, large-corpus throughput, private-document handling, or
model quality.

## Start Self-Hosted MinerU

Example local service:

```bash
mineru-api \
  --host 127.0.0.1 \
  --port 18000 \
  --enable-vlm-preload true
```

Check health:

```bash
curl http://127.0.0.1:18000/health
```

Run Molly's endpoint preflight before parsing:

```bash
python -m ai4s_agent.mineru_endpoint_preflight \
  --profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --policy-name manual-primary \
  --output /tmp/molly-mineru-preflight \
  --run-id "mineru-preflight-$(date +%Y%m%d-%H%M%S)"
```

The preflight checks health reachability, protocol version, redacted endpoint
metadata, and node45-oriented CUDA/vLLM environment diagnostics. It does not
parse PDFs or perform fallback routing.

MinerU model downloads and VLM preload can make the first startup slow. Treat
first startup separately from steady-state acceptance behavior.

## Bind A Preflight Report

Corpus live acceptance can optionally bind a prior `preflight_report.json`
before it submits any parsing task:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-$(date +%Y%m%d-%H%M%S)" \
  --preflight-report /tmp/molly-mineru-preflight/<run-id>/preflight_report.json
```

By default, preflight mismatches are recorded as warnings so existing direct
manual acceptance workflows are not broken. Add `--require-preflight-match` to
make mismatches fail before parsing:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-$(date +%Y%m%d-%H%M%S)" \
  --preflight-report /tmp/molly-mineru-preflight/<run-id>/preflight_report.json \
  --preflight-artifact-sha256 sha256:<artifact-sha256> \
  --require-preflight-match
```

The binding checks that the preflight decision is `passed`, health status is
`healthy` or `ok`, protocol version is `2`, the redacted API origin matches the
resolved acceptance endpoint, and profile/policy names match when profile mode
is used. Invalid or unreadable preflight reports are structured failures. The
acceptance report records only a safe report filename, preflight run id,
decision, health status, protocol version, redacted origin, profile/policy
names, mismatch codes, and the optional artifact SHA-256.

## SSH Tunnel

If MinerU runs on a workstation or GPU node, tunnel the loopback API to the
local machine:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 18000:127.0.0.1:18000 \
  user@node45
```

Then run Molly against:

```text
http://127.0.0.1:18000
```

## Endpoint Profiles

Direct CLI mode remains supported. Existing commands that pass `--api-url`,
`--endpoint-kind`, `--backend`, `--effort`, and timing flags continue to work.

For repeatable manual runs, the live acceptance CLIs can also resolve endpoint
settings from a checked-in example profile or a user-supplied local profile:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-$(date +%Y%m%d-%H%M%S)"
```

The example profile uses only a loopback URL:

```text
http://127.0.0.1:18000
```

This is the recommended shape for node45-style manual runs: start MinerU on the
remote node bound to loopback, establish the SSH tunnel, and keep Molly pointed
at the local loopback endpoint. Do not put real public IPs, host credentials,
tokens, authorization headers, local artifact paths, or generated acceptance
bundles in profile files.

Profile-based unconfirmed corpus run:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --endpoint-profile node45-loopback \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-unconfirmed-$(date +%Y%m%d-%H%M%S)"
```

Profile-based confirmed synthetic run:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-confirmed-$(date +%Y%m%d-%H%M%S)" \
  --confirm-synthetic-dataset \
  --confirmed-by test-fixture \
  --n-bits 64 \
  --topn 3 \
  --min-numeric-ratio 0.5 \
  --min-nonempty 1
```

Explicit CLI flags override profile values. This lets a user keep a stable
profile while overriding a local port, backend, effort, parse method, upload
setting, pdfplumber comparison setting, or timeout for a specific run.

Routing policies are declarative and manual in this PR. A policy can record the
intended primary profile and fallback profile names for operator visibility,
but the runner does not perform automatic live fallback, retry orchestration,
canary routing, rollback, scheduling, or worker-pool dispatch.

Tokens still come only from environment variables:

- `MINERU_API_TOKEN`
- `AI4S_MINERU_API_TOKEN`

Profile files must not contain secrets. Acceptance reports include only a
redacted endpoint profile summary with profile names, routing policy names,
resolved non-secret options, and redacted API origin.

## Run Without Confirmation

This mode parses the corpus and builds candidate/rejected/manifest artifacts,
but it intentionally stops before Phase 1:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --api-url http://127.0.0.1:18000 \
  --endpoint-kind mineru-api \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-$(date +%Y%m%d-%H%M%S)" \
  --backend hybrid-engine \
  --effort medium \
  --allow-remote-upload \
  --compare-pdfplumber
```

Expected decision:

```text
awaiting_confirmation
```

This is the correct result when `--confirm-synthetic-dataset` is absent.

## Run With Synthetic Confirmation

Only pass `--confirm-synthetic-dataset` for the generated synthetic corpus:

```bash
python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --api-url http://127.0.0.1:18000 \
  --endpoint-kind mineru-api \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id "mineru-corpus-live-$(date +%Y%m%d-%H%M%S)" \
  --backend hybrid-engine \
  --effort medium \
  --allow-remote-upload \
  --compare-pdfplumber \
  --confirm-synthetic-dataset \
  --confirmed-by test-fixture
```

Expected decision:

```text
passed
```

That means all MinerU parses succeeded, protocol version 2 was observed,
ParsedDocument outputs were written, the corpus conflict audit detected the
expected duplicate/conflict behavior, the confirmed synthetic dataset reached
Phase 1, and corpus replay/report artifacts exist.

## Output Layout

Each run writes a fresh run-specific directory:

```text
<output>/<run-id>/
  generated_pdfs/
    paper_a.pdf
    paper_b.pdf
    paper_c.pdf
  mineru_bundles/
    paper_a/
    paper_b/
    paper_c/
  parsed_documents/
    paper_a_parsed_document.json
    paper_b_parsed_document.json
    paper_c_parsed_document.json
  pdfplumber_baselines/
    paper_a/
    paper_b/
    paper_c/
  corpus_workflow/
    corpus_workflow_report.json
    corpus_conflict_report.json
    candidate_dataset.csv
    training_dataset.csv
    rejected_records.json
    dataset_manifest.json
    corpus_lineage_manifest.json
    corpus_replay_manifest.json
    corpus_reproducibility_report.json
    corpus_report.json
    corpus_report.md
  acceptance_report.json
  acceptance_summary.md
```

Existing non-empty run directories are rejected. Use a fresh `--run-id` for
reruns so evidence cannot be silently reused.

## Confirmation Boundary

There is no automatic dataset confirmation.

- `--confirm-synthetic-dataset` is required to enter Phase 1.
- `--confirmed-by` must be non-empty when confirmation is enabled.
- Without confirmation, parsing and corpus dataset construction may succeed,
  but the acceptance decision is `awaiting_confirmation`.
- Future custom/private PDFs must not be auto-confirmed.

The Phase 1 manifest-to-training-CSV binding remains enforced by the existing
Phase 1 pipeline.

## Credentials

The CLI never accepts API tokens as command-line arguments. Tokens are read only
from:

- `MINERU_API_TOKEN`
- `AI4S_MINERU_API_TOKEN`

Tokens and authorization headers must not appear in stdout, stderr,
`acceptance_report.json`, or `acceptance_summary.md`.

## Troubleshooting

`failed` with `invalid_api_url`:

- Use an HTTP/HTTPS origin such as `http://127.0.0.1:18000`.
- Do not include userinfo, query strings, fragments, or tokens in the URL.

`failed` with `unsupported_protocol_version`:

- The endpoint did not report MinerU protocol version 2.
- Confirm the self-hosted service is the expected task API implementation.

`failed` with `mineru_parse_failed`:

- Check MinerU server logs.
- Verify the SSH tunnel is still alive.
- Increase `--task-timeout-sec` and `--max-poll-attempts` for slow cold starts.

`failed` with `expected_conflict_missing`:

- MinerU parsed the PDFs but did not preserve enough table structure for the
  corpus conflict audit to detect the synthetic duplicate/conflict evidence.
- Inspect `parsed_documents/` and `mineru_bundles/`.

`failed` with `invalid_preflight_report`:

- The path passed to `--preflight-report` was missing, unreadable, or not a
  valid `mineru_endpoint_preflight.v1` report.
- Re-run `python -m ai4s_agent.mineru_endpoint_preflight` and pass the new
  `preflight_report.json`.

`failed` with `preflight_match_failed`:

- `--require-preflight-match` was enabled and the preflight report did not
  match the acceptance endpoint, protocol, health status, or selected
  profile/policy.
- Inspect `preflight_binding.mismatches` in `acceptance_report.json`.

`awaiting_confirmation`:

- This is expected when `--confirm-synthetic-dataset` is absent.
- Re-run with `--confirm-synthetic-dataset --confirmed-by test-fixture` only
  for the generated synthetic corpus.

## Boundaries

This runner does not:

- add a MinerU Cloud API provider
- change MinerU provider protocol behavior
- change ZIP extraction logic
- change `ParsedDocument` schema
- modify Phase 3 extraction
- modify dataset builder logic
- modify Phase 1 model internals
- weaken `DatasetConfirmation`
- bypass manifest-to-training-CSV binding
- add live CI dependencies
- call live MinerU in tests
- use LLMs or external APIs
- change queued-canary, retry, rollback, or worker queue behavior
- add UI or API routes
