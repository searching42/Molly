# MinerU Manual Live Acceptance Gate

This runbook defines a reusable manual release gate for validating a
self-hosted MinerU-compatible endpoint against Molly's synthetic live corpus
acceptance path.

The first known-good reference instance is node45, but the process is not
node45-specific. Operators can reuse the same gate on other GPU hosts when
their service exposes the expected MinerU task API and `/health` behavior.

## Purpose

The gate validates that a running self-hosted MinerU endpoint can support
Molly's integration path:

```text
endpoint profile
-> endpoint preflight diagnostics
-> preflight report artifact
-> preflight-bound synthetic live corpus acceptance
-> corpus conflict audit
-> explicit synthetic DatasetConfirmation gate
-> Phase 1 pipeline
-> redacted evidence summary
```

This is integration evidence for Molly's parser-to-corpus workflow. It is not
production scientific accuracy evidence, a throughput benchmark, or a private
document handling validation.

## What This Is / Is Not

This is:

- a manual, opt-in, operator-run validation flow
- scriptable by shell commands
- reusable for any compatible self-hosted GPU MinerU service
- designed to produce redacted evidence suitable for code review
- designed to keep full artifacts outside git

This is not:

- ordinary CI
- automatic deployment
- a MinerU installer
- CUDA or cuDNN configuration management
- model download automation
- service startup, supervision, or daemon management
- MinerU Cloud API support
- fallback, retry, rollback, queueing, scheduling, or canary routing
- environment repair for node-specific runtime issues

Operators remain responsible for installing MinerU, configuring CUDA/cuDNN,
downloading models, starting the service, and keeping the service healthy.

## Required Operator Inputs

Before running the gate, prepare:

- a running self-hosted MinerU-compatible endpoint
- a loopback or SSH-tunneled HTTP endpoint, for example
  `http://127.0.0.1:18000`
- a local endpoint profile based on
  `docs/examples/mineru-endpoint-profiles.example.json`
- optional API token supplied only through `MINERU_API_TOKEN` or
  `AI4S_MINERU_API_TOKEN`
- a clean preflight output directory
- a clean corpus acceptance output directory
- fresh run ids for preflight and corpus acceptance
- external artifact storage outside git
- a plan for retaining tarballs and SHA-256 values outside git

Do not put tokens, auth headers, private home paths, generated artifact paths,
or public IPs into checked-in profile files.

## Recommended Generic Workflow

1. Start the MinerU service on the GPU host.
2. Confirm `/health` manually with `curl`.
3. Create or adapt a local endpoint profile.
4. Run `python -m ai4s_agent.mineru_endpoint_preflight`.
5. Tar/gzip the preflight artifact directory outside git.
6. Compute SHA-256 for the preflight artifact bundle.
7. Run `python -m ai4s_agent.document_parse_corpus_live_acceptance` with:
   - `--endpoint-profile-file`
   - `--routing-policy` or `--endpoint-profile`
   - `--preflight-report`
   - `--preflight-artifact-sha256`
   - `--require-preflight-match`
   - `--confirm-synthetic-dataset`
   - `--confirmed-by test-fixture`
8. Tar/gzip the corpus acceptance artifact directory outside git.
9. Compute SHA-256 for the corpus artifact bundle.
10. Commit only a redacted evidence summary.

Use
`docs/evidence/templates/mineru-preflight-bound-live-corpus-evidence-template.md`
as the starting point for the redacted evidence summary.

## Example Endpoint And Tunnel

Local loopback health check:

```bash
curl http://127.0.0.1:18000/health
```

SSH tunnel from a local machine to a GPU host:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 18000:127.0.0.1:18000 \
  operator@gpu-host
```

After the tunnel is up, Molly should still use the local loopback origin:

```text
http://127.0.0.1:18000
```

## Example Profile

Start from `docs/examples/mineru-endpoint-profiles.example.json` and keep the
profile local if it contains machine-specific details. A reusable profile shape
looks like:

```json
{
  "schema_version": "mineru_endpoint_profiles.v1",
  "profiles": [
    {
      "name": "local-loopback",
      "api_url": "http://127.0.0.1:18000",
      "endpoint_kind": "mineru-api",
      "backend": "hybrid-engine",
      "effort": "medium",
      "parse_method": "auto",
      "allow_remote_upload": true,
      "compare_pdfplumber": true,
      "expected_protocol_version": "2",
      "health_path": "/health"
    }
  ],
  "routing_policies": [
    {
      "name": "manual-primary",
      "default_profile": "local-loopback",
      "fallback_profiles": [],
      "mode": "manual"
    }
  ]
}
```

The routing policy is declarative. It records operator intent; it does not
perform automatic fallback.

## Run Preflight

```bash
PREFLIGHT_RUN_ID="mineru-preflight-$(date +%Y%m%d-%H%M%S)"
PREFLIGHT_OUT="/tmp/molly-mineru-preflight"

PYTHONPATH=src python -m ai4s_agent.mineru_endpoint_preflight \
  --profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --policy-name manual-primary \
  --output "$PREFLIGHT_OUT" \
  --run-id "$PREFLIGHT_RUN_ID"
```

Expected preflight outputs:

```text
$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID/preflight_report.json
$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID/preflight_summary.md
```

## Package Preflight Artifacts Outside Git

Linux:

```bash
tar -C "$PREFLIGHT_OUT" \
  -czf "$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID.tar.gz" \
  "$PREFLIGHT_RUN_ID/preflight_report.json" \
  "$PREFLIGHT_RUN_ID/preflight_summary.md"

sha256sum "$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID.tar.gz"
```

macOS:

```bash
tar -C "$PREFLIGHT_OUT" \
  -czf "$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID.tar.gz" \
  "$PREFLIGHT_RUN_ID/preflight_report.json" \
  "$PREFLIGHT_RUN_ID/preflight_summary.md"

shasum -a 256 "$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID.tar.gz"
```

Record the SHA-256 as:

```text
sha256:<64-hex-digest>
```

Do not commit the tarball or full preflight report.

## Run Preflight-Bound Corpus Acceptance

```bash
CORPUS_RUN_ID="mineru-corpus-live-bound-$(date +%Y%m%d-%H%M%S)"
CORPUS_OUT="/tmp/molly-mineru-corpus-acceptance"
PREFLIGHT_REPORT="$PREFLIGHT_OUT/$PREFLIGHT_RUN_ID/preflight_report.json"
PREFLIGHT_SHA256="sha256:<preflight-artifact-sha256>"

PYTHONPATH=src python -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output "$CORPUS_OUT" \
  --run-id "$CORPUS_RUN_ID" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --preflight-artifact-sha256 "$PREFLIGHT_SHA256" \
  --require-preflight-match \
  --confirm-synthetic-dataset \
  --confirmed-by test-fixture \
  --n-bits 64 \
  --topn 3 \
  --min-numeric-ratio 0.5 \
  --min-nonempty 1
```

`--confirm-synthetic-dataset` is valid only for Molly's generated synthetic
corpus. Do not use it to auto-confirm private or custom documents.

## Package Corpus Artifacts Outside Git

Package a reviewable evidence bundle outside git. Include report, manifest,
conflict, Phase 1 summary, and reproducibility files. Exclude generated PDFs,
MinerU bundles, `ParsedDocument` outputs, and pdfplumber baselines.

Example:

```bash
tar -C "$CORPUS_OUT" \
  -czf "$CORPUS_OUT/$CORPUS_RUN_ID.tar.gz" \
  "$CORPUS_RUN_ID/acceptance_report.json" \
  "$CORPUS_RUN_ID/acceptance_summary.md" \
  "$CORPUS_RUN_ID/corpus_workflow/corpus_workflow_report.json" \
  "$CORPUS_RUN_ID/corpus_workflow/conflicts/corpus_conflict_report.json" \
  "$CORPUS_RUN_ID/corpus_workflow/conflicts/conflict_summary.json" \
  "$CORPUS_RUN_ID/corpus_workflow/dataset/dataset_manifest.json" \
  "$CORPUS_RUN_ID/corpus_workflow/phase1/full_phase1_pipeline.json" \
  "$CORPUS_RUN_ID/corpus_workflow/report/corpus_report.json" \
  "$CORPUS_RUN_ID/corpus_workflow/report/corpus_report.md" \
  "$CORPUS_RUN_ID/corpus_workflow/reproducibility/corpus_replay_manifest.json" \
  "$CORPUS_RUN_ID/corpus_workflow/reproducibility/corpus_reproducibility_report.json"
```

Linux:

```bash
sha256sum "$CORPUS_OUT/$CORPUS_RUN_ID.tar.gz"
```

macOS:

```bash
shasum -a 256 "$CORPUS_OUT/$CORPUS_RUN_ID.tar.gz"
```

Do not commit the tarball or full acceptance report.

## Pass Criteria

The gate passes only when all of these are true:

- preflight decision is `passed`
- health status is `healthy` or `ok`
- protocol version is `2`
- preflight artifact SHA-256 is recorded
- live corpus acceptance decision is `passed`
- `preflight_binding.matched` is `true`
- `preflight_binding.mismatches` is empty
- all MinerU parses are `ok`
- all MinerU parse protocol versions are `2`
- Phase 1 status is `success`
- expected consistent duplicate evidence is present
- expected conflict evidence is present
- corpus artifact SHA-256 is recorded
- full artifacts are retained outside git
- only a redacted markdown evidence summary is committed

## Fail / Rerun Criteria

Rerun or investigate before accepting the gate if any of these occur:

- `/health` is unreachable
- health status is not `healthy` or `ok`
- protocol version is missing or not `2`
- preflight report is invalid or missing
- `preflight_binding` mismatches when `--require-preflight-match` is set
- `ParsedDocument` output is missing
- any MinerU parse fails
- Phase 1 does not run when synthetic confirmation is expected
- expected duplicate/conflict evidence is missing
- preflight artifact SHA-256 is missing or invalid
- corpus artifact SHA-256 is missing or invalid
- secrets, auth headers, remote task ids, public IPs, or private paths appear
  in the redacted evidence summary

Do not weaken the gate by removing `--require-preflight-match` for release
evidence. A warning-only binding can be useful during debugging, but it is not
the formal gate.

## Redaction Checklist

Before committing evidence, verify that the committed markdown does not include:

- tokens
- auth headers
- full `preflight_report.json`
- full `acceptance_report.json`
- generated PDFs
- MinerU bundles
- `ParsedDocument` outputs
- pdfplumber baselines
- private home paths
- public IPs unless explicitly intended and reviewed
- remote task ids
- complete artifact tarballs

Commit only a redacted markdown summary.

## Evidence Template

Copy this template and fill in only redacted values:

```text
docs/evidence/templates/mineru-preflight-bound-live-corpus-evidence-template.md
```

Recommended committed evidence path:

```text
docs/evidence/mineru-live-corpus-bound-preflight-<host-label>-<date>.md
```

Use a local or redacted host label. Do not encode private hostnames, user names,
public IP addresses, or private paths in the filename unless they are intended
for review.

## Next Boundary: Custom Corpus Intake

This manual gate validates infrastructure using Molly-generated synthetic PDFs.
Real, custom, public, private, or mixed PDF corpora require a separate intake
contract before any dry-run runner is implemented.

See:

```text
docs/custom-corpus-intake-contract.md
```

No custom or private corpus should use `--confirm-synthetic-dataset`. Future
custom corpus dry-runs must remain unconfirmed until human review and a
separate dataset admission mechanism exist.

## Relationship To Existing Docs

- `docs/mineru-live-corpus-acceptance.md` describes individual live corpus
  acceptance commands and troubleshooting.
- `docs/mineru-endpoint-preflight.md` describes endpoint preflight diagnostics.
- This runbook combines those pieces into a reusable manual gate.
- Node45 evidence under `docs/evidence/` is the first known-good reference
  instance, not a requirement that other operators use node45.
