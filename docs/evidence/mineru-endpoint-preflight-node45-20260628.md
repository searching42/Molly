# Node45 MinerU Endpoint Preflight Evidence - 2026-06-28

This document records a redacted evidence summary for a manual MinerU endpoint
preflight run on node45. It intentionally does not include the full
`preflight_report.json`, `preflight_summary.md`, or the artifact bundle.

## Run Summary

| Field | Value |
| --- | --- |
| Run ID | `mineru-preflight-node45-20260628-144842` |
| Date | 2026-06-28 |
| Generated at | `2026-06-28T06:48:42.982175Z` |
| Node | `node45` |
| SSH target used | `workstation2` (hostname reported as `node45`) |
| Endpoint profile | `node45-loopback` |
| Routing policy | `manual-primary` |
| Redacted API origin | `http://127.0.0.1:18000` |
| Endpoint kind | `mineru_api` |
| Health path | `/health` |
| Decision | `passed` |

The run used the PR #150 preflight source synchronized to a temporary node45
directory because node45 did not have a Molly checkout at the time of evidence
capture. The command executed the same module entry point:

```bash
PYTHONPATH=src python \
  -m ai4s_agent.mineru_endpoint_preflight \
  --profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --policy-name manual-primary \
  --output /tmp/molly-mineru-preflight \
  --run-id mineru-preflight-node45-20260628-144842
```

## Health Check

| Metric | Value |
| --- | --- |
| HTTP status | `200` |
| Health status | `healthy` |
| MinerU version | `3.4.0` |
| Protocol version | `2` |
| Response schema valid | `true` |
| Elapsed seconds | `0.09738082299008965` |

The report recorded no preflight errors.

## Environment Diagnostics

| Field | Value |
| --- | --- |
| Python environment | `mineru34` conda environment |
| Torch version | `2.11.0+cu130` |
| Torch CUDA version | `13.0` |
| Torch CUDA available | `true` |
| CUDA device | `NVIDIA GeForce RTX 5090` |
| CUDA capability | `12.0` |
| `nvidia-smi` available | `true` |
| NVIDIA driver version | `580.126.20` |
| `nvidia-smi` reported CUDA version | `13.0` |
| `CUDA_HOME` | not set in the preflight shell |
| `LD_LIBRARY_PATH` | empty in the preflight shell |
| `VLLM_USE_FLASHINFER_SAMPLER` | not set in the preflight shell |

Warnings recorded by the preflight report:

- `node45_hint_vllm_use_flashinfer_sampler`
- `node45_hint_ld_library_path_cuda_cudnn_ordering`

These warnings indicate that the preflight shell did not inherit the launch
environment safeguards documented in the live corpus evidence. The successful
MinerU service itself was already running and reachable at the loopback
endpoint.

## Artifact Evidence

The complete preflight artifact bundle was retained outside git on node45:

```text
/tmp/molly-mineru-preflight/mineru-preflight-node45-20260628-144842/
```

A tarball was created outside git for bundle integrity tracking:

```text
/tmp/molly-mineru-preflight/mineru-preflight-node45-20260628-144842.tar.gz
```

Artifact bundle SHA-256:

```text
3f944b48bb75daf97401eee16ac785984882559181f054e553345642b9948e36
```

The bundle contained:

- `preflight_report.json`
- `preflight_summary.md`

Only this redacted summary is committed. The full preflight artifact bundle is
not committed.

## Boundaries

- This evidence validates the endpoint preflight path only:
  endpoint profile -> `/health` -> protocol/schema check -> environment
  diagnostics -> report artifacts.
- It does not parse PDFs.
- It does not run MinerU task submission.
- It does not download or extract MinerU result bundles.
- It does not run corpus acceptance.
- It does not invoke Phase 3 extraction or Phase 1 training.
- It does not change routing, fallback, retry, rollback, queue, or
  `DatasetConfirmation` behavior.
- It does not justify auto-confirming future private or custom corpora.
