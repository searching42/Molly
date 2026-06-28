# Node45 MinerU Preflight-Bound Live Corpus Acceptance Evidence - 2026-06-28

This document records a redacted evidence summary for a manual preflight-bound
MinerU live corpus acceptance run against the node45 self-hosted endpoint. It
intentionally does not include full preflight artifacts, full live corpus
artifacts, generated PDFs, MinerU bundles, or `ParsedDocument` outputs.

## Scope

The run validated this manual chain:

```text
node45 endpoint preflight
-> preflight_report.json binding
-> live synthetic corpus parsing through self-hosted MinerU
-> corpus conflict audit
-> explicit synthetic DatasetConfirmation gate
-> Phase 1 pipeline
-> corpus report and reproducibility artifacts
```

The endpoint profile was `node45-loopback` through the `manual-primary`
routing policy. The redacted API origin was `http://127.0.0.1:18000`.

## Preflight Run

| Field | Value |
| --- | --- |
| Run ID | `mineru-preflight-node45-20260628-154527` |
| Generated at | `2026-06-28T07:45:27.957941Z` |
| Node | `node45` |
| Decision | `passed` |
| Health status | `healthy` |
| MinerU version | `3.4.0` |
| Protocol version | `2` |
| Endpoint profile | `node45-loopback` |
| Routing policy | `manual-primary` |
| Redacted API origin | `http://127.0.0.1:18000` |

The preflight was executed on node45 using the same module entry point from a
temporary synchronized Molly preflight runtime:

```bash
PYTHONPATH=src python \
  -m ai4s_agent.mineru_endpoint_preflight \
  --profile-config docs/examples/mineru-endpoint-profiles.example.json \
  --policy-name manual-primary \
  --output /tmp/molly-mineru-preflight \
  --run-id mineru-preflight-node45-20260628-154527
```

Node45 environment summary recorded by the preflight:

| Field | Value |
| --- | --- |
| Torch CUDA version | `13.0` |
| Torch CUDA available | `true` |
| CUDA device | `NVIDIA GeForce RTX 5090` |
| CUDA capability | `12.0` |
| NVIDIA driver version | `580.126.20` |
| `nvidia-smi` reported CUDA version | `13.0` |

Preflight warnings:

- `node45_hint_vllm_use_flashinfer_sampler`
- `node45_hint_ld_library_path_cuda_cudnn_ordering`

The full preflight artifact bundle was retained outside git:

```text
/tmp/molly-mineru-preflight/mineru-preflight-node45-20260628-154527/
```

Preflight artifact tarball retained outside git:

```text
/tmp/molly-node45-remote-preflight/mineru-preflight-node45-20260628-154527.tar.gz
```

Preflight artifact SHA-256:

```text
sha256:05071590c3eea76f3a5f169770d2ad320820271662b4f885beef3319558628fa
```

## Bound Live Corpus Run

| Field | Value |
| --- | --- |
| Run ID | `mineru-corpus-live-bound-20260628-154606` |
| Generated at | `2026-06-28T07:46:07.015028Z` |
| Decision | `passed` |
| Phase 1 status | `success` |
| Parser backend | `mineru_api:hybrid-engine` |
| Requested effort | `medium` |
| Synthetic confirmation | `confirmed_by=test-fixture` |
| Top ranked candidate count | `3` |

The live corpus acceptance was run with the node45 preflight report as a hard
pre-parse binding:

```bash
PYTHONPATH=src python \
  -m ai4s_agent.document_parse_corpus_live_acceptance \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-mineru-corpus-acceptance \
  --run-id mineru-corpus-live-bound-20260628-154606 \
  --preflight-report /tmp/molly-node45-remote-preflight/mineru-preflight-node45-20260628-154527/preflight_report.json \
  --preflight-artifact-sha256 sha256:05071590c3eea76f3a5f169770d2ad320820271662b4f885beef3319558628fa \
  --require-preflight-match \
  --confirm-synthetic-dataset \
  --confirmed-by test-fixture \
  --n-bits 64 \
  --topn 3 \
  --min-numeric-ratio 0.5 \
  --min-nonempty 1
```

Preflight binding result:

| Field | Value |
| --- | --- |
| `matched` | `true` |
| `mismatches` | `[]` |
| `require_preflight_match` | `true` |
| Bound preflight run id | `mineru-preflight-node45-20260628-154527` |
| Bound preflight decision | `passed` |
| Bound preflight health status | `healthy` |
| Bound preflight protocol version | `2` |

## Corpus Results

| Metric | Value |
| --- | --- |
| Documents parsed | `3` |
| MinerU parse successes | `3 / 3` |
| MinerU protocol versions | `2, 2, 2` |
| Extracted records | `9` |
| Accepted records | `5` |
| Candidate records | `5` |
| Training records | `5` |
| Rejected records | `5` |
| Consistent duplicate count | `1` |
| Conflict count | `1` |
| Unresolved conflict count | `1` |
| Valid record count before conflict rejection | `8` |

Conflict/rejection summary:

| Reason | Count |
| --- | --- |
| `duplicate_conflict` | `2` |
| `invalid_smiles` | `1` |
| `missing_lambda_em_nm` | `1` |
| `missing_plqy` | `1` |

The corpus workflow detected the expected consistent duplicate and conflict:

- consistent duplicate SMILES: `CCO`
- conflicted SMILES: `CCN`

The live corpus evidence artifact tarball was retained outside git:

```text
/tmp/molly-mineru-corpus-acceptance/mineru-corpus-live-bound-20260628-154606.tar.gz
```

Corpus evidence artifact SHA-256:

```text
sha256:ce2cc4a2a2bdd74069184af169ba959758848cc21a5d4cd1f636f310e028fc2f
```

The tarball contains only selected report, manifest, conflict, and
reproducibility evidence files. It excludes generated PDFs, MinerU bundles,
`ParsedDocument` outputs, and pdfplumber baseline directories.

## Boundaries

- This is manual opt-in evidence, not CI evidence.
- Full preflight artifacts are not committed.
- Full live corpus artifacts are not committed.
- Generated PDFs are not committed.
- MinerU bundles are not committed.
- `ParsedDocument` outputs are not committed.
- pdfplumber baseline outputs are not committed.
- Remote task ids and full parser artifacts are not copied into this document.
- No tokens, auth headers, full sensitive URLs, or private documents
  are included.
- This evidence does not add a MinerU Cloud API provider.
- This evidence does not add automatic fallback, retry, queue, rollback, or
  routing behavior.
- This evidence does not weaken `DatasetConfirmation` or
  manifest-to-training-CSV binding.
