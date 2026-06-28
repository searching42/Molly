# Node45 MinerU Live Corpus Acceptance Evidence - 2026-06-28

This document records a redacted evidence summary for the manual MinerU live
corpus acceptance run on node45. It intentionally does not include generated
PDFs, MinerU bundles, parsed document JSON, or the full acceptance artifact
bundle.

## Run Summary

| Field | Value |
| --- | --- |
| Run ID | `mineru-corpus-live-confirmed-20260628-101009` |
| Date | 2026-06-28 |
| Generated at | `2026-06-28T02:10:10.376875Z` |
| Node | `node45` |
| SSH target used | `workstation2` (`211.86.155.63`, hostname reported as `node45`) |
| Endpoint kind | `mineru-api` |
| MinerU version | `3.4.0` |
| Protocol version | `2` |
| Requested backend | `hybrid-engine` |
| Requested effort | `medium` |
| Dataset confirmation | `--confirm-synthetic-dataset --confirmed-by test-fixture` |
| Decision | `passed` |

## Acceptance Metrics

| Metric | Value |
| --- | ---: |
| Document count | 3 |
| Parse success count | 3 |
| Phase 1 status | `success` |
| Extracted record count | 9 |
| Accepted record count | 5 |
| Rejected record count | 5 |
| Candidate record count | 5 |
| Training record count | 5 |
| Consistent duplicate count | 1 |
| Conflict count | 1 |
| Unresolved conflict count | 1 |
| Top ranked candidate count | 3 |

All three live MinerU parses reported `ok=true` and `protocol_version="2"` in
the acceptance report:

| Document | Parse result | Protocol version | ParsedDocument |
| --- | --- | --- | --- |
| `paper_a` | `ok=true` | `2` | `parsed_documents/paper_a_parsed_document.json` |
| `paper_b` | `ok=true` | `2` | `parsed_documents/paper_b_parsed_document.json` |
| `paper_c` | `ok=true` | `2` | `parsed_documents/paper_c_parsed_document.json` |

The report recorded no acceptance errors.

## Artifact Evidence

The complete run artifact bundle was retained outside git:

```text
/tmp/molly-mineru-corpus-acceptance/mineru-corpus-live-confirmed-20260628-101009/
```

A local tarball was created outside git for bundle integrity tracking:

```text
/tmp/molly-mineru-corpus-acceptance/mineru-corpus-live-confirmed-20260628-101009.tar.gz
```

Artifact bundle SHA-256:

```text
ca932fa7fd93146edd4fd864e54c77f5b1433584abe3e927cddbbe8ceaefd84f
```

The bundle contained the runner-generated evidence, including:

- `acceptance_report.json`
- `acceptance_summary.md`
- `generated_pdfs/`
- `mineru_bundles/`
- `parsed_documents/`
- `pdfplumber_baselines/`
- `corpus_workflow/reproducibility/corpus_replay_manifest.json`
- `corpus_workflow/reproducibility/corpus_reproducibility_report.json`
- `corpus_workflow/report/corpus_report.json`
- `corpus_workflow/report/corpus_report.md`

Only this redacted summary is committed. The generated PDFs, MinerU bundles,
ParsedDocument outputs, and full artifact bundle are not committed.

## Environment Notes

The service was started from the `mineru34` conda environment on node45. The
working launch required two environment safeguards:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This avoids a vLLM FlashInfer sampler startup failure on the node45 RTX 5090
GPU (`SM 12.0`).

The launch also had to put the `mineru34` NVIDIA runtime libraries before the
system CUDA libraries in `LD_LIBRARY_PATH`. Without this, the service loaded
`/usr/local/cuda-12.4` cuDNN libraries while the environment's PyTorch build was
CUDA 13.0, and live parsing stalled after a cuDNN symbol error.

The successful service health check reported:

```json
{
  "status": "healthy",
  "version": "3.4.0",
  "protocol_version": 2
}
```

## Known Limitations

- This evidence validates the live integration path only:
  MinerU task API -> ParsedDocument -> corpus audit -> confirmed synthetic
  dataset -> Phase 1 -> report/replay artifacts.
- The source PDFs are synthetic fixtures. This run is not production scientific
  accuracy evidence.
- This run is not a large-corpus throughput benchmark.
- The runner uses explicit synthetic dataset confirmation. It does not justify
  auto-confirming future private or custom corpora.
- The artifact storage location above is a local external path, not a durable
  object-store location.
- The recorded bundle SHA-256 identifies the local tarball created for this
  evidence capture; the full bundle is intentionally excluded from git.
