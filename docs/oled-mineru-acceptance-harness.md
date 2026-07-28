# OLED MinerU Parsed-Output Acceptance Harness

This harness runs a read-only offline smoke test over local MinerU parsed-output bundles. It is intended to audit whether parsed JSON/MD sidecars can flow through:

```text
MinerU parsed outputs
-> OledMineruCandidate
-> OledSchemaCandidate
-> proposed OledLayeredRecord candidates
-> validation / summary report
```

It does not produce curated data.

## Manifest Format

```json
{
  "manifest_id": "oled-mineru-smoke-001",
  "bundles": [
    {
      "paper_id": "paper-003",
      "content_list_path": "paper-003_content_list.json",
      "content_list_v2_path": "paper-003_content_list_v2.json",
      "md_path": "paper-003.md",
      "source_label": "ncomms5016"
    }
  ]
}
```

Paths are resolved relative to the manifest file. JSON inputs must be parsed-output files such as `content_list` or `content_list_v2`. Markdown is optional and is used only as nearby context.

## Command

```bash
PYTHONPATH=src:. python -m ai4s_agent.domains.oled_mineru_acceptance_harness \
  --manifest /path/to/manifest.json \
  --output-report /path/to/report.json \
  --confirm-read-only-parsed-outputs
```

Without `--output-report`, the CLI prints a compact JSON summary to stdout. The confirmation flag is required for any run.

## What It Does

- loads manifest-listed parsed JSON and optional MD sidecars
- extracts deterministic MinerU evidence candidates
- maps evidence candidates to intermediate schema candidates
- compiles schema candidates to proposed layered-record candidates
- aggregates counts, statuses, representative anchors, and finding codes
- writes a redacted JSON acceptance report when requested

## What It Does Not Do

- does not call MinerU
- does not call LLMs
- does not read PDFs
- does not read or inspect images
- does not create gold records
- does not write curated datasets
- does not run model backends
- does not add external dependencies

## Report Fields

- `mineru_candidate_count`: number of deterministic evidence candidates extracted from parsed JSON.
- `semantic_candidate_count`: number of intermediate `OledSchemaCandidate` objects produced by deterministic mapping.
- `compiled_record_count`: number of proposed layered-record candidates produced by the compiler.
- `compiled_status_counts`: counts by compiler status such as `compiled`, `partial`, `needs_review`, and `rejected`.
- `finding_code_counts`: aggregate taxonomy of warning/error/reason codes from semantic mapping, compilation, and harness-level checks.
- `metadata.metadata_key_counts`: audit counts for metadata keys emitted by per-paper results, intended to catch uncontrolled metadata growth.

Reports intentionally include record ids, counts, statuses, finding codes, and representative evidence anchors. They do not include full paper text or full compiled layered-record payloads.

## Recommended First Smoke Test

Start with 1-5 OLED papers whose MinerU parsed outputs are already present locally. Run the harness in read-only mode, inspect the report manually, and treat the output as an acceptance diagnostic only. Do not treat it as a curated dataset or gold benchmark.
