# OLED Curated Promoted Registry Publication Preflight

## Purpose

`oled_curated_promoted_registry_publication_preflight.py` is a read-only publication-readiness preflight for local promoted candidate registry artifacts.

Promoted artifacts remain local review artifacts with `promotion_status="promoted_candidate"`. Passing this preflight does not publish results, mutate a global registry, validate benchmark performance, or create scientific conclusions.

## Inputs

The preflight reads:

- promotion writer manifest JSON
- promoted registry entry JSON
- promoted registry index JSONL

The promoted entry and index are discovered through the promotion writer manifest. SHA256 values are verified when present.

## Relationship To Promotion Writer

`oled_curated_benchmark_registry_promotion_writer.py` writes the local promoted candidate registry entry, promoted index, and writer manifest.

This publication preflight checks those artifacts before any later final/public registry writer is allowed to exist. It does not write final registry files.

## SHA256 Checks

For manifest file results with status `written`, the loader:

- resolves relative artifact paths against the provided base directory
- verifies promoted entry JSON SHA256 when available
- verifies promoted index JSONL SHA256 when available
- raises `promoted_registry_entry_sha256_mismatch:` or `promoted_registry_index_sha256_mismatch:` on mismatch

## Promoted Entry And Index Checks

The main preflight checks that:

- the promoted entry is present when required
- the promoted index is present when required
- the promoted entry status is `promoted_candidate`
- the promoted index record status is `promoted_candidate`
- the promoted entry is referenced by the promoted index
- a single promoted index record is present by default
- run-card and metric-card counts are nonzero when required
- required caveats are present

The CLI flag `--allow-multiple-promoted-index-records` relaxes the single-record check.

## Source Chain Checks

The report verifies source identifiers needed by a later publication gate:

- source registry writer manifest id
- source registry entry id
- source registry promotion preflight status
- source candidate report id
- source benchmark report manifest id

Promotion preflight status must be `passed` or `passed_with_warnings` by default.

## Safety Boundary

The preflight rejects metadata or payloads that claim:

- `benchmark_validated=True`
- `scientific_claim_validated=True`
- benchmark publication
- benchmark registration
- final/global registry mutation

It also checks for raw prediction payload keys, feature dictionaries, raw text fields, and absolute local paths.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_promoted_registry_publication_preflight \
  --promotion-writer-manifest /path/to/promoted_registry_manifest.json \
  --promoted-registry-base-dir /path/to/promoted_registry \
  --output-report /path/to/promoted_registry_publication_preflight_report.json
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--allow-multiple-promoted-index-records`

## What This Does Not Do

This preflight does not:

- publish benchmark registry entries
- write final/global registry files
- append to or mutate an existing registry
- mark outputs benchmark validated
- validate scientific performance claims
- rerun baseline or model backends
- train, predict, or recompute metrics
- call LLMs or MinerU
- read PDFs or images

The final/public registry writer remains a later explicit gate.
