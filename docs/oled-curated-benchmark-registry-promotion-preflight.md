# OLED Curated Benchmark Registry Promotion Preflight

## Purpose

`oled_curated_benchmark_registry_promotion_preflight.py` is a read-only readiness check for OLED benchmark candidate registry artifacts.

It verifies whether local candidate registry artifacts are internally consistent and ready for a later promotion/final-validation gate. Passing this preflight does not promote, validate, publish, or register a benchmark.

## Inputs

The preflight reads:

- registry writer manifest JSON
- registry entry JSON
- registry index JSONL

The entry and index artifacts are discovered from the registry writer manifest.

## Relationship To Registry Writer

The registry writer creates standalone candidate artifacts:

- `oled_benchmark_registry_entry.json`
- `oled_benchmark_registry_index.jsonl`
- registry writer manifest JSON

This preflight checks those artifacts after writing. It does not mutate them and does not append to any global registry.

## SHA256 Checks

When the writer manifest includes SHA256 values for the registry entry or index artifacts, the loader verifies the exact bytes on disk.

SHA mismatch errors use these prefixes:

```text
benchmark_registry_entry_sha256_mismatch:
benchmark_registry_index_sha256_mismatch:
```

## Entry And Index Consistency Checks

The preflight checks that:

- the registry entry JSON is present when required
- the registry index JSONL is present when required
- the entry status is `candidate`
- index record statuses are `candidate`
- the entry id appears in the index
- a single-entry index is used by default
- run-card and metric-card counts are nonzero when required

Multiple index records can be allowed for local inspection with:

```bash
--allow-multiple-index-records
```

## Source Chain Checks

The preflight checks source chain identifiers including:

- source benchmark report manifest id
- source benchmark registry preflight status
- source candidate report id

The source registry preflight status must be `passed` or `passed_with_warnings` by default.

## Candidate-Only Status Checks

Registry artifacts remain candidate artifacts. The preflight rejects source metadata that claims:

- `benchmark_validated=True`
- `scientific_claim_validated=True`

Required caveats include:

```text
baseline_candidate_report_only
not_benchmark_validated
not_scientific_performance_claim
```

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_benchmark_registry_promotion_preflight \
  --registry-writer-manifest /path/to/benchmark_registry_manifest.json \
  --registry-base-dir /path/to/benchmark_registry \
  --output-report /path/to/benchmark_registry_promotion_preflight_report.json
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`

## Safety Boundary

This preflight does not:

- promote benchmark registry entries
- mark outputs benchmark validated
- validate scientific claims
- write final or global registry files
- rerun baseline or model backends
- train, predict, or recompute metrics
- call LLMs or MinerU
- read PDFs or images

Promotion and final validation remain later explicit gates.
