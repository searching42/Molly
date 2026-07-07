# OLED Curated Benchmark Registry Promotion Writer

## Purpose

`oled_curated_benchmark_registry_promotion_writer.py` is the controlled writer gate for local promoted candidate registry artifacts.

It reads candidate registry artifacts that passed promotion-readiness preflight and writes:

- a standalone promoted registry entry JSON file
- a standalone promoted registry index JSONL file
- a promotion writer manifest with SHA256 hashes and gate metadata

Promotion here means `promoted_candidate`. It does not mean benchmark validation, scientific performance validation, publication, or registration in a final/global registry.

## Inputs

The writer reads:

- registry writer manifest JSON
- registry entry JSON
- registry index JSONL
- registry promotion preflight report JSON

The registry entry and index are loaded through the registry writer manifest. SHA256 values are verified by the promotion-preflight loader when present.

## Relationship To Promotion-Readiness Preflight

`oled_curated_benchmark_registry_promotion_preflight.py` is the read-only gate that checks candidate status, source chain, caveats, run/metric counts, safety metadata, and entry/index consistency.

The promotion writer requires that preflight to be valid by default. Warnings are allowed by default but can be rejected by policy.

## Confirmation Requirement

Writing promoted artifacts requires explicit confirmation:

```bash
--confirm-benchmark-registry-promotion-write
```

Without confirmation, the selection API raises:

```text
confirmation_required:benchmark_registry_promotion_write
```

## Dry-Run Mode

`--dry-run` builds the promoted entry and promoted index records in memory. It writes no promoted entry JSON and no promoted index JSONL. If `--output-manifest` is supplied, the manifest can be written with dry-run reason codes.

## Output Files

Default output names:

```text
oled_benchmark_promoted_registry_entry.json
oled_benchmark_promoted_registry_index.jsonl
```

The promotion writer manifest path is supplied explicitly with `--output-manifest`.

## Promoted-Candidate Status

The only supported promotion status is:

```text
promoted_candidate
```

The writer rejects policies or source metadata that attempt to set:

- `benchmark_validated=True`
- `scientific_claim_validated=True`

Every promoted entry preserves the source caveats and records safety metadata.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_benchmark_registry_promotion_writer \
  --registry-writer-manifest /path/to/benchmark_registry_manifest.json \
  --registry-promotion-preflight-report /path/to/benchmark_registry_promotion_preflight_report.json \
  --registry-base-dir /path/to/benchmark_registry \
  --output-dir /path/to/promoted_registry \
  --output-manifest /path/to/promoted_registry_manifest.json \
  --confirm-benchmark-registry-promotion-write
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--entry-only`
- `--index-only`

## Safety Boundary

This writer does not:

- validate benchmark performance
- create scientific conclusions
- publish or register benchmark results
- write final/global registry files
- append to or mutate an existing registry in place
- rerun baseline or model backends
- train, predict, or recompute metrics
- call LLMs or MinerU
- read PDFs or images

Final validated/public registry promotion remains a later explicit gate.
