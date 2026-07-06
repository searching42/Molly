# OLED Curated Benchmark Registry Writer

## Purpose

`oled_curated_benchmark_registry_writer.py` is the controlled writer gate for local OLED benchmark candidate registry artifacts.

It converts a benchmark candidate report that passed registry-readiness preflight into:

- a standalone registry entry JSON file
- a standalone registry index JSONL file
- an audit manifest with SHA256 hashes and gate metadata

The artifacts remain candidate registry artifacts. They are not benchmark-validated records and do not make scientific performance claims.

## Inputs

The writer reads:

- benchmark report writer manifest JSON
- benchmark candidate report JSON
- benchmark candidate report Markdown
- benchmark registry preflight report JSON

The benchmark report JSON and Markdown are loaded through the report writer manifest. File SHA256 values are verified when present.

## Relationship To Registry Preflight

The writer uses `oled_curated_benchmark_registry_preflight.py` as the readiness gate. By default, the registry preflight report must be valid. Warnings are allowed by default but can be rejected through policy.

The writer does not rerun baseline backends, recompute metrics, or perform final benchmark validation.

## Confirmation Requirement

Writing registry artifacts requires explicit confirmation:

```bash
--confirm-benchmark-registry-write
```

Without confirmation, the selection API raises:

```text
confirmation_required:benchmark_registry_write
```

## Dry-Run Mode

`--dry-run` builds the registry entry and index records in memory. It writes no registry entry JSON and no registry index JSONL. If `--output-manifest` is supplied, the manifest can be written with dry-run reason codes.

## Output Files

Default output names:

```text
oled_benchmark_registry_entry.json
oled_benchmark_registry_index.jsonl
```

The manifest path is supplied explicitly with `--output-manifest`.

## Candidate Registry Status

The only supported registry status is:

```text
candidate
```

The writer rejects policies or source metadata that attempt to set:

- `benchmark_validated=True`
- `scientific_claim_validated=True`

Every registry entry preserves caveats from the candidate benchmark report and records safety metadata.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_benchmark_registry_writer \
  --benchmark-report-manifest /path/to/benchmark_report_manifest.json \
  --benchmark-registry-preflight-report /path/to/benchmark_registry_preflight_report.json \
  --benchmark-report-base-dir /path/to/benchmark_report \
  --output-dir /path/to/benchmark_registry \
  --output-manifest /path/to/benchmark_registry_manifest.json \
  --confirm-benchmark-registry-write
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
- append to a global registry in place
- rerun baseline or model backends
- train models or predict
- call LLMs or MinerU
- read PDFs or images

Promotion, final validation, and publication remain later explicit gates.
