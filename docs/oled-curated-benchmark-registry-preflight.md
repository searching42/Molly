# OLED Curated Benchmark Registry Preflight

This read-only gate checks whether OLED benchmark candidate report artifacts are ready for a later benchmark registry writer. Passing this preflight does not register benchmark results, mark outputs as benchmark validated, or validate scientific conclusions.

## Inputs

- Benchmark report writer manifest JSON
- Benchmark candidate report JSON
- Benchmark candidate report Markdown

The file runner loads the manifest, resolves the JSON and Markdown artifacts from manifest file results, verifies SHA256 values when present, and runs registry-readiness checks in memory.

## Relationship To Benchmark Report Writer

The benchmark report writer creates reviewable candidate report artifacts from baseline run outputs that passed benchmark-readiness preflight. Registry preflight consumes those report artifacts and checks whether they still preserve the benchmark boundary before a later explicit registry writer is allowed to materialize registry entries.

## SHA256 Checks

For each written report artifact referenced by the manifest:

- JSON report SHA mismatches raise `benchmark_report_json_sha256_mismatch:`
- Markdown report SHA mismatches raise `benchmark_report_markdown_sha256_mismatch:`

Missing report artifacts raise their corresponding loader errors.

## Caveat Checks

By default the report must contain:

- `baseline_candidate_report_only`
- `not_benchmark_validated`
- `not_scientific_performance_claim`

Missing caveats are hard errors because registry readiness depends on preserving the distinction between candidate reports and benchmark-validated records.

## Source Chain Checks

The preflight checks:

- source baseline run manifest id is present
- source benchmark preflight status is present
- source benchmark preflight status is registry-ready
- manifest source ids match candidate report source ids
- report has run cards and metric cards for the selected filters

## Markdown Consistency

The Markdown report must include the candidate report id, required caveats, and a safety statement indicating that the report is not a benchmark registration record. The Markdown loader and preflight reject raw payload markers such as `raw_text`, `full_text`, `features`, `prediction_id`, and `training_row_id`.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_benchmark_registry_preflight \
  --benchmark-report-manifest /path/to/benchmark_report_manifest.json \
  --benchmark-report-base-dir /path/to/benchmark_report \
  --output-report /path/to/benchmark_registry_preflight_report.json
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--allow-missing-markdown`

## Safety Boundary

This preflight does not register benchmark results, write benchmark registry entries, mark outputs as benchmark validated, rerun baseline/model backends, train models, predict, call LLMs or MinerU, or read PDFs/images.
