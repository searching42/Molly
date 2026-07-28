# OLED Curated Baseline Benchmark Report Writer

This controlled writer gate materializes reviewable OLED baseline benchmark candidate reports from baseline run artifacts that have passed benchmark-readiness preflight. The outputs are candidate reports only. They are not benchmark registration records and do not validate scientific performance.

## Inputs

- Baseline run manifest JSON from `oled_curated_training_package_baseline_runner.py`
- Prediction JSONL files referenced by completed baseline run results
- Metrics JSON files referenced by completed baseline run results
- Benchmark preflight report JSON from `oled_curated_baseline_benchmark_preflight.py`

The combined runner uses the existing baseline artifact loaders, including SHA256 checks when hashes are present in the baseline run manifest.

## Relationship To Benchmark Preflight

Benchmark preflight is the read-only gate that checks prediction coverage, split coverage, evidence refs, benchmark-validation claims, and metric consistency. This writer consumes that preflight result as the gate for producing candidate report artifacts. If the preflight failed, the writer rejects report materialization by default. Preflight warnings are allowed by default, but can be rejected with policy.

## Confirmation Requirement

The writer requires explicit confirmation before writing report JSON or Markdown artifacts:

```bash
--confirm-benchmark-report-write
```

Dry-run mode builds the report object and may write a manifest, but does not write benchmark report JSON or Markdown files.

## Output Files

Default report filenames:

- `oled_baseline_benchmark_candidate_report.json`
- `oled_baseline_benchmark_candidate_report.md`

The manifest records the policy, output file count, per-file SHA256 values, source manifest id, source preflight status, and safety metadata.

## JSON Report Structure

The JSON report contains:

- source baseline run manifest id
- source benchmark preflight status
- selected baseline kinds, target properties, feature views, and splits
- input prediction and metric counts
- run cards with artifact hashes and split row counts
- metric cards with reported metric values and preflight status/reason codes
- preflight coverage, metric, and finding counts
- caveats that the report is not benchmark validated

The JSON report does not include full prediction rows, feature dictionaries, raw paper text, benchmark registry entries, or benchmark-validated claims.

## Markdown Report Structure

The Markdown report contains:

- source ids and preflight status
- caveats
- a run table
- a metric table
- an explicit boundary statement that the report is not a benchmark registration record and does not validate scientific performance

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_baseline_benchmark_report_writer \
  --baseline-run-manifest /path/to/baseline_run_manifest.json \
  --benchmark-preflight-report /path/to/benchmark_preflight_report.json \
  --baseline-run-base-dir /path/to/baseline_run \
  --output-dir /path/to/benchmark_report \
  --output-manifest /path/to/benchmark_report_manifest.json \
  --confirm-benchmark-report-write
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--json-only`
- `--markdown-only`

## Safety Boundary

This gate does not register benchmark results, write benchmark-validated registry entries, rerun baseline/model backends, train models, predict, recompute new metrics beyond summarizing preflight-checked metrics, call LLMs or MinerU, or read PDFs/images.
