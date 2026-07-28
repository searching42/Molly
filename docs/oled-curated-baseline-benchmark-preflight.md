# OLED Curated Baseline Benchmark Preflight

This read-only gate checks whether OLED baseline run artifacts are ready for a later benchmark-reporting step. It verifies the run manifest, prediction JSONL files, and metrics JSON files without running baselines, training models, writing benchmark results, or marking outputs as benchmark validated.

## Inputs

- Baseline run manifest JSON from `oled_curated_training_package_baseline_runner.py`
- Prediction JSONL artifacts referenced by completed run results
- Metrics JSON artifacts referenced by completed run results

The loader verifies SHA256 values when present in the manifest and rejects PDF/image inputs, raw feature payload leakage, and obvious absolute path leakage in metadata.

## Relationship to the Baseline Runner

The baseline runner is the execution gate that may fit/predict controlled baselines and write prediction and metrics artifacts. This benchmark preflight only reads those outputs and checks consistency. Passing this preflight does not validate scientific performance and does not register benchmark results.

## Checks

- Completed, skipped, and failed run status handling
- Missing prediction or metrics artifacts
- Prediction and metrics SHA256 integrity
- Duplicate prediction ids
- Missing predictions, targets, and evidence refs
- Source claims of `benchmark_validated=True`
- Train/evaluation split coverage
- Reported metric row counts and values

## Metric Recalculation

The preflight recomputes deterministic regression metrics from existing predictions:

- row count
- MAE
- RMSE
- R2
- bias
- target mean
- prediction mean

Only rows with numeric `y_true` and `y_pred` are used. Recomputed values are rounded to six decimals and compared with the reported metrics using the configured tolerance.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_baseline_benchmark_preflight \
  --baseline-run-manifest /path/to/baseline_run_manifest.json \
  --baseline-run-base-dir /path/to/baseline_run \
  --output-report /path/to/benchmark_preflight_report.json
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--split`

Each option may be repeated or comma-separated.

## Output

The report includes coverage summaries, metric consistency summaries, finding counts, and safety metadata. It does not include full prediction payloads, feature dictionaries, raw paper text, benchmark results, or full source artifacts.

## Boundary

This gate does not register benchmark results, write benchmark-validated reports, rerun baseline/model backends, train models, predict, evaluate new metrics beyond deterministic consistency recomputation, call LLMs or MinerU, or read PDFs/images.
