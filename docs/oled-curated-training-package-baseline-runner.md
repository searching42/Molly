# OLED Curated Training Package Baseline Runner

This writer gate runs controlled baseline models on curated OLED training package artifacts.

It is the first explicit baseline execution step in the Phase 1 real-literature pipeline. It requires explicit confirmation, writes baseline prediction JSONL and metrics JSON artifacts, and emits an audit manifest with SHA256 values. The outputs are baseline-run artifacts only. They are not benchmark-validated results and are not registered benchmark entries.

## Inputs

The combined runner expects:

- training package writer manifest JSON
- training row JSONL files referenced by that manifest
- training package schema JSON referenced by that manifest
- backend preflight report JSON

The backend preflight should come from `oled_curated_training_package_backend_preflight.py`. By default, this runner requires that preflight to be valid.

## Confirmation

Write mode requires explicit confirmation:

```bash
python -m ai4s_agent.domains.oled_curated_training_package_baseline_runner \
  --training-package-manifest /path/to/training_package_manifest.json \
  --backend-preflight-report /path/to/backend_preflight_report.json \
  --training-package-base-dir /path/to/training_package \
  --output-dir /path/to/baseline_run \
  --output-manifest /path/to/baseline_run_manifest.json \
  --confirm-baseline-run
```

Without `--confirm-baseline-run`, write mode exits with `confirmation_required:baseline_run`.

## Dry Run

Dry-run mode assembles the run report and may write the manifest, but writes no prediction or metrics files:

```bash
python -m ai4s_agent.domains.oled_curated_training_package_baseline_runner \
  --training-package-manifest /path/to/training_package_manifest.json \
  --backend-preflight-report /path/to/backend_preflight_report.json \
  --output-manifest /path/to/baseline_run_manifest.json \
  --dry-run
```

## Supported Baselines

The default baseline is:

- `mean_baseline`

It uses the numeric target mean from the train split and predicts that value for train, validation, and test rows.

Optional sklearn baselines are available only when sklearn is installed:

- `tabular_ridge_sklearn`
- `tabular_random_forest_sklearn`

For sklearn baselines, features are flattened deterministically in memory, models are fit only on train rows, and predictions are produced for available splits. If sklearn is unavailable, those baselines are skipped with `optional_dependency_unavailable:sklearn`.

## Output Files

Prediction files are named:

```text
oled_baseline_predictions__<baseline_kind>__<target_property_id>__<feature_view>.jsonl
```

Metrics files are named:

```text
oled_baseline_metrics__<baseline_kind>__<target_property_id>__<feature_view>.json
```

The audit manifest records:

- source training package manifest id
- source backend preflight status
- output file count and prediction count
- per-run prediction and metrics SHA256 values
- policy
- status counts
- reason code counts
- safety metadata

Prediction rows do not include full training rows, full feature dictionaries, raw paper text, parsed JSON, PDFs, images, or absolute local paths.

## Metrics

Regression metrics are computed per split when numeric targets are present:

- row count
- MAE
- RMSE
- R2
- bias
- target mean
- prediction mean

Nonnumeric targets are ignored for metric computation. If the train split has no numeric target rows, the baseline run is blocked.

## Warning

This gate runs baselines, but it does not validate scientific performance, register benchmark results, call LLMs, call MinerU, read PDFs, or read images. Benchmark reporting and registration remain later explicit gates.
