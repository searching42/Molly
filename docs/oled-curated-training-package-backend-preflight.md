# OLED Curated Training Package Backend Preflight

This read-only preflight checks whether curated OLED training package artifacts are ready for later baseline or tabular backend execution.

It loads training package manifests, training row JSONL files, and schema JSON files. It verifies SHA256 values where available, previews feature flattening in memory, checks split and target readiness, and reports optional backend dependency availability. It does not fit models, predict, evaluate, write predictions, write benchmark results, call LLMs, call MinerU, read PDFs, or read images.

## Inputs

The file runner expects:

- training package writer manifest JSON
- training row JSONL files referenced by that manifest
- training package schema JSON referenced by that manifest

The manifest is resolved relative to its parent directory unless `--training-package-base-dir` is provided.

## Relationship To The Training Package Writer

`oled_curated_split_training_package_writer.py` writes ML-ready training rows, schema JSON, and a package manifest.

This backend preflight consumes those artifacts and checks whether they are shaped for backend consumption. Passing this preflight does not run any backend and does not validate benchmark performance.

## Feature Flattening Preview

Feature dictionaries are flattened deterministically in memory:

- booleans become `0.0` or `1.0`
- numbers become floats
- strings become one-hot style keys such as `feature=value`
- lists use indexed keys
- dictionaries use nested keys
- missing or empty values are omitted

This flattening is only a readiness preview. It does not write feature tables or model inputs.

## Backend Readiness Checks

The report includes feature matrix summaries by target property and feature view:

- train, validation, and test row counts
- numeric, nonnumeric, and missing target counts
- raw and flattened feature column counts
- missing feature row counts
- flattened feature column preview

It also reports backend readiness for:

- `tabular_ridge_sklearn`
- `tabular_random_forest_sklearn`

## Optional Dependencies

For sklearn-backed tabular backends, the preflight checks `importlib.util.find_spec("sklearn")`.

If sklearn is unavailable, backend readiness is marked `skipped` with `optional_dependency_unavailable:sklearn`. This is not a hard failure by default. No sklearn model is instantiated, fit, or used for prediction.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_training_package_backend_preflight \
  --training-package-manifest /path/to/training_package_manifest.json \
  --training-package-base-dir /path/to/training_package \
  --output-report /path/to/backend_preflight_report.json
```

Optional filters:

```bash
--backend-kind tabular_ridge_sklearn
--target-property-id eqe_percent
--feature-view full_context
--split train,validation,test
```

Repeated and comma-separated values are supported.

## Output

The output is a deterministic, redacted JSON report with:

- preflight status
- feature matrix summaries
- backend readiness results
- rows by split
- finding taxonomy
- safety metadata

Reports do not include full training rows, full feature dictionaries, raw paper text, parsed JSON, PDFs, images, or absolute local paths.

## Warning

This preflight is only a backend-readiness gate. It does not run baselines, train models, evaluate models, write predictions, register benchmark results, or mark outputs as benchmark validated.
