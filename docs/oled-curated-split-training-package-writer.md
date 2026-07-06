# OLED Curated Split Training Package Writer

This writer gate materializes OLED split feature row artifacts into an ML-ready training package directory after the read-only training-package preflight has passed.

It writes training row JSONL artifacts, a package schema JSON, and an audit manifest. It does not run baseline backends, train models, evaluate models, call LLMs, call MinerU, read PDFs, or read images. The output is not benchmark-validated.

## Inputs

The combined runner expects:

- split feature writer manifest JSON
- split feature row JSONL files referenced by that manifest
- training-package preflight report JSON

The split feature manifest is used to load feature row artifacts and verify row file SHA256 values when present. The preflight report is used as the readiness gate for split, target, feature schema, missingness, evidence refs, duplicate rows, and manifest integrity.

## Relationship To Preflight

`oled_curated_split_training_package_preflight.py` checks that split feature rows are ready to become a training package. It is read-only.

This writer consumes the preflight report and selects only rows that satisfy the writer policy. If the preflight failed, the default policy rejects the write. If the preflight has warnings, the default policy allows writing while preserving the warning state in the manifest.

## Confirmation

Write mode requires explicit confirmation:

```bash
python -m ai4s_agent.domains.oled_curated_split_training_package_writer \
  --split-feature-manifest /path/to/split_feature_manifest.json \
  --training-preflight-report /path/to/training_package_preflight_report.json \
  --split-feature-base-dir /path/to/split_features \
  --output-dir /path/to/training_package \
  --output-manifest /path/to/training_package_manifest.json \
  --confirm-training-package-write
```

Without `--confirm-training-package-write`, write mode exits with `confirmation_required:training_package_write`.

## Dry Run

Dry-run mode performs selection and may write the manifest, but writes no training rows or schema file:

```bash
python -m ai4s_agent.domains.oled_curated_split_training_package_writer \
  --split-feature-manifest /path/to/split_feature_manifest.json \
  --training-preflight-report /path/to/training_package_preflight_report.json \
  --output-manifest /path/to/training_package_manifest.json \
  --dry-run
```

## Output Layout

Training row files are grouped by split, target property, and feature view:

```text
oled_training_rows__<split>__<target_property_id>__<feature_view>.jsonl
```

Example:

```text
oled_training_rows__train__eqe_percent__full_context.jsonl
```

The package schema is written as:

```text
oled_training_schema.json
```

Each training row preserves:

- split and source row ids
- record id and source record ids
- target property, value, and unit
- feature view and feature object
- condition hash and confidence score
- evidence refs
- safety metadata

Rows do not include full gold records, raw paper text, parsed JSON, PDFs, images, or absolute local paths.

## Schema JSON

The schema JSON records:

- target property ids
- feature views
- splits
- target columns
- feature columns
- metadata columns
- inferred feature column kinds
- required columns

Feature values remain JSON objects so downstream package consumers can flatten them according to the selected modeling backend.

## Manifest

The audit manifest records:

- source split feature manifest id
- source training preflight status
- output file count and row count
- per-file SHA256 values
- rows by split, target, and feature view
- policy
- safety metadata

Safety metadata records that no baseline backend or model backend was run, and that the package is not benchmark-validated.

## Warning

This gate writes ML-ready training package artifacts, but it does not run baselines, train models, evaluate models, register benchmark results, or validate benchmark performance. Those actions remain separate explicit gates.
