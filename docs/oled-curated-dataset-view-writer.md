# OLED Curated Dataset-View Writer Gate

This module materializes selected OLED dataset-view rows from curated gold records.

Dataset-view rows are analysis artifacts. They are not training data, split datasets, or model-ready benchmark inputs.

## Purpose

Use this layer after:

1. curated gold records have been written by `oled_curated_gold_writer.py`
2. curated gold dataset-view readiness has been checked by `oled_curated_gold_view_preflight.py`

The writer:

- loads curated gold-record JSONL
- optionally loads the curated gold writer manifest
- runs dataset-view preflight
- selects configured view kinds and target properties
- writes deterministic dataset-view row JSONL files under explicit confirmation
- writes an audit manifest with row counts, file SHA256 values, policy, and reason codes

It does not write training data or run split/model workflows.

## Inputs

Required:

- curated gold-record JSONL

Recommended:

- curated gold writer manifest JSON

The manifest allows SHA256 integrity checking before rows are selected for writing.

## Relationship To Preflight

The writer runs `run_oled_curated_gold_view_preflight_from_files()` internally.

If the preflight is invalid and `require_preflight_valid=True`, selection is blocked with reason code `preflight_failed`.

If the preflight has warnings and `allow_preflight_warnings=False`, selection is blocked with reason code `preflight_warnings_present`.

## Confirmation Requirement

Writing row JSONL files requires `--confirm-dataset-view-write`:

```bash
python -m ai4s_agent.domains.oled_curated_dataset_view_writer \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --curated-gold-manifest /path/to/curated_gold_manifest.json \
  --output-dir /path/to/dataset_views \
  --output-manifest /path/to/dataset_view_manifest.json \
  --confirm-dataset-view-write
```

Without confirmation, the CLI exits with `confirmation_required:dataset_view_write`.

## Dry-Run Mode

Dry-run mode runs preflight and selection but writes no row JSONL files:

```bash
python -m ai4s_agent.domains.oled_curated_dataset_view_writer \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --curated-gold-manifest /path/to/curated_gold_manifest.json \
  --output-manifest /path/to/dataset_view_manifest.json \
  --dry-run
```

The manifest records `dry_run_no_rows_written`.

## Output Files

The writer creates one JSONL file per non-empty selected view/target pair.

Filename format:

```text
oled_view__<view_kind>__<target_property_id>.jsonl
```

Example:

```text
oled_view__raw_all_measurements__eqe_percent.jsonl
```

Each line is an `OledCuratedDatasetViewRowArtifact`.

## SHA256 Manifest

Every written row file has an exact-byte SHA256 stored in the writer manifest.

The manifest also records:

- source curated gold SHA256
- source curated gold manifest id
- source preflight status
- output directory
- output file count
- output row count
- status counts
- reason code counts
- writer policy
- safety metadata

## Feature Payload Policy

By default, row artifacts omit feature payloads:

- `features={}`
- `metadata.feature_payload_omitted=True`

This avoids turning dataset-view rows into heavy feature materialization outputs.

Use `--include-feature-payload` only when a downstream audit explicitly needs the in-memory feature context.

## Supported Views And Targets

By default, the writer considers all existing `OledDatasetViewKind` values and the target property ids:

- `eqe_percent`
- `plqy`
- `delta_e_st_ev`

Restrict output with:

```bash
--view-kind raw_all_measurements \
--target-property-id eqe_percent
```

Both options may be repeated or comma-separated.

## Boundary

This writer does not write training data, write ML-ready split datasets, run leakage splits, write feature materialization outputs, run model backends, call LLMs, call MinerU, read PDFs, read images, use OCR, or mutate curated gold files or manifests in place.

`best_reported` keeps the semantics implemented in `oled_dataset_views.py`; it is not reinterpreted as numeric argmax here.
