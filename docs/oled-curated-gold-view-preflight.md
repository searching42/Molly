# OLED Curated Gold Dataset-View Preflight

This module checks whether curated OLED gold records can enter the existing dataset-view layer.

It is read-only. It builds dataset views in memory and writes only a redacted preflight report when requested.

## Purpose

Use this layer after the curated gold writer gate has produced:

- curated gold-record JSONL
- curated gold writer manifest JSON

The preflight:

- loads curated `OledGoldDatasetRecord` JSONL
- optionally loads the writer manifest
- verifies JSONL SHA256 against the manifest
- reruns `validate_oled_gold_dataset()`
- calls `build_oled_dataset_view(...)` in memory
- reports per-view row counts, status, finding codes, and safety metadata

It does not write dataset view rows or training data.

## Input Files

The main input is curated gold-record JSONL:

```bash
python -m ai4s_agent.domains.oled_curated_gold_writer \
  --gold-candidates /path/to/gold_candidates.jsonl \
  --output-jsonl /path/to/curated_gold_records.jsonl \
  --output-manifest /path/to/curated_gold_manifest.json \
  --confirm-curated-gold-write
```

The manifest is optional but recommended.

## SHA256 Integrity Check

`sha256_file()` computes the SHA256 of the exact curated JSONL bytes.

`check_oled_curated_gold_manifest_integrity()` returns:

- `matched`: manifest SHA256 matches the JSONL.
- `mismatched`: manifest SHA256 differs from the JSONL.
- `missing_sha256`: manifest has no `output_sha256`.
- `missing_output_path`: manifest has no `output_jsonl_path`.
- `not_provided`: no manifest was provided.

The path comparison is intentionally conservative. The preflight does not require an absolute output path match.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_gold_view_preflight \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --manifest /path/to/curated_gold_manifest.json \
  --output-report /path/to/view_preflight_report.json
```

If `--output-report` is omitted, the CLI prints a compact summary only.

## Supported Dataset Views

By default the preflight runs all `OledDatasetViewKind` values:

- `raw_all_measurements`
- `curated_device_baseline`
- `best_reported`
- `curated_intrinsic`

Rows are built in memory only. The preflight does not persist row payloads.

## Target Properties

Default target properties are:

- `eqe_percent`
- `plqy`
- `delta_e_st_ev`

Python callers can override `OledCuratedGoldViewPreflightPolicy.target_property_ids`.

## Empty Views

An empty view is not automatically a failure. Some target properties are not meaningful for every view kind.

For example:

- `curated_intrinsic` may be empty for device-only properties.
- `best_reported` may be empty when no record is explicitly flagged as best-reported.
- device views may be empty for intrinsic-only targets.

With `include_empty_views=True`, empty views are reported as `passed_with_warnings` with reason code `empty_view`.

With `include_empty_views=False`, empty views fail the preflight.

## Output Report

The JSON report contains:

- overall preflight status
- input record count
- manifest integrity status
- input and manifest SHA256 values
- gold validation error and warning codes
- per-view row counts and statuses
- finding code counts
- safety metadata

It does not include full curated record payloads, raw paper text, absolute local paths, or dataset view rows.

## Boundary

This module does not write dataset view rows, write training data, run leakage splits, run feature materialization outputs, run model backends, call LLMs, call MinerU, read PDFs, read images, use OCR, or mutate curated gold files or manifests in place.

Passing this preflight means the curated records are ready for a later explicit dataset-view materialization step. It does not create ML-ready training data.
