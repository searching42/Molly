# OLED Curated Split Dataset-View Writer

This writer materializes split-assigned OLED dataset-view row artifacts after the read-only leakage-split preflight has passed.

The output is still not ML-ready training data. It is split-specific dataset-view row JSONL plus an audit manifest.

## Inputs

- Dataset-view writer manifest from `oled_curated_dataset_view_writer.py`
- Dataset-view row JSONL files referenced by that manifest
- Split preflight report from `oled_curated_dataset_split_preflight.py`

The split preflight report supplies the authoritative row-to-split assignments. Rows without a valid `assigned` status are not materialized by default.

## Relationship To Split Preflight

Run the split preflight first:

```bash
python -m ai4s_agent.domains.oled_curated_dataset_split_preflight \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --dataset-view-manifest /path/to/dataset_view_manifest.json \
  --dataset-view-base-dir /path/to/dataset_views \
  --output-report /path/to/split_preflight_report.json
```

Then run this writer only after inspecting the preflight report.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_split_dataset_view_writer \
  --dataset-view-manifest /path/to/dataset_view_manifest.json \
  --split-preflight-report /path/to/split_preflight_report.json \
  --dataset-view-base-dir /path/to/dataset_views \
  --output-dir /path/to/split_dataset_views \
  --output-manifest /path/to/split_dataset_view_manifest.json \
  --confirm-split-dataset-view-write
```

At least one output path is required. The confirmation flag is required unless `--dry-run` is used.

## Dry Run

```bash
python -m ai4s_agent.domains.oled_curated_split_dataset_view_writer \
  --dataset-view-manifest /path/to/dataset_view_manifest.json \
  --split-preflight-report /path/to/split_preflight_report.json \
  --output-manifest /path/to/split_dataset_view_manifest.json \
  --dry-run
```

Dry-run mode performs selection and may write the manifest, but writes no split row JSONL files.

## Output Naming

The writer creates one JSONL file per `(split, view_kind, target_property_id)` group:

```text
oled_split_view__train__raw_all_measurements__eqe_percent.jsonl
oled_split_view__validation__raw_all_measurements__eqe_percent.jsonl
oled_split_view__test__raw_all_measurements__eqe_percent.jsonl
```

Each row preserves:

- original dataset-view `row_id`
- deterministic `split_row_id`
- split assignment
- source record ids and source record splits
- target value, unit, layer, condition hash, and dedup key hash
- evidence refs and confidence score

## Manifest

The audit manifest records:

- source dataset-view manifest id
- source split preflight status
- output file count and row count
- rows by split
- per-file SHA256 hashes
- policy and reason-code counts
- safety metadata

The manifest does not include full row payloads, gold records, raw paper text, PDFs, or images.

## Feature Payload Policy

Feature payloads are omitted by default:

```text
include_feature_payload = false
```

When omitted, row artifacts keep `features={}` and mark `metadata.feature_payload_omitted=true`.

Use `--include-feature-payload` only when the row-level analysis context is explicitly needed. This still does not run feature materialization writers or model backends.

## Boundary

This writer does not:

- write ML-ready training data
- write feature tables
- run feature materialization output writers
- run leakage split generation
- run model backends
- call LLMs or MinerU
- read PDFs or images

Feature materialization and training-data writers remain later explicit gates.
