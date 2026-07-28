# OLED Curated Split Feature Writer

This writer gate materializes aligned OLED split feature rows into deterministic JSONL artifacts after the read-only split feature preflight has passed.

It writes feature row artifacts only. It does not create ML-ready training packages, run baseline backends, train models, evaluate models, call LLMs, call MinerU, read PDFs, or read images.

## Inputs

The combined runner expects:

- curated gold-record JSONL from the curated gold writer gate
- split dataset-view writer manifest
- split dataset-view row JSONL files referenced by that manifest
- split feature preflight report

The split dataset-view manifest is used to load split row artifacts and verify row file SHA256 values when present. The feature preflight report supplies the allowlist of aligned rows.

## Relationship To Preflight

`oled_curated_split_feature_preflight.py` builds feature tables in memory and classifies row alignment:

- `matched`
- `missing_feature_row`
- `ambiguous_feature_row`
- `target_mismatch`

This writer rebuilds feature materialization tables in memory, then writes only rows allowed by the preflight report and policy. By default, only matched rows are written.

## Confirmation

Write mode requires explicit confirmation:

```bash
python -m ai4s_agent.domains.oled_curated_split_feature_writer \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --split-dataset-view-manifest /path/to/split_dataset_view_manifest.json \
  --feature-preflight-report /path/to/split_feature_preflight_report.json \
  --split-dataset-view-base-dir /path/to/split_dataset_views \
  --output-dir /path/to/split_features \
  --output-manifest /path/to/split_feature_manifest.json \
  --confirm-split-feature-write
```

Without `--confirm-split-feature-write`, write mode exits with `confirmation_required:split_feature_write`.

## Dry Run

Dry-run mode performs selection and may write the manifest, but writes no feature JSONL rows:

```bash
python -m ai4s_agent.domains.oled_curated_split_feature_writer \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --split-dataset-view-manifest /path/to/split_dataset_view_manifest.json \
  --feature-preflight-report /path/to/split_feature_preflight_report.json \
  --output-manifest /path/to/split_feature_manifest.json \
  --dry-run
```

## Output Files

Feature row files are written one JSON object per line. File names are deterministic:

```text
oled_split_features__<split>__<target_property_id>__<feature_view>.jsonl
```

Example:

```text
oled_split_features__train__eqe_percent__full_context.jsonl
```

Each row preserves:

- split and split row id
- source record ids
- target property, value, unit, and condition hash
- feature view and feature values
- missing/present feature columns
- evidence refs
- alignment status and reason codes

Rows do not include full gold records, raw paper text, parsed JSON, PDFs, images, or absolute local paths.

## Manifest

The audit manifest records:

- source split dataset-view manifest id
- source feature preflight status
- output file count and row count
- per-file SHA256
- rows by split
- policy
- reason code counts
- safety metadata

Safety metadata explicitly records that no ML-ready training data or backend outputs were produced.

## Missing Feature Policy

Missing feature values are allowed by default because early real-literature extractions can be sparse. Set policy `allow_missing_feature_values=False` to reject matched rows with missing feature columns.

Missing feature rows, ambiguous feature rows, and target mismatches are rejected by default.

## Warning

These artifacts are training-adjacent, not final benchmark-ready training data. A later explicit gate should package final train/validation/test feature tables and run model backends.
