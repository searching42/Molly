# OLED Curated Split Feature-Materialization Preflight

This read-only harness checks whether curated split dataset-view row artifacts can be aligned with OLED feature materialization rows built in memory.

It does not write feature tables, ML-ready training data, train/validation/test feature files, or model-backend inputs.

## Inputs

- Curated gold-record JSONL
- Split dataset-view writer manifest from `oled_curated_split_dataset_view_writer.py`
- Split dataset-view row JSONL files referenced by that manifest

The split row manifest should include `output_jsonl_path` and `output_sha256` for each written split/view/target file.

## Relationship To Split Dataset-View Writer

Run the split dataset-view writer first:

```bash
python -m ai4s_agent.domains.oled_curated_split_dataset_view_writer \
  --dataset-view-manifest /path/to/dataset_view_manifest.json \
  --split-preflight-report /path/to/split_preflight_report.json \
  --dataset-view-base-dir /path/to/dataset_views \
  --output-dir /path/to/split_dataset_views \
  --output-manifest /path/to/split_dataset_view_manifest.json \
  --confirm-split-dataset-view-write
```

Then run this preflight to check whether those split rows can be matched to feature materialization output in memory.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_split_feature_preflight \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --split-dataset-view-manifest /path/to/split_dataset_view_manifest.json \
  --split-dataset-view-base-dir /path/to/split_dataset_views \
  --output-report /path/to/split_feature_preflight_report.json
```

Optional filters:

```bash
--feature-view molecule_only
--feature-view molecule_interaction
--feature-view full_context
--target-property-id eqe_percent
```

If no feature views are provided, all supported `OledBaselineFeatureView` values are checked.

## Feature Views

The preflight uses existing materialization logic:

- `molecule_only`
- `molecule_interaction`
- `full_context`

Feature tables are created only in memory through `materialize_oled_baseline_feature_table(...)`.

## Alignment Rules

For each split dataset-view row:

1. Match `record_id`.
2. Match `target_property_id`.
3. If `condition_hash` is present, require the same feature row condition hash.
4. If `condition_hash` is absent and exactly one feature row exists for the record/target, use it.
5. If multiple feature rows exist without a disambiguating condition hash, mark the row as ambiguous.

Alignment statuses:

- `matched`
- `missing_feature_row`
- `ambiguous_feature_row`
- `target_mismatch`

## Target Mismatch

If the split row target value or unit differs from the feature row target value or unit, the alignment status is `target_mismatch`.

By default, target mismatches are errors.

## Missing Feature Values

For matched rows, the preflight counts missing feature values. A value is considered missing when it is `None`, an empty string, an empty list, or an empty dict.

Missing feature values are warnings by default because early real-literature records may be sparse. They can be promoted to errors through policy.

## Report

The report includes:

- gold validation error/warning codes
- row alignments
- split/target/feature-view summaries
- alignment status counts
- missing feature column counts
- finding code counts

Reports are redacted and do not include full gold records, full row payloads, raw paper text, PDFs, or images.

## Boundary

This preflight does not:

- write feature tables
- write ML-ready training data
- write split feature files
- run baseline/model backends
- call LLMs or MinerU
- read PDFs or images

Feature table writing and training-data writing remain later explicit gates.
