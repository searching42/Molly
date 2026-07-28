# OLED Curated Dataset Leakage-Split Preflight

This read-only harness checks whether curated OLED gold records and curated dataset-view row artifacts can be assigned to leakage-safe splits.

It does not write split datasets, training data, feature materialization outputs, or model-backend inputs.

## Inputs

- Curated gold-record JSONL from `oled_curated_gold_writer.py`
- Dataset-view writer manifest from `oled_curated_dataset_view_writer.py`
- Dataset-view row JSONL files referenced by that manifest
- Optional curated gold writer manifest for SHA256 integrity checks

The dataset-view manifest should contain one written file result per view/target pair, including `output_jsonl_path` and `output_sha256`.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_dataset_split_preflight \
  --curated-gold-jsonl /path/to/curated_gold_records.jsonl \
  --dataset-view-manifest /path/to/dataset_view_manifest.json \
  --dataset-view-base-dir /path/to/dataset_views \
  --output-report /path/to/split_preflight_report.json
```

Optional filters:

```bash
--split-name train --split-name validation --split-name test
--leakage-group-kind molecule_inchikey
--leakage-group-kind paper_evidence
--leakage-group-kind device_stack
```

If split names are not supplied, the default split names are `train`, `validation`, and `test`.

## Split Groups

The preflight uses the existing leakage guard:

- `molecule_inchikey`: groups records by molecule inchikey or canonical SMILES fallback.
- `paper_evidence`: groups records by evidence refs and paper-id prefixes.
- `device_stack`: groups records by normalized device stack strings.

If no leakage group kinds are configured, all supported groups are used.

## Row-To-Split Mapping

Each dataset-view row is mapped using:

```text
effective_source_record_ids = row.source_record_ids or [row.record_id]
```

Outcomes:

- `assigned`: all effective source records map to exactly one split.
- `unassigned`: no usable source record id exists, or a source record is not in the split plan.
- `cross_split_source_records`: source records for one row span multiple splits.

Cross-split rows are errors by default because a single analysis row would mix records from multiple split partitions.

## Report

The report includes:

- gold validation error/warning codes
- split leakage error/warning codes
- proposed split plan
- row assignments
- rows by split
- per-view row split summaries
- finding code counts

The report is redacted and omits full gold records, raw paper text, PDFs, images, and parsed paper payloads.

## Boundary

Passing this preflight means the curated records and dataset-view row artifacts are ready for a later explicit split writer gate.

It does not create train/validation/test JSONL files, training data, feature materialization outputs, or model evaluation inputs.
