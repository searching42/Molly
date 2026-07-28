# OLED Curated Split Training-Package Preflight

This read-only preflight checks whether split OLED feature row artifacts are ready for a later ML-ready training package writer.

It does not write final training JSONL, CSV, Parquet, benchmark packages, or model inputs. It does not run baseline backends, train models, evaluate models, call LLMs, call MinerU, read PDFs, or read images.

## Inputs

The file runner expects:

- split feature writer manifest JSON
- split feature row JSONL files referenced by that manifest

The manifest is loaded through the preflight module and each written file result is resolved relative to the manifest directory or an explicit base directory. SHA256 values are verified when present.

## Relationship To The Split Feature Writer

`oled_curated_split_feature_writer.py` writes split-specific feature row artifacts. Those rows are training-adjacent but not ML-ready training data.

This preflight inspects those row artifacts and reports whether a later explicit training-package writer can safely package them.

## Checks

The preflight validates:

- duplicate `feature_row_id`
- same `feature_row_id` across multiple splits
- missing target values
- nonnumeric targets
- missing evidence refs
- unknown split names
- missing or empty expected splits
- inconsistent feature columns within each `(target_property_id, feature_view)`
- required feature columns
- required feature values
- optional feature missingness

It also warns when a `row_id` or `record_id` appears across splits. The leakage split preflight remains the authoritative leakage gate, but these warnings are useful before training package creation.

## Summaries

The report includes:

- rows by split
- target property coverage
- feature view coverage
- per-split missing target/evidence counts
- per-target numeric and nonnumeric target counts
- target unit distribution
- feature column kind
- feature column missingness
- short deterministic example values

Reports do not include full split feature row payloads.

## CLI

```bash
python -m ai4s_agent.domains.oled_curated_split_training_package_preflight \
  --split-feature-manifest /path/to/split_feature_manifest.json \
  --split-feature-base-dir /path/to/split_features \
  --output-report /path/to/training_package_preflight_report.json
```

Optional filters:

```bash
--target-property-id eqe_percent
--feature-view full_context
--required-feature-column molecule.canonical_smiles
```

Repeated and comma-separated values are supported.

## Output

The output is a deterministic, redacted JSON report with:

- preflight status
- split summaries
- target summaries
- feature column summaries
- finding taxonomy
- safety metadata

Safety metadata records:

- `training_package_preflight_only=True`
- `training_package_written=False`
- `ml_ready_training_data_written=False`
- `baseline_backend_run=False`
- `model_backends_run=False`

## Warning

Passing this preflight does not create benchmark-ready data. A later explicit writer gate must create any final ML-ready package, and model backend execution must remain a separate explicit step.
