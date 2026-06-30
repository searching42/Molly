# Custom Corpus Property Training Dataset Row Contract

The property training dataset row contract defines the semantic shape of a
future training dataset row before any dataset writer is allowed to create
JSONL, Parquet, LMDB, or CSV artifacts.

It answers what a valid row-shaped training sample must carry: property value
fields, molecule identifiers, provenance hashes, split and dedup keys, quality
flags, model-family compatibility labels, and output-format compatibility
labels.

It does not write a dataset.

## Relationship To Plan Precheck

The upstream plan precheck is documented in:

```text
docs/custom-corpus-property-training-dataset-materialization-plan-precheck.md
```

The row contract builder reads a passed
`custom_corpus_property_training_dataset_materialization_plan_precheck.v1`
package, the materialization plan and planner summary, and the upstream
ledger, dry-run, execution request, request draft, request plan, readiness,
and quarantine candidate evidence. It validates the full chain again before
writing a contract-only artifact.

The future dataset writer remains separate. A row contract is necessary but
not sufficient for dataset writing.

## Output Artifacts

The builder writes a run-scoped directory:

```text
<output-dir>/<row-contract-id>/
  property_training_dataset_row_contract.json
  property_training_dataset_row_contract_summary.json
  redacted_property_training_dataset_row_contract_evidence.md
```

Schemas:

- `custom_corpus_property_training_dataset_row_contract.v1`
- `custom_corpus_property_training_dataset_row_contract_builder.v1`

The contract keeps:

- `contract_mode=training_dataset_row_contract_only`
- `training_dataset_materialized=false`
- `dataset_artifact_created=false`
- `phase1_status=not_run`
- `dataset_confirmation_changed=false`

## Required Row Fields

Future rows must include:

- `dataset_record_id`
- `candidate_record_id`
- `record_id`
- `document_id`
- `field_name`
- `property_name`
- `property_value`
- `property_unit`
- `property_value_normalized`
- `property_unit_normalized`
- `task_type`
- `compound_id`
- `canonical_smiles`
- `source_artifact_sha256`
- `review_artifact_sha256`
- `admission_request_sha256`
- `training_admission_execution_ledger_sha256`
- `training_dataset_materialization_plan_sha256`

## Optional Row Fields

Future writers may include:

- `inchi`
- `inchi_key`
- `molecular_formula`
- `molecular_weight`
- `temperature`
- `solvent`
- `method`
- `aggregation_state`
- `device_context`
- `paper_id`
- `doi`
- `property_uncertainty`
- `quality_flags`
- `split_group_key`
- `dedup_key`
- `model_family_compatibility`

## Field Types

The contract uses safe type descriptors only:

- `string`
- `number`
- `boolean`
- `array[string]`
- `nullable[string]`
- `nullable[number]`

It does not include serialized training rows.

## Provenance Contract

Every future row must preserve:

- ledger record id
- planned dataset record id
- review id
- admission record id
- source artifact SHA-256
- materialization plan SHA-256
- row contract SHA-256

## Quality Flags

Allowed quality flag labels are:

- `unit_normalized`
- `value_normalized`
- `source_reviewed`
- `human_review_bound`
- `ledger_admitted`
- `needs_unit_review`
- `needs_structure_review`
- `needs_property_review`

These are labels only. They do not perform validation, correction, or
training.

## Split And Dedup

Future rows must support:

- `dedup_key`
- `split_group_key`

The `dedup_key` must be derived from canonical molecule identity, property
name, normalized value, normalized unit, and source artifact hash.

The `split_group_key` must default to canonical molecule identity, not row id,
to reduce molecule-level leakage.

## Model-Family Compatibility

Compatibility labels:

- `generic_property_predictor`
- `unimol`
- `dpa3`

`generic_property_predictor` requires canonical SMILES and a scalar property
value. `unimol` requires canonical SMILES and later conformer generation or a
conformer reference, but this contract does not generate conformers. `dpa3`
requires later structure or geometry-compatible data, but this contract does
not create DPA3 artifacts.

## Output Format Compatibility

Output format labels:

- `jsonl`
- `parquet`
- `lmdb`
- `csv`

These labels describe future writer compatibility only. This PR does not
create files in any of those formats.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_dataset_row_contract \
  --training-dataset-materialization-plan-precheck /tmp/property_training_dataset_materialization_plan_precheck_summary.json \
  --training-dataset-materialization-plan /tmp/property_training_dataset_materialization_plan.json \
  --training-dataset-materialization-planner-summary /tmp/property_training_dataset_materialization_planner_summary.json \
  --training-admission-execution-ledger-precheck /tmp/property_training_admission_execution_ledger_precheck_summary.json \
  --training-admission-execution-ledger /tmp/property_training_admission_execution_ledger.json \
  --training-admission-execution-ledger-summary /tmp/property_training_admission_execution_ledger_summary.json \
  --training-admission-execution-dry-run-precheck /tmp/property_training_admission_execution_dry_run_precheck_summary.json \
  --training-admission-execution-dry-run-report /tmp/property_training_admission_execution_dry_run_report.json \
  --training-admission-execution-request /tmp/property_training_admission_execution_request.json \
  --training-admission-execution-request-summary /tmp/property_training_admission_execution_request_summary.json \
  --training-admission-execution-request-preflight /tmp/property_training_admission_execution_request_preflight_summary.json \
  --training-admission-request-draft /tmp/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /tmp/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /tmp/property_training_admission_request_draft_precheck_summary.json \
  --training-admission-request-plan /tmp/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /tmp/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /tmp/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /tmp/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /tmp/property_quarantine_candidate_records.json \
  --output-dir /tmp/property-training-dataset-row-contract \
  --row-contract-id property-training-dataset-row-contract-001 \
  --created-by operator-redacted \
  --confirm-training-dataset-row-contract
```

Return codes:

- `0` for `written` or `needs_review`
- `1` for `blocked`

## Redaction

The contract, summary, and Markdown evidence contain safe ids, SHA-256 hashes,
schema/status fields, row-field names, compatibility labels, aggregate counts,
and safe error codes only. They must not contain raw table rows, raw article
text, PDF names or paths, local paths, token-like values, serialized dataset
rows, or future dataset output paths.

## Boundaries

- This is a training dataset row contract only.
- No training dataset artifact is created.
- No training CSV/JSONL/Parquet/LMDB artifact is created.
- No candidate CSV/JSONL/Parquet/LMDB artifact is created.
- No conformers are generated.
- No DPA3 structures are generated.
- No Phase 1 execution occurs.
- `DatasetConfirmation` is not changed.
- No model training or evaluation is run.
- No LLM, agent, MinerU, PDF, ParsedDocument, or corpus workflow call is made.
