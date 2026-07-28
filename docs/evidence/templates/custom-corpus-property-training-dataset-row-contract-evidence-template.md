# Custom Corpus Property Training Dataset Row Contract Evidence Template

## Training Dataset Row Contract Summary

- contract status:
- row contract id:
- dataset name:
- contract version label:
- planned dataset record count:
- contract record reference count:

## Input Artifact Summary

- materialization plan precheck SHA-256:
- materialization plan SHA-256:
- materialization planner summary SHA-256:
- ledger precheck SHA-256:
- ledger SHA-256:
- ledger summary SHA-256:
- dry-run precheck SHA-256:
- dry-run report SHA-256:
- execution request SHA-256:
- execution request summary SHA-256:
- execution request preflight SHA-256:
- request draft SHA-256:
- request draft summary SHA-256:
- request draft precheck SHA-256:
- request plan SHA-256:
- request preflight SHA-256:
- readiness summary SHA-256:
- quarantine candidate preflight SHA-256:
- quarantine candidate records SHA-256:

## Required Row Fields

- dataset_record_id
- candidate_record_id
- record_id
- document_id
- field_name
- property_name
- property_value
- property_unit
- property_value_normalized
- property_unit_normalized
- task_type
- compound_id
- canonical_smiles
- source_artifact_sha256
- review_artifact_sha256
- admission_request_sha256
- training_admission_execution_ledger_sha256
- training_dataset_materialization_plan_sha256

## Optional Row Fields

- inchi
- inchi_key
- molecular_formula
- molecular_weight
- temperature
- solvent
- method
- aggregation_state
- device_context
- paper_id
- doi
- property_uncertainty
- quality_flags
- split_group_key
- dedup_key
- model_family_compatibility

## Provenance Contract

- ledger record id preserved:
- planned dataset record id preserved:
- review id preserved:
- admission record id preserved:
- source artifact SHA preserved:
- materialization plan SHA preserved:
- row contract SHA required for future rows:

## Quality Flag Contract

- unit_normalized
- value_normalized
- source_reviewed
- human_review_bound
- ledger_admitted
- needs_unit_review
- needs_structure_review
- needs_property_review

## Split And Dedup Contract

- dedup key rule:
- split group key rule:

## Model Family Compatibility

- generic_property_predictor:
- unimol:
- dpa3:

## Output Format Compatibility

- jsonl:
- parquet:
- lmdb:
- csv:

## Contract Errors And Warnings

- contract errors:
- warnings:

## Redaction Statement

- no raw PDFs
- no ParsedDocument outputs
- no MinerU bundles
- no raw article text
- no raw table rows
- no private paths
- no tokens/auth/cookies
- no serialized training rows
- no training CSV/JSONL/Parquet/LMDB artifacts
- no candidate CSV/JSONL/Parquet/LMDB artifacts

## Boundary Statement

- training dataset row contract only
- no training dataset artifact
- no training CSV/JSONL/Parquet/LMDB
- no candidate CSV/JSONL/Parquet/LMDB
- no conformer generation
- no DPA3 structure generation
- no Phase 1
- no DatasetConfirmation change
- no model training or evaluation
