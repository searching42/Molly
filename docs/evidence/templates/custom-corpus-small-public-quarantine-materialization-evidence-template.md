# Small Public Quarantine Materialization Evidence Template

## Scope

| Field | Value |
| --- | --- |
| evidence_id | <evidence_id> |
| date | <date> |
| operator | <operator> |
| corpus_id | <corpus_id> |
| dataset_name | <dataset_name> |
| candidate_record_count | <candidate_record_count> |
| quarantined_record_count | <quarantined_record_count> |
| admitted_record_count | <admitted_record_count> |
| training_dataset_materialized | false |
| dataset_artifact_created | false |
| phase1_status | not_run |
| dataset_confirmation_changed | false |
| model_training_run | false |
| evaluation_run | false |

## Public/Low-Risk Corpus Boundary

Document the public source boundary using safe source labels only. Do not add
file names, local paths, exact values, molecular strings, table text, article
body text, output locations, credentials, or auth header material.

## Input Evidence Chain

| Gate | Status | Evidence |
| --- | --- | --- |
| property candidate review | <review_status> | redacted ids only |
| quarantine candidate materialization | <quarantine_status> | redacted ids only |
| training admission readiness | <training_admission_readiness_status> | redacted ids only |
| materialization dry-run | <materialization_dry_run_status> | row-preview counts only |
| materialization dry-run precheck | <materialization_dry_run_precheck_status> | redacted summary only |
| controlled writer value resolution dry-run | <value_resolution_dry_run_status> | no values emitted |
| controlled writer value resolution dry-run precheck | <value_resolution_dry_run_precheck_status> | report/summary validated |

## Quarantine Materialization Evidence

| Field | Value |
| --- | --- |
| quarantine_candidate_record_count | <quarantined_record_count> |
| admitted_record_count | <admitted_record_count> |
| quarantine_status | <quarantine_status> |
| candidate_record_ids | <candidate_record_ids> |
| quarantine_record_ids | <quarantine_record_ids> |

## Training Dataset Boundary

This evidence packet does not create a training dataset.
This evidence packet does not execute a controlled writer.
This evidence packet does not serialize training rows.
This evidence packet does not create CSV/JSONL/Parquet/LMDB artifacts.
This evidence packet does not run Phase 1.
This evidence packet does not modify DatasetConfirmation.
This evidence packet does not run model training or evaluation.

## Value Resolution Readiness

| Field | Value |
| --- | --- |
| value_resolution_dry_run_status | <value_resolution_dry_run_status> |
| value_resolution_dry_run_precheck_status | <value_resolution_dry_run_precheck_status> |
| redaction_status | <redaction_status> |
| source_payloads_read_by_evidence_packet | false |
| values_emitted | false |
| values_materialized | false |
| row_serialization_created | false |

## Redaction Review

- public source boundary confirmed
- no private paths
- no exact property values
- no canonical SMILES
- no InChI/InChIKey
- no PDF names
- no article/table text
- no row serialization
- no dataset artifact paths
- no conformer/DPA3 artifacts
- no model training
- no Phase 1
- no DatasetConfirmation mutation
- no credential or auth header material

## Residual Risks

<residual_risks>

## Next Gate

<next_gate_decision>

## Operator Checklist

- [ ] public source boundary confirmed
- [ ] no private paths
- [ ] no exact property values
- [ ] no canonical SMILES
- [ ] no InChI/InChIKey
- [ ] no PDF names
- [ ] no article/table text
- [ ] no row serialization
- [ ] no dataset artifact paths
- [ ] no conformer/DPA3 artifacts
- [ ] no model training
- [ ] no Phase 1
- [ ] no DatasetConfirmation mutation
- [ ] reviewer notes recorded
