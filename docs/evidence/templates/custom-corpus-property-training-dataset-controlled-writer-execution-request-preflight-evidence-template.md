# Property Training Dataset Controlled Writer Execution Request Preflight Evidence Template

## Request Preflight Metadata

| Field | Value |
| --- | --- |
| controlled_writer_execution_request_preflight_evidence_id | <controlled_writer_execution_request_preflight_evidence_id> |
| date | <date> |
| operator | <operator> |
| corpus_id | <corpus_id> |
| dataset_name | <dataset_name> |
| request_id | <request_id> |
| request_status | <request_status> |
| preflight_status | <preflight_status> |
| request_basename | <request_basename> |
| request_sha256 | <request_sha256> |
| request_summary_basename | <request_summary_basename> |
| requested_by | <requested_by> |

## Dry-Run Precheck Binding

| Field | Value |
| --- | --- |
| dry_run_precheck_summary_basename | <dry_run_precheck_summary_basename> |
| dry_run_precheck_summary_sha256 | <dry_run_precheck_summary_sha256> |
| dry_run_precheck_status | <dry_run_precheck_status> |
| dry_run_status | <dry_run_status> |

## Aggregate Counts

| Field | Value |
| --- | --- |
| accepted_candidate_record_count | <accepted_candidate_record_count> |
| needs_review_candidate_record_count | <needs_review_candidate_record_count> |
| blocked_candidate_record_count | <blocked_candidate_record_count> |
| missing_required_field_count | <missing_required_field_count> |
| would_write_row_count | <would_write_row_count> |
| would_write_field_count | <would_write_field_count> |

## Authorization State

| Field | Value |
| --- | --- |
| redaction_status | <redaction_status> |
| writer_execution_authorized | <writer_execution_authorized> |
| explicit_confirmation_required | <explicit_confirmation_required> |
| next_gate_decision | <next_gate_decision> |
| residual_risks | <residual_risks> |

## Boundary Checklist

- This controlled writer execution request preflight does not explicitly confirm execution.
- This controlled writer execution request preflight does not execute the controlled writer.
- This controlled writer execution request preflight does not authorize writer execution by itself.
- This controlled writer execution request preflight keeps explicit confirmation required.
- This controlled writer execution request preflight does not emit raw values.
- This controlled writer execution request preflight does not materialize values.
- This controlled writer execution request preflight does not serialize training rows.
- This controlled writer execution request preflight does not create training dataset artifacts.
- This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.
- This controlled writer execution request preflight does not generate conformers.
- This controlled writer execution request preflight does not generate DPA3 structures.
- This controlled writer execution request preflight does not run Phase 1.
- This controlled writer execution request preflight does not modify DatasetConfirmation.
- This controlled writer execution request preflight does not run model training or evaluation.

## Redaction Checklist

- [ ] Safe ids only.
- [ ] Safe basenames only.
- [ ] SHA-256 hashes only.
- [ ] Aggregate counts only.
- [ ] No raw values.
- [ ] No exact numeric extracted values.
- [ ] No molecular strings.
- [ ] No row payloads.
- [ ] No source payloads.
- [ ] No local or output paths.
- [ ] No credentials.
