# Property Training Dataset Controlled Writer Dry-Run Precheck Evidence Template

## Precheck Summary

| Field | Value |
| --- | --- |
| evidence_id | <controlled_writer_dry_run_precheck_evidence_id> |
| date | <date> |
| operator | <operator> |
| corpus_id | <corpus_id> |
| dataset_name | <dataset_name> |
| dry_run_id | <dry_run_id> |
| dry_run_status | <dry_run_status> |
| precheck_status | <precheck_status> |
| dry_run_report_basename | <dry_run_report_basename> |
| dry_run_report_sha256 | <dry_run_report_sha256> |
| dry_run_summary_basename | <dry_run_summary_basename> |
| accepted_candidate_record_count | <accepted_candidate_record_count> |
| needs_review_candidate_record_count | <needs_review_candidate_record_count> |
| blocked_candidate_record_count | <blocked_candidate_record_count> |
| required_field_count | <required_field_count> |
| resolved_required_field_count | <resolved_required_field_count> |
| missing_required_field_count | <missing_required_field_count> |
| would_write_row_count | <would_write_row_count> |
| would_write_field_count | <would_write_field_count> |
| redaction_status | <redaction_status> |
| next_gate_decision | <next_gate_decision> |
| residual_risks | <residual_risks> |

## Boundary Checklist

- This controlled writer dry-run precheck does not rerun the dry-run.
- This controlled writer dry-run precheck does not execute the controlled writer.
- This controlled writer dry-run precheck does not emit raw values.
- This controlled writer dry-run precheck does not materialize values.
- This controlled writer dry-run precheck does not serialize training rows.
- This controlled writer dry-run precheck does not create training dataset artifacts.
- This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.
- This controlled writer dry-run precheck does not generate conformers.
- This controlled writer dry-run precheck does not generate DPA3 structures.
- This controlled writer dry-run precheck does not run Phase 1.
- This controlled writer dry-run precheck does not modify DatasetConfirmation.
- This controlled writer dry-run precheck does not run model training or evaluation.

## Validation Checklist

- [ ] Report schema is valid.
- [ ] Summary schema is valid.
- [ ] Report checksum matches summary.
- [ ] Report basename is basename-only.
- [ ] Report and summary ids match.
- [ ] Counts are aggregate and consistent.
- [ ] Boundary flags remain false or `not_run`.
- [ ] Evidence Markdown is redacted.
- [ ] No raw values or row payloads appear.
- [ ] No concrete dataset artifact paths appear.

## Validation Commands

```bash
python -m pytest tests/test_custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.py -q
python -m compileall -q src/ai4s_agent tests
git diff --check
python -m pytest -q
```

## Reviewer Notes

<reviewer_notes>
