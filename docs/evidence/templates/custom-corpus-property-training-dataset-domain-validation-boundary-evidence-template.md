# Custom Corpus Property Training Dataset Domain Validation Boundary Evidence Template

## Domain Validation Boundary Summary

| Field | Value |
| --- | --- |
| domain_validation_boundary_evidence_id | <domain_validation_boundary_evidence_id> |
| date | <date> |
| operator | <operator> |
| corpus_id | <corpus_id> |
| dataset_name | <dataset_name> |
| quarantined_candidate_admission_boundary_status | <quarantined_candidate_admission_boundary_status> |
| candidate_record_count | <candidate_record_count> |
| accepted_candidate_record_count | <accepted_candidate_record_count> |
| needs_review_candidate_record_count | <needs_review_candidate_record_count> |
| blocked_candidate_record_count | <blocked_candidate_record_count> |
| property_unit_compatibility_status | <property_unit_compatibility_status> |
| property_unit_compatibility_pass_count | <property_unit_compatibility_pass_count> |
| property_unit_compatibility_needs_review_count | <property_unit_compatibility_needs_review_count> |
| property_unit_compatibility_fail_count | <property_unit_compatibility_fail_count> |
| numeric_plausibility_status | <numeric_plausibility_status> |
| numeric_plausibility_pass_count | <numeric_plausibility_pass_count> |
| numeric_plausibility_needs_review_count | <numeric_plausibility_needs_review_count> |
| numeric_plausibility_fail_count | <numeric_plausibility_fail_count> |
| provenance_consistency_status | <provenance_consistency_status> |
| condition_completeness_status | <condition_completeness_status> |
| compound_alias_association_status | <compound_alias_association_status> |
| duplicate_conflict_status | <duplicate_conflict_status> |
| redaction_status | <redaction_status> |
| next_gate_decision | <next_gate_decision> |

## Governance Chain Position

```text
property training dataset controlled writer value resolution dry-run
-> property training dataset controlled writer value resolution dry-run precheck
-> small public quarantine materialization evidence
-> property training dataset quarantined candidate admission boundary
-> property training dataset domain validation boundary
-> future controlled training dataset writer
```

## Domain Validation Checklist

- [ ] property and unit labels are compatible
- [ ] numeric plausibility status is label-only
- [ ] provenance labels are consistent
- [ ] condition context is passed, not applicable, or recorded as needs-review
- [ ] compound and alias associations are explicit
- [ ] duplicate and conflict counts are reviewed
- [ ] accepted candidates are separated from needs-review candidates
- [ ] blocked and rejected candidates are absent from accepted ids

## Boundary Checklist

This domain validation boundary does not execute a controlled writer.
This domain validation boundary does not emit raw values.
This domain validation boundary does not materialize values.
This domain validation boundary does not serialize training rows.
This domain validation boundary does not create training dataset artifacts.
This domain validation boundary does not create CSV/JSONL/Parquet/LMDB artifacts.
This domain validation boundary does not generate conformers.
This domain validation boundary does not generate DPA3 structures.
This domain validation boundary does not run Phase 1.
This domain validation boundary does not modify DatasetConfirmation.
This domain validation boundary does not run model training or evaluation.

## Redaction Checklist

- [ ] no raw property values
- [ ] no exact numeric extracted values
- [ ] no canonical molecular strings
- [ ] no structure identifiers
- [ ] no table row payloads
- [ ] no article body text
- [ ] no paper titles
- [ ] no source file names
- [ ] no local paths
- [ ] no output paths
- [ ] no row payloads
- [ ] no conformer or DPA3 structure data
- [ ] no credential material

## Residual Risks

<residual_risks>
