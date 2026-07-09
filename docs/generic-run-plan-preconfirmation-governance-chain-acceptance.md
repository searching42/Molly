# Generic RunPlan Pre-Confirmation Governance Chain Acceptance

This acceptance coverage proves the pre-confirmation candidate governance chain can execute in one generic
`run_plan_execute` queue job.

```text
run_plan_execute
→ extract_records
→ normalize_extracted_units
→ track_citation_provenance
→ merge_extracted_records
→ final candidate_training_dataset
```

The test uses synthetic local parsed document, evidence hit, and evidence chunk fixtures. It validates artifact
handoff across extraction, normalization, provenance, and merge, then checks candidate extracted records, normalized
candidate records, citation/license provenance audit output, merged records, conflict report, and the final merged
candidate dataset.

The chain remains candidate-only. Records are not confirmed or promoted, conflicted values are excluded pending human
review, and citation/license tracking does not grant reuse permission.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No model training or prediction is performed.
- No confirmation, promotion, publication, release, or global append artifact is produced.
