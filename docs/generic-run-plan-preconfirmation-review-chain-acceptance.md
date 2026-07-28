# Generic RunPlan Pre-Confirmation Review Chain Acceptance

This acceptance coverage proves the full pre-confirmation review chain can execute in one generic `run_plan_execute`
queue job.

```text
run_plan_execute
→ extract_records
→ normalize_extracted_units
→ track_citation_provenance
→ merge_extracted_records
→ evaluate_extraction_benchmark
→ check_public_dataset_leakage
→ pre-confirmation review reports
```

The test uses synthetic local parsed document, evidence hit, evidence chunk, gold-record, public benchmark CSV, and
model-metric fixtures. It validates artifact handoff across extraction, normalization, provenance auditing, record
merge, benchmark evaluation, and public leakage checking.

The chain writes candidate records, normalized records, citation provenance audit outputs, conflict reports, benchmark
reports, leakage reports, and the final candidate dataset. It remains pre-confirmation review only: records are not
confirmed or promoted, benchmark metrics are review signals, and public dataset overlap requires review before any
downstream use.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No model training or prediction is performed.
- No confirmation, promotion, publication, release, or global append artifact is produced.
