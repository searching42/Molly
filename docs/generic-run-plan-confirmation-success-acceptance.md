# Generic RunPlan Confirmation Success Acceptance

This acceptance coverage proves `confirm_extracted_dataset` can succeed through the generic `run_plan_execute` queue path
only after explicit DATA_MINING approval.

```text
run_plan_execute
→ confirm_extracted_dataset
→ WAITING_USER
→ DATA_MINING approval + actor
→ confirmed_training_dataset + extraction_confirmation_record
```

The test uses synthetic local candidate dataset, conflict report, and citation provenance report fixtures. It first
verifies the queued generic run pauses at `WAITING_USER` with a frozen gate snapshot and no executed tasks. It then
resumes with explicit DATA_MINING approval and a non-empty actor, confirming only a run-scoped synthetic dataset.

The clean fixture has an existing candidate dataset, `conflict_count == 0`, and `unknown_license_count == 0`. Additional
coverage verifies a wrong gate is rejected and dirty review reports still block confirmation.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- Only a synthetic run-scoped candidate dataset is confirmed.
- No model training or prediction is performed.
- No publication, release, or global append artifact is produced.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
