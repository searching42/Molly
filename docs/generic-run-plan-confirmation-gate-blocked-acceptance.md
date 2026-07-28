# Generic RunPlan Confirmation Gate Blocked Acceptance

This acceptance coverage proves `confirm_extracted_dataset` is blocked safely in the generic `run_plan_execute` queue
path unless the DATA_MINING gate and review requirements are satisfied.

```text
run_plan_execute
→ confirm_extracted_dataset
→ WAITING_USER / confirmation_blocked
→ no confirmed artifacts
```

The test uses synthetic local candidate training dataset, conflict report, and citation provenance report fixtures. It
first verifies generic queue execution pauses at `WAITING_USER` before DATA_MINING approval, with a frozen execution
snapshot and no executed tasks. It then verifies that explicit DATA_MINING approval still fails when unresolved
conflicts or license-review items remain.

The coverage also checks missing confirmation intent directly at the adapter boundary. No confirmed dataset is created,
no confirmation record is written, and no human confirmation report is emitted.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- No dataset is positively confirmed.
- No confirmed dataset artifacts are emitted.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No model training or prediction is performed.
- No promotion, publication, release, or global append artifact is produced.
