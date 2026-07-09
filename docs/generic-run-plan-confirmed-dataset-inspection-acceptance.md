# Generic RunPlan Confirmed Dataset Inspection Acceptance

This acceptance coverage proves `inspect_dataset` can consume a `confirmed_training_dataset` artifact through the
generic `run_plan_execute` queue path.

```text
run_plan_execute
→ inspect_dataset
→ dataset_profile + property_catalog
```

The test uses a synthetic local confirmed CSV fixture with numeric OLED-like property columns. It writes and registers
only the `dataset_profile` and `property_catalog` artifacts, and verifies the existing `uploaded_dataset` path still
works.

This is a post-confirmation, pre-training inspection step. It does not clean data, run trainability checks, train,
predict, publish, release, or globally append data.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- Only `inspect_dataset_service` executes.
- Only a synthetic local confirmed CSV fixture is read.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No subprocess or daemon behavior is introduced.
