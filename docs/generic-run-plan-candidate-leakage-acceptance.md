# Generic RunPlan Candidate Leakage Acceptance

This acceptance coverage proves `check_public_dataset_leakage` can execute through the generic `run_plan_execute`
queue infrastructure before candidate data is confirmed.

```text
run_plan_execute
→ check_public_dataset_leakage_adapter
→ benchmark_contamination_report
```

The test uses synthetic local candidate training dataset and public benchmark CSV fixtures. It writes only benchmark
contamination report outputs and registers the report through ProjectStorage. Canonical SMILES overlap is checked as the
Phase 3 MVP leakage signal.

This remains a pre-confirmation review step. Overlap means possible public benchmark contamination or train/test leakage
that requires review; it does not confirm, promote, publish, train, or predict anything.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No model training or prediction is performed.
- No confirmation, promotion, publication, release, or global append artifact is produced.
