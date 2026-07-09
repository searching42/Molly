# Generic RunPlan Extraction Benchmark Acceptance

This acceptance coverage proves `evaluate_extraction_benchmark` can execute through the generic `run_plan_execute`
queue infrastructure.

```text
run_plan_execute
→ evaluate_extraction_benchmark_adapter
→ extraction_benchmark_report
```

The test uses synthetic local benchmark fixtures: evidence hits, normalized candidate records, gold records, conflict
report metadata, extraction confidence metadata, citation provenance metadata, unit normalization metadata, candidate
CSV rows, and before/after model metric summaries. It writes only extraction benchmark report outputs and registers the
benchmark report through ProjectStorage.

Missing gold labels are reported as missing rather than inferred. The benchmark estimates confirmation workload from
conflicts, rejected records, unknown-license sources, and unit-normalization warnings, but it does not confirm or promote
anything.

Safety boundary:

- No new CLI or queue operation is added.
- The OLED local demo allowlist is unchanged.
- No PDFs are read or parsed.
- No MinerU or GROBID calls are made.
- No network, DOI resolution, LLM, sentence-transformers, or model-download path is used.
- No model training or prediction is performed.
- No confirmation, promotion, publication, release, or global append artifact is produced.
