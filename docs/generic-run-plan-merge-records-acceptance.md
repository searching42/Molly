# Generic RunPlan Merge Records Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute conflict-aware candidate record merging:

```text
run_plan_execute
-> merge_extracted_records_adapter
-> merged_records + conflict_report + candidate_training_dataset
```

The test uses synthetic local normalized extracted records JSONL and citation provenance report fixtures. No PDF fixture is created or read.

## Safety Boundary

This path groups records by exact normalized SMILES, writes merged candidate records, writes a conflict report, and writes a merged candidate training dataset CSV. Conflicted values are excluded from the candidate CSV pending human review. The test does not confirm, promote, train, predict, parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses direct synthetic normalized-record and provenance fixtures so the focused test executes only `merge_extracted_records_adapter`.
