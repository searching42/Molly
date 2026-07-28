# Generic RunPlan Candidate Dataset Chain Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute candidate dataset construction in one job:

```text
run_plan_execute
-> extract_records
-> normalize_extracted_units
-> normalized candidate_training_dataset
```

The test uses synthetic local evidence hits JSON and evidence chunks JSONL fixtures. The evidence points to one table chunk with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` rows.

## Safety Boundary

This path writes candidate extracted records, rejected records, extraction confidence reports, normalized candidate records, normalized candidate training datasets, and unit normalization reports. Records remain candidate and unconfirmed only. The test does not confirm, promote, train, predict, parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses direct synthetic evidence fixtures so the focused chain executes only `extract_records_adapter` followed by `normalize_extracted_units_adapter`.
