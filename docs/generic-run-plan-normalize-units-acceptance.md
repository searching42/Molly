# Generic RunPlan Normalize Units Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute deterministic unit normalization:

```text
run_plan_execute
-> normalize_extracted_units_adapter
-> normalized_extracted_records + normalized_candidate_training_dataset + unit_normalization_report
```

The test uses synthetic local extracted records JSONL fixtures. The fixture records remain `candidate` and include OLED table values such as `PLQY (%)`, `HOMO`, and `LUMO`.

## Safety Boundary

This path writes normalized candidate records, a normalized candidate training dataset CSV, and unit normalization reports. Records remain candidate and unconfirmed only. The test does not confirm, promote, train, predict, parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses direct synthetic extracted-record fixtures so the focused test executes only `normalize_extracted_units_adapter`.
