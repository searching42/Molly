# Generic RunPlan Extract Records Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute deterministic table-record extraction:

```text
run_plan_execute
-> extract_records_adapter
-> extracted_records + candidate_training_dataset
```

The test uses synthetic local evidence hits JSON and evidence chunks JSONL fixtures. The table chunk contains `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` metadata and fixture rows. No PDF fixture is created or read.

## Safety Boundary

This path writes candidate extracted records, rejected records, an extraction confidence report, an extraction summary, and a candidate training dataset CSV. Records are candidate and unconfirmed only. The test does not confirm, promote, train, predict, parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses direct synthetic evidence fixtures so the focused test executes only `extract_records_adapter`.
