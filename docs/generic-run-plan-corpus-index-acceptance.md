# Generic RunPlan Corpus Index Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute the low-risk `index_corpus` task:

```text
run_plan_execute
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> index_corpus_adapter
-> corpus_index + evidence_chunks
```

The test uses a synthetic local `ParsedDocument` JSON fixture built from schema models. The fixture contains one OLED-focused paragraph and one table with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` fields. No PDF fixture is created or read.

## Safety Boundary

This path writes corpus index JSON, evidence chunks JSONL, and an index report JSON. It does not parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses the general RunPlan queue infrastructure directly for `index_corpus`.
