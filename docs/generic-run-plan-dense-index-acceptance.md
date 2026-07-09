# Generic RunPlan Dense Index Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute the low-risk `build_dense_index` task:

```text
run_plan_execute
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> build_dense_index_adapter
-> dense_index
```

The test uses a synthetic local evidence chunks JSONL fixture built from schema models. The fixture contains an OLED-focused paragraph chunk and a table chunk with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` metadata. No PDF fixture is created or read.

## Safety Boundary

This path writes dense index JSON and summary Markdown using deterministic hash embeddings only. It does not parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses the general RunPlan queue infrastructure directly for `build_dense_index`.
