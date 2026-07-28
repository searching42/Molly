# Generic RunPlan Multi-Index Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute the low-risk `build_multi_index` task:

```text
run_plan_execute
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> build_multi_index_adapter
-> multi_index
```

The test uses a synthetic local evidence chunks JSONL fixture built from schema models. The fixture contains an OLED-focused paragraph chunk and a table chunk with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` metadata. No PDF fixture is created or read.

## Safety Boundary

This path writes a deterministic multi-index JSON and summary Markdown. It does not parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses the general RunPlan queue infrastructure directly for `build_multi_index`.
