# Generic RunPlan Evidence Retrieval Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute the low-risk `retrieve_evidence` task:

```text
run_plan_execute
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> retrieve_evidence_adapter
-> evidence_hits + retrieval_log
```

The test uses synthetic local corpus index, evidence chunks, and multi-index fixtures. The chunks include an OLED-focused paragraph and a table with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` metadata. No PDF fixture is created or read.

## Safety Boundary

This path writes evidence hits JSON and a retrieval log JSONL. It does not parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use dense retrieval or sentence-transformers, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses the general RunPlan queue infrastructure directly for `retrieve_evidence`.
