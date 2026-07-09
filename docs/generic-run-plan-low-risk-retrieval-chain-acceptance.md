# Generic RunPlan Low-Risk Retrieval Chain Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute a continuous low-risk retrieval chain in one queue job:

```text
run_plan_execute
-> index_corpus
-> build_multi_index
-> build_dense_index
-> retrieve_evidence
-> evidence_hits + retrieval_log
```

The test uses a synthetic local `ParsedDocument` JSON fixture built from schema models. The fixture contains one OLED-focused paragraph and one table with `Compound`, `SMILES`, `PLQY (%)`, `HOMO`, and `LUMO` fields. No PDF fixture is created or read.

## Safety Boundary

This path writes corpus index, evidence chunks, multi-index, deterministic dense index, evidence hits, and retrieval log outputs. It does not parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses deterministic hash embeddings only and exercises artifact handoff through the general RunPlan queue infrastructure.
