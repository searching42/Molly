# Generic RunPlan Source Manifest Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute a second low-risk real task:

```text
run_plan_execute
-> RunPlanExecutorTaskRunner
-> RunPlanExecutor
-> prepare_literature_corpus_sources_adapter
-> corpus_source_manifest
```

The test uses synthetic local source metadata only. It creates a one-task RunPlan for `prepare_literature_corpus_sources`, enqueues it through the generic RunPlan queue helper, runs the bounded local queue worker, and verifies queue job state, lease state, ProjectStorage stage state, artifact registry entries, and deterministic source manifest outputs.

## Safety Boundary

This path only writes source manifest JSON/Markdown outputs. It does not download PDFs, resolve DOIs online, fetch URLs, parse PDFs, scan corpora, call MinerU or GROBID, call LLMs, train models, predict candidates, approve gates, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

The source manifest records source intent and leaves acquisition explicit. Search queries, DOIs, URLs, and dataset registry entries remain metadata until a separate reviewed acquisition step runs.

## Scope

This acceptance test does not add a wrapper CLI and does not modify the OLED local demo generic allowlist. The OLED allowlist remains limited to `execute_oled_local_demo`; this test uses the general RunPlan queue infrastructure directly for `prepare_literature_corpus_sources`.
