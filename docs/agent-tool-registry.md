# Agent Tool Registry

## Purpose

PR #275 added `OLEDDiscoveryLoopAgent`, a review-only run card that makes OLED discovery progress visible. `AgentToolRegistry` is the next small step: a deterministic capability map that helps the Agent reason about which existing tool, task, or adapter category belongs at each discovery stage.

The registry answers what tools exist, what stage they support, which artifacts they require, what they produce, whether they need gates, what risk level they carry, and why they are or are not ready for the current run card.

## What It Does

`AgentToolRegistry` is declarative and review-only. It can:

- list built-in OLED discovery tool specs
- filter tools by `OLEDDiscoveryStage`
- recommend tools from a current stage and available artifact names
- mark missing inputs, gated tools, and tools above the current risk budget
- render deterministic Markdown for review
- emit compact CLI summaries for planning/debugging

It does not import or execute governance writer modules. It describes existing Molly concepts in broad Agent-planning terms instead of expanding another registry, promotion, publication, release, or global append layer.

## What It Does Not Execute

The registry does not execute adapters, call LLMs, call MinerU, read PDFs/images, perform external network access, run model training, predict, validate benchmarks, mutate artifacts, or write registry/promotion/publication/release outputs.

Every `AgentToolSpec`, `AgentToolRecommendation`, and `AgentToolRegistrySnapshot` is non-executable and must keep `executable=false`.

## Relationship To OLEDDiscoveryLoopAgent

`OLEDDiscoveryLoopAgent` summarizes where an OLED discovery run stands. `AgentToolRegistry` maps that stage to review-only capability recommendations.

Example:

```python
from ai4s_agent.agents.tool_registry import AgentToolRegistry

registry = AgentToolRegistry()
recommendations = registry.recommend_tools(
    current_stage="diagnostics_ready",
    available_artifacts=["diagnostics_report", "training_package_artifacts"],
    risk_budget="medium",
)
```

For `diagnostics_ready`, the registry can recommend `candidate_generation_or_prediction` when both diagnostics and training-package artifacts are available. If the training package artifact name is missing, the same recommendation is returned as not ready with `missing_required_inputs`.

## Stage Mapping

The built-in registry maps tools to the discovery-loop stages introduced by `OLEDDiscoveryLoopAgent`:

- `intent_captured`: research source proposal
- `research_plan_proposed`: acquisition preparation and corpus-source planning
- `acquisition_prepared`: literature/data acquisition, parsing, indexing, evidence retrieval, extraction, normalization, provenance, and dataset confirmation
- `dataset_ready`: leakage split, feature materialization, training package, and modeling brief preparation
- `training_package_ready`: baseline runner and modeling brief preparation
- `baseline_ready`: diagnostics report
- `diagnostics_ready`: candidate generation or prediction and candidate ranking
- `candidates_ready`: candidate ranking and critic review
- `critic_reviewed`: next action proposal

## AtomicTaskRegistry Difference

`AtomicTaskRegistry` is the execution/task dependency registry. It belongs to controlled task execution and gate enforcement.

`AgentToolRegistry` is a review-only capability map for Agent planning. It helps Molly explain the next safe action without running the action.

## Safety Boundary

This module is a planning aid only:

- no adapter execution
- no registry, promotion, publication, release, or global append mutation
- no model training or prediction
- no benchmark validation
- no LLM calls
- no MinerU calls
- no PDF/image reads
- no external network access
- no real corpus dependency
- no new dependencies
