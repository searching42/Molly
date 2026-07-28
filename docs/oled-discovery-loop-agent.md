# OLED Discovery Loop Agent

## Purpose

PR #275 intentionally pauses the expansion of registry, promotion, publication, and release governance layers. The next project priority is making Molly's Agent progress observable for actual OLED discovery work.

`OLEDDiscoveryLoopAgent` provides a deterministic, review-only state machine and run-card skeleton for OLED discovery runs. It summarizes the current stage, available artifacts, missing artifacts, blockers, risks, and the next recommended agent action.

## State Machine

The loop follows this review sequence:

```text
user goal
  -> intent understanding
  -> research/source planning
  -> acquisition/data readiness
  -> modeling readiness
  -> baseline/model diagnostics
  -> candidate screening readiness
  -> critic review
  -> next action proposal
```

The run card reports stages such as `intent_captured`, `research_plan_proposed`, `dataset_ready`, `training_package_ready`, `baseline_ready`, `diagnostics_ready`, `candidates_ready`, and `critic_reviewed`. It can summarize a later stage when downstream artifacts are supplied even if earlier planning summaries are absent; missing earlier artifacts remain visible in the run card.

## Inputs

The agent accepts synthetic summary dictionaries rather than requiring real files:

```python
card = OLEDDiscoveryLoopAgent().build_run_card(
    run_id="demo",
    goal="Find OLED emitters with high PLQY and red-shifted emission",
    dataset_artifacts={"dataset_view_rows": "rows.jsonl"},
    training_package_artifacts={"training_rows": "training_rows.jsonl"},
    baseline_artifacts={"metrics": "metrics.json"},
    diagnostics_report={"status": "acceptable"},
)
```

The resulting card is an `OLEDDiscoveryRunCard` with `executable=False`.

## Relationship To Existing Agents

- `ConversationAgent` captures user intent and can provide conversation decisions.
- `ResearchAgent` proposes sources and acquisition preparation.
- `ModelingAgent` prepares modeling briefs and diagnostics.
- `RunPlanExecutor` remains responsible for gated execution snapshots.
- Existing OLED artifact governance remains the audit trail for curated data and benchmark artifacts.

The discovery loop does not replace those components. It provides a compact Agent-level run card so a reviewer can see where a discovery run stands and which existing agent or gate should be used next.

## Markdown Run Cards

`write_run_card()` writes deterministic JSON and Markdown artifacts through `ProjectStorage`:

- `oled_discovery_run_card.json`
- `oled_discovery_run_card.md`

The Markdown includes stage status, available artifacts, blockers, risks, recommended actions, and a safety boundary.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_discovery \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY and red-shifted emission" \
  --output-dir /tmp/oled-loop-demo
```

## Safety Boundary

This module is review-only. It does not execute research acquisition, parse PDFs/images, call MinerU, call LLMs, train models, run model backends, predict, mutate registry/promotion/publication/release artifacts, write global registry files, or use external network access.

It is a small Agent-loop skeleton intended to prepare later work on ToolRegistry, CriticAgent, and controlled OLED discovery-loop execution.
