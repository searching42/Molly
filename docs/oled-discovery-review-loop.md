# OLED Discovery Review Loop

## Purpose

PR #275 made OLED discovery run state visible with `OLEDDiscoveryLoopAgent`. PR #276 added `AgentToolRegistry` so the Agent can reason about relevant tools and gates. PR #277 added `CriticAgent` so the Agent can critique whether to continue, revise, rerun, request evidence, block overclaims, or run candidate review.

`OLEDDiscoveryReviewLoopAgent` connects those pieces into the first integrated review-only OLED discovery loop artifact:

```text
OLEDDiscoveryRunCard
+ AgentToolRecommendation list
+ CriticReview
= OLEDDiscoveryLoopReview
```

## What The Loop Answers

The loop review summarizes:

- where the run is now
- which tools are relevant
- which tools are ready or blocked
- what the critic decided
- what risk flags and blocked reasons remain
- which single next action should be reviewed

The output is deterministic and non-executable.

## Inputs

The harness accepts the same synthetic summaries used by the component agents:

- conversation decision
- research source proposal
- research acquisition preparation
- target modeling brief
- dataset, training package, baseline, diagnostics, and candidate artifacts
- dataset, training package, baseline, candidate, provenance, and model-package summaries
- risk budget
- gated-tool policy

It does not require real corpus files.

## Decision Flow

The harness builds a run card, asks `AgentToolRegistry` for stage-aware recommendations, then asks `CriticAgent` for a review.

Next-action selection is deterministic:

- non-`continue` critic decisions win first
- otherwise, choose the first ready tool recommendation by deterministic order
- otherwise, recommend resolving the first blocked tool input or gate
- otherwise, resolve a run-card blocker
- otherwise, require human review

Examples include `request_more_evidence`, `revise_data`, `rerun_baseline`, `block_promotion`, `candidate_generation_or_prediction`, and `human_review_required`.

## JSON And Markdown

`write_review()` writes:

- `oled_discovery_loop_review.json`
- `oled_discovery_loop_review.md`

The Markdown includes run-card summary, tool recommendation table, critic findings, risk flags, blocked reasons, and a safety boundary.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.oled_review_loop \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --diagnostics-status acceptable
```

The CLI prints a compact JSON summary only. It does not execute tools.

## Safety Boundary

This is a review-only loop. It does not execute adapters, call `RunPlanExecutor`, train models, predict, validate benchmarks, call LLMs, call MinerU, read PDFs/images, perform external network access, mutate registry/promotion/publication/release/global append artifacts, or add another governance gate.

This prepares a later controlled execution bridge while keeping PR #278 as pure loop summarization.
