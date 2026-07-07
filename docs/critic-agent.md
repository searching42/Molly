# CriticAgent

## Purpose

`OLEDDiscoveryLoopAgent` makes the current OLED discovery state visible. `AgentToolRegistry` maps that state to review-only tool recommendations. `CriticAgent` adds the missing critique layer: it inspects the run card, tool recommendations, diagnostics, candidate summaries, and provenance signals, then recommends whether the discovery loop should continue, revise, rerun, request evidence, run candidate review, block overclaims, or stop.

The critic is deterministic and review-only. It does not execute the recommended action.

## Decisions

`CriticAgent` can emit these decisions:

- `continue`
- `revise_data`
- `revise_model`
- `rerun_baseline`
- `request_more_evidence`
- `run_candidate_review`
- `block_promotion`
- `stop`

Every decision includes a reason, target stage, suggested tools, findings, risk flags, blocked reasons, and recommended next actions.

## Evidence Inputs

The critic accepts synthetic summaries only:

- `OLEDDiscoveryRunCard`
- `AgentToolRecommendation` list
- dataset summary
- training package summary
- baseline summary
- diagnostics report
- candidate summary
- provenance or evidence summary
- optional model package review

These inputs are treated as already-redacted dictionaries. The critic does not open files or inspect corpus content.

## Deterministic Rules

The critic prioritizes safety boundaries and review quality:

- missing objective or blocked run card -> `request_more_evidence`
- missing provenance/evidence beyond dataset readiness -> `request_more_evidence`
- leakage, split contamination, or train/test overlap -> `revise_data`
- weak, blocked, failed, compressed, or rerun-needed diagnostics -> `rerun_baseline`
- out-of-domain candidates, invalid SMILES, or missing predictions -> `run_candidate_review`
- promotion/publication/validation wording with weak diagnostics or missing provenance -> `block_promotion`
- acceptable diagnostics without candidates -> `continue` with `candidate_generation_or_prediction`
- candidate artifacts without critic review -> `run_candidate_review`

The critic can preserve specific diagnostic risk flags such as `high_value_underprediction`, `prediction_range_compression`, and `weak_generalization`.

## Markdown And JSON

`write_review()` writes deterministic review artifacts through `ProjectStorage`:

- `critic_review.json`
- `critic_review.md`

The Markdown includes findings, risk flags, blocked reasons, recommended next actions, and a safety boundary.

## CLI Example

```bash
PYTHONPATH=src python -m ai4s_agent.agents.critic \
  --run-id demo \
  --goal "Find OLED emitters with high PLQY" \
  --current-stage diagnostics_ready \
  --diagnostics-status weak
```

The CLI prints a compact JSON summary only. It does not execute tools.

## Safety Boundary

`CriticAgent` does not execute adapters, run model training, run prediction, validate benchmarks, call LLMs, call MinerU, read PDFs/images, use external network access, mutate registry/promotion/publication/release/global append artifacts, or require a real corpus.

This prepares PR #278 to connect `OLEDDiscoveryLoopAgent`, `AgentToolRegistry`, and `CriticAgent` into a minimal review-only closed-loop harness.
