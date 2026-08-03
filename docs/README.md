# Molly documentation map

This directory contains durable technical contracts, operator guidance,
machine-readable schemas, examples, and reviewed evidence summaries.

[`../todo.md`](../todo.md) is the only normative source for roadmap scope,
milestone status, priorities, execution order, and decisions. Documents here
may describe how a capability works or how it is accepted; they do not supersede
the current state recorded in `todo.md`.

## Core technical guides

| Area | Entry points |
|---|---|
| Execution and human gates | [Resume-intent validation](resume-intent-validation-semantics.md), [user-confirmed replan application](user-confirmed-replan-application-semantics.md), [generic confirmation acceptance](generic-run-plan-confirmation-success-acceptance.md) |
| Session, recovery, publication, and replay | [Bounded discovery session](oled-bounded-discovery-session.md), [bounded controller](oled-bounded-discovery-controller.md), [remote execution lifecycle](stage6b-remote-execution-lifecycle.md) |
| Literature intake and parsing | [Literature intake](literature-intake.md), [document parsing providers](document-parsing-providers.md), [real PDF ingestion](real-pdf-ingestion.md) |
| Remote resources | [Local deployment](local_deployment.md), [remote worker setup](remote_worker_setup.md), [resource profiles](stage6a-resource-profiles.md) |
| Observation and trajectory integrity | [Unified Scientific Agent run inspection](scientific_agent_run_inspection_v1.md), [privacy-safe Harness observability](scientific_agent_observability_v1.md), [Control-plane event projection and SSE](control-plane-event-projection.md), [deterministic trajectory audit metrics](oled-scientific-agent-trajectory-audit-metrics.md), [failure attribution](oled-scientific-agent-failure-attribution.md), and [read-only trajectory inspection](oled-scientific-agent-trajectory-inspection.md); current status remains in `todo.md` under M2/M3/M3.5 |
| Scientific workflow | [Structured Dataset Canary v1](structured_dataset_canary_v1.md), [Phase 3 to Phase 1 pipeline](phase-3-to-phase-1-pipeline.md), [OLED bounded discovery session](oled-bounded-discovery-session.md), [OLED inverse design](oled-inverse-design.md) |
| Acceptance and operations | [Private Structured Dataset Canary runbook](private_structured_dataset_canary_runbook.md), [OLED MVP quickstart](oled-mvp-demo-quickstart.md), [queued-canary rollback runbook](queued-canary-operational-rollback-runbook.md), [Uni-Mol compatibility acceptance](manual-real-unimol-acceptance.md) |

Focused documents beside these entry points define individual adapters,
schemas, preflights, publication writers, and acceptance fixtures. Their scope
is intentionally narrow so they can be checked against the corresponding source
module and test.

## Supporting material

- `schemas/` contains generated or maintained public JSON contracts.
- `examples/` contains synthetic or public-safe example inputs.
- `evidence/` contains reviewed, sanitized summaries only. Raw papers, runtime
  bundles, concrete infrastructure locators, and user data stay outside Git.
- `evidence/templates/` contains purpose-specific evidence templates. Similar
  templates are retained when their schemas and gate positions differ.

## Historical context

The following migrated documents remain because tests or current contracts
refer to their stable identifiers. They are historical implementation evidence,
not active roadmaps:

- [OPEN issue register](open-issues.md)
- [Phase 1–4 milestone snapshot](phase-1-4-milestone-status.md)
- [Post-OPEN hardening record](post-open-hardening.md)
- [Bounded closed-loop delivery record](oled-bounded-closed-loop-roadmap.md)
- [Custom-corpus governance stage snapshot](custom-corpus-governance-stage-summary-20260628.md)

Pre-migration PR numbers in those records refer to the private audit archive
unless an explicit public GitHub URL is present. Public PR numbering restarted
after the 2026-07-27 migration.

Dated task-by-task implementation plans and tool-specific Agent instructions
are not retained here after implementation. The private audit archive preserves
their original history for authorized review.

## Documentation rules

- Prefer relative Markdown links and keep referenced repository paths valid.
- Use logical resource IDs and synthetic examples; never publish resolved
  machine paths, credentials, private source material, or raw runtime bundles.
- Mark historical snapshots as non-normative.
- Update `todo.md`, not a topic document, when roadmap status or execution order
  changes.
