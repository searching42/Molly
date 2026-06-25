# Phase 1-4 Milestone Status

Date: 2026-06-24

This document consolidates the current AI4Science/OLED workflow milestone
status after the Phase 1-4 fixture and review-loop work. It is a status and
boundary document only. It does not introduce execution behavior.

## Summary

The project now has a low-risk, local, auditable fixture path that demonstrates
the shape of the intended workflow:

```text
Phase 1 baseline workflow
-> Phase 2 deterministic generation-to-screening
-> Phase 3 literature/table-to-dataset
-> OLED property profile and multi-objective ranking
-> Phase 4 observer-verifier, reviewable replan proposal, review artifacts,
   review card, and project-memory summary
```

This is not yet a production autonomous science system. The fixtures prove
control-plane wiring, artifact provenance, and review-loop contracts around
small local examples. Heavy model training, remote execution, large-scale
literature acquisition, and default-route migration remain future work.

## Current Status

| Area | Status | Main evidence |
| --- | --- | --- |
| Phase 1 queued workflow fixture | Completed for local lightweight fixture | `tests/test_phase1_queued_workflow_demo.py` |
| Phase 2 deterministic generation screening fixture | Completed for deterministic local generation bridge | `tests/test_phase2_generation_screening_demo.py` |
| Phase 3 literature-to-dataset fixture | Completed for parsed-table fixture to confirmed dataset | `tests/test_phase3_literature_dataset_demo.py` |
| OLED property profile + multi-objective screening | Completed for data-configured OLED fixture and weighted ranking | `tests/test_oled_multiobjective_screening_demo.py` |
| Phase 4 observer-verifier | Completed as read-only fixed schema | `src/ai4s_agent/run_plan_artifact_verifier.py` |
| Phase 4 reviewable replan proposal | Completed as deterministic non-executable proposal | `src/ai4s_agent/run_plan_replan_proposal.py` |
| Phase 4 review artifacts | Completed as review-only artifact materialization | `src/ai4s_agent/run_plan_review_artifacts.py` |
| Phase 4 review card | Completed as read-only UI/API aggregation schema | `src/ai4s_agent/run_plan_review_card.py` |
| Phase 4 project memory summary | Completed as compact memory record, summary only | `src/ai4s_agent/run_plan_review_memory.py` |
| Phase 4 replan application review artifacts | Completed as non-executing user-confirmed application drafts | `src/ai4s_agent/run_plan_replan_application_artifacts.py` |
| Phase 4 replan application audit/memory summary | Completed as append-only audit and compact memory refs only | `src/ai4s_agent/run_plan_replan_application_audit_memory.py` |
| Phase 4 internal replan application route | Completed as feature-flagged review-only route | `src/ai4s_agent/routes/internal_run_plan_queue.py` |
| Phase 4 resume intent validation semantics | Completed as docs-only validation contract | `docs/resume-intent-validation-semantics.md` |
| Phase 4 resume intent state binding | Completed as validation-only integrity hardening | `src/ai4s_agent/run_plan_state_fingerprint.py` |
| Phase 4 strict resume stage/gate validation | Completed as validation-only waiting-stage and executor-gate hardening | `src/ai4s_agent/run_plan_resume_stage_gate.py` |
| Phase 4 internal resume intent execution bridge | Completed as feature-flagged one-time internal bridge | `src/ai4s_agent/routes/internal_run_plan_queue.py` |
| Phase 4 user-confirmed resume loop | Completed as review → application → validation → actual resume → post-resume review (PR #118) | `tests/test_user_confirmed_resume_loop_e2e.py` |
| Phase 4 queued execute canary | Completed as feature-flagged observability/rollback evidence with a low-risk task-chain allowlist, not default migration | `tests/test_run_plan_executor.py` |

## Phase 1: Queued Workflow Fixture

Phase 1 productizes the existing baseline workflow through the internal queued
execution bridge in a small, local fixture.

Completed behavior:

- Uses `tests/fixtures/phase1_queued_workflow_demo/`.
- Invokes the feature-flagged internal queued execution route.
- Requires actor identity and a `run_plan_queue_execute` server grant.
- Runs a lightweight Phase 1 chain through existing local adapters.
- Writes real fixture artifacts:
  - cleaned dataset
  - baseline metrics
  - lightweight baseline model metadata
  - candidate predictions
  - ranked candidates
  - report files
  - artifact registry entries
  - queue status
  - requested/succeeded audit records

Boundaries:

- Does not replace `/api/run-plan/execute`.
- Does not connect remote workers.
- Does not use SQLite.
- Does not run heavy Uni-Mol, DPA-3, or GPU training.
- Does not prove production model quality.

## Phase 2: Deterministic Generation Screening Fixture

Phase 2 verifies the minimum bridge from generated candidates into the Phase 1
screening chain. The generator is deterministic and local.

Completed behavior:

- Uses `tests/fixtures/phase2_generation_screening_demo/`.
- Runs deterministic candidate generation.
- Registers `generated_candidates.csv` as the candidate dataset.
- Feeds generated candidates into Phase 1 prediction.
- Runs filtering/ranking and report rendering.
- Writes real fixture artifacts:
  - `generation_report.json`
  - `generated_candidates.csv`
  - `candidate_predictions.csv`
  - `ranked_candidates.csv`
  - `report.md`
  - `report.json`
  - queue status and audit records

Boundaries:

- Does not execute REINVENT4.
- Does not use external generation backends.
- Does not let an LLM generate executable code.
- Does not claim full inverse-design automation.
- Does not replace the default synchronous run-plan execution route.

## Phase 3: Literature-To-Dataset Fixture

Phase 3 verifies that structured literature/table records can become a
provenance-backed, trainability-ready dataset without doing live web or PDF
mining.

Completed behavior:

- Uses `tests/fixtures/phase3_literature_dataset_demo/`.
- Loads a parsed document/table fixture.
- Extracts table rows into structured records.
- Normalizes PLQY percent values to fractions.
- Preserves provenance fields such as paper/source/table/row context.
- Merges duplicate molecule/property records.
- Writes conflict and benchmark reports.
- Exports a confirmed dataset CSV.
- Feeds the confirmed dataset into Phase 1 dataset inspection and
  trainability checks.
- Writes real fixture artifacts:
  - `extracted_records.jsonl`
  - `extracted_records.json`
  - `unit_normalization_report.json`
  - `conflict_report.json`
  - `merged_records.json`
  - `confirmed_dataset.csv`
  - `extraction_benchmark_report.json`
  - `report.md`
  - `report.json`

Boundaries:

- Does not perform Web Search.
- Does not download papers or crawl PDFs.
- Does not run real MinerU large-model parsing.
- Does not process large literature corpora.
- Does not run heavy training on the confirmed dataset.

## OLED Property Profile And Multi-Objective Screening

The OLED fixture moves property configuration out of hardcoded core logic and
proves a small multi-objective screening flow.

Completed behavior:

- Uses `tests/fixtures/oled_property_profiles/oled_properties.json`.
- Defines OLED property metadata as data:
  - `plqy`
  - `lambda_em_nm`
  - `homo_ev`
  - `lumo_ev`
  - `delta_e_st_ev`
- Captures aliases, canonical units, optimization direction, ranking defaults,
  risk notes, and recommended task types.
- Uses `tests/fixtures/oled_multiobjective_screening_demo/`.
- Trains/predicts multiple single-property lightweight baselines.
- Merges predictions into `multi_property_predictions.csv`.
- Computes profile-driven objective score contributions.
- Ranks candidates using weighted multi-objective scores.
- Renders a report.

Boundaries:

- Does not implement full multi-task model training.
- Does not guarantee OLED-only scope in the core schema.
- Does not prevent future non-OLED domains or arbitrary property IDs.
- Does not execute external generation, Web Search, MinerU, remote workers, or
  heavy training.

## Phase 4: Observer, Replan, Review, And Memory Loop

Phase 4 is a review loop around artifacts. It is intentionally not an automatic
execution loop.

Completed layers:

1. Observer-Verifier
   - Module: `src/ai4s_agent/run_plan_artifact_verifier.py`
   - Reads queue summary/status, audit records, artifact registry, and known
     reports.
   - Evaluates trainability, model metrics, generation reports, extraction
     benchmarks, and multi-objective ranking outputs.
   - Produces `RunPlanArtifactVerification`.
   - Decision set: `continue`, `needs_review`, `rerun_recommended`, `blocked`.

2. Reviewable Replan Proposal
   - Module: `src/ai4s_agent/run_plan_replan_proposal.py`
   - Consumes only `RunPlanArtifactVerification`.
   - Produces `RunPlanReplanProposal`.
   - Uses deterministic rule-based mapping.
   - Keeps `executable=false`.
   - Produces an advisory, unapplied `proposed_run_plan_patch`.

3. Review Artifacts
   - Module: `src/ai4s_agent/run_plan_review_artifacts.py`
   - Writes:
     - `review/observer_verification.json`
     - `review/replan_proposal.json`
     - `review/replan_review.md`
   - Registers these files in the artifact registry.

4. Review Card
   - Module: `src/ai4s_agent/run_plan_review_card.py`
   - Reads the review artifacts.
   - Returns one `RunPlanReviewCard` schema for UI, report, or memory
     consumers.
   - Internal route:
     `GET /api/internal/run-plan/review-card?project_id=...&run_id=...`
   - Route is feature-flagged and requires actor identity plus
     `run_plan_queue_execute` permission.

5. Project Memory Summary
   - Module: `src/ai4s_agent/run_plan_review_memory.py`
   - Saves a compact `ProjectMemoryRecord` with category `run_plan_review`.
   - Stores only:
     - verifier decision
     - proposed action
     - affected tasks
     - required user decisions
     - artifact references
   - Avoids raw data, full artifact contents, markdown bodies, and full
     verifier/proposal payloads.

6. Replan Application Review Artifacts
   - Module: `src/ai4s_agent/run_plan_replan_application_artifacts.py`
   - Reads a user-confirmed application request and proposal artifact.
   - Verifies `proposal_hash` and selected `operation_id` values.
   - Writes review-only application artifacts:
     - `review/replan_application_record.json`
     - `review/replan_resume_intent.json`, `review/run_plan_revision.json`, or
       `review/blocked_acknowledgement.json`
   - Keeps `executable=false` and does not apply the advisory patch.

7. Replan Application Audit And Memory Summary
   - Module: `src/ai4s_agent/run_plan_replan_application_audit_memory.py`
   - Appends compact `replan_application_requested`,
     `replan_application_completed`, or `replan_application_failed` audit
     records.
   - Saves compact project memory records with category
     `run_plan_replan_application`.
   - Stores only summary fields, selected operation ids, artifact references,
     and audit references.

8. Internal Replan Application Review Route
   - Route: `POST /api/internal/run-plan/replan/apply-review`
   - Requires the internal feature flag, actor identity, and
     `run_plan_replan_apply` permission grant.
   - Accepts a `ReplanApplicationRequest`, writes requested/completed/failed
     audit records, materializes review-only application artifacts, and saves a
     compact memory summary.
   - Does not execute, enqueue, auto-resume, apply patches, call LLMs, mutate
     `RunPlan`, or replace `/api/run-plan/execute`.

9. Resume Intent Validation Semantics
   - Document: `docs/resume-intent-validation-semantics.md`
   - Defines how future gate/resume paths should validate
     `review/replan_resume_intent.json`.
   - Covers source application id, proposal hash, artifact refs, current
     `RunPlan` compatibility, rerun task presence, stale-intent detection,
     gate checks, resume audit, and default-route compatibility.
   - Does not add a resume route, enqueue work, execute adapters, write gate
     decisions, mutate `RunPlan`, call LLMs, or replace `/api/run-plan/resume`
     or `/api/run-plan/execute`.

10. Strict Resume Stage/Gate Compatibility
   - Module: `src/ai4s_agent/run_plan_resume_stage_gate.py`
   - Validates that resume intents bind to the current `WAITING_USER` stage,
     a known atomic task, a complete execution snapshot, and executor gates from
     `AtomicTaskRegistry`.
   - Separates application gates from executor gates and rejects embedded
     executor approvals in resume intent artifacts.
   - Does not call `RunPlanExecutor.resume_after_gate(...)`, write gate
     decisions, enqueue work, execute adapters, mutate `RunPlan`, call LLMs, or
     replace default routes.

11. Internal Resume Intent Execution Bridge
   - Route: `POST /api/internal/run-plan/resume-intent/execute`
   - Requires `AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE`, actor
     identity, and `run_plan_resume_execute` permission.
   - Server-loads artifacts and current state, reruns strict validation, writes
     `resume_intent_consumed` before execution, calls the existing
     `RunPlanExecutor.resume_after_gate(...)`, and records completed/failed
     audit plus compact memory.
   - Consumes each intent once. It does not enqueue work, call LLMs, mutate
     `RunPlan`, write custom gate decisions, or replace default routes.

12. User-confirmed resume loop e2e
   - `tests/test_user_confirmed_resume_loop_e2e.py` connects verifier findings,
     replan application, resume-intent validation, one-time resume execution,
     and post-resume review artifacts.
   - Confirms `resume_intent` can be consumed once, stage transitions to success,
     and post-resume artifacts/review card can be refreshed.

Phase 4 boundaries:

- Does not execute proposals.
- Does not apply patches.
- Does not call LLMs.
- Does not enqueue jobs.
- Does not mutate `RunPlan`.
- Does not automatically rerun tasks.
- Does not replace `/api/run-plan/execute`.

## Cross-Cutting Controls Now In Place

- Internal run-plan queue execute route is feature-flagged.
- Internal execute/status/review-card routes require actor identity.
- Internal routes require explicit server grant for `run_plan_queue_execute`.
- Requested and terminal audit records are written for queued execution.
- Queue paths are internal to the workspace and safe path components are
  required.
- Review/replan layers are non-executable by schema and contract.
- Project memory integration stores only compact review summaries and artifact
  references.

## Still Not Complete

The following items remain explicitly out of scope and should not be implied by
the completed fixtures:

- Default route migration:
  `/api/run-plan/execute` remains synchronous and is not replaced.
- Full queued resume:
  `WAITING_USER` remains terminal-compatible in the queue for now; resumable
  queued WAITING_USER resume remains future work.
- Remote worker:
  no remote worker contract, lease handoff, heartbeat service, or remote GPU
  execution is connected.
- SQLite:
  queue/job/project state remains file-backed; no SQLite migration has
  started.
- Real MinerU and Web Search:
  Phase 3 uses local parsed-table fixtures, not live search, downloads, or
  full PDF/miner parsing.
- REINVENT4 or external generation:
  Phase 2 uses deterministic local generation only.
- Heavy Uni-Mol/DPA-3:
  fixtures use lightweight local baselines, not production-grade heavy model
  training.
- Multi-task model training:
  OLED multi-objective screening uses multiple single-property lightweight
  baselines and weighted ranking, not a trained multi-task model.
- Autonomous replanning:
  verifier and proposal outputs are reviewable only. User confirmation and a
  future gate/resume or modified-run-plan path are still required before
  execution. The first design contract for that path is documented in
  `docs/user-confirmed-replan-application-semantics.md`.

## Suggested Next Milestones

Recommended next work should keep the same safety posture:

1. Add review-card or review-memory consumption tests in Planner/Observer
   without allowing automatic execution.
2. Implement user-confirmed proposal application schemas and tests from
   `docs/user-confirmed-replan-application-semantics.md`, still without
   automatic execution.
3. Decide whether queued `WAITING_USER` should remain terminal-compatible or
   move to a resumable non-terminal state.
4. Only after local controls stay green, revisit remote worker contracts and
  storage migration design.
5. Target-job acquisition is implemented at the queue/poller layer, while the
   run-plan service helper is constrained to the job it just enqueued. A
   feature-flagged `/api/run-plan/execute` queued canary can exercise that path,
   and rollback evidence shows disabling the flag returns to the sync response
   shape. Synchronous execution remains the default until the migration gates
   are green.
6. The queued execute canary is now restricted to selected low-risk task chains:
   `inspect_dataset`, `clean_dataset`, `check_trainability`, `run_baseline`,
   and `render_report`. Non-allowlisted tasks, including `train_model`,
   generation, literature/mining, and unknown tasks, fall back to synchronous
   execution without queued response fields.
7. Remaining canary migration work includes rollout policy, default migration
   decision, remote worker contract, SQLite or storage migration decision, and
   production scientific adapter validation.
