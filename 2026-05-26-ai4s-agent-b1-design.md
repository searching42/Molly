# AI4S Agent Design (B1, Semi-Automation)

- Date: 2026-05-26
- Scope baseline: existing `claude` progress (not PRD-first)
- Owner decision set:
  - Main architecture: `B` (Planner + Expert Agents)
  - First implementation mode: `B1` (same-process modularization)
  - Future migration: `B2` (split orchestrator service)
  - First-line capability priority: discriminative line already mostly working
  - Data mining phase-1: local data auto-mining/alignment only
  - Operation mode: semi-automation with human-in-the-loop
  - Required gates: 5 confirmation gates
  - Generative line in v1: REINVENT4 minimal closed loop
  - Reward/selection target: reuse current screening objectives (`lambda_em`, `PLQY`, `Mw`)

## 1. Objective

Build an AI4S agent on top of current MVP scripts/UI so it can:

1. Auto-run local data mining/alignment to produce trainable datasets
2. Run training and inference/screening with controlled confirmation gates
3. Run REINVENT4 generation and feed candidates back into the same screening objective
4. Keep full traceability through artifacts and stage history

## 2. B1 Architecture (Same-Process, Modular)

Keep Flask app process for first release, but split logic into explicit modules:

1. `Planner`
   - Converts prompt + panel inputs into executable plan
   - Produces `plan.json` (steps, required artifacts, gate checkpoints)
2. `Gatekeeper`
   - Handles all human confirmations
   - Blocks downstream execution until approved
   - Writes `gate_decisions.json`
3. `DataMinerAgent`
   - Local dataset discovery, column mapping, cleaning preparation
   - Reuses existing `prepare_training_entry_from_prompt.py` and mapping logic
4. `TrainerAgent`
   - Builds/validates training config and triggers training steps
   - Reuses current training and auto-train flow
5. `ScreenerAgent`
   - Inference, filtering, scoring, TopN generation, statistics/summary
   - Reuses `run_mvp_flow.py` and downstream reports
6. `GeneratorAgent` (REINVENT4)
   - Runs minimal generation workflow
   - Outputs candidates for screener re-ranking
7. `Adapter Layer`
   - Encapsulates script invocation + error taxonomy mapping (`REMOTE/WF/VAL/DATA/PRED/GEN`)
8. `Artifact Store`
   - All task artifacts under `runs/<run_id>/`

## 3. Required 5 Confirmation Gates

1. Gate-1: task parsing confirmation
   - Confirm objectives, weights, constraints, model choice, TopN
2. Gate-2: data mining result confirmation
   - Confirm selected local dataset(s), mapping quality, cleaning plan
3. Gate-3: training config confirmation
   - Confirm train properties, target columns, runtime/training setup
4. Gate-4: post-inference stats confirmation
   - Confirm distribution quality, anomalies, confidence summary
5. Gate-5: final threshold/output confirmation
   - Confirm final hard constraints and publish final top candidates

## 4. Phase-1 Data Mining Scope

In-scope:

1. Local file discovery/indexing in known workspace paths
2. Header alias matching and mapping suggestions
3. Cleaning/alignment preparation and training-entry generation

Out-of-scope:

1. External web data crawling/downloading
2. Public dataset registry auto-ingestion

## 5. Generative Minimal Closed Loop (REINVENT4)

1. Input
   - Current screening objective (`lambda_em`, `PLQY`, `Mw`, weights, constraints)
2. Generation
   - REINVENT4 generates candidate structures
3. Re-screening
   - Candidates pass through same screener/scoring path
4. Output
   - `generation_result.json`, re-ranked candidate list, traceable merge report

## 6. Execution Flow

1. Parse input and build plan
2. Gate-1 approve
3. DataMinerAgent runs
4. Gate-2 approve
5. TrainerAgent runs (or reuse model path if applicable)
6. Gate-3 approve
7. ScreenerAgent runs
8. Gate-4 approve
9. GeneratorAgent (REINVENT4) runs and feeds candidates back
10. Gate-5 approve
11. Final reports and artifacts emitted

## 7. Artifacts (v1)

Per run (under `runs/<run_id>/`):

1. `plan.json`
2. `gate_decisions.json`
3. `stage.json`
4. `task_info.json`
5. `data_mining_report.json`
6. `training_report.json` (or pointers to train artifacts)
7. `screening_report.json`
8. `generation_result.json`
9. `mvp_result.json` (existing-compatible final summary)

## 8. B2 Migration Preparation (Do Now)

Even in B1, enforce stable interfaces:

1. Planner API contract
2. Gatekeeper contract
3. Agent input/output contracts
4. Adapter command contract

With these boundaries, B2 migration is primarily process split and transport change, not logic rewrite.

## 9. Storage Convention

From now on, files created for this AI4S agent design/implementation cycle should be placed under:

- `/Users/benton/openclaw-docker/workspace/agent/`

This folder is the canonical location for design docs, plans, interface drafts, migration notes, and related artifacts for this stream.

## 10. Implementation Documents

- B1 architecture: `/Users/benton/openclaw-docker/workspace/agent/docs/architecture-b1.md`
- B2 migration readiness: `/Users/benton/openclaw-docker/workspace/agent/docs/migration-b2-ready.md`
