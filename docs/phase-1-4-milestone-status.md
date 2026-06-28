# Phase 1-4 Milestone Status

Date: 2026-06-27

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
| Phase 3 document parsing provider layer | Completed as provider/API/normalizer/baseline infrastructure plus opt-in live acceptance evidence, not yet the full scientific closed loop | `src/ai4s_agent/document_parse_service.py`, `src/ai4s_agent/document_parse_live_acceptance.py`, `docs/document-parsing-providers.md` |
| Phase 3 to Phase 1 scientific dataset pipeline | Completed for deterministic `ParsedDocument` consumption, explicit dataset confirmation, Phase 1 baseline execution, and candidate ranking | `src/ai4s_agent/phase3_scientific_extractor.py`, `src/ai4s_agent/scientific_dataset_builder.py`, `src/ai4s_agent/phase3_to_phase1_bridge.py`, `src/ai4s_agent/workflows/phase3_to_phase1_workflow.py`, `tests/test_phase3_to_phase1_workflow.py`, `docs/phase-3-to-phase-1-pipeline.md` |
| Phase 1 training and ranking stabilization | Completed for confirmed-dataset-only training orchestration, deterministic model-based candidate ranking, and scientific report generation | `src/ai4s_agent/phase1_training_orchestrator.py`, `src/ai4s_agent/phase1_candidate_ranker.py`, `src/ai4s_agent/phase1_report_generator.py`, `src/ai4s_agent/workflows/phase1_full_pipeline.py`, `tests/test_phase1_full_pipeline.py`, `docs/phase-1-training-and-ranking-pipeline.md` |
| Multi-paper corpus evaluation and reproducibility audit | Completed for offline multi-document `ParsedDocument` fixtures, cross-paper conflict rejection, corpus replay manifests, and confirmed Phase 1 execution | `src/ai4s_agent/phase3_corpus_extractor.py`, `src/ai4s_agent/corpus_conflict_auditor.py`, `src/ai4s_agent/corpus_reproducibility_auditor.py`, `src/ai4s_agent/workflows/corpus_to_phase1_workflow.py`, `tests/test_corpus_to_phase1_workflow.py`, `docs/corpus-evaluation-and-reproducibility-audit.md` |
| MinerU live corpus acceptance bridge | Completed as a manual opt-in bridge and reusable operator gate from self-hosted MinerU parsing to the corpus workflow, with offline-tested endpoint profile/routing policy resolution, endpoint preflight diagnostics, optional preflight-report binding, and no CI live-service dependency | `src/ai4s_agent/document_parse_corpus_live_acceptance.py`, `src/ai4s_agent/corpus_live_acceptance_fixtures.py`, `src/ai4s_agent/mineru_endpoint_profiles.py`, `src/ai4s_agent/mineru_endpoint_preflight.py`, `tests/test_document_parse_corpus_live_acceptance.py`, `tests/test_mineru_endpoint_profiles.py`, `tests/test_mineru_endpoint_preflight.py`, `docs/mineru-live-corpus-acceptance.md`, `docs/mineru-endpoint-preflight.md`, `docs/mineru-manual-live-acceptance-gate.md` |
| Custom corpus dry-run runner | Implemented as a controlled manifest-described local PDF dry-run path; preserves `DatasetConfirmation.confirmed=false`, verifies Phase 1 remains `not_run`, and produces redacted dry-run evidence without production dataset admission | `src/ai4s_agent/custom_corpus_manifest.py`, `src/ai4s_agent/custom_corpus_dry_run.py`, `tests/test_custom_corpus_manifest.py`, `tests/test_custom_corpus_dry_run.py`, `docs/custom-corpus-dry-run.md`, `docs/custom-corpus-intake-contract.md` |
| Custom corpus human review schema | Introduced an offline review artifact schema and validator for custom corpus records; review artifacts still do not admit training data and do not change Phase 1 or `DatasetConfirmation` behavior | `src/ai4s_agent/custom_corpus_review.py`, `tests/test_custom_corpus_review.py`, `docs/custom-corpus-human-review.md`, `docs/examples/custom-corpus-review-manifest.example.json` |
| Custom corpus admission gate contract | Introduced an offline admission request schema and validator for structurally checking reviewed custom corpus packages; no dataset materialization, Phase 1 execution, or `DatasetConfirmation` change | `src/ai4s_agent/custom_corpus_admission.py`, `tests/test_custom_corpus_admission.py`, `docs/custom-corpus-dataset-admission-gate.md`, `docs/examples/custom-corpus-admission-request.example.json` |
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
| Phase 4 queued execute canary | Completed as feature-flagged, allowlisted, rollout-policy documented, first and second chain parity started, artifact registry parity fixture started, failure classification parity fixture started, repeated-run stability coverage started, queue recovery/stale lease coverage started, cancellation coverage started, retry/requeue semantics documented, atomic one-shot explicit retry-child creation implemented for eligible allowlisted local queue jobs, operational rollback drill and runbook documented, production-sized fixture boundary documented, optional nightly production-sized fixture lane designed, telemetry/observability checklist documented, minimal structured telemetry implemented, optional manual/nightly workflow skeleton added, and default-migration readiness checklist documented; not default migrated | `tests/test_run_plan_executor.py`, `tests/test_queued_execute_canary_artifact_parity.py`, `tests/test_queued_execute_canary_failure_parity.py`, `tests/test_queued_execute_canary_second_chain_parity.py`, `tests/test_queued_execute_canary_cancellation_retry.py`, `tests/test_queued_execute_canary_retry_requeue_semantics_docs.py`, `tests/test_queued_execute_canary_explicit_retry.py`, `tests/test_worker_queue_explicit_retry.py`, `tests/test_queued_execute_canary_operational_rollback.py`, `tests/test_queued_canary_operational_rollback_runbook_docs.py`, `tests/test_queued_execute_canary_minimal_telemetry.py`, `tests/test_queued_execute_canary_production_sized_boundary.py`, `tests/test_queued_execute_canary_nightly_fixture_lane_docs.py`, `tests/test_queued_execute_canary_observability_checklist_docs.py`, `tests/test_queued_execute_canary_repeated_run_stability.py`, `tests/test_queued_execute_canary_queue_recovery.py`, `tests/test_queued_execute_canary_default_migration_readiness_docs.py`, `tests/test_queued_canary_manual_nightly_workflow_skeleton.py`, `.github/workflows/queued-canary-manual-nightly.yml`, `docs/queued-execute-canary-rollout-policy.md`, `docs/queued-canary-retry-requeue-semantics.md`, `docs/queued-canary-operational-rollback-runbook.md` |

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

Supporting infrastructure now exists for the next parsing step:

- a stable document parsing provider contract
- a direct MinerU task-API client
- safe output-bundle extraction
- official-style MinerU output normalization into `ParsedDocument`
- a deterministic pdfplumber baseline provider
- a manual CLI and benchmark fixture
- an opt-in live MinerU API acceptance runner that writes redacted
  service/protocol, normalization, benchmark, and comparison evidence

That provider layer is not yet wired into the full Phase 3 scientific closed
loop by default and does not change current route defaults.

## Phase 3 To Phase 1 Scientific Dataset Pipeline

The first deterministic bridge from parsed scientific documents into the Phase
1 baseline stack is now available as a local workflow.

Completed behavior:

- Consumes `ParsedDocument` only; MinerU and pdfplumber remain upstream parser
  providers.
- Extracts structured scientific records from parsed tables without LLM calls,
  external APIs, MinerU calls, or PDF parsing.
- Extracts and normalizes:
  - `SMILES`
  - `PLQY`
  - `lambda_em_nm`
- Preserves mandatory training-data provenance:
  - `paper_id`
  - `page`
  - `table_id`
  - `row_id`
- Detects duplicate SMILES and conflicting property values.
- Builds candidate and training datasets through RDKit SMILES validation,
  numeric sanity checks, duplicate resolution, and rejection reason tracking.
- Requires explicit `DatasetConfirmation` before Phase 1 is invoked.
- Reuses existing Phase 1 adapters for inspection, cleaning, trainability,
  baseline evaluation, baseline training, prediction, ranking, and report
  rendering.
- Writes deterministic workflow artifacts:
  - `full_pipeline_report.json`
  - `scientific_dataset_manifest.json`
  - `phase1_baseline_report.json`
  - `candidate_ranking.json`

Evidence:

- `tests/test_phase3_scientific_extractor.py`
- `tests/test_scientific_dataset_builder.py`
- `tests/test_phase3_to_phase1_bridge.py`
- `tests/test_phase3_to_phase1_workflow.py`
- `tests/fixtures/phase3_to_phase1/`
- `docs/phase-3-to-phase-1-pipeline.md`

Boundaries:

- Does not modify MinerU providers or the document parsing layer.
- Does not add APIs, routes, queued-canary behavior, retry behavior, rollback
  behavior, or worker queue behavior.
- Does not use LLM-based extraction.
- Does not call live services.
- Does not modify Phase 1 model implementations or add ML frameworks.
- Does not trust extraction output automatically; `DatasetConfirmation` is the
  required boundary before model training.

## Phase 1 Training And Ranking Stabilization

Phase 1 now has a deterministic local pipeline for confirmed scientific
datasets.

Completed behavior:

- Accepts only datasets with explicit `DatasetConfirmation.confirmed=True`.
- Also checks the dataset manifest confirmation/status before training.
- Raises `DatasetNotConfirmedError` for unconfirmed input and does not fallback
  into training.
- Uses existing Phase 1 adapters for dataset inspection, cleaning,
  trainability, baseline evaluation, model training, candidate prediction,
  ranking, and reporting.
- Keeps baseline evaluation outputs separate from trained model outputs.
- Enforces RDKit Morgan fingerprints for the stabilized feature pipeline.
- Writes reproducibility hashes for:
  - confirmed dataset bytes
  - training configuration
  - trained model artifacts
  - ranking outputs
- Produces:
  - `training_metadata.json`
  - `feature_config.json`
  - model package artifacts
  - `ranked_candidates.csv`
  - `ranking_metadata.json`
  - `report.json`
  - `report.md`
  - `report_summary.json`
  - `full_phase1_pipeline.json`

Evidence:

- `tests/test_phase1_training_orchestrator.py`
- `tests/test_phase1_candidate_ranker.py`
- `tests/test_phase1_report_generator.py`
- `tests/test_phase1_full_pipeline.py`
- `tests/fixtures/phase1_training_and_ranking/`
- `docs/phase-1-training-and-ranking-pipeline.md`

Boundaries:

- Does not modify MinerU providers or parsing.
- Does not modify Phase 3 extraction or dataset building.
- Does not introduce LLM calls, external APIs, new ML frameworks, remote
  training services, or GPU requirements.
- Does not change queued-canary, retry, rollback, or worker queue behavior.

## Multi-Paper Corpus Evaluation And Reproducibility Audit

The corpus layer proves deterministic behavior across multiple parsed
scientific documents before any confirmed corpus dataset reaches Phase 1.

Completed behavior:

- Consumes multiple `ParsedDocument` fixtures offline.
- Reuses the single-document Phase 3 scientific extractor without modifying
  parsing providers or extraction semantics.
- Preserves corpus-level provenance:
  - `paper_id`
  - `source_document_id`
  - `parsed_document_path`
  - `parser_provider`
  - `parser_backend`
- Detects consistent duplicates and unresolved cross-paper conflicts.
- Rejects unresolved conflicts before dataset confirmation and training.
- Carries invalid SMILES and missing-property rows into rejected-record audit
  output with deterministic reason codes.
- Builds corpus candidate/training/rejected dataset artifacts through the
  existing dataset builder.
- Keeps `DatasetConfirmation` mandatory before Phase 1.
- Preserves the Phase 1 manifest-to-training-CSV binding.
- Runs the stabilized Phase 1 full pipeline only on confirmed, non-conflicting
  corpus training data.
- Writes lineage, replay, reproducibility, and corpus summary reports.

Evidence:

- `tests/test_phase3_corpus_extractor.py`
- `tests/test_corpus_conflict_auditor.py`
- `tests/test_corpus_reproducibility_auditor.py`
- `tests/test_corpus_report_generator.py`
- `tests/test_corpus_to_phase1_workflow.py`
- `tests/fixtures/corpus_multi_paper/`
- `docs/corpus-evaluation-and-reproducibility-audit.md`

Boundaries:

- Does not modify MinerU providers or document parsing infrastructure.
- Does not call live MinerU, LLMs, or external APIs.
- Does not modify Phase 1 model internals or introduce new ML frameworks.
- Does not weaken `DatasetConfirmation`.
- Does not bypass manifest-to-training-CSV binding.
- Does not change queued-canary, retry, rollback, or worker queue behavior.

## MinerU Live Corpus Acceptance Bridge

The live corpus bridge is a manual acceptance runner that connects real
self-hosted MinerU parsing to the corpus workflow without adding live CI
dependencies.

Completed behavior:

- Generates a deterministic three-document synthetic PDF corpus locally.
- Parses each PDF through the existing `DocumentParseService` and
  `MinerUApiDocumentParseProvider`.
- Optionally parses each PDF with `PdfPlumberDocumentParseProvider` as a local
  baseline.
- Copies MinerU `ParsedDocument` outputs into a corpus acceptance directory.
- Runs `corpus_to_phase1_workflow` on those parsed documents.
- Preserves the explicit `DatasetConfirmation` boundary:
  - without `--confirm-synthetic-dataset`, the decision is
    `awaiting_confirmation` and Phase 1 does not run
  - with `--confirm-synthetic-dataset --confirmed-by ...`, the synthetic
    dataset may reach Phase 1
- Resolves optional MinerU endpoint profiles and declarative manual routing
  policies from local JSON without reading tokens from profile files.
- Records only redacted endpoint profile metadata in acceptance reports.
- Provides a manual endpoint preflight that checks `/health`, protocol version
  2, response schema, redacted endpoint metadata, and node45-oriented
  CUDA/vLLM environment diagnostics before parsing.
- Optionally binds a prior `preflight_report.json` before corpus parsing:
  mismatches are warnings by default, while `--require-preflight-match` makes
  endpoint/profile/protocol/health mismatches fail before parse submission.
- Documents the self-hosted MinerU live acceptance path as a reusable manual
  operator gate, with artifact packaging, SHA-256 recording, pass/fail
  criteria, and a generic redacted evidence template.
- Implements the next custom corpus dry-run boundary:
  custom/private dry-runs keep `DatasetConfirmation.confirmed` set to `false`,
  verify Phase 1 remains `not_run`, and preserve redaction requirements before
  any real/custom records can be considered for future training admission.
- Introduces the custom corpus human review artifact boundary:
  review manifests are validated offline, but they do not admit training data,
  do not set `DatasetConfirmation.confirmed=true`, and do not run Phase 1.
- Introduces the custom corpus admission gate contract:
  admission requests can be structurally validated offline, but no dataset is
  materialized, Phase 1 is not run, and `DatasetConfirmation` is unchanged.
- Writes corpus-level acceptance evidence:
  - `acceptance_report.json`
  - `acceptance_summary.md`
  - generated PDFs
  - parsed documents
  - MinerU bundles
  - optional pdfplumber baselines
  - corpus workflow outputs
  - corpus report and replay/reproducibility manifests

Evidence:

- `src/ai4s_agent/document_parse_corpus_live_acceptance.py`
- `src/ai4s_agent/corpus_live_acceptance_fixtures.py`
- `src/ai4s_agent/mineru_endpoint_profiles.py`
- `src/ai4s_agent/mineru_endpoint_preflight.py`
- `tests/test_document_parse_corpus_live_acceptance.py`
- `tests/test_mineru_endpoint_profiles.py`
- `tests/test_mineru_endpoint_preflight.py`
- `docs/mineru-live-corpus-acceptance.md`
- `docs/mineru-endpoint-preflight.md`
- `docs/mineru-manual-live-acceptance-gate.md`
- `docs/evidence/templates/mineru-preflight-bound-live-corpus-evidence-template.md`
- `src/ai4s_agent/custom_corpus_manifest.py`
- `src/ai4s_agent/custom_corpus_dry_run.py`
- `tests/test_custom_corpus_manifest.py`
- `tests/test_custom_corpus_dry_run.py`
- `docs/custom-corpus-dry-run.md`
- `docs/custom-corpus-intake-contract.md`
- `docs/examples/custom-corpus-manifest.example.json`
- `docs/evidence/templates/custom-corpus-dry-run-evidence-template.md`
- `src/ai4s_agent/custom_corpus_review.py`
- `tests/test_custom_corpus_review.py`
- `docs/custom-corpus-human-review.md`
- `docs/examples/custom-corpus-review-manifest.example.json`
- `docs/evidence/templates/custom-corpus-human-review-evidence-template.md`
- `src/ai4s_agent/custom_corpus_admission.py`
- `tests/test_custom_corpus_admission.py`
- `docs/custom-corpus-dataset-admission-gate.md`
- `docs/examples/custom-corpus-admission-request.example.json`
- `docs/evidence/templates/custom-corpus-admission-gate-evidence-template.md`

Boundaries:

- Does not add a MinerU Cloud API provider.
- Does not change MinerU provider protocol, ZIP extraction, output
  normalization, document parsing schema, Phase 3 extraction, dataset builder,
  or Phase 1 model internals.
- Does not implement automatic live fallback, retry orchestration, canary
  routing, rollback, scheduling, or worker-pool dispatch.
- The reusable manual gate remains outside CI and does not add Cloud API
  support, automatic deployment, fallback, queues, rollback, or scheduling.
- Does not call live MinerU in tests or CI.
- Does not use LLMs or external APIs.
- Does not weaken `DatasetConfirmation`.
- Does not bypass manifest-to-training-CSV binding.
- Does not change queued-canary, retry, rollback, or worker queue behavior.
- Does not admit custom/private corpora to Phase 1 automatically.
- Does not implement human review or production dataset admission for custom
  corpora.
- Human review artifacts do not change `DatasetConfirmation`, do not create
  training datasets, and do not implement admission.
- Admission gate validation does not materialize datasets, create
  candidate/training CSVs, run Phase 1, or admit training data.
- Does not commit real PDFs or private artifacts.

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
7. The queued execute canary rollout policy and decision matrix are documented
   in `docs/queued-execute-canary-rollout-policy.md`. The policy requires
   response parity, artifact registry parity, failure classification parity,
   queue safety, rollback evidence, and no hidden scope expansion before the
   allowlist can grow.
8. Artifact registry parity fixture coverage has started for an existing
   allowlisted chain. The fixture compares sync and queued canary logical
   artifact ids plus artifact file existence, without requiring run-specific
   paths or hashes to match.
9. Failure classification parity fixture coverage has started for an existing
   allowlisted chain. The fixture compares sync and queued canary failed status,
   failed task, and useful error message fields without moving `train_model` or
   other excluded tasks into the queued canary.
10. A second allowlisted chain parity fixture has started. Because the current
    planner expansion for `render_report` still reaches non-allowlisted tasks,
    the second real all-allowlisted chain is currently
    `inspect_dataset -> clean_dataset -> check_trainability`. The fixture uses
    that actual second chain without expanding the allowlist.
11. Cancellation coverage has started for existing allowlisted queued execute
    chains. Cancelled queued jobs are not treated as the target job, and sync
    fallback does not process or mutate cancelled queued jobs.
12. Retry/requeue semantics are now documented explicitly and partially
    implemented. Lease attempts, stale recovery, explicit retry, and rerun/new
    execution are separate concepts. Stale recovery keeps the same `job_id`;
    PR #138 adds atomic one-shot explicit retry-child creation with a new
    `job_id`, preserved source-job immutability, and no public retry/requeue
    API.
13. Production-sized fixture boundary documentation has started. Current
    parity fixtures remain small and deterministic; they are useful for
    control-plane confidence, but they are not production-sized proof. A
    larger nightly or offline fixture policy is still future work.
14. Telemetry/observability checklist documentation has started, and the
    queued canary now emits a minimal structured telemetry log line for local
    review/tests. This still does not mean production telemetry, dashboards,
    alerting, or centralized sinks are implemented.
15. Optional nightly production-sized fixture lane design has started. The
    design remains docs-only: no nightly workflow is enabled, no large fixture
    data is committed, and no default presubmit test is made slower by this
    work.
16. Optional manual/nightly workflow skeleton has started. The workflow is
    manual-only via `workflow_dispatch`, uploads bounded pytest evidence, and
    does not run on `pull_request`, `push`, or a schedule.
17. Repeated-run stability coverage has started for existing allowlisted queued
    execute chains. The fixture checks project/run queue isolation, stable
    response shape, stable logical artifact ids, and rollback-to-sync behavior
    that does not touch existing queued jobs.
18. Operational rollback drill evidence has started. PR #139 proves that
    disabling `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` returns new requests
    to sync while leaving existing queued jobs, retry children, and lease
    records unchanged.
19. Remaining canary migration work includes broader observability wiring if
    needed, optional scheduled/nightly enablement only after policy gates are
    satisfied, any future actor/audit/public-route hardening for retry if
    queued execution needs it, the default migration decision, remote worker
    contract, SQLite or storage migration decision, and production scientific
    adapter validation.
