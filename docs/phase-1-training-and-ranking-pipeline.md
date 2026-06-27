# Phase 1 Training And Ranking Pipeline

Date: 2026-06-27

This document describes the stabilized Phase 1 scientific modeling pipeline.
It is strictly downstream of confirmed datasets produced by the Phase 3 to
Phase 1 dataset boundary. It does not change MinerU, document parsing, Phase 3
extraction, queued-canary behavior, retry behavior, rollback behavior, worker
queue behavior, or model internals.

## Boundary

```text
confirmed_training_dataset.csv
candidate_dataset.csv
dataset_manifest.json
DatasetConfirmation(confirmed=True)
        |
        v
Phase 1 training orchestrator
        |
        v
Phase 1 candidate ranker
        |
        v
Phase 1 scientific report generator
```

No dataset can enter this pipeline unless the caller supplies an explicit
`DatasetConfirmation` with `confirmed=True` and the dataset manifest is also
confirmed. Unconfirmed datasets raise `DatasetNotConfirmedError`; there is no
fallback training path.

## Training Orchestrator

Module: `src/ai4s_agent/phase1_training_orchestrator.py`

Responsibilities:

- accept only confirmed training datasets
- enforce RDKit Morgan fingerprint availability
- run existing Phase 1 adapters:
  - `inspect_dataset_service`
  - `execute_cleaning_adapter`
  - `check_trainability_service`
  - `run_baseline_service`
  - `train_model_baseline_adapter`
- keep baseline evaluation outputs separate from trained model outputs
- persist:
  - `feature_config.json`
  - `training_metadata.json`
  - model package artifacts from the existing baseline trainer
- compute reproducibility hashes:
  - dataset hash
  - training configuration hash
  - model hash per trained property

The training configuration fixes the feature pipeline to RDKit Morgan
fingerprints. It does not introduce dynamic feature selection or new model
frameworks.

## Candidate Ranking

Module: `src/ai4s_agent/phase1_candidate_ranker.py`

Responsibilities:

- load trained model metadata
- predict candidate properties using existing baseline prediction adapters
- rank candidates using model prediction columns
- write:
  - `ranked_candidates.csv`
  - `ranking_metadata.json`

Ranking is model-based: the score is computed from predicted property columns,
not from a heuristic over raw candidate metadata. The default scoring function
uses `filter_rank_adapter` with deterministic weights and maximize directions.

## Report Generation

Module: `src/ai4s_agent/phase1_report_generator.py`

Responsibilities:

- combine training metadata, model artifact info, ranking metadata, and dataset
  manifest provenance
- preserve the confirmation gate status
- include model configuration and available training metrics
- include candidate ranking summary
- include reproducibility hashes
- write:
  - `report.json`
  - `report.md`
  - `report_summary.json`

## Full Workflow

Module: `src/ai4s_agent/workflows/phase1_full_pipeline.py`

The workflow runs:

1. confirmation gate
2. deterministic training orchestration
3. model artifact persistence
4. candidate prediction
5. model-based ranking
6. report generation

It writes `full_phase1_pipeline.json` with artifact references and hashes.

## Test Fixture

Fixture directory: `tests/fixtures/phase1_training_and_ranking/`

The fixture reuses the synthetic confirmed dataset shape from the Phase 3 to
Phase 1 bridge. It contains:

- `confirmed_training_dataset.csv`
- `candidate_dataset.csv`
- `dataset_manifest.json`

No new molecule generation is introduced by this pipeline.

## Explicit Non-Goals

This pipeline does not:

- modify MinerU or document parsing layers
- modify Phase 3 extraction or dataset building logic
- introduce LLM usage
- introduce external APIs or training services
- introduce new ML frameworks
- bypass `DatasetConfirmation`
- change queued-canary behavior
- change retry, rollback, or worker queue behavior
- require GPU training
