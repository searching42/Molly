# Phase 3 To Phase 1 Scientific Dataset Pipeline

Date: 2026-06-27

This document describes the deterministic bridge from parsed scientific
documents into the existing Phase 1 baseline modeling stack. It is a boundary
document only. It does not add public routes, queued-canary behavior, live
MinerU calls, or new model implementations.

## Pipeline Boundary

```text
MinerU or pdfplumber provider
        |
        v
ParsedDocument
        |
        v
Phase 3 scientific extraction
        |
        v
OLED review packet generation
        |
        v
candidate scientific dataset
        |
        v
explicit DatasetConfirmation gate
        |
        v
Phase 1 baseline training and candidate ranking
```

MinerU is only an upstream parser in this flow. The Phase 3 extractor consumes
the normalized `ParsedDocument` schema and does not call MinerU, parse PDFs,
access external services, or use an LLM.

## Phase 3 Extraction Layer

Module: `src/ai4s_agent/phase3_scientific_extractor.py`

The extractor converts a `ParsedDocument` into deterministic
`StructuredScientificRecord` items for the legacy molecule-level training
path. That path still targets structured table evidence with columns that
identify:

- `SMILES`
- `PLQY`
- `lambda_em_nm`
- optional row confidence

Responsibilities:

- select table evidence from `ParsedDocument.tables`
- extract SMILES and photophysical property values
- normalize PLQY percent values to fractions
- normalize emission wavelength values to nanometers
- preserve mandatory provenance:
  - `paper_id`
  - `page`
  - `table_id`
  - `row_id`
- reject rows with missing SMILES, low confidence, or missing required
  properties
- detect duplicate SMILES conflicts using deterministic tolerances
- produce an extraction report and conflict report

The same extractor also emits OLED evidence/schema candidates for real OLED
papers that do not expose molecule-level SMILES tables:

- `oled_candidates.json` contains table, text, figure, and chart evidence
  candidates from `ParsedDocument.tables` and `ParsedDocument.elements`.
- `oled_text_evidence_candidates.json` contains deterministic text-derived
  review candidates from paragraphs, section text, figure captions, and chart
  captions. Candidates preserve the evidence span, lightweight compound
  mentions, property id, raw value, normalized numeric value, unit, nearby
  condition text, confidence, and provenance.
- `oled_schema_candidates.json` contains deterministic OLED layered-schema
  candidates for mapped table columns such as host, dopant, PLQY, EQE,
  `delta_e_st_ev`, doping ratio, voltage, luminance, and device context.
- `oled_compiled_records.json` contains proposed layered record candidates
  compiled from those schema candidates.

The table-first path remains the only deterministic route toward structured
OLED schema and compiled candidates in this workflow. Text evidence candidates
are separate, review-only recall aids for papers where MinerU extracts useful
property statements but no structured tables. They are not compiled into
layered records, do not enter the RDKit/SMILES training dataset, and do not
bypass `DatasetConfirmation`.

No extraction output is trusted automatically for training.

## OLED Review Packet Layer

Module: `src/ai4s_agent/oled_review_packet_generator.py`

After candidate extraction, the corpus workflow writes review artifacts under
`review/`:

- `oled_review_packet.json`
- `oled_review_packet.md`
- `oled_reviewer_decision_template.json`
- `oled_review_summary.json`

The review packet generator reads only run-scoped candidate artifacts:

- `extraction/oled_candidates.json`
- `extraction/oled_text_evidence_candidates.json`
- `extraction/oled_schema_candidates.json`
- `extraction/oled_compiled_records.json`
- `extraction/corpus_extraction_manifest.json`

It deterministically converts those artifacts into pending review items for
human adjudication. Review items keep source candidate ids, paper ids,
candidate type, priority, property/value/unit fields when available,
compound/material mentions, condition/device context, evidence text, page or
location fields, provenance, warnings, and suggested review questions.

`oled_review_packet.md` is the reviewer-facing packet. Reviewers should compare
each item against the original PDF and fill decisions in
`oled_reviewer_decision_template.json`. The decision template intentionally
starts with empty pending decisions.

This layer does not create a gold dataset, does not confirm data, does not
create training rows, and does not auto-accept any candidate. Accepted review
decisions are for a later adjudication PR to consume explicitly. The
`DatasetConfirmation` gate remains the only path to confirmed training rows.

Review packet paths and counts are propagated through:

- `corpus_workflow_report.json`
- `dataset/dataset_manifest.json`
- `report/corpus_report.json`
- `report/corpus_report.md`
- `reproducibility/corpus_replay_manifest.json`

## Dataset Builder

Module: `src/ai4s_agent/scientific_dataset_builder.py`

The builder converts extracted records into dataset artifacts:

- `candidate_dataset.csv`
- `training_dataset.csv`
- `rejected_records.json`
- `dataset_manifest.json`

Validation rules:

- SMILES must be valid under RDKit canonicalization.
- PLQY must be within the configured fraction range.
- `lambda_em_nm` must be within the configured wavelength range.
- consistent duplicate SMILES records are merged.
- inconsistent duplicate SMILES records are rejected with reason
  `duplicate_conflict`.

Only confirmed records enter `training_dataset.csv`. Rejected records are
fully traceable through reason codes and provenance fields.

## Confirmation Gate

The gate is represented by `DatasetConfirmation`:

```python
DatasetConfirmation(
    confirmed=True,
    confirmed_by="manual-reviewer-or-test-fixture",
    confirmation_source="manual-review",
    confirmation_timestamp="2026-06-27T00:00:00Z",
)
```

The bridge layer enforces the gate. If `confirmation.confirmed` is false, the
pipeline stops after dataset artifact generation and does not invoke Phase 1.

There is no implicit trust fallback. A clean-looking candidate dataset is still
not authorized for model training unless the confirmation context is explicit.

## Phase 1 Bridge

Module: `src/ai4s_agent/phase3_to_phase1_bridge.py`

The bridge reuses existing Phase 1 adapters only:

- `inspect_dataset_service`
- `execute_cleaning_adapter`
- `check_trainability_service`
- `run_baseline_service`
- `train_model_baseline_adapter`
- `predict_candidates_baseline_adapter`
- `filter_rank_adapter`
- `render_report_adapter`

It does not modify Phase 1 model logic, does not add model architectures, and
does not introduce new ML frameworks.

## Workflow Entrypoint

Module: `src/ai4s_agent/workflows/phase3_to_phase1_workflow.py`

The deterministic workflow writes:

- `full_pipeline_report.json`
- `scientific_dataset_manifest.json`
- `phase1_baseline_report.json`
- `candidate_ranking.json`

When the confirmation gate is not satisfied, only the Phase 3 extraction and
dataset artifacts are produced. Phase 1 outputs remain empty.

## Test Fixture

Fixture directory: `tests/fixtures/phase3_to_phase1/`

The fixture is fully synthetic and contains:

- `parsed_document.json`
- `expected_extraction.json`
- `expected_dataset.csv`
- `expected_conflicts.json`

It is designed to cover successful extraction, unit normalization, provenance,
duplicate merging, duplicate conflicts, invalid SMILES rejection, numeric range
validation, and the explicit confirmation gate.

## Explicit Non-Goals

This pipeline does not:

- modify MinerU providers
- modify the document parsing layer
- add public APIs or routes
- call MinerU
- call external services
- perform PDF parsing
- use LLM-based extraction
- change queued-canary behavior
- change retry, rollback, worker queue, or route semantics
- modify Phase 1 model implementations
- introduce new ML frameworks
