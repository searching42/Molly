# Corpus Evaluation And Reproducibility Audit

Date: 2026-06-27

This document describes the offline multi-paper corpus evaluation layer. It
consumes `ParsedDocument` fixtures and reuses the Phase 3 extraction, confirmed
dataset, and Phase 1 training/ranking layers. It does not add parsing behavior,
live MinerU calls, LLM extraction, external APIs, public routes, queued-canary
behavior, retry behavior, rollback behavior, worker queue behavior, model
architectures, ML frameworks, or GPU requirements.

## Boundary

```text
ParsedDocument fixtures
        |
        v
corpus extraction
        |
        v
cross-paper duplicate/conflict audit
        |
        v
candidate dataset + rejected records + manifest
        |
        v
explicit DatasetConfirmation gate
        |
        v
Phase 1 full pipeline
        |
        v
lineage, replay, reproducibility, and corpus reports
```

MinerU and pdfplumber remain upstream parsers only. The corpus workflow starts
after parsing, from normalized `ParsedDocument` inputs.

## Corpus Extraction

Module: `src/ai4s_agent/phase3_corpus_extractor.py`

Responsibilities:

- accept multiple `ParsedDocument` paths or objects
- run the existing single-document Phase 3 scientific extractor for each input
- preserve document-level provenance:
  - `paper_id`
  - `source_document_id`
  - `parsed_document_path`
  - `parser_provider`
  - `parser_backend`
- aggregate `StructuredScientificRecord` items in deterministic order
- write:
  - `corpus_records.json`
  - `per_document_extraction_reports.json`
  - `corpus_extraction_manifest.json`

The extractor does not call MinerU, parse PDFs, call external services, or use
LLMs.

## Conflict Audit

Module: `src/ai4s_agent/corpus_conflict_auditor.py`

Responsibilities:

- canonicalize SMILES with RDKit
- detect consistent duplicates across papers
- detect unresolved PLQY and `lambda_em_nm` conflicts across papers
- retain invalid SMILES and missing-property extraction failures as rejected
  records with reason codes
- ensure no unresolved conflicting records are passed to confirmed training
- write:
  - `corpus_conflict_report.json`
  - `conflict_summary.json`
  - `conflict_table.csv`

Consistent duplicates may be merged downstream by the dataset builder.
Conflicting duplicates are rejected. The workflow does not silently average
records outside tolerance.

## Dataset And Confirmation Gate

The workflow reuses `build_scientific_dataset(...)` from
`src/ai4s_agent/scientific_dataset_builder.py`.

The candidate dataset is rule-validated, but it is not automatically trusted.
Phase 1 is invoked only when the caller supplies:

```python
DatasetConfirmation(
    confirmed=True,
    confirmed_by="reviewer-or-test-fixture",
    confirmation_source="manual-review-or-fixture",
)
```

Unconfirmed corpus runs stop before Phase 1. There is no fallback path where a
clean-looking corpus dataset trains automatically.

## Reproducibility Audit

Module: `src/ai4s_agent/corpus_reproducibility_auditor.py`

The auditor computes deterministic SHA-256 hashes for input documents and
workflow artifacts, including:

- input `ParsedDocument` fixtures
- corpus records
- conflict reports
- candidate dataset
- confirmed training dataset
- rejected records
- dataset manifest
- Phase 1 pipeline output
- ranked candidates
- reports

It writes:

- `corpus_lineage_manifest.json`
- `corpus_replay_manifest.json`
- `corpus_reproducibility_report.json`

The replay manifest is the reproducibility boundary. It contains the fixture
inputs, artifact references, hashes, replay steps, and the explicit statement
that no external services are required.

## Workflow

Module: `src/ai4s_agent/workflows/corpus_to_phase1_workflow.py`

The workflow runs:

1. corpus extraction
2. cross-paper conflict audit
3. scientific dataset building
4. explicit confirmation gate
5. Phase 1 full pipeline when confirmed
6. reproducibility audit
7. corpus-level report generation

Outputs include:

- `corpus_workflow_report.json`
- `corpus_extraction_manifest.json`
- `corpus_conflict_report.json`
- `candidate_dataset.csv`
- `training_dataset.csv`
- `rejected_records.json`
- `dataset_manifest.json`
- `full_phase1_pipeline.json` when confirmed
- `report.json` and `report.md` when confirmed
- `corpus_lineage_manifest.json`
- `corpus_replay_manifest.json`
- `corpus_reproducibility_report.json`
- `corpus_report.json`
- `corpus_report.md`
- `corpus_summary.json`

## Test Fixture

Fixture directory: `tests/fixtures/corpus_multi_paper/`

The fixture is fully synthetic. It contains three parsed documents:

- paper A: valid OLED-like records
- paper B: one consistent duplicate, one conflicting duplicate, and one new
  valid molecule
- paper C: invalid SMILES, missing PLQY, missing emission wavelength, and one
  valid molecule

Expected outputs cover corpus records, conflict summary, dataset manifest
shape, and replay manifest shape.

## Explicit Non-Goals

This pipeline does not:

- modify MinerU providers
- modify document parsing infrastructure
- add a MinerU Cloud API provider
- call live MinerU
- call external APIs
- use LLM extraction
- modify Phase 1 model internals
- introduce new ML frameworks
- weaken `DatasetConfirmation`
- bypass manifest-to-training-CSV binding
- change queued-canary behavior
- change retry, rollback, or worker queue behavior
- add UI or API routes
