# Real PDF Ingestion

`ai4s_agent.run_pdf_to_dataset` is a single-paper ingestion runner for engineering validation of Molly's PDF-to-dataset path. It accepts one scientific PDF, parses it through the existing document parse service, writes a `ParsedDocument` JSON file, then hands that parsed document to the existing corpus-to-Phase-1 workflow.

This version supports engineering validation of the PDF-to-dataset pipeline. Scientific accuracy still requires human review and real corpus acceptance.

## Architecture

```text
PDF
  |
  v
DocumentParseService
  |
  +-- MinerU API provider when configured
  +-- pdfplumber baseline when selected or auto fallback applies
  |
  v
parsed_document.json
  |
  v
run_corpus_to_phase1_workflow
  |
  +-- extract_corpus_records
  +-- audit_corpus_conflicts
  +-- build_scientific_dataset
  +-- audit_corpus_reproducibility
  +-- generate_corpus_report
  |
  v
candidate/training datasets, manifest, conflict report, reproducibility artifacts
```

The runner is a composition layer. It does not implement a new parser, extractor, validator, dataset schema, or writer.

## CLI Usage

```bash
python -m ai4s_agent.run_pdf_to_dataset \
  --pdf ./paper.pdf \
  --output-dir ./runs/paper001 \
  --run-id paper001
```

MinerU can be selected explicitly when an API-compatible endpoint is available:

```bash
python -m ai4s_agent.run_pdf_to_dataset \
  --pdf ./paper.pdf \
  --output-dir ./runs/paper001 \
  --run-id paper001 \
  --provider mineru-api \
  --mineru-api-url http://localhost:8000 \
  --backend hybrid-engine \
  --parse-method auto
```

By default, the run is not confirmed. This preserves the existing `DatasetConfirmation` boundary: candidate records and manifests are written, while training rows remain gated unless explicit confirmation metadata is supplied.

```bash
python -m ai4s_agent.run_pdf_to_dataset \
  --pdf ./paper.pdf \
  --output-dir ./runs/paper001 \
  --run-id paper001 \
  --confirm-dataset \
  --confirmed-by "reviewer@example.org" \
  --confirmation-source "manual-review-2026-07-09"
```

Use confirmation only after scientific and provenance review. Confirmation may allow the existing downstream workflow to proceed beyond candidate-only materialization according to existing Molly rules.

## Output Layout

For `--output-dir ./runs/paper001`, the runner writes a run-scoped directory:

```text
runs/paper001/
├── input/
│   └── paper.pdf
├── parsed_documents/
│   └── paper001_parsed_document.json
├── extraction/
│   ├── corpus_records.json
│   ├── corpus_extraction_manifest.json
│   └── extraction_manifest.json
├── conflicts/
│   ├── corpus_conflict_report.json
│   ├── conflict_report.json
│   └── conflict_summary.json
├── dataset/
│   ├── candidate_dataset.csv
│   ├── training_dataset.csv
│   ├── rejected_records.json
│   └── dataset_manifest.json
├── report/
│   ├── corpus_report.json
│   ├── corpus_report.md
│   └── corpus_summary.json
├── reproducibility/
│   ├── corpus_lineage_manifest.json
│   ├── corpus_replay_manifest.json
│   └── corpus_reproducibility_report.json
├── corpus_workflow_report.json
└── workflow_report.json
```

`extraction/extraction_manifest.json` and `conflicts/conflict_report.json` are stable aliases for the existing workflow artifacts `corpus_extraction_manifest.json` and `corpus_conflict_report.json`.

## Artifact Description

- `input/paper.pdf`: run-scoped copy of the input PDF.
- `parsed_documents/<run_id>_parsed_document.json`: normalized `ParsedDocument` handoff from the parse service.
- `extraction/corpus_records.json`: deterministic extracted corpus records.
- `extraction/extraction_manifest.json`: extraction manifest alias for user-facing discovery.
- `conflicts/conflict_report.json`: validation and conflict report alias.
- `conflicts/conflict_summary.json`: aggregate validation/conflict counts.
- `dataset/candidate_dataset.csv`: candidate dataset rows requiring review.
- `dataset/training_dataset.csv`: training dataset artifact governed by `DatasetConfirmation`.
- `dataset/dataset_manifest.json`: dataset status, confirmation, provenance fields, validation rules, and artifact paths.
- `report/corpus_report.json` and `report/corpus_report.md`: corpus summary report.
- `reproducibility/*`: lineage, replay, and artifact hash records.
- `workflow_report.json`: top-level PDF ingestion report with input, parse, workflow, and governance metadata.

## Governance

The runner preserves existing boundaries:

- Run artifacts are isolated under the requested `--output-dir`.
- The source PDF is copied into the run before parsing.
- The workflow report records source PDF hash, parser selection, parsed-document path, artifact paths, and confirmation metadata.
- Candidate data is not silently promoted.
- `DatasetConfirmation` remains the control point for confirmed training rows and any downstream confirmed behavior.
- Existing provenance fields from parsed documents, extraction records, conflict audit, dataset manifest, and reproducibility reports are preserved.

## Limitations

- A live MinerU endpoint is not required for tests; unit tests use a mocked parse service.
- Real MinerU use requires endpoint configuration and any required upload policy settings.
- The runner handles one PDF per invocation.
- The molecule-level training dataset writer still depends on recognizable
  SMILES and required property columns.
- OLED-specific evidence and schema candidate artifacts are emitted separately
  from tables, text, figure captions, and chart captions. These artifacts are
  not silently promoted into training rows.
- Scientific correctness, license review, conflict review, and dataset confirmation remain human responsibilities.
