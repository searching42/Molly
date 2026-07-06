# OLED Reviewed Gold Candidates

This module converts reviewed OLED extraction candidates into `OledGoldDatasetRecord` candidates and runs the existing gold validation contract.

Gold candidates are preflight artifacts. They are not final benchmark data, curated dataset rows, or training data.

## Purpose

Use this layer after reviewed extraction candidate staging.

The converter:

- accepts staged reviewed extraction candidates
- converts eligible accepted/corrected candidates into `OledGoldDatasetRecord` candidates
- reconstructs layered OLED records from effective review packets
- preserves review provenance, packet ids, evidence anchors, and correction metadata
- runs `validate_oled_gold_dataset()`
- reports conversion status and validation codes
- writes redacted JSONL/report artifacts only when explicitly requested

## Input

The input is reviewed extraction candidate JSONL:

```bash
python -m ai4s_agent.domains.oled_reviewed_extraction_candidates \
  --adjudication-report /path/to/adjudication_report.json \
  --output-candidates /path/to/reviewed_candidates.jsonl \
  --output-report /path/to/reviewed_staging_report.json
```

## CLI Example

```bash
python -m ai4s_agent.domains.oled_reviewed_gold_candidates \
  --reviewed-candidates /path/to/reviewed_candidates.jsonl \
  --output-candidates /path/to/gold_candidates.jsonl \
  --output-report /path/to/gold_conversion_report.json
```

At least one output path is required. The CLI prints a compact summary only.

## Conversion Policy

The default policy is conservative:

- include accepted reviewed candidates
- include corrected reviewed candidates
- exclude rejected candidates
- exclude needs-source-check candidates
- require no source schema errors
- require no adjudication errors
- require evidence anchors
- keep all outputs candidate-only

Policy overrides are available through the Python API via `OledReviewedGoldConversionPolicy`.

## Validation Behavior

Every constructed `OledGoldDatasetRecord` candidate is passed through `validate_oled_gold_dataset()`.

Validation findings are copied into:

- `validation_error_codes`
- `validation_warning_codes`
- report-level `validation_code_counts`
- report findings

Candidates with validation errors remain candidates and are marked invalid. They are not accepted gold records.

## Output Meaning

Statuses:

- `converted`: converted and gold validation had no findings.
- `converted_with_warnings`: converted with validation warnings only.
- `invalid`: converted or attempted but blocked by validation/conversion errors.
- `rejected`: blocked by conservative policy or explicitly included rejected source status.
- `skipped`: source status is not eligible for conversion.

## Recommended Workflow

1. Stage reviewed extraction candidates.
2. Convert reviewed candidates to gold candidates.
3. Inspect the gold conversion and validation report.
4. Resolve invalid candidates or review decisions.
5. Only later run an explicit curated gold writer.

This module does not write curated datasets, write training data, run dataset views, run split/leakage workflows, run model backends, call LLMs, call MinerU, read PDFs/images, or mark records as final accepted benchmark data.
