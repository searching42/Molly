# OLED Reviewed Extraction Candidates

Reviewed extraction candidates are staged, auditable extraction objects produced from OLED MinerU review adjudication reports.

They are not gold records, curated dataset rows, training data, or automatically accepted scientific facts.

## Purpose

Use this layer after human review adjudication and before any future gold-candidate conversion.

The staging layer:

- stages accepted adjudicated review packets
- optionally stages rejected or source-check packets for audit trails
- applies supported structured correction proposals to packet-level fields
- preserves original packet snapshots and corrected packet snapshots
- records correction application status and finding codes
- writes redacted JSONL and staging reports

## Input

The input is an `OledReviewAdjudicationReport`, usually written by:

```bash
python -m ai4s_agent.domains.oled_mineru_review_adjudication \
  --packets-jsonl /path/to/review_packets.jsonl \
  --decisions /path/to/review_decisions.json \
  --output-report /path/to/adjudication_report.json \
  --require-all-reviewed
```

## CLI Example

```bash
python -m ai4s_agent.domains.oled_reviewed_extraction_candidates \
  --adjudication-report /path/to/adjudication_report.json \
  --output-candidates /path/to/reviewed_candidates.jsonl \
  --output-report /path/to/reviewed_staging_report.json
```

At least one output path is required. The command prints only a compact summary.

## Correction Field Paths

Supported deterministic field paths are:

- `properties[N].value`
- `properties[N].unit`
- `properties[N].property_label`
- `properties[N].property_id`
- `material_roles[N].role`
- `material_roles[N].material_name`
- `device_stack`
- `device_stack[N]`
- `reviewer_notes`
- `metadata.<key>`

Unsupported paths are not silently applied. They produce correction findings and failed application status.

If `original_value` is supplied and does not match the current packet value, staging emits `correction_original_value_mismatch` and still applies the proposed value. This keeps reviewer intent deterministic while preserving an auditable warning.

## Candidate Status Meanings

- `accepted`: an accepted adjudicated packet was staged without correction application.
- `corrected`: all supplied correction proposals were applied.
- `rejected`: a rejected packet was staged only because `include_rejected=True`.
- `needs_source_check`: a source-check packet was staged only because `include_needs_source_check=True`.
- `needs_correction`: correction proposals were absent, disabled, or not fully applied.
- `invalid`: reserved for invalid staged objects; invalid adjudicated packets are not staged by default.

## Correction Application Status

- `not_applicable`: no correction application was relevant.
- `applied`: the correction was applied deterministically.
- `partially_applied`: reserved for future compound corrections.
- `not_applied`: the correction was recognized but not applied.
- `failed`: the correction could not be applied safely.

## Recommended Workflow

1. Generate review packets.
2. Adjudicate reviewer decisions.
3. Stage reviewed extraction candidates.
4. Inspect the staging report and correction findings.
5. Only then run a later explicit gold-candidate conversion step.

This module does not call MinerU, call LLMs, read PDFs/images, use OCR, run model backends, create `OledGoldDatasetRecord`, call gold validation, write curated datasets, or write training data.
