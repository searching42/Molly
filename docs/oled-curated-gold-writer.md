# OLED Curated Gold Writer Gate

This module is the first explicit materialization gate for reviewed OLED gold candidates.

It writes curated gold-record JSONL only after conservative policy checks, explicit confirmation, and a final `validate_oled_gold_dataset()` pass. The output is curated gold records, not training data or ML-ready feature tables.

## Purpose

Use this layer after reviewed extraction candidates have been converted into `OledReviewedGoldCandidate` preflight artifacts.

The writer:

- loads reviewed gold candidate JSONL
- selects writable candidates under `OledCuratedGoldWriterPolicy`
- re-runs gold validation before writing
- writes deterministic curated gold-record JSONL
- writes an audit manifest with policy, counts, reason codes, written record ids, and SHA256
- records safety metadata showing no training data, dataset views, splits, feature materialization, model backends, LLMs, MinerU, PDFs, or images were used

## Input Gold Candidate JSONL

The input is one `OledReviewedGoldCandidate` JSON object per line:

```bash
python -m ai4s_agent.domains.oled_reviewed_gold_candidates \
  --reviewed-candidates /path/to/reviewed_candidates.jsonl \
  --output-candidates /path/to/gold_candidates.jsonl \
  --output-report /path/to/gold_conversion_report.json
```

The curated writer rejects missing files, invalid JSON lines, PDF/image inputs, and obvious absolute-path leakage in candidate metadata.

## Confirmation Requirement

Curated gold writing requires explicit confirmation:

```bash
python -m ai4s_agent.domains.oled_curated_gold_writer \
  --gold-candidates /path/to/gold_candidates.jsonl \
  --output-jsonl /path/to/curated_gold_records.jsonl \
  --output-manifest /path/to/curated_gold_manifest.json \
  --confirm-curated-gold-write
```

Without `--confirm-curated-gold-write`, the CLI exits with `confirmation_required:curated_gold_write`.

## Dry-Run Mode

Dry-run mode performs selection and validation but does not write curated gold-record JSONL:

```bash
python -m ai4s_agent.domains.oled_curated_gold_writer \
  --gold-candidates /path/to/gold_candidates.jsonl \
  --output-manifest /path/to/curated_gold_manifest.json \
  --dry-run
```

Use dry-run mode to inspect policy decisions, rejected candidates, and reason codes before materializing curated records.

## Policy Gates

Default policy is conservative:

- require explicit confirmation
- require `converted` status
- reject `converted_with_warnings` unless explicitly allowed
- reject validation warnings unless explicitly allowed
- require no validation errors
- require top-level source evidence anchors
- require `gold_record.evidence_refs`
- require candidate-only source provenance
- reject any source claiming `final_gold_dataset=True`
- do not write training data
- do not run dataset views or model backends

Common reason codes:

- `missing_gold_record`
- `status_not_writable`
- `validation_errors_present`
- `validation_warnings_present`
- `missing_evidence_refs`
- `missing_reviewer`
- `source_not_candidate_only`
- `source_claims_final_gold_dataset`
- `selected_for_write`
- `post_selection_validation_error`

## Output Files

The curated record JSONL contains one redacted `OledGoldDatasetRecord` per line with deterministic key ordering.

The manifest JSON contains:

- manifest id
- input candidate count
- output record count
- redacted output JSONL path
- SHA256 of the exact written JSONL bytes
- status counts
- reason code counts
- written record ids
- per-candidate write results
- writer policy
- safety metadata

## SHA256 Manifest

`write_oled_curated_gold_records_jsonl()` returns the SHA256 hash of the exact bytes written. `run_oled_curated_gold_writer()` stores that hash in the manifest when an output JSONL path is supplied.

Use this hash to verify that a curated gold snapshot has not changed.

## Boundary

This module does not write training data, run dataset views, create split assignments, materialize features, run model backends, call LLMs, call MinerU, read PDFs, read images, use OCR, or mutate source candidate files in place.

The next explicit step should decide how curated gold records become task-specific dataset views or training artifacts.
