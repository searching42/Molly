# OLED MinerU Review Adjudication

The review adjudication gate validates human decisions over OLED MinerU review packets.

It is a gate between reviewer-facing extraction packets and a future gold-candidate conversion step. It does not create gold records, accepted dataset rows, or curated training data.

## Purpose

Use this gate after generating review packets and after a human reviewer has filled decisions or correction notes.

The gate checks:

- whether packets were accepted, rejected, marked for correction, or marked for source check
- whether every required packet has a decision
- whether decision manifests contain duplicate or unknown packet ids
- whether accepted packets still carry schema errors or warnings
- whether correction entries have enough structure to be auditable

## Decision Manifest Format

```json
{
  "review_manifest_id": "oled-review-round-001",
  "packet_source_label": "ncomms5016-smoke",
  "decisions": [
    {
      "packet_id": "review:compiled-oled:abc123",
      "review_decision": "accept",
      "reviewer_notes": "Values match Table 2."
    },
    {
      "packet_id": "review:compiled-oled:def456",
      "review_decision": "needs_correction",
      "reviewer_notes": "PLQY unit should be fraction, not percent.",
      "corrections": [
        {
          "correction_type": "property_unit",
          "field_path": "properties[0].unit",
          "original_value": "%",
          "proposed_value": "fraction",
          "reason": "Original table reports Phi_PL as fraction."
        }
      ]
    }
  ]
}
```

Duplicate packet ids are reported as adjudication errors instead of being silently ignored.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_mineru_review_adjudication \
  --packets-jsonl /path/to/review_packets.jsonl \
  --decisions /path/to/review_decisions.json \
  --output-report /path/to/adjudication_report.json \
  --require-all-reviewed
```

If `--decisions` is omitted, embedded packet decisions are used. `--output-report` is always required.

## Decision Meanings

- `accept`: reviewer believes the packet is eligible for the next conversion stage.
- `reject`: reviewer believes the packet should not move forward.
- `needs_correction`: reviewer found a fixable extraction issue.
- `needs_source_check`: reviewer needs another source/evidence check before deciding.
- `unreviewed`: no final reviewer decision has been made.

An accepted packet is not a gold record. It is only eligible for a later explicit conversion step.

## Validation Rules

The gate emits errors for:

- unknown decision packet ids
- duplicate decision packet ids
- missing decisions when `require_all_reviewed=True`
- unreviewed packets when `require_all_reviewed=True`
- accepted packets with schema errors unless explicitly allowed
- accepted packets with schema warnings when warnings are disallowed
- corrections without a field path

The gate emits warnings for:

- rejects without notes or corrections
- needs-correction decisions without structured corrections
- source-check decisions without notes
- corrections without original or proposed values

## Recommended Workflow

1. Run the review packet writer to produce JSONL and Markdown packets.
2. Have a human reviewer fill decisions and correction notes.
3. Run the adjudication gate.
4. Inspect the adjudication report and fix review manifest issues.
5. Only then run a later explicit gold-candidate conversion step.

This module does not call MinerU, call LLMs, read PDFs/images, run model backends, create `OledGoldDatasetRecord`, or write curated datasets.
