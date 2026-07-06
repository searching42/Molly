# OLED MinerU Review Packets

This module builds reviewer-facing packets from proposed OLED layered-record candidates produced by the offline MinerU parsed-output pipeline.

The packets are for manual inspection only. They are not gold records, accepted records, curated training data, or dataset outputs.

## Manifest Usage

Use the same parsed-output manifest shape as the acceptance harness:

```json
{
  "manifest_id": "oled-mineru-review-smoke-001",
  "bundles": [
    {
      "paper_id": "paper-003",
      "content_list_path": "paper-003_content_list.json",
      "content_list_v2_path": "paper-003_content_list_v2.json",
      "md_path": "paper-003.md",
      "source_label": "ncomms5016"
    }
  ]
}
```

Paths are resolved by the manifest loader. Bundle inputs must be local parsed-output JSON or optional Markdown sidecars. PDF and image inputs are rejected.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_mineru_review_packets \
  --manifest /path/to/manifest.json \
  --output-jsonl /path/to/review_packets.jsonl \
  --output-md /path/to/review_packets.md \
  --confirm-read-only-parsed-outputs
```

At least one output path is required. The confirmation flag is required before reading local parsed-output files.

## What The Builder Does

The runner executes the read-only parsed-output pipeline:

```text
MinerU parsed JSON/MD
-> OledMineruCandidate
-> OledSchemaCandidate
-> proposed OledLayeredRecord candidate
-> OledMineruReviewPacket
```

Each packet includes:

- paper id and optional source label
- compiled record id and compiled status
- schema warning/error codes
- source evidence anchors
- material roles and raw material names
- properties, values, units, and condition summaries
- device stack
- review decision placeholder

## What It Does Not Do

The review packet builder does not:

- call MinerU
- call LLMs
- read PDFs or images
- use OCR
- create `OledGoldDatasetRecord`
- write curated datasets
- run model backends
- treat any packet as accepted data

## JSONL Format

`write_oled_mineru_review_packets_jsonl` writes one redacted packet per line with deterministic key ordering.

Use JSONL when reviewers or downstream tools need a machine-readable checklist. The payload intentionally omits full raw paper text, full parsed JSON, absolute local paths, and full layered-record payloads.

## Markdown Review Workflow

`write_oled_mineru_review_packets_markdown` writes a human-readable checklist with one section per packet.

Recommended workflow:

1. Run the review packet builder on 1-5 OLED papers with read-only parsed outputs.
2. Open the Markdown file and compare values against the listed source evidence anchors.
3. Fill the review decision and notes manually outside the automated pipeline.
4. Do not treat reviewed packets as a curated dataset until a later explicit gold-conversion step exists.

## Review Decisions

`review_decision` defaults to `unreviewed`.

Allowed values are:

- `unreviewed`
- `accept`
- `reject`
- `needs_correction`
- `needs_source_check`

Changing this field is a human review action. It does not create gold records by itself.
