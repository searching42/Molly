# OLED Human Review Workflow

This workflow is the human gate between deterministic OLED evidence extraction and any later adjudication, gold-candidate conversion, curated dataset writing, or training use.

## Safety Boundary

- Review decisions do not create training data.
- Review decisions do not confirm a dataset or run Phase 1.
- Every decision must be checked against the original PDF, not only the extracted text.
- Missing context must remain missing; reviewers must not infer values that are not supported by the paper.
- Only accepted `oled_compiled_record` items can enter the existing adjudication/gold-candidate bridge. Accepted text, schema, and raw candidates remain extraction-quality evidence until a separate materialization rule exists.

## Review Inputs

For a prepared run, use these files together:

1. Original PDF: the run-scoped `input/<paper>.pdf` or the explicitly supplied source PDF.
2. Human-readable admission packet: `review/oled_compiled_admission_packet.md`.
3. Machine-readable admission packet: `review/oled_compiled_admission_packet.json`.
4. Editable admission decisions: `review/oled_compiled_admission_decision_template.json`.
5. Admission readiness report: `review/oled_compiled_admission_readiness.md` and `.json`.
6. Full QA packet when supporting extraction detail is needed: `review/oled_review_packet.md` and `.json`.

Do not edit packet or readiness files. For dataset admission, edit only the compiled-admission decision JSON.
The full QA decision template is optional and does not gate compiled-record adjudication.

When a regenerated run includes `oled_reviewer_decisions_for_review.json` and
`oled_delta_review_packet.md`, use those files instead. The migrated decision
file contains prior decisions only for items whose review content is unchanged;
the delta packet contains the items reset for fresh review.

Those legacy names apply to full QA packet migration. Dataset-admission decisions
must be bound to `oled_compiled_admission_packet.json`; only a prior compiled-admission
packet may be used as its decision-migration source.

## Recommended Order

Review admission records first:

1. Complete every item in `oled_compiled_admission_packet.md`.
2. Consult the full QA packet, original PDF, schema candidates, and text evidence when a record needs source detail.
3. Review medium/low QA items separately only when measuring extractor quality or investigating an error class.

Finish all compiled-admission items before requesting downstream adjudication. A partial admission review can
be saved and revalidated, but it remains `awaiting_human_review`. Pending items in the separate full QA packet
do not block compiled-record admission.

## Regenerated Packet Review

Do not copy every decision blindly after an extractor fix. Use
`migrate_unchanged_oled_review_decisions()` to compare the old and new packets.
It requires the source decision file's `source_packet_digest` to match the
source packet, matches stable candidate identities, compares both the review
item and its complete source candidate/record payload, and carries forward only
unchanged completed decisions. Changed, new, missing, invalid, or previously
pending decisions are recreated as blank `pending` entries.

Each generated review item also stores a `source_payload_digest` for the full
candidate or compiled record, including all merged text-candidate payloads.
Readiness validation checks that digest against the current source artifact,
and the post-review bridge rechecks the actual compiled record passed to
adjudication. Replacing or regenerating an artifact after review therefore
fails closed even when its record ID is unchanged.

The migration report must record:

- source and target run IDs;
- source and target packet digests;
- the full-source-payload content-binding policy;
- migrated item and reviewed counts;
- reset pending count;
- every reset item, candidate type, source candidate ID, and reset reason.

Validate the migrated decision file without `--require-all-reviewed` before
handing the delta packet to a reviewer. The expected state is
`awaiting_human_review`, with `invalid_count` equal to zero and `pending_count`
equal to the delta packet item count.

## Per-Item Checklist

For every review item:

1. Locate the cited page, block, table, figure, or caption in the original PDF.
2. Confirm the material or compound identity. Check whether the text refers to the claimed emitter, host, dopant, device, or comparison compound.
3. Confirm the property meaning, not only the number. Distinguish PLQY, EQE, wavelength, lifetime, energy level, and device conditions.
4. Confirm the numeric value and unit, including percent-versus-fraction and maximum-versus-operating-point semantics.
5. Confirm solvent, host, doping ratio, film state, device stack, luminance/current density, temperature, and other nearby conditions when relevant.
6. Check that the evidence span actually supports the extracted claim and that the page/location is correct.
7. Choose one decision and record the audit fields below.

## Decision Rules

Interpret `accept` at the candidate's own level:

- `oled_compiled_record`: accept only when the complete layered record, including device/material context, is correct enough for downstream adjudication. A valid sentence fragment alone is not sufficient.
- `oled_schema_candidate`: accept when the source evidence is mapped to the correct layer, property/material role, value, and unit.
- `oled_text_evidence`: accept when the quoted sentence supports the extracted property/value/unit and the compound/condition association is correct. A true number attached only to a generic term such as “OLEDs” may still need more context.
- `oled_raw_candidate`: accept when the block is genuinely useful OLED evidence. This records extraction relevance only and never makes the block gold or training-ready.

### Accept unchanged

Use when the candidate is fully supported as written.

```json
{
  "review_status": "reviewed",
  "decision": "accept",
  "reviewer": "your-name",
  "reviewed_at": "2026-07-10T12:00:00+08:00"
}
```

### Accept with corrections

Use `decision: "accept"`, fill only the corrected fields, and explain the correction in `comment`.

Supported review fields are:

- `corrected_property_id`
- `corrected_value`
- `corrected_unit`
- `corrected_compound`
- `corrected_condition`

Corrections are preserved in the review report. Some corrections may still require explicit downstream materialization after review; the bridge fails closed when it cannot apply one deterministically.

### Reject

Use when the candidate is wrong, irrelevant, duplicated in a misleading way, assigned to the wrong material, or not supported by the cited evidence. A rejection requires a concise `comment`.

### Needs more context

Use when the claim may be correct but the main PDF evidence is insufficient, such as when supplementary information, a figure, an unresolved abbreviation, or an unavailable experimental condition is required. A context request requires a concise `comment` describing what is missing.

## Required Audit Fields

For every completed item:

- set `review_status` to `reviewed`;
- set `decision` to `accept`, `reject`, or `needs_more_context`;
- set `reviewer`;
- set `reviewed_at` to an ISO-8601 timestamp;
- add `comment` for rejects, context requests, and accepted corrections.

Leave unused correction fields as empty strings. Do not delete review entries or change `review_item_id`.
Do not change `run_id`, `generated_at`, or `source_packet_digest`; they bind the
decision file to the exact packet under review.

## Validate A Partial Review

```bash
PYTHONPATH=src python -m ai4s_agent.oled_review_adjudication_bridge \
  --packet runs/<run_id>/review/oled_review_packet.json \
  --decisions runs/<run_id>/review/oled_reviewer_decision_template.json \
  --output-report runs/<run_id>/review/oled_review_readiness.json \
  --output-markdown runs/<run_id>/review/oled_review_readiness.md
```

The expected status during review is `awaiting_human_review`.

## Validate A Completed Review

Add `--require-all-reviewed`. The expected status is `ready_for_adjudication`.

```bash
PYTHONPATH=src python -m ai4s_agent.oled_review_adjudication_bridge \
  --packet runs/<run_id>/review/oled_review_packet.json \
  --decisions runs/<run_id>/review/oled_reviewer_decision_template.json \
  --output-report runs/<run_id>/review/oled_review_readiness.json \
  --output-markdown runs/<run_id>/review/oled_review_readiness.md \
  --require-all-reviewed
```

## Post-Review Bridge

After a completed review, the same command can produce legacy compiled-record adjudication artifacts by adding:

```bash
  --compiled-records runs/<run_id>/extraction/oled_compiled_records.json \
  --output-legacy-packets runs/<run_id>/review/oled_mineru_review_packets.jsonl \
  --output-legacy-decisions runs/<run_id>/review/oled_legacy_decision_manifest.json \
  --output-adjudication-report runs/<run_id>/review/oled_legacy_adjudication_report.json
```

This bridge processes compiled-record items only. It does not write gold records, curated datasets, or training data.
