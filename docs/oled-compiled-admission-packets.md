# OLED Compiled-Record Admission Packets

The compiled-record admission packet is the dataset-admission review surface for OLED literature extraction.
It contains only `oled_compiled_record` items, because those are the only review items that can enter the
existing adjudication and gold-candidate path.

The full OLED review packet is still generated and retained as an extraction-quality QA artifact. Raw,
schema, and text evidence remain available for debugging, recall analysis, and source inspection, but they
do not need individual decisions before compiled-record adjudication.

## Generated Artifacts

Every OLED corpus workflow writes both packet families under `review/`:

```text
oled_review_packet.json
oled_review_packet.md
oled_reviewer_decision_template.json
oled_review_summary.json

oled_compiled_admission_packet.json
oled_compiled_admission_packet.md
oled_compiled_admission_decision_template.json
oled_compiled_admission_summary.json
```

Use the first group for full extraction QA. Use the second group for dataset-admission review.

## Safety Model

The admission packet reuses `oled_review_packet.v1` and the existing decision-template schema. Each item
retains the complete compiled-record source-payload digest. The admission packet itself has an independent
packet digest, so decisions cannot be copied from the full QA packet without rebinding and validation.

The existing `require_all_reviewed` behavior is unchanged. It applies to every item in the selected packet.
Because the admission packet contains only compiled records, every compiled record must be reviewed while
QA-only raw, schema, and text items remain outside the admission gate.

Passing admission review still does not create gold data or training rows. The explicit adjudication,
gold-candidate conversion, curated-writer confirmation, dataset-view, split, and modeling gates remain
separate.

## Review Content

The Markdown admission packet shows, for each compiled record:

- record and compilation status;
- device and system labels;
- material roles and device stack;
- every molecular, interaction, device, and measurement observation;
- values, units, conditions, and source schema candidate ids;
- evidence anchors, confidence, schema errors, schema warnings, and source-payload digest.

Reviewers must decide whether the complete layered record is correct enough for adjudication. A correct
number with an incorrect compound, role, device, or condition association must not be accepted.

## Validation

Validate an in-progress admission review with the existing bridge:

```bash
PYTHONPATH=src python -m ai4s_agent.oled_review_adjudication_bridge \
  --packet runs/<run_id>/review/oled_compiled_admission_packet.json \
  --decisions runs/<run_id>/review/oled_compiled_admission_decision_template.json \
  --output-report runs/<run_id>/review/oled_compiled_admission_readiness.json \
  --output-markdown runs/<run_id>/review/oled_compiled_admission_readiness.md
```

After every admission item has a final decision, add `--require-all-reviewed`. The expected status is
`ready_for_adjudication`. The post-review legacy bridge may then consume the same admission packet and the
run's `oled_compiled_records.json`.

If a run has no compiled records, readiness is `no_eligible_items`, not `ready_for_adjudication`. Such a run
must not enter the adjudication or gold-candidate path. An admission packet containing any non-compiled item
is blocked as invalid.
