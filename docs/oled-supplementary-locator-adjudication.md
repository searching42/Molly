# OLED supplementary locator adjudication MVP

This step records human decisions over a content-bound supplementary locator
review artifact produced by the preceding locator-review step. It is an
offline adjudication boundary: accepting a locator confirms only that the
selected parsed table is the requested source table.

Locator acceptance does not validate the physical interpretation of table
labels or values. It also does not regenerate candidates, stage evidence,
create gold records, admit device-only records, or write a dataset.

## Exact input binding

The adjudicator consumes two JSON files:

1. the complete PR-E supplementary locator review artifact; and
2. a separate human decision manifest bound to both the exact review-artifact
   bytes and its canonical content digest.

The decision manifest has this shape:

```json
{
  "schema_version": "oled_supplementary_locator_decision_manifest.v1",
  "run_id": "supp-mineru-run-001",
  "paper_id": "paper016",
  "review_artifact_sha256": "sha256:<exact file hash>",
  "review_artifact_digest": "sha256:<canonical artifact digest>",
  "adjudication_confirmed": true,
  "decisions": [
    {
      "review_item_id": "supplementary-locator-review:<bound item id>",
      "decision": "accept_locator",
      "reviewed_by": "reviewer-id",
      "reviewed_at": "2026-07-13T15:30:00+08:00",
      "review_note": "The locator, page, and caption identify Supplementary Table S1.",
      "semantic_note": "HOMO/LUMO labels are preserved as reported but require semantic review"
    }
  ]
}
```

`adjudication_confirmed` must be `true`. `run_id` and `paper_id` must match the
review artifact, and the two review-artifact digests must match independently:
reserializing otherwise equivalent JSON changes the exact file hash and
invalidates the manifest. Decisions must cover every review item exactly once;
unknown, duplicate, or missing item IDs fail closed. Reviewer timestamps must
include a timezone. Duplicate JSON object keys are rejected instead of using
last-value-wins parsing, and `adjudication_confirmed` must be the literal JSON
boolean `true`.

The manifest is intentionally not a correction interface. Extra fields for a
replacement locator, page, header, cell, unit, or value are rejected. If the
locator or transcription is wrong, the reviewer must reject it or request a
source check; a corrected, newly bound locator-review artifact must be
generated before acceptance.

## Decisions

The only decision values are:

- `accept_locator`: the requested locator is an `exact_match` and its selected
  table binding was confirmed against the source;
- `reject_locator`: the selection is not acceptable and must not advance;
- `needs_source_check`: the source evidence is insufficient for a final
  locator decision.

Only an `exact_match` item may receive `accept_locator`. An unresolved,
ambiguous, unsupported-kind, or unsupported-format item cannot be accepted.
Both `reject_locator` and `needs_source_check` require a non-empty
`review_note`.

The aggregate artifact status is `all_locators_accepted`,
`partially_accepted`, or `no_locators_accepted`. A valid reject or source-check
decision is a successful adjudication result, not a CLI contract failure.

## Semantic-review boundary

`semantic_note` preserves a scientific caveat without modifying the source
evidence. A non-empty note sets `semantic_review_required=true` on that item
and contributes to `semantic_review_required_count`.

For example, the note
`HOMO/LUMO labels are preserved as reported but require semantic review`
records that the literal labels were retained while their physical meaning is
still unresolved. The slash in `HOMO/LUMO` is normal scientific text; it is
not a local path. Notes must not contain URLs, absolute paths, credentials, or
control characters.

Credential filtering is applied identically to `reviewed_by`, `review_note`,
and `semantic_note`. It rejects bounded credential assignments such as
`token=...`, `access_token: ...`, `api key=...`, and `private key: ...`, plus
Bearer credentials and common `sk-...` key forms. Unassigned scientific or
descriptive uses of words such as “token” and “secret” are not rejected merely
for containing those words.

Even when the locator is accepted:

- `physical_semantics_validated` remains `false`;
- `table_transcription_validated` and `scientific_content_validated` remain
  `false`;
- `semantic_correction_applied` remains `false`;
- `direct_admission_eligible` remains `false`; and
- the source table is not interpreted as schema-ready data.

Acceptance sets only
`eligible_for_later_scoped_candidate_proposal=true`. A later, separately
reviewed proposal stage must carry the semantic-review requirement forward and
must not treat this adjudication as approval of HOMO/LUMO or any other
scientific interpretation.

## Redacted output

The output is one JSON adjudication artifact. It keeps the chain of evidence
through hashes and locator bindings, including:

- exact and canonical digests for the review artifact and decision manifest;
- upstream execution, locator-manifest, and preflight digests;
- a digest of each original review item;
- source and parsed-document hashes;
- target kind, reported locator, canonical locator, and match status; and
- matched table ID, page, and table-content digest for an exact match.

It does not copy table captions, headers, rows, footnotes, source bounding
boxes, or other matched-table content from the PR-E packet. The original review
artifact remains the authoritative object for inspecting those values.

The artifact records that human decisions are complete while keeping all
downstream actions disabled. It does not read parsed output or PDFs, access the
network, call MinerU or an LLM, apply locator/content corrections, perform
schema mapping, regenerate or merge candidates, stage reviewed evidence,
admit device-only data, create gold records, or write a dataset.

## Run the adjudicator

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_locator_adjudication \
  --review-artifact /operator/local/supplementary_locator_review.json \
  --decision-manifest /operator/local/supplementary_locator_decisions.json \
  --output /operator/local/supplementary_locator_adjudication.json
```

Both inputs are read as stable regular files with `O_NOFOLLOW`. The output must
be fresh and may not overwrite either input. CLI output contains only a
redacted status/count summary; review notes, table content, and local paths are
not printed.

After a successful adjudication, only accepted locator bindings are eligible
for a later explicitly scoped candidate-proposal step. That later step remains
responsible for semantic interpretation and its own human-review boundary.
