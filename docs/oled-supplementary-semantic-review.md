# OLED supplementary semantic review and adjudication MVP

This stage turns one exact-bound PR-H response into a compact packet for human
semantic review, then records a complete human decision manifest as an
exact-bound adjudication artifact. It reduces repeated review work without
discarding cell-level coverage. It does not correct the response inline,
resolve material identity, extend the ontology, create schema candidates, or
admit data.

## Mandatory evidence chain

Packet generation requires all three original upstream files:

- the complete PR-G scoped candidate request, including source identity,
  locator, caption, headers, rows, footnotes, pages, and bounding boxes;
- the original response manifest supplied to PR-H, including its authorship
  provenance; and
- the validated PR-H response artifact.

The packet builder replays the PR-H request/response validation instead of
trusting copied summaries. It verifies the exact file SHA-256 and canonical
digest for the request and response manifest, verifies the PR-H artifact's
exact bytes and canonical digest, and rechecks run, paper, scope, table, and
cell bindings across the chain. Replacing any upstream file after validation
therefore invalidates the packet.

Adjudication requires the same three upstream files plus the exact review
packet and a separate complete human decision manifest. The decision manifest
binds the packet's exact file SHA-256 and canonical digest. The adjudicator
replays the complete upstream chain before accepting decisions; it does not
treat the packet's display copies as independent source evidence.

Inputs are bounded stable regular files, read without following symlinks.
Duplicate JSON keys, non-finite JSON numbers, input/output path collisions,
changed bytes, and incomplete or inconsistent bindings fail closed. Outputs
must be fresh. Unsafe authored response and reviewer text also fails closed;
literal source-table text instead stays unchanged in the JSON evidence and is
HTML/table escaped with display-control characters made visible in the
read-only Markdown.

## Compact review packet

The packet shows each source table once and preserves the complete PR-G review
context. Numeric-cell dispositions from PR-H are grouped only when they share
the same bound scope, table, source column, disposition kind, and proposed
mapping semantics. A deterministic review-item digest binds every grouped
cell and every displayed proposal field.

Grouping is presentation compression, not evidence compression. The packet
enforces a strict cell partition:

- every PR-H numeric-cell disposition belongs to exactly one mapping group;
- no bound cell can be omitted, duplicated, moved to another table, or added;
- the union of group cells equals the complete independently derived numeric
  roster; and
- each adjudicated group decision is expanded back to all of its exact cell
  bindings in the output artifact.

Each non-empty PR-G semantic note is a separate semantic-note review item. It
is never hidden inside a mapping group and cannot be silently treated as
resolved by accepting a column disposition.

For the paper016 Supplementary Table S1 canary, 49 numeric dispositions are
presented as seven column-level mapping groups plus one semantic-note item.
The seven groups still partition all 49 cells exactly once. The HOMO/LUMO note
remains an independent decision, and the two unsupported columns remain
ontology-review dispositions rather than becoming schema properties.

## Human decisions

The decision manifest must be explicitly marked complete, cover every review
item exactly once, preserve the exact item ID and digest, identify the human
reviewer, and use timezone-aware review timestamps. Unknown, duplicate,
missing, stale, or kind-incompatible decisions fail closed. Decision entries
may follow the human-friendly order shown in the Markdown; canonical hashing
sorts them by `review_item_id` internally. The rendered Markdown displays the
exact packet-file SHA-256 plus every item ID, item kind, and item digest needed
to construct this file. For ontology-review summaries, an empty
`reported_unit` is displayed as `no explicit unit in source header`; this is a
display explanation only, and the packet retains the exact empty string. An
explicit source literal such as `unitless` remains displayed verbatim.

The decision manifest has this exact shape (repeat `decisions` once for every
rendered item and copy all bound values literally):

```json
{
  "schema_version": "oled_supplementary_semantic_decision_manifest.v1",
  "run_id": "<copy packet run_id>",
  "paper_id": "<copy packet paper_id>",
  "review_packet_sha256": "<exact packet-file SHA-256 shown by render>",
  "review_packet_digest": "<copy packet digest>",
  "reviewed_by": "<reviewer identifier>",
  "reviewed_at": "2026-07-13T22:20:00+08:00",
  "adjudication_confirmed": true,
  "decisions": [
    {
      "review_item_id": "<copy item ID>",
      "review_item_digest": "<copy item digest>",
      "item_kind": "column_mapping_group",
      "decision": "accept_known_mapping",
      "review_note": ""
    }
  ]
}
```

For the independent semantic item, use `item_kind: "scope_semantic_note"`
and one of its semantic decisions below. The manifest is an input file, not a
place to change a property, unit, context, source literal, or group membership.

The positive decision is specific to the bound PR-H disposition kind:

- `propose_known_property` allows `accept_known_mapping`;
- `needs_ontology_review` allows `confirm_ontology_review`;
- `needs_source_check` allows `confirm_source_check`; and
- `exclude_from_dataset` allows `accept_exclusion`.

Known-property, ontology-review, and exclusion groups also allow
`needs_source_check` or `reject_group`. A group already proposed as
`needs_source_check` allows only `confirm_source_check` or `reject_group`, so
the same unresolved outcome is not represented by two aliases. Source-check
decisions retain the whole group for additional evidence inspection;
rejection makes it ineligible. Correcting either outcome requires a new
external response and a newly validated PR-H artifact. A positive decision
for one disposition kind is invalid for every other kind.

Semantic-note items allow only:

- `resolve_semantic_note_as_reported`: resolve the note by accepting the
  reported labels and values without rewriting them;
- `needs_source_check`: retain the note as unresolved pending source review;
  or
- `reject_scope`: reject the affected scope.

`reject_group`, `reject_scope`, and `needs_source_check` decisions require a
bounded explanatory review note. `confirm_source_check` may retain the exact
reason already present in PR-H without duplicating it. Semantic resolution
also records the reviewer's explicit reason.
Reviewer fields reject paths, URLs, credentials, control characters, code
fences, and high-confidence executable content.

There is deliberately no inline correction syntax. A changed property, unit,
cell, context value, semantic interpretation, or proposal note must be made in
a new response manifest and must pass PR-H validation again. This prevents a
review decision from creating an unbound shadow response.

`confirm_ontology_review` confirms only that the column remains outside the
pinned ontology. `accept_exclusion` does not admit the excluded cells.
`confirm_source_check` confirms that the PR-H source-check disposition is the
correct unresolved outcome; it cannot make the group eligible.

## Adjudication boundary

The adjudication artifact records the exact evidence chain, packet and
decision bindings, compact decisions, and their deterministic cell-level
expansion. It distinguishes a completed review with no remaining unresolved
mapping obstacle from one that still contains ontology-review or source-check
work. Explicit rejection is a completed negative decision, not an unresolved
item, and its groups remain ineligible.

Even the cleanest successful adjudication keeps every downstream gate closed:

- no material or molecular identity resolution;
- no ontology change or schema-candidate materialization;
- no response correction or automatic merge;
- no reviewed-evidence or gold-candidate staging;
- no direct or device-only admission;
- no curated dataset or training-data write; and
- no network, LLM, MinerU, PDF, ParsedDocument, or executable-script call.

Human review here confirms only the exact displayed response dispositions and
semantic decisions. It does not establish source exhaustiveness, universal
physical correctness, or cross-paper comparability. Ontology-review and
source-check results must remain outside every later materialization path until
their own explicit gates are implemented and satisfied.

## Run the workflow

Generate the exact-bound JSON review packet:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_semantic_review packet \
  --request-artifact /operator/local/supplementary_scoped_candidate_request.json \
  --response-manifest /operator/local/supplementary_scoped_candidate_response_manifest.json \
  --response-artifact /operator/local/supplementary_scoped_candidate_response.json \
  --output /operator/local/supplementary_semantic_review_packet.json
```

Render a reviewer-facing Markdown copy from the validated packet:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_semantic_review render \
  --review-packet /operator/local/supplementary_semantic_review_packet.json \
  --output-markdown /operator/local/supplementary_semantic_review_packet.md
```

After completing every decision, build the exact-bound adjudication artifact:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_semantic_review adjudicate \
  --request-artifact /operator/local/supplementary_scoped_candidate_request.json \
  --response-manifest /operator/local/supplementary_scoped_candidate_response_manifest.json \
  --response-artifact /operator/local/supplementary_scoped_candidate_response.json \
  --review-packet /operator/local/supplementary_semantic_review_packet.json \
  --decision-manifest /operator/local/supplementary_semantic_review_decisions.json \
  --output /operator/local/supplementary_semantic_adjudication.json
```

CLI success output is redacted. It reports only status and bounded counts; it
does not print local paths, table cells, semantic notes, reviewer notes,
credentials, or external model identifiers.

## Next boundary

A later, separate stage may consume only fully eligible exact-bound results to
perform material-identity review and schema-candidate materialization. That
stage must define its own provenance and conflict rules. It must also add an
explicit source-transcription gate, or keep transcription-dependent
materialization disabled, because PR-I intentionally does not claim that the
parsed table was checked against the authoritative PDF. Every unresolved
ontology, source-check, rejected-group, rejected-scope, and device-only outcome
must remain excluded. PR-I itself grants no dataset-admission authority.
