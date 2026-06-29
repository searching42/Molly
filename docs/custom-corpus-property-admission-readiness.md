# Custom Corpus Property Admission Readiness

The property admission readiness planner reads a property review binding
summary and a manually-created `custom_corpus_review.v1` manifest, then emits a
safe JSON or Markdown summary describing which reviewed property records are
ready to be considered for a future admission request.

Admission readiness is not admission. It is not materialization. A readiness
summary is necessary but not sufficient for admission.

## Relationship To Review Binding

The previous boundary is documented in:

```text
docs/custom-corpus-property-review-binding.md
```

Binding validation checks queue-to-review consistency. Readiness planning uses
that binding summary to identify accepted, queue-bound review records that may
be considered by a future explicit admission request. It still does not create
admission actions or a `custom_corpus_admission.v1` artifact.

## Inputs

The planner reads two local JSON files:

- a `custom_corpus_property_review_binding.v1` summary
- a manually-created `custom_corpus_review.v1` manifest

It does not read PDFs, ParsedDocument outputs, MinerU bundles, property
candidate manifests, review queue JSON, admission requests, materialization
plans, or training artifacts.

## Readiness Rules

- `accept` review decisions become planned future admission candidates only
  when they are present in binding `reviewed_queue_record_ids`, not present in
  blocked or unknown binding lists, and include extracted value, normalized
  value, and provenance summaries.
- `reject` review decisions become planned future exclusions.
- `needs_review` decisions are blocked from admission readiness.
- A failed binding summary blocks readiness.
- A `needs_review` binding summary blocks readiness when
  `--require-complete-binding` is set.

## Status Meanings

| Status | Meaning |
| --- | --- |
| `ready` | Binding status is `passed`, no readiness errors exist, and at least one accepted record is admission-ready. |
| `partial` | Binding status is `needs_review`, completeness is not required, no readiness errors exist, and at least one accepted record is admission-ready. |
| `blocked` | Binding failed, completeness is required but missing, no accepted records are ready, or readiness errors exist. |

The CLI returns `0` for `ready` and `partial`, and `1` for `blocked`.

## Summary Schema

The summary schema is:

```text
custom_corpus_property_admission_readiness.v1
```

The summary includes safe basenames, SHA-256 values, review queue and manifest
ids, binding status, counts, planned admission candidate ids, planned exclusion
ids, blocked ids, binding carry-through lists, readiness errors, warnings, and
redaction status.

It does not include raw value summaries, raw table rows, raw article text,
full provenance text, local absolute paths, private paths, ParsedDocument text,
MinerU bundle paths, tokens, cookies, Authorization headers, or private emails.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_readiness \
  --review-binding-summary /tmp/custom-corpus-property-review-binding-summary.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --output-summary /tmp/custom-corpus-property-admission-readiness-summary.json \
  --output-markdown /tmp/custom-corpus-property-admission-readiness-summary.md \
  --require-complete-binding
```

## Markdown Output

When `--output-markdown` is supplied, the planner writes a concise safe report
with readiness status, ids, counts, planned candidate ids, planned exclusion
ids, blocked ids, and a boundary statement.

## Redaction Behavior

Before printing or writing any summary or Markdown report, the planner scans
serialized output for private path and credential markers. If unsafe material
is detected, it fails closed with:

```text
property_admission_readiness_summary_redaction_failed
```

## Boundaries

- Admission readiness is not admission.
- The readiness planner does not create `custom_corpus_admission.v1`.
- The readiness planner does not create admission actions.
- The readiness planner does not create admission requests.
- The readiness planner does not materialize data.
- The readiness planner does not create dataset candidate/training CSVs.
- The readiness planner does not run Phase 1.
- The readiness planner does not modify `DatasetConfirmation`.
- A readiness summary is necessary but not sufficient for admission.
