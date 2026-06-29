# Custom Corpus Property Admission Request Planner

The property admission request planner reads a
`custom_corpus_property_admission_readiness.v1` readiness summary and a
manually-created `custom_corpus_review.v1` manifest, then emits a safe JSON or
Markdown planning summary for what a future `custom_corpus_admission.v1`
request would need to contain.

Admission request planning is not admission. A planned admission action is not
an admission action. A request plan is necessary but not sufficient for
admission.

## Relationship To Readiness

The previous boundary is documented in:

```text
docs/custom-corpus-property-admission-readiness.md
```

Readiness planning identifies accepted, queue-bound human review records that
may be considered for future admission. The admission request planner turns
that readiness evidence into a safe request-planning summary. It still does
not create a `custom_corpus_admission.v1` artifact and does not create
executable admission actions.

## Inputs

The planner reads two local JSON files:

- a `custom_corpus_property_admission_readiness.v1` summary
- a manually-created `custom_corpus_review.v1` manifest

It does not read PDFs, ParsedDocument outputs, MinerU bundles, property
candidate manifests, review queue JSON, property review binding summaries
other than fields carried by the readiness summary, admission requests, or
materialization plans.

## Readiness-To-Plan Rules

- `readiness_status=ready` can produce `planner_status=planned` when at least
  one planned admission candidate or planned exclusion exists.
- `readiness_status=partial` can produce `planner_status=partial` when
  `--require-ready-status` is not set.
- `readiness_status=partial` is blocked when `--require-ready-status` is set.
- `readiness_status=blocked` is blocked.
- Non-empty readiness errors block planning.
- If no planned admission candidates and no planned exclusions exist, planning
  is blocked.

## Planned Record Mapping

Records listed in `planned_admission_candidate_record_ids` are planned as
future `admit` records only when their review decision is `accept` and the
review record includes extracted value, normalized value, provenance, and
source artifact hash summaries.

Records listed in `planned_exclusion_record_ids` are planned as future
`exclude` records only when their review decision is `reject`.

Records listed in `blocked_from_admission_record_ids`, and any review records
with `decision=needs_review`, remain blocked from admission planning.

The planner may emit planning records with `planned_action=admit`, `exclude`,
or `blocked`. These are planning labels only. They are not admission actions
and they are not a `custom_corpus_admission.v1` request.

## Status Meanings

| Status | Meaning |
| --- | --- |
| `planned` | Readiness is ready, review bindings match, at least one record is planned for future admit or exclude, and no planning errors exist. |
| `partial` | Readiness is partial, complete readiness is not required, at least one record is planned, and no planning errors exist. |
| `blocked` | Readiness is blocked, complete readiness is required but missing, no records are planned, or planning errors exist. |

The CLI returns `0` for `planned` and `partial`, and `1` for `blocked`.

## Summary Schema

The summary schema is:

```text
custom_corpus_property_admission_request_plan.v1
```

The summary includes safe basenames, SHA-256 values, readiness and binding
status, review manifest and queue ids, counts, planned admit ids, planned
exclude ids, blocked ids, planning errors, warnings, and short redacted
planning record summaries.

It must not include raw table rows, raw article text, local absolute paths,
private paths, ParsedDocument text, MinerU bundle paths, tokens, cookies,
Authorization headers, or private emails.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_request_planner \
  --admission-readiness-summary /tmp/custom-corpus-property-admission-readiness-summary.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --output-summary /tmp/custom-corpus-property-admission-request-plan-summary.json \
  --output-markdown /tmp/custom-corpus-property-admission-request-plan-summary.md \
  --require-ready-status
```

## Markdown Output

When `--output-markdown` is supplied, the planner writes a concise safe report
with planner status, readiness status, binding status, ids, counts, planned
admit ids, planned exclude ids, blocked ids, planning errors, and a boundary
statement.

## Redaction Behavior

Before printing or writing JSON or Markdown, the planner scans serialized
output for private path and credential markers. If unsafe material is detected,
it fails closed with:

```text
property_admission_request_plan_summary_redaction_failed
```

## Boundaries

- Admission request planning is not admission.
- The planner does not create `custom_corpus_admission.v1`.
- The planner does not create admission actions.
- The planner does not create admission requests.
- The planner does not materialize data.
- The planner does not create dataset candidate/training CSVs.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- A request plan is necessary but not sufficient for admission.
