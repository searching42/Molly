# OLED reviewed-evidence facet review request (PR-U)

## Purpose

PR-U converts one exact, verified PR-T ledger publication into a bounded human
review request for the two remaining evidence facets:

1. scientific consistency; and
2. confidence sufficiency for later Gold consideration.

It records no human decisions and does not mutate reviewed evidence, assign a
numeric confidence score, create Gold, write a dataset, or enable training.

## Why confidence is not numeric here

The current project has no calibrated probability model that makes one numeric
confidence score comparable across papers and properties. PR-U therefore asks
for a categorical sufficiency disposition:

```text
sufficient | insufficient | needs_source_check
```

It explicitly states that this is not a calibrated probability. A later
adjudication may clear or retain the blocker, but must not manufacture a score
such as the historical fallback `0.5`.

Scientific consistency uses:

```text
consistent | inconsistent | needs_source_check
```

The source PDF remains authoritative for both facets.

## Exact input and eligibility

The controlled file entry consumes one exact
`oled_reviewed_evidence_ledger_postwrite_verifier.v1` artifact and records its
file SHA-256 and semantic digest.

An observation is included only when all of the following hold:

- it belongs to the exact PR-R scope embedded through PR-T;
- the published ledger entry is `active`;
- comparison context is `complete` or `not_required`, never `incomplete`;
- the causal layer is not `device`; and
- its only remaining Gold blockers are
  `missing_confidence_assessment` and
  `scientific_consistency_not_reviewed`.

Quarantined conflicts and incomplete-context evidence are counted but excluded.
Exact-replay entries remain eligible when they are active and the two facets
are still unfinished; replay does not silently imply that a review occurred.

## Source-row grouping

Eligible observations are grouped by the exact PR-R source-row group. Each
observation preserves:

- material ID and the exact Registry entry;
- property ID/label and causal layer;
- reported value text, decimal places, and unit;
- normalized value and unit;
- comparison-context status, payload, and digest;
- PDF SHA-256, page, table, row, column, header, and source-cell digest; and
- source candidate, Registry entry, ledger entry, and projection digests.

For properties where comparison context is not required, the context remains
JSON `null`; PR-U does not turn an absent context into an inferred empty one.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_reviewed_evidence_facet_review_request \
  --postwrite-verification /operator/local/pr-t.json \
  --output /operator/local/pr-u-facet-review-request.json
```

The output must be fresh and cannot overwrite the input. Symbolic paths,
timestamp reversal, changed output parents, model/count/group tamper, and
partial publication fail closed. CLI failures are redacted.

## Automated boundary

The paper016-shaped fixture produces one source-row group with five eligible
active observations. Tests also cover exact-replay eligibility, conflict
quarantine exclusion, reported precision/provenance preservation, null
not-required context, derived-count tamper, timestamp reversal, input overwrite,
and CLI redaction.

This remains fixture-level validation. The request has not yet been answered by
a human reviewer, and no Gold eligibility changes.

## Next boundary

The next safe step is an exact-roster human decision manifest plus adjudication
artifact for these two categorical facets. It must bind every observation in
PR-U, require one decision per observation, reject partial/extra rosters, and
keep inconsistent, insufficient, or source-check-required evidence out of Gold.
