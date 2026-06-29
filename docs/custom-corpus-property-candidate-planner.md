# Custom Corpus Property Candidate Planner

The offline property candidate planner reads a validated
`custom_corpus_property_candidate.v1` manifest and produces a safe
review-planning summary. It helps operators understand which property
candidates should enter a future human review queue and which records are
blocked from review or admission.

The planner output is necessary but not sufficient for future human review. A
future review artifact must still be created separately.

## Relationship To Property Candidate Schema

The candidate schema is documented in:

```text
docs/custom-corpus-property-candidate-schema.md
```

Candidate validation checks structure, numeric representation, units, entity
binding, provenance summaries, confidence, trainability decision fields, and
redaction safety. The planner consumes that validated manifest and emits a
concise summary for review preparation.

## Planner Input

The CLI requires one local JSON manifest:

```text
custom_corpus_property_candidate.v1
```

The planner does not read PDFs, ParsedDocument content, MinerU bundles, corpus
workflow outputs, review manifests, admission requests, materialization plans,
or training artifacts.

## Planner Outputs

The planner can write:

- a safe JSON planner summary
- a safe Markdown planner summary

These are planner summaries only. They are not review queues, human review
manifests, admission requests, materialization plans, candidate CSVs, or
training CSVs.

## Planner Summary Schema

The JSON summary uses:

```text
custom_corpus_property_candidate_planner.v1
```

It includes:

- planner status: `planned` or `blocked`
- property candidate manifest basename and SHA-256
- manifest, corpus, and dry-run ids
- total, candidate, needs-review, and rejected counts
- review queue count and blocked record count
- safe record id lists for review queue, blocked, candidate, needs-review, and
  rejected records
- unique document, entity, and field counts
- aggregate counts by field name, property family, value kind, unit status,
  and extraction source
- candidate policy, extraction scope, source artifact hashes
- planned review output labels
- blocking reasons, warnings, and redaction status

It does not include raw values, provenance summaries, raw table rows, raw
article text, local paths, ParsedDocument text, MinerU bundle paths, tokens,
cookies, Authorization headers, or private emails.

## Review Queue Planning

Reviewable records are:

- `trainability_decision=candidate` with `review_required=true`
- `trainability_decision=needs_review` with `review_required=true`

Blocked records are:

- `trainability_decision=reject`
- any record with `review_required=false`

If no reviewable records exist, planner status is `blocked` with:

```text
no_reviewable_property_candidates
```

The planner does not override the validator and does not turn rejected records
into review or admission candidates.

## Planned Review Output Labels

The planner may list future output labels:

- `property_candidate_review_queue.json`
- `property_candidate_review_queue.md`
- `property_candidate_review_summary.json`
- `redacted_property_candidate_evidence.md`

These are labels only. The planner must not create these files unless the user
explicitly requested the planner's own JSON or Markdown summary path.

## Redaction And Fail-Closed Behavior

Before writing or printing planner summaries, the planner scans serialized
summary content for forbidden private path and credential markers. If unsafe
material is detected, it returns a minimal blocked summary with:

```json
{
  "schema_version": "custom_corpus_property_candidate_planner.v1",
  "planner_status": "blocked",
  "blocking_reasons": ["property_candidate_planner_summary_redaction_failed"],
  "redaction_status": "failed"
}
```

The CLI returns `1` on redaction failure and must not write unsafe summary
content.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_candidate_planner \
  --property-candidates docs/examples/custom-corpus-property-candidates.example.json \
  --output-summary /tmp/custom-corpus-property-candidate-planner-summary.json \
  --output-markdown /tmp/custom-corpus-property-candidate-planner-summary.md
```

Return codes:

- `0` when a valid manifest produces a safe planner summary, whether status is
  `planned` or `blocked`
- `1` when manifest validation fails or planner summary redaction fails

## Boundaries

- The planner does not implement property extraction.
- The planner does not call an LLM or agent.
- The planner does not create a human review manifest.
- The planner does not perform human review.
- The planner does not create admission requests.
- The planner does not materialize data.
- The planner does not create candidate/training CSVs.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner output is necessary but not sufficient for future human review.
