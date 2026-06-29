# Custom Corpus Property Candidate Review Queue

The offline property candidate review queue builder reads a validated
`custom_corpus_property_candidate.v1` manifest, reuses the property candidate
planner, and writes safe review-preparation artifacts for future human review.

A review queue is not a review decision. It is not admission. It is not
materialization. It is necessary but not sufficient for human review.

## Relationship To Existing Artifacts

The builder consumes:

```text
custom_corpus_property_candidate.v1
```

It uses the same planning rules documented in:

```text
docs/custom-corpus-property-candidate-planner.md
```

The builder writes queue-preparation artifacts only. It does not create a
`custom_corpus_review.v1` human review manifest.

## Queue Input

The CLI requires:

- `--property-candidates`: validated property candidate manifest path
- `--output-dir`: local output root
- `--review-queue-id`: safe run-specific queue id

Optional:

- `--allow-empty-queue`: allow a blocked empty queue to return success

The run directory is:

```text
<output-dir>/<review-queue-id>/
```

The directory must be absent or empty. This PR does not implement overwrite.

## Queue Artifacts

The builder writes:

- `property_candidate_review_queue.json`
- `property_candidate_review_queue.md`
- `property_candidate_review_summary.json`
- `redacted_property_candidate_evidence.md`

These artifacts are review-preparation artifacts only.

## Reviewable And Blocked Records

Reviewable records are:

- `trainability_decision=candidate` with `review_required=true`
- `trainability_decision=needs_review` with `review_required=true`

Blocked records are:

- `trainability_decision=reject`
- any record with `review_required=false`

Rejected records are not converted into review or admission candidates.

## Queue Record Content

Queue records include safe review context:

- source corpus, dry-run, document, source record, and artifact hashes
- table, row, column, and page labels when available
- property labels and field name
- numeric value summary and normalized numeric fields
- unit information
- entity information
- method, condition, and provenance summaries
- extraction source, confidence, trainability decision, and decision reason
- a constant review instruction for future review-manifest preparation

Queue records must not include:

- reviewer labels
- reviewed timestamps
- human review decisions
- admission actions
- materialization actions
- training decisions

## Redaction And Fail-Closed Behavior

Before writing queue artifacts, the builder scans serialized output for private
path and credential markers. If unsafe material is detected, it fails closed
and writes only a minimal blocked summary when possible.

The summary redaction failure code is:

```text
property_candidate_review_queue_redaction_failed
```

Full local artifacts should remain outside git unless explicitly reviewed and
redacted.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_candidate_review_queue \
  --property-candidates docs/examples/custom-corpus-property-candidates.example.json \
  --output-dir /tmp/custom-corpus-property-review-queue \
  --review-queue-id property-review-queue-example-001
```

Return codes:

- `0` when review queue artifacts are safely prepared
- `0` when the queue is empty and `--allow-empty-queue` is set
- `1` when manifest validation fails, the run directory is non-empty,
  redaction fails, or the queue is empty without `--allow-empty-queue`

## Boundaries

- The review queue builder does not implement property extraction.
- The review queue builder does not call an LLM or agent.
- The review queue builder does not perform human review.
- The review queue builder does not create a `custom_corpus_review.v1`
  manifest.
- The review queue builder does not create review decisions.
- The review queue builder does not create admission requests.
- The review queue builder does not materialize data.
- The review queue builder does not create dataset candidate/training CSVs.
- The review queue builder does not run Phase 1.
- The review queue builder does not modify `DatasetConfirmation`.
- Review queue artifacts are necessary but not sufficient for human review.
