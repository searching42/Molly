# Custom Corpus Property Admission Draft Builder

The property admission draft builder reads a
`custom_corpus_property_admission_request_plan.v1` summary and a
manually-created `custom_corpus_review.v1` manifest, then writes a reviewable
`custom_corpus_admission.v1` draft JSON artifact plus safe summary and evidence
files.

A generated admission request draft is not training admission. It is not
materialization. It is not Phase 1 confirmation. It still requires explicit
downstream validation.

## Relationship To Request Planning

The previous planning boundary is documented in:

```text
docs/custom-corpus-property-admission-request-planner.md
```

The request planner emits planning summaries only. The draft builder is the
first property-candidate path component allowed to write a
`custom_corpus_admission.v1`-shaped artifact, and it frames that artifact as a
draft through the output filename, summary, evidence, policy, and reason text.

## Inputs

The builder reads:

- a `custom_corpus_property_admission_request_plan.v1` request plan summary
- a manually-created `custom_corpus_review.v1` manifest

It does not read PDFs, ParsedDocument outputs, MinerU bundles, property
candidate manifests, review queues, admission package validation summaries,
materialization plans, or training artifacts.

## Explicit Confirmation Requirement

The CLI requires:

```text
--confirm-admission-draft-output
```

Without this flag the builder returns a blocked summary and writes no draft
admission JSON. This avoids accidentally materializing a reviewable admission
draft from planning evidence.

## Draft Mapping Rules

Records with `planned_action=admit` become draft admission records with
`action=admit` only when the plan carries `review_decision=accept`,
normalized value summary, provenance summary, source artifact SHA-256, review
manifest SHA-256, and a planned reason.

Records with `planned_action=exclude` become draft admission records with
`action=exclude` only when the plan carries `review_decision=reject`, a
planned reason, source artifact SHA-256, and review manifest SHA-256.

Records with `planned_action=blocked` are not included in
`custom_corpus_admission.draft.json`. They are reported in the builder summary
as blocked records.

The generated draft is validated through the existing
`custom_corpus_admission.v1` validator. The builder does not duplicate or
weaken that schema.

## Output Artifacts

The builder creates a clean run directory:

```text
<output-dir>/<admission-request-id>/
  custom_corpus_admission.draft.json
  property_admission_draft_summary.json
  redacted_property_admission_draft_evidence.md
```

The run directory must be absent or empty. This PR does not implement
overwrite.

## Validation Behavior

Draft generation is blocked when:

- the confirmation flag is missing
- the request plan is blocked
- the request plan is partial and `--allow-partial-plan` is not set
- the request plan carries planning errors
- review manifest ids or source hashes do not match
- no draft admission records are produced
- generated `custom_corpus_admission.v1` validation fails
- redaction checks fail

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_draft_builder \
  --admission-request-plan /tmp/custom-corpus-property-admission-request-plan-summary.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --output-dir /tmp/custom-corpus-property-admission-draft \
  --admission-request-id property-admission-draft-example-001 \
  --dataset-target example-candidate-target \
  --created-by operator-redacted \
  --confirm-admission-draft-output
```

Use `--allow-partial-plan` only when partial request-plan evidence is
acceptable for draft creation. The output remains a draft and still requires
downstream validation.

## Redaction Behavior

Before writing any artifact, the builder serializes the planned draft, summary,
and evidence and scans for private path and credential markers. If unsafe
material is detected, it fails closed with:

```text
property_admission_draft_redaction_failed
```

No draft admission JSON is written on redaction failure.

## Boundaries

- The builder creates a reviewable admission request draft only.
- The builder does not admit training data.
- The builder does not run admission package binding.
- The builder does not materialize data.
- The builder does not create dataset candidate/training CSVs.
- The builder does not run Phase 1.
- The builder does not modify `DatasetConfirmation`.
- A generated draft is necessary but not sufficient for package validation or
  materialization.
