# Custom Corpus Governance Stage Summary - 2026-06-28

## Summary

Molly now has a full custom corpus governance chain through offline
materialization planning:

```text
custom_corpus_manifest.v1
-> custom_corpus_dry_run.v1
-> custom_corpus_property_candidate.v1
-> custom_corpus_property_candidate_planner.v1
-> custom_corpus_property_candidate_review_queue.v1
-> custom_corpus_review.v1
-> custom_corpus_property_review_binding.v1
-> custom_corpus_property_admission_readiness.v1
-> custom_corpus_admission.v1
-> custom_corpus_admission_package_validation.v1
-> custom_corpus_materialization.v1
-> custom_corpus_materialization_planner.v1
-> future materializer
```

The chain supports controlled custom corpus intake, unconfirmed dry-runs,
open-ended numeric property candidate manifests, property candidate
review-planning summaries, property candidate review queue artifacts, human
review artifacts, queue-to-review binding validation, admission readiness
planning, admission-intent validation, cross-artifact package binding
validation, materialization plan validation, and safe offline planning. It does
not materialize records into datasets and does not run Phase 1.

## Completed PRs

- **#155 intake contract**: defined the custom/private corpus intake contract,
  corpus class policy, redaction expectations, and dry-run boundary. It did
  not implement parsing, review, admission, or dataset materialization.
- **#156 dry-run runner**: implemented the controlled local PDF dry-run path
  that keeps `DatasetConfirmation.confirmed=false` and verifies Phase 1 remains
  `not_run`. It did not admit training data.
- **#157 public dry-run evidence**: recorded redacted evidence for a small
  public custom corpus dry-run. It did not commit PDFs, full reports, MinerU
  bundles, or ParsedDocument outputs.
- **#158 human review artifact schema**: added `custom_corpus_review.v1` and
  an offline validator for redacted review metadata. Review artifacts still do
  not admit training data.
- **#159 admission request gate contract**: added `custom_corpus_admission.v1`
  and an offline validator for admission intent. It did not create candidate or
  training datasets.
- **#160 package binding validator**: added
  `custom_corpus_admission_package_validation.v1` summary generation across
  manifest, dry-run report, review manifest, and admission request. It did not
  materialize datasets or run Phase 1.

## Current Trust Boundary

Custom corpora can be described, parsed in dry-run mode, reviewed, and checked
for admission intent. The package binding validator can verify that the review
and admission request are tied to the expected manifest and dry-run report.

No records are yet materialized into training data. Phase 1 remains protected
by the absence of any custom corpus dataset materialization implementation and
by the unchanged `DatasetConfirmation` boundary.

## Why This Matters

This governance chain separates parsing from review, review from admission
intent, and admission intent from materialization. It prevents silent
training-data admission, preserves reproducibility through artifact hashes, and
keeps redaction requirements explicit before any future dataset builder can
consume custom corpus records.

## Remaining Gaps

- no materialization implementation
- no reviewed-record-to-dataset transform
- no production data deletion/rollback story
- no full real scientific extraction quality benchmark
- no private corpus operational certification
- no MinerU Cloud API provider

## Recommended Next PR

```text
docs: design custom corpus dataset materialization boundary
```

The next step should be a design-only PR that defines how package-validated
admission records could become materialized candidate/training artifacts,
which explicit operator gates are required, and how rollback/deletion and
provenance binding will work before any runtime implementation is added.

## Post-Runbook Design Note

The materialization boundary design was added after the #155-#161 governance
runbook work:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

It documents the future materialization boundary, but still does not implement
materialization. The custom corpus path remains protected before candidate or
training artifacts are created.

## Post-Design Schema Note

The materialization plan schema was added after the materialization boundary
design:

```text
docs/custom-corpus-materialization-schema.md
```

It validates candidate-only materialization intent and source binding, but
still does not implement a materializer or create candidate/training artifacts.

## Post-Schema Planner Note

The offline materialization planner was added after the materialization plan
schema:

```text
docs/custom-corpus-materialization-planner.md
```

It reads a valid `custom_corpus_materialization.v1` plan and produces safe
JSON/Markdown planning summaries. It still does not implement a materializer,
create candidate/training artifacts, run Phase 1, or change
`DatasetConfirmation`.

## Pre-Review Property Candidate Note

The property candidate schema was added as a pre-review layer:

```text
docs/custom-corpus-property-candidate-schema.md
```

It represents open-ended numeric scientific property candidates without a
fixed property whitelist. It still does not implement a property extraction
runner, call an LLM or agent, evaluate extraction accuracy, implement Agentic
RL, materialize data, create candidate/training CSVs, run Phase 1, or change
`DatasetConfirmation`.

## Pre-Review Property Candidate Planner Note

The property candidate planner was added as a pre-review planning layer:

```text
docs/custom-corpus-property-candidate-planner.md
```

It reads a validated property candidate manifest and produces safe
review-planning summaries. It still does not implement an extraction runner,
call an LLM or agent, perform evaluation/RL, generate human review artifacts,
admit data, materialize data, create candidate/training CSVs, run Phase 1, or
change `DatasetConfirmation`.

## Pre-Review Property Candidate Review Queue Note

The property candidate review queue builder was added as a pre-review artifact
generator:

```text
docs/custom-corpus-property-candidate-review-queue.md
```

It reads validated property candidate manifests through the planner and writes
safe review-preparation artifacts. It still does not create human review
decisions, generate `custom_corpus_review.v1`, admit data, materialize data,
call an LLM or agent, perform evaluation/RL, create candidate/training CSVs,
run Phase 1, or change `DatasetConfirmation`.

## Property Review Binding Validator Note

The property review binding validator was added as a queue-to-review
consistency layer:

```text
docs/custom-corpus-property-review-binding.md
```

It validates manually-created `custom_corpus_review.v1` manifests against
property candidate review queue artifacts. It still does not create review
decisions, generate review manifests, admit data, materialize data, call an LLM
or agent, perform evaluation/RL, create candidate/training CSVs, run Phase 1,
or change `DatasetConfirmation`.

## Property Admission Readiness Planner Note

The property admission readiness planner was added as a pre-admission planning
layer:

```text
docs/custom-corpus-property-admission-readiness.md
```

It summarizes accepted, queue-bound human review records as future admission
candidates. It still does not generate admission requests, create admission
actions, admit data, materialize data, call an LLM or agent, perform
evaluation/RL, create candidate/training CSVs, run Phase 1, or change
`DatasetConfirmation`.

## Property Admission Request Planner Note

The property admission request planner was added as the next pre-admission
planning layer:

```text
docs/custom-corpus-property-admission-request-planner.md
```

It summarizes admission-ready review records into safe future admission request
plans. It still does not generate `custom_corpus_admission.v1`, create
admission actions, admit data, materialize data, call an LLM or agent, perform
evaluation/RL, create candidate/training CSVs, run Phase 1, or change
`DatasetConfirmation`.

## Property Admission Draft Builder Note

The property admission draft builder was added after the request planner:

```text
docs/custom-corpus-property-admission-draft-builder.md
```

It can generate reviewable `custom_corpus_admission.v1` draft artifacts from
request plans and manually-created review manifests. It still does not run
package binding, admit training data, materialize data, call an LLM or agent,
perform evaluation/RL, create candidate/training CSVs, run Phase 1, or change
`DatasetConfirmation`.

## Property Admission Draft Package Precheck Note

The property admission draft package precheck was added after the draft
builder:

```text
docs/custom-corpus-property-admission-draft-package-precheck.md
```

It checks reviewable admission drafts against upstream property evidence before
formal package binding. It still does not run formal package binding, create
`custom_corpus_admission_package_validation.v1`, admit training data,
materialize data, call an LLM or agent, perform evaluation/RL, create
candidate/training CSVs, run Phase 1, or change `DatasetConfirmation`.

## Property-Aware Package Binding Runner Note

The property-aware package binding runner was added after precheck:

```text
docs/custom-corpus-property-package-binding.md
```

It gates formal package validation on property precheck evidence and writes a
formal `custom_corpus_admission_package_validation.v1` summary plus a
property-aware wrapper summary. It still does not materialize data, admit
training data, call an LLM or agent, perform evaluation/RL, create
candidate/training CSVs, run Phase 1, or change `DatasetConfirmation`.

## Property Materialization Plan Draft Builder Note

The property materialization plan draft builder was added after property-aware
package binding:

```text
docs/custom-corpus-property-materialization-plan-draft.md
```

It maps formal package-validated property admissions into a reviewable
`custom_corpus_materialization.v1` draft plus safe summary/evidence artifacts.
It still does not run materialization, invoke the offline materialization
planner, run any materializer, admit training data, call an LLM or agent,
perform evaluation/RL, create candidate/training CSVs, run Phase 1, or change
`DatasetConfirmation`.
