# Custom Corpus Dataset Materialization Boundary

## Purpose

This document defines the future boundary between package-validated custom
corpus admission records and materialized dataset artifacts.

This is a design document only. It does not implement dataset materialization,
admit training data, run Phase 1, set `DatasetConfirmation.confirmed=true`,
create candidate/training CSVs, or certify scientific correctness.

## Current Governance Chain

```text
custom corpus manifest
-> custom corpus dry-run
-> property candidate manifest
-> property candidate planner
-> property candidate review queue
-> human review artifact
-> property review binding validator
-> property admission readiness planner
-> property admission request planner
-> admission request
-> admission package binding validation
-> future materialization boundary
-> future candidate/training artifacts
```

Existing artifact schemas:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_property_candidate.v1`
- `custom_corpus_property_candidate_planner.v1`
- `custom_corpus_property_candidate_review_queue.v1`
- `custom_corpus_review.v1`
- `custom_corpus_property_review_binding.v1`
- `custom_corpus_property_admission_readiness.v1`
- `custom_corpus_property_admission_request_plan.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`

All current steps stop before materialization.

The property candidate schema represents open-ended numeric scientific
property candidates before review. It does not define a property whitelist,
call LLMs or agents, evaluate extraction accuracy, or materialize data.

The property candidate planner also sits before human review. It creates safe
review-planning summaries only; materialization still consumes reviewed,
admitted, package-validated, and materialization-plan records rather than raw
property candidate manifests directly.

The property candidate review queue builder sits after the planner and before
human review. It creates review-preparation artifacts only. Raw property
candidates and review queue artifacts do not directly materialize; future
materialization still requires review, admission, package validation, and a
materialization plan.

The property review binding validator sits after human review and before
admission. It validates queue-to-review consistency only. Review queue and
binding evidence do not directly materialize data.

The property admission readiness planner sits after review binding and before
admission. It summarizes accepted, queue-bound human review records for future
admission planning only. Readiness evidence does not directly materialize data.

The property admission request planner sits after readiness and before the
actual admission request. It prepares request-plan evidence only. Materialization
still requires actual admission, package validation, and a materialization plan;
request-plan evidence does not directly materialize data.

## Materialization Definition

Materialization means transforming package-validated admitted records into
durable dataset artifacts that could later be consumed by Phase 1 or
downstream dataset builders.

Examples of future materialized artifacts may include:

- candidate records JSON/JSONL
- training candidate CSV
- manifest-to-record binding file
- provenance report
- reviewer/admission binding report
- rollback manifest
- redacted evidence summary

This PR does not create any of these artifacts.

## Materialization Plan Schema

The pre-materialization plan schema is documented in:

```text
docs/custom-corpus-materialization-schema.md
```

Safe example plan:

```text
docs/examples/custom-corpus-materialization-plan.example.json
```

Future plan evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-plan-evidence-template.md
```

This schema is still pre-materialization. It validates operator intent,
source hash binding, candidate-only mode, explicit confirmation metadata,
record selection, and dry-run/package boundaries. It does not create outputs,
candidate CSVs, training CSVs, or Phase 1 inputs.

## Offline Materialization Planner

The offline planner is documented in:

```text
docs/custom-corpus-materialization-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-planner-evidence-template.md
```

The planner reads a valid `custom_corpus_materialization.v1` plan and produces
a safe JSON or Markdown planning summary. Planner output is not candidate
data, does not imply training admission, and does not create materialized
records or candidate/training artifacts.

## Required Inputs For A Future Materializer

A future materializer must require:

1. custom corpus manifest
2. custom corpus dry-run report
3. human review manifest
4. admission request
5. admission package validation summary
6. explicit operator materialization confirmation
7. materialization output directory
8. materialization run id

Required input conditions:

- package validation status must be `passed`
- admission decision must be `eligible`
- no `needs_review` admission records may be materialized
- no rejected records may be materialized
- every admitted record must trace to an accepted review record
- dry-run `DatasetConfirmation.confirmed` must remain `false`
- dry-run Phase 1 status must remain `not_run`
- dry-run training dataset admitted must remain `false`

## Explicit Operator Confirmation

A future materializer must require explicit materialization confirmation. It
must not reuse synthetic dataset confirmation.

Suggested future concept:

```text
CustomCorpusMaterializationConfirmation
```

Design-only fields:

- `confirmed: bool`
- `confirmed_by: str`
- `confirmed_at: str`
- `confirmation_source: str`
- `package_validation_sha256: str`
- `admission_request_sha256: str`
- `review_manifest_sha256: str`
- `dry_run_report_sha256: str`
- `manifest_sha256: str`
- `corpus_id: str`
- `dry_run_id: str`
- `review_manifest_id: str`
- `admission_request_id: str`
- `reason: str`

Rules:

- `confirmed=true` must be explicit.
- `confirmed_by` must be non-empty and redacted if needed.
- confirmation must bind to exact artifact SHA-256 values.
- confirmation must not override failed package validation.
- confirmation must not override records with `needs_review`.
- confirmation must not bypass review completeness checks.
- confirmation must not set `DatasetConfirmation.confirmed=true` by itself.

This document only proposes the future concept. It does not implement it.

## Review Completeness Gate

A future materializer must verify:

- every materialized record has an admission record with `action=admit`
- every admitted record has a matched review record
- matched review record decision is `accept`
- review/admission document id, record id, field name, and source artifact
  SHA-256 match
- admitted records have non-empty provenance summary
- admitted records have non-empty normalized value summary
- admitted records have non-empty admission reason
- no `needs_review` record is materialized
- no rejected record is materialized
- no duplicate materialized target exists
- materialization record count equals admit count from package validation

## Provenance Binding

Each future materialized record must include safe provenance fields:

- corpus id
- dry-run id
- document id
- record id
- field name
- review id
- admission record id
- source manifest SHA-256
- dry-run report SHA-256
- review manifest SHA-256
- admission request SHA-256
- package validation summary SHA-256
- source artifact SHA-256
- review artifact SHA-256
- normalized value summary
- provenance summary
- materialization run id

Do not include:

- raw PDF path
- raw article text
- ParsedDocument text
- MinerU bundle path
- private home path
- token/auth/cookie/signed URL
- local absolute paths

## Proposed Future Output Artifacts

Design-only output directory layout:

```text
custom_corpus_materialization_<run_id>/
  materialization_summary.json
  materialized_records.jsonl
  materialized_records.csv
  provenance_bindings.jsonl
  rollback_manifest.json
  redacted_evidence_summary.md
```

Artifact policy:

| Artifact | Purpose | Commit policy | Redaction requirements |
| --- | --- | --- | --- |
| `materialization_summary.json` | Machine-readable run summary and counts. | Stay outside git by default. | No raw text, paths, tokens, or private details. |
| `materialized_records.jsonl` | Candidate materialized record payloads. | Stay outside git by default. | Safe summaries and provenance ids only. |
| `materialized_records.csv` | Tabular candidate records for future gates. | Stay outside git by default. | No raw article text or private paths. |
| `provenance_bindings.jsonl` | Per-record source/review/admission bindings. | Stay outside git by default. | SHA-256 values and safe ids only. |
| `rollback_manifest.json` | Deletion/rollback planning for generated outputs. | Stay outside git by default. | Redacted path labels where needed. |
| `redacted_evidence_summary.md` | Human-readable evidence for a future PR. | May be committed after review. | Must pass redaction checklist. |

Raw text, PDFs, ParsedDocuments, MinerU bundles, and private paths must never
be committed.

## Candidate/Training Artifact Boundary

Future materialization must use a strict two-step boundary.

Step A: materialize package-validated records into candidate artifacts only.

Step B: a separate future gate may decide whether candidate artifacts can
become training artifacts.

Rules:

- candidate artifact creation must not imply training admission
- training CSV creation must require a separate explicit gate
- Phase 1 must not run automatically after materialization
- `DatasetConfirmation.confirmed=true` must not be set by materialization alone
- any future training admission must bind to materialization summary SHA-256

## Phase 1 Boundary

Materialization must not run Phase 1. It must not call
`corpus_to_phase1_workflow` in confirmed mode, must not reuse synthetic
confirmation flags, and must not set `DatasetConfirmation.confirmed=true`.
Phase 1 remains a separate explicit future gate. Materialization evidence must
show Phase 1 was not run.

## Deletion And Rollback Design

A future materializer must produce a rollback manifest containing:

- materialization run id
- output artifact paths or redacted path labels
- output artifact SHA-256 values
- list of materialized record ids
- source package validation SHA-256
- deletion instructions
- rollback safety notes

Rules:

- operators must be able to delete materialized candidate artifacts
- rollback must not touch source PDFs
- rollback must not delete external original corpora
- rollback must distinguish local generated artifacts from committed evidence
- committed redacted evidence should be immutable unless a follow-up correction
  PR is made

## Redaction Requirements

Future materialization summaries and evidence must not include:

- raw PDFs
- local absolute paths
- private home paths
- `/Users/`
- `/home/`
- `C:\`
- tokens
- Authorization headers
- bearer tokens
- cookies
- x-api-key
- signed URLs
- raw article text
- ParsedDocument content
- MinerU bundle content
- remote task ids unless explicitly reviewed
- private emails unless redacted

Allowed:

- safe ids
- safe basenames
- SHA-256 values
- counts
- decision/status strings
- safe provenance summaries
- safe normalized value summaries
- safe binding error codes

## Pass Criteria For A Future Materialization Run

A future materialization run may pass only if:

- all source artifacts validate
- package validation status is `passed`
- admission decision is `eligible`
- explicit materialization confirmation is present
- confirmation binds exact source artifact hashes
- all materialized records are admitted and accepted
- no rejected or `needs_review` records are materialized
- materialized record count equals admit count
- output artifacts are created under a clean output directory
- redaction scan passes
- rollback manifest is written
- Phase 1 remains not run
- `DatasetConfirmation` remains unchanged

## Fail Criteria For A Future Materialization Run

A future materialization run must fail if:

- package validation failed
- admission decision is `needs_review` or `ineligible`
- any source artifact hash mismatches
- confirmation is missing or not bound to source hashes
- any admitted record lacks accepted review binding
- any rejected record is materialized
- any `needs_review` record is materialized
- provenance summary is missing
- normalized value summary is missing
- private paths or token-like values appear in output summary
- output directory is non-empty unless explicitly allowed by future design
- Phase 1 runs
- `DatasetConfirmation` changes unexpectedly

## Evidence Requirements

A future materialization evidence PR should commit only a redacted Markdown
summary.

Evidence should include:

- materialization run id
- source artifact SHA-256 values
- package validation summary SHA-256
- materialized candidate count
- excluded count
- needs_review count
- redaction scan result
- rollback manifest SHA-256
- Phase 1 status: `not_run`
- `DatasetConfirmation` changed: `false`
- statement that full artifacts are retained outside git

Evidence must not commit:

- raw PDFs
- raw extracted text
- ParsedDocument outputs
- MinerU bundles
- full local materialization outputs
- private paths
- tokens
- private emails
- signed URLs

## Future Implementation Plan

Recommended future sequence:

1. `docs/test: add custom corpus property candidate schema`
2. `test/docs: add offline custom corpus property candidate planner`
3. `docs/test: add custom corpus materialization schema`
4. `test: add offline materialization planner`
5. `test: add dry-run-only materializer writing candidate artifacts outside git`
6. `docs: record small public materialization dry-run evidence`
7. `docs/test: design training admission boundary from materialized candidates`
8. only later: implement explicit training artifact builder if all previous
   gates pass

Direct implementation of training materialization should not happen in the
next PR.

## Non-Goals

- no code implementation
- no materializer
- no candidate CSV
- no training CSV
- no Phase 1 execution
- no `DatasetConfirmation` change
- no automatic training admission
- no bypass of review/admission/package validation
- no scientific correctness certification
- no private corpus certification
- no MinerU Cloud API provider
- no live CI
- no fallback, retry, queue, rollback scheduler implementation
