# Custom Corpus Governance Runbook

## Purpose

This runbook describes the governance path for user-supplied custom corpora in
Molly. It covers local, custom, public, private, or mixed PDF corpus intake,
dry-run parsing, human review, admission request validation, package binding
validation, and the future dataset materialization boundary.

This is not a training-data admission mechanism. It is not a Phase 1 trigger,
not production scientific validation, and not automatic private-document
certification. The current path is designed to keep parsing, review, admission
intent, and future materialization separated.

## Governance Chain

```text
custom corpus manifest
-> custom corpus dry-run
-> property candidate manifest
-> property candidate planner
-> property candidate review queue
-> human review artifact
-> property review binding validator
-> admission request
-> package binding validator
-> materialization plan
-> offline materialization planner
-> future materializer
```

Concrete artifact schemas:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_property_candidate.v1`
- `custom_corpus_property_candidate_planner.v1`
- `custom_corpus_property_candidate_review_queue.v1`
- `custom_corpus_review.v1`
- `custom_corpus_property_review_binding.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_materialization.v1`
- `custom_corpus_materialization_planner.v1`

## Step 1: Custom Corpus Manifest

The operator prepares local PDFs outside git and creates a
`custom_corpus_manifest.v1` manifest. The manifest records safe ids, local PDF
paths, source policy, access/provenance notes, and PDF SHA-256 values when
possible. Real PDFs should not be committed. Private, restricted, unknown, or
mixed corpora must be treated conservatively.

References:

- `docs/custom-corpus-intake-contract.md`
- `docs/examples/custom-corpus-manifest.example.json`

Pass criteria:

- manifest validates
- corpus class is explicit
- document ids are safe
- local PDFs exist
- supplied hashes match local files
- raw artifact commit flags are false for real, private, or mixed corpora

Fail criteria:

- unsafe ids
- private path in commit-bound evidence
- token-like values
- missing PDFs
- hash mismatch
- real/custom corpus attempting synthetic confirmation

## Step 2: Custom Corpus Dry-Run

The dry-run parses local PDFs and writes local artifacts under the selected
output/run-id directory. It can use a MinerU endpoint profile and optional
preflight binding. It remains unconfirmed: `DatasetConfirmation.confirmed` is
`false`, Phase 1 must remain `not_run`, and training dataset admission must be
`false`.

References:

- `docs/custom-corpus-dry-run.md`
- `docs/evidence/custom-corpus-dry-run-public-20260628.md`

Example command shape:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_dry_run \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --endpoint-profile-file docs/examples/mineru-endpoint-profiles.example.json \
  --routing-policy manual-primary \
  --output /tmp/molly-custom-corpus-dry-run \
  --run-id custom-corpus-dry-run-<date> \
  --preflight-report /path/outside/git/preflight_report.json \
  --preflight-artifact-sha256 sha256:<digest> \
  --require-preflight-match
```

Pass criteria:

- dry-run decision is `passed`
- parse summary is recorded
- `DatasetConfirmation.confirmed=false`
- Phase 1 status is `not_run`
- training dataset admitted is `false`
- dry-run summary is redacted

Fail criteria:

- Phase 1 runs
- `DatasetConfirmation` is true
- training dataset is admitted
- private paths leak into report
- manifest, hash, or preflight mismatch

## Step 3: Property Candidate Manifest

The property candidate manifest records open-ended numeric scientific property
candidates before human review. It defines reviewable numeric property
candidates without defining a fixed whitelist of accepted scientific fields.
The schema validates evidence structure, numeric representation, units, entity
binding, provenance summaries, confidence, and trainability decision status.

References:

- `docs/custom-corpus-property-candidate-schema.md`
- `docs/examples/custom-corpus-property-candidates.example.json`
- `docs/evidence/templates/custom-corpus-property-candidates-evidence-template.md`

Pass criteria:

- manifest validates
- `field_name` and `canonical_property_guess` are safe labels
- candidate records have finite numeric values and entity/provenance binding
- units are explicit, inferred, not applicable, or review-blocking
- rejected records include reasons
- needs-review records include notes or decision reasons
- no raw text, private path, or token leakage

Fail criteria:

- duplicate property candidate ids or targets
- candidate records with unknown or non-finite numeric values
- candidate records with missing unit evidence outside allowed cases
- rejected records without rejection reasons
- needs-review records without explanatory text
- unsafe labels, private paths, credential-like strings, or raw article text

## Step 4: Property Candidate Planner

The property candidate planner produces review queue planning only. It reads a
validated `custom_corpus_property_candidate.v1` manifest, groups safe counts by
field/property family/value kind/unit status/source, and identifies reviewable
and blocked record ids. It does not create the review manifest.

References:

- `docs/custom-corpus-property-candidate-planner.md`
- `docs/evidence/templates/custom-corpus-property-candidate-planner-evidence-template.md`

Pass criteria:

- planner status is `planned` when reviewable candidates exist
- review queue includes candidate and needs-review records with
  `review_required=true`
- rejected records and `review_required=false` records are blocked
- output contains safe ids and aggregate counts only
- no raw values, provenance summaries, private paths, or token leakage

Fail criteria:

- source candidate manifest is invalid
- planner summary redaction fails
- no reviewable property candidates exist, in which case planner status is
  `blocked`

## Step 5: Property Candidate Review Queue

The property candidate review queue builder creates review-preparation
artifacts from a validated property candidate manifest and planner decisions.
It queues only candidate or needs-review records with `review_required=true`.
It does not create the review manifest.

References:

- `docs/custom-corpus-property-candidate-review-queue.md`
- `docs/evidence/templates/custom-corpus-property-candidate-review-queue-evidence-template.md`

Pass criteria:

- queue status is `prepared` when reviewable records exist
- queue contains candidate and needs-review records with `review_required=true`
- rejected records and `review_required=false` records are blocked
- generated artifacts are under the run-specific queue directory
- queue artifacts contain no review decisions, admission actions, or
  materialization actions
- redaction checks pass

Fail criteria:

- source candidate manifest is invalid
- output queue directory is non-empty
- queue is empty without `--allow-empty-queue`
- queue artifact redaction fails
- queue artifacts imply human review, admission, materialization, or training

## Step 6: Human Review Artifact

Human review summarizes extracted custom corpus records. The review artifact
does not admit data. Review decisions are `accept`, `reject`, and
`needs_review`. Review records bind reviewed targets to corpus id, dry-run id,
source artifact hashes, and short redacted evidence summaries.

References:

- `docs/custom-corpus-human-review.md`
- `docs/examples/custom-corpus-review-manifest.example.json`
- `docs/evidence/templates/custom-corpus-human-review-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_review \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --output-summary /tmp/custom-corpus-review-summary.json
```

Pass criteria:

- review manifest validates
- record ids are safe
- duplicate review ids and duplicate review targets are absent
- `reject` has a rejection reason
- `accept` has no rejection reason
- `needs_review` has notes or a confidence note
- no raw text, private path, or token leakage

Fail criteria:

- duplicate review ids or targets
- unsafe reviewer label
- private paths
- credential-like strings
- overlong raw article text
- invalid decision-specific fields

## Step 7: Property Review Binding Validator

The property review binding validator verifies that a manually-created
`custom_corpus_review.v1` manifest corresponds to the property candidate review
queue. It does not create review decisions.

References:

- `docs/custom-corpus-property-review-binding.md`
- `docs/evidence/templates/custom-corpus-property-review-binding-evidence-template.md`

Pass criteria:

- queue and review manifest corpus/dry-run ids match
- source manifest and dry-run report hashes match
- every reviewed property candidate exists in the queue
- no blocked queue record is reviewed
- accepted records include extracted, normalized, and provenance summaries
- completeness requirements are satisfied when enabled

Fail criteria:

- queue or review manifest is invalid
- source hashes or ids mismatch
- unknown or blocked records are reviewed
- accepted records lack required summaries
- complete queue binding is required but records are missing reviews
- binding summary redaction fails

## Step 8: Admission Request

The admission request records governance intent. It may mark reviewed accepted
records as `admit`, or mark records as `exclude` or `needs_review`. It does
not materialize data, create candidate/training CSVs, set `DatasetConfirmation`,
or run Phase 1.

References:

- `docs/custom-corpus-dataset-admission-gate.md`
- `docs/examples/custom-corpus-admission-request.example.json`
- `docs/evidence/templates/custom-corpus-admission-gate-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_admission \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --output-summary /tmp/custom-corpus-admission-summary.json
```

Pass criteria:

- request validates
- required hashes are present
- action/review decision rules hold
- dataset target is a safe label
- summary decision is recorded as `eligible`, `needs_review`, or `ineligible`

Fail criteria:

- admit without accepted review
- rejected record admitted
- needs-review record admitted
- missing reasons
- missing required hashes
- private path or token-like content

## Step 9: Admission Package Binding Validator

The package validator checks the four artifacts together:

- custom corpus manifest
- dry-run report
- human review manifest
- admission request

It validates hash binding, id consistency, review/admission record matching,
and the dry-run confirmation boundary. It still does not materialize data.

References:

- `docs/custom-corpus-admission-package-binding.md`
- `docs/evidence/templates/custom-corpus-admission-package-validation-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_admission_package \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --output-summary /tmp/admission-package-validation-summary.json
```

Pass criteria:

- package validation status is `passed`
- artifact hashes match
- `corpus_id`, `dry_run_id`, and `review_manifest_id` match
- dry-run decision is `passed`
- dry-run `DatasetConfirmation` is false
- dry-run Phase 1 status is `not_run`
- dry-run training admitted is false
- admission records match review records
- admitted records trace to accepted review records
- no redaction failure

Fail criteria:

- any hash mismatch
- id mismatch
- dry-run was confirmed
- Phase 1 ran
- training was admitted
- review record is missing
- review/action mismatch
- rejected or needs-review record is admitted
- summary redaction failure

## Artifact Retention Policy

Full local artifacts remain outside git. Committed PRs should include only
redacted Markdown evidence. Full reports may contain local paths and should not
be committed by default. Raw PDFs, `ParsedDocument` outputs, MinerU bundles,
pdfplumber baselines, and raw extracted text should not be committed.
Artifact SHA-256 values should be recorded where safe.

## Redaction Checklist

Committed docs or evidence must not include:

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

Allowed:

- safe ids
- safe basenames
- SHA-256 values
- counts
- decision/status strings
- redacted API origin
- safe binding error codes

## Current Implemented Capabilities

- manifest contract and validator
- custom corpus dry-run runner
- public dry-run evidence
- property candidate schema and validator
- property candidate planner
- property candidate review queue builder
- human review schema and validator
- property review binding validator
- admission request schema and validator
- admission package binding validator
- materialization plan schema and validator
- offline materialization planner
- redacted summary/evidence templates

## Current Non-Goals

- no dataset materialization
- no candidate/training CSV
- no Phase 1
- no `DatasetConfirmation` change
- no automatic training admission
- no scientific correctness certification
- no private-document handling certification
- no MinerU Cloud API provider
- no live CI
- no automatic fallback, retry, queue, rollback, or scheduler

## Recommended Next Implementation Boundary

The next implementation, if pursued, should remain carefully constrained to a
dry-run-only materializer that writes candidate artifacts outside git. It
should not jump directly to training CSV generation or automatic Phase 1
execution.

Recommended next PR:

```text
test: add dry-run-only custom corpus materializer
```

That implementation should consume the validated plan and planner output,
write candidate-only artifacts under a clean local output directory, prove
rollback and redaction behavior, and still keep training admission and Phase 1
disabled.

## Materialization Boundary Design

The materialization boundary design is documented in:

```text
docs/custom-corpus-dataset-materialization-boundary.md
```

Future evidence should use:

```text
docs/evidence/templates/custom-corpus-materialization-evidence-template.md
```

This design is now documented, but implementation remains intentionally
absent. A future implementation should begin with schema and planner tests, not
training CSV generation or automatic Phase 1 admission.

## Materialization Plan Schema

The pre-materialization plan schema is documented in:

```text
docs/custom-corpus-materialization-schema.md
```

Safe example:

```text
docs/examples/custom-corpus-materialization-plan.example.json
```

Future evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-plan-evidence-template.md
```

Materialization plan validation still does not create candidate artifacts,
training artifacts, or Phase 1 inputs. It records and validates candidate-only
operator intent before any future materializer exists.

## Offline Materialization Planner

The offline planner is documented in:

```text
docs/custom-corpus-materialization-planner.md
```

Future planner evidence template:

```text
docs/evidence/templates/custom-corpus-materialization-planner-evidence-template.md
```

Planner output summarizes intended future output labels, rollback labels, and
candidate/excluded counts from a validated materialization plan. It still does
not create candidate artifacts, candidate/training CSVs, materialized records,
or Phase 1 inputs.
