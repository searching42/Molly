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
-> property admission readiness planner
-> property admission request planner
-> property admission request draft
-> property admission draft package precheck
-> property-aware package binding validator
-> property materialization plan draft
-> property materialization plan preflight
-> property-aware offline materialization planner
-> property materialization dry-run
-> materializer execution request
-> materializer execution request preflight
-> property quarantine materializer
-> property quarantine candidate preflight
-> property training admission readiness
-> property training admission request planner
-> property training admission request preflight
-> property training admission request draft
-> property training admission request draft precheck
-> property training admission execution request
-> future training admission execution
```

Concrete artifact schemas:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_property_candidate.v1`
- `custom_corpus_property_candidate_planner.v1`
- `custom_corpus_property_candidate_review_queue.v1`
- `custom_corpus_review.v1`
- `custom_corpus_property_review_binding.v1`
- `custom_corpus_property_admission_readiness.v1`
- `custom_corpus_property_admission_request_plan.v1`
- `custom_corpus_property_admission_draft_builder.v1`
- `custom_corpus_property_admission_draft_package_precheck.v1`
- `custom_corpus_property_package_binding.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_admission_package_validation.v1`
- `custom_corpus_property_materialization_plan_draft_builder.v1`
- `custom_corpus_property_materialization_plan_preflight.v1`
- `custom_corpus_materialization.v1`
- `custom_corpus_materialization_planner.v1`
- `custom_corpus_property_materialization_planner_runner.v1`
- `custom_corpus_property_materialization_dry_run.v1`
- `custom_corpus_property_materializer_execution_request.v1`
- `custom_corpus_property_materializer_execution_request_builder.v1`
- `custom_corpus_property_materializer_execution_preflight.v1`
- `custom_corpus_property_quarantine_materialization.v1`
- `custom_corpus_property_quarantine_materializer.v1`
- `custom_corpus_property_quarantine_candidate_preflight.v1`
- `custom_corpus_property_training_admission_readiness.v1`
- `custom_corpus_property_training_admission_request_plan.v1`
- `custom_corpus_property_training_admission_request_preflight.v1`
- `custom_corpus_property_training_admission_request_draft.v1`
- `custom_corpus_property_training_admission_request_draft_builder.v1`
- `custom_corpus_property_training_admission_request_draft_precheck.v1`
- `custom_corpus_property_training_admission_execution_request.v1`
- `custom_corpus_property_training_admission_execution_request_builder.v1`

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

## Step 8: Property Admission Readiness Planner

The property admission readiness planner reads a property review binding
summary and a manually-created `custom_corpus_review.v1` manifest. It
summarizes accepted, queue-bound review records as future admission candidates
and rejected records as future exclusions. It does not create admission
actions.

References:

- `docs/custom-corpus-property-admission-readiness.md`
- `docs/evidence/templates/custom-corpus-property-admission-readiness-evidence-template.md`

Pass criteria:

- binding summary validates
- review manifest ids, corpus id, dry-run id, and source hashes match the
  binding summary
- accepted records are queue-bound and include extracted, normalized, and
  provenance summaries
- at least one accepted record is admission-ready
- redaction checks pass

Fail criteria:

- binding summary is failed
- complete binding is required but binding is incomplete
- no accepted records are admission-ready
- accepted records are missing required summaries or binding membership
- readiness summary redaction fails

## Step 9: Property Admission Request Planner

The property admission request planner reads an admission readiness summary and
a manually-created `custom_corpus_review.v1` manifest. It produces a safe plan
for what a future admission request would need to contain. It does not create
admission actions or a `custom_corpus_admission.v1` artifact.

References:

- `docs/custom-corpus-property-admission-request-planner.md`
- `docs/evidence/templates/custom-corpus-property-admission-request-plan-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_request_planner \
  --admission-readiness-summary /path/outside/git/property-admission-readiness-summary.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --output-summary /tmp/custom-corpus-property-admission-request-plan-summary.json \
  --output-markdown /tmp/custom-corpus-property-admission-request-plan-summary.md \
  --require-ready-status
```

Pass criteria:

- readiness summary validates
- review manifest ids, corpus id, dry-run id, and source hashes match the
  readiness summary
- accepted records planned for future admit have extracted, normalized, and
  provenance summaries
- rejected records planned for future exclude have reject decisions
- at least one future admit or exclude record is planned
- redaction checks pass

Fail criteria:

- readiness status is blocked
- ready status is required but readiness is partial
- readiness errors are present
- future admit is planned from a non-accept review
- future exclude is planned from a non-reject review
- required accepted-record summaries are missing
- request-plan summary redaction fails

## Step 10: Property Admission Request Draft

The property admission draft builder writes a reviewable
`custom_corpus_admission.v1` draft from a request plan and a manually-created
review manifest. The draft records governance intent, but it does not admit
training data, run package binding, materialize data, create
candidate/training CSVs, set `DatasetConfirmation`, or run Phase 1.

References:

- `docs/custom-corpus-property-admission-draft-builder.md`
- `docs/custom-corpus-dataset-admission-gate.md`
- `docs/evidence/templates/custom-corpus-property-admission-draft-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_draft_builder \
  --admission-request-plan /path/outside/git/property-admission-request-plan-summary.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --output-dir /tmp/custom-corpus-property-admission-draft \
  --admission-request-id property-admission-draft-<date> \
  --dataset-target example-candidate-target \
  --created-by operator-redacted \
  --confirm-admission-draft-output
```

Pass criteria:

- request plan validates and is not blocked
- explicit draft output confirmation is present
- review manifest ids and source hashes match the request plan
- generated draft validates as `custom_corpus_admission.v1`
- blocked records are not included in the draft admission request
- redaction checks pass

Fail criteria:

- confirmation flag is missing
- request plan is blocked
- request plan is partial and partial output is not explicitly allowed
- no draft admission records are created
- generated draft fails `custom_corpus_admission.v1` validation
- private path or token-like content appears in draft artifacts

## Step 11: Property Admission Draft Package Precheck

The property admission draft package precheck reads the admission draft plus
upstream property governance summaries and reports whether the draft appears
ready for later formal package binding. It is not formal package binding and
does not create `custom_corpus_admission_package_validation.v1`.

References:

- `docs/custom-corpus-property-admission-draft-package-precheck.md`
- `docs/evidence/templates/custom-corpus-property-admission-draft-package-precheck-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_admission_draft_package_precheck \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-draft /path/outside/git/custom_corpus_admission.draft.json \
  --draft-summary /path/outside/git/property_admission_draft_summary.json \
  --request-plan-summary /path/outside/git/property_admission_request_plan_summary.json \
  --readiness-summary /path/outside/git/property_admission_readiness_summary.json \
  --review-binding-summary /path/outside/git/property_review_binding_summary.json \
  --output-summary /tmp/property-admission-draft-package-precheck-summary.json
```

Pass criteria:

- precheck status is `passed`
- dry-run decision is `passed`
- `DatasetConfirmation.confirmed=false`
- Phase 1 status is `not_run`
- training dataset admitted is `false`
- draft, request plan, readiness, and review binding ids match
- non-empty artifact hashes match actual input files
- admission draft record ids match upstream admit/exclude ids
- reviewed blocked or unknown review records are not in the draft
- redaction checks pass

Fail criteria:

- any input schema is invalid
- dry-run was confirmed, Phase 1 ran, or training was admitted
- draft, request plan, readiness, or review binding is blocked or failed
- required source hashes mismatch
- admission draft records do not match upstream property evidence
- needs-review records appear as admitted draft records
- summary redaction fails

## Step 12: Property-Aware Package Binding Validator

The property-aware package binding runner checks the property precheck summary
and then calls the existing formal package validator on the four standard
artifacts:

- custom corpus manifest
- dry-run report
- human review manifest
- admission request

It validates hash binding, id consistency, review/admission record matching,
and the dry-run confirmation boundary through formal package validation. It
still does not materialize data.

References:

- `docs/custom-corpus-property-package-binding.md`
- `docs/custom-corpus-admission-package-binding.md`
- `docs/evidence/templates/custom-corpus-property-package-binding-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_package_binding \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --property-precheck-summary /path/outside/git/property_precheck_summary.json \
  --output-dir /tmp/property-package-binding \
  --binding-run-id property-package-binding-<date> \
  --confirm-formal-package-binding
```

Pass criteria:

- property-aware binding status is `passed`
- property precheck status is `passed`
- formal package validation status is `passed`
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
- property precheck failed or was not explicitly allowed
- summary redaction failure

## Step 13: Property Materialization Plan Draft

The property materialization plan draft builder reads a formally
package-validated property admission package and writes a reviewable
`custom_corpus_materialization.v1` draft. It maps admitted, accepted property
records into candidate-only materialization plan records. It does not run the
offline materialization planner, run a materializer, create candidate/training
CSVs, admit training data, run Phase 1, or modify `DatasetConfirmation`.

References:

- `docs/custom-corpus-property-materialization-plan-draft.md`
- `docs/custom-corpus-materialization-schema.md`
- `docs/evidence/templates/custom-corpus-property-materialization-plan-draft-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_plan_draft \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --output-dir /tmp/property-materialization-plan-draft \
  --materialization-plan-id property-materialization-plan-draft-<date> \
  --dataset-target example-candidate-target \
  --created-by operator-redacted \
  --confirm-materialization-plan-draft-output
```

Pass criteria:

- explicit draft output confirmation is present
- property-aware package binding status is `passed`
- formal package validation status is `passed`
- artifact hashes match actual input files
- corpus, dry-run, review manifest, and admission request ids match
- dry-run decision is `passed`
- dry-run `DatasetConfirmation` is false
- dry-run Phase 1 status is `not_run`
- dry-run training admitted is false
- at least one admitted record becomes a materialization draft record
- generated draft validates as `custom_corpus_materialization.v1`
- redaction checks pass

Fail criteria:

- confirmation flag is missing
- package binding failed
- package binding is `needs_review` without explicit allowance
- formal package validation failed
- source hashes or ids mismatch
- dry-run was confirmed, Phase 1 ran, or training was admitted
- no materialization draft records are produced
- generated draft fails materialization schema validation
- private path or token-like content appears in draft artifacts

## Step 14: Property Materialization Plan Preflight

The property materialization plan preflight reads a reviewable
`custom_corpus_materialization.v1` draft, its draft-builder summary, and
upstream package/admission evidence. It checks whether the draft appears ready
for offline materialization planner submission. It is not planner execution,
does not run a materializer, does not execute materialization, does not create
candidate/training CSVs, does not admit training data, does not run Phase 1,
and does not modify `DatasetConfirmation`.

References:

- `docs/custom-corpus-property-materialization-plan-preflight.md`
- `docs/custom-corpus-materialization-planner.md`
- `docs/evidence/templates/custom-corpus-property-materialization-plan-preflight-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_plan_preflight \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan-draft /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-draft-summary /path/outside/git/property_materialization_plan_draft_summary.json \
  --output-summary /tmp/property-materialization-plan-preflight-summary.json \
  --output-markdown /tmp/property-materialization-plan-preflight-summary.md
```

Pass criteria:

- preflight status is `passed`
- source hashes match actual input files
- corpus, dry-run, review manifest, admission request, and materialization
  plan ids match
- package binding status is `passed`
- formal package validation status is `passed`
- materialization draft status is `written`
- materialization decision is `planned`
- dry-run `DatasetConfirmation` is false
- dry-run Phase 1 status is `not_run`
- dry-run training admitted is false
- materialization records derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not materialization records
- redaction checks pass

Fail criteria:

- any input schema is invalid
- package binding failed
- package binding is `needs_review` while strict package-binding pass is
  required
- formal package validation failed
- draft status is not `written`
- materialization decision is not `planned`
- source hashes or ids mismatch
- dry-run was confirmed, Phase 1 ran, or training was admitted
- materialization records include excluded, blocked, or needs-review records
- materialization record counts or ids mismatch the draft summary
- summary redaction fails

## Step 15: Property-Aware Offline Materialization Planner

The property-aware offline materialization planner runner reads the
preflight-checked `custom_corpus_materialization.v1` plan draft and upstream
package/admission evidence, requires explicit operator confirmation, invokes
the existing offline materialization planner, and writes a property-aware
wrapper summary. It is planner execution only. It does not run a materializer,
execute materialization, create candidate/training CSVs, admit training data,
run Phase 1, or modify `DatasetConfirmation`.

References:

- `docs/custom-corpus-property-materialization-planner-runner.md`
- `docs/custom-corpus-materialization-planner.md`
- `docs/evidence/templates/custom-corpus-property-materialization-planner-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_planner_runner \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-draft-summary /path/outside/git/property_materialization_plan_draft_summary.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --output-dir /tmp/property-materialization-planner \
  --planner-run-id property-materialization-planner-<date> \
  --confirm-offline-materialization-planner
```

Pass criteria:

- planner status is `planned`
- explicit planner confirmation flag is present
- preflight status is `passed`, or `needs_review` is explicitly allowed
- preflight errors are empty
- source hashes match actual input files
- materialization plan validates
- materialization records derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not materialization records
- offline planner output status is `planned`
- wrapper redaction checks pass

Fail criteria:

- confirmation flag is missing
- preflight failed or has errors
- preflight is `needs_review` without explicit allowance
- package binding or formal package validation failed
- draft status is not `written`
- materialization decision is not `planned`
- source hashes or ids mismatch
- dry-run was confirmed, Phase 1 ran, or training was admitted
- offline planner reports blocked/failed status or claims materialized output
- summary redaction fails

## Step 16: Property Materialization Dry-Run

The property materialization dry-run runner reads the offline planner output,
the property planner wrapper summary, the materialization plan, and upstream
package/admission evidence. It validates future materializer-readiness and
writes a no-data dry-run report plus redacted evidence. It is not materializer
execution. It does not execute materialization, create materialized records,
create candidate/training CSV/JSONL/Parquet/LMDB artifacts, admit training
data, run Phase 1, or modify `DatasetConfirmation`.

References:

- `docs/custom-corpus-property-materialization-dry-run.md`
- `docs/custom-corpus-property-materialization-planner-runner.md`
- `docs/evidence/templates/custom-corpus-property-materialization-dry-run-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materialization_dry_run \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --output-dir /tmp/property-materialization-dry-run \
  --dry-run-id property-materialization-dry-run-<date> \
  --confirm-materialization-dry-run
```

Pass criteria:

- dry-run status is `passed`
- explicit dry-run confirmation flag is present
- planner summary status is `planned`, or `needs_review` is explicitly
  allowed
- offline planner output status is `planned`
- source hashes match actual input files
- materialization plan validates
- materialization record ids and counts match upstream planner/preflight
  evidence
- materialization records derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not materialization records
- no materialized output or candidate/training artifact paths are claimed
- dry-run report and evidence redaction checks pass

Fail criteria:

- confirmation flag is missing
- planner summary failed or has errors
- planner summary is `needs_review` without explicit allowance
- offline planner output is failed/blocked or schema-invalid
- offline planner output claims materialized records, candidate/training
  artifacts, Phase 1 execution, training admission, or `DatasetConfirmation`
  mutation
- source hashes or ids mismatch
- dry-run was confirmed, Phase 1 ran, or training was admitted
- materialization records include excluded, blocked, or needs-review records
- output directory is not clean
- dry-run redaction fails

## Step 17: Property Materializer Execution Request

The property materializer execution request builder reads the property
materialization dry-run report and the same upstream package/planner evidence.
It writes a request-only packet for a future materializer plus a safe summary
and redacted evidence. It is not materializer execution and does not execute
materialization, create materialized records, create candidate/training
CSV/JSONL/Parquet/LMDB artifacts, admit training data, run Phase 1, or modify
`DatasetConfirmation`.

References:

- `docs/custom-corpus-property-materializer-execution-request.md`
- `docs/custom-corpus-property-materialization-dry-run.md`
- `docs/evidence/templates/custom-corpus-property-materializer-execution-request-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materializer_execution_request \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --output-dir /tmp/property-materializer-execution-request \
  --execution-request-id property-materializer-execution-request-<date> \
  --created-by operator-redacted \
  --confirm-materializer-execution-request-output
```

Pass criteria:

- explicit execution-request confirmation flag is present
- property materialization dry-run status is `passed`, or `needs_review` is
  explicitly allowed
- offline planner output status is `planned`
- source hashes match actual input files
- materialization plan validates
- materialization record ids and counts match upstream planner/preflight/dry-run
  evidence
- execution records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not execution request records
- execution request records contain safe ids and hashes only
- request summary and evidence redaction checks pass

Fail criteria:

- confirmation flag is missing
- property materialization dry-run failed or has errors
- dry-run is `needs_review` without explicit allowance
- source hashes or ids mismatch
- materialization records include excluded, blocked, or needs-review records
- output directory is not clean
- request redaction fails

## Step 18: Property Materializer Execution Request Preflight

The property materializer execution request preflight reads the request-only
execution packet, execution request builder summary, materialization dry-run
report, and upstream package/planner evidence. It validates request
consistency before future materializer submission. It is not materializer
execution, is not materialization, and produces no candidate/training
artifact.

References:

- `docs/custom-corpus-property-materializer-execution-preflight.md`
- `docs/custom-corpus-property-materializer-execution-request.md`
- `docs/evidence/templates/custom-corpus-property-materializer-execution-preflight-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_materializer_execution_preflight \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --execution-request /path/outside/git/property_materializer_execution_request.json \
  --execution-request-summary /path/outside/git/property_materializer_execution_request_summary.json \
  --output-summary /tmp/property-materializer-execution-preflight-summary.json \
  --output-markdown /tmp/property-materializer-execution-preflight-summary.md
```

Pass criteria:

- execution request status is `written`
- execution mode is `request_only`
- materializer status is `not_run`
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` changed is false
- dry-run status is `passed`, or `needs_review` is allowed by non-strict mode
- all source hashes match actual input files
- materialization and execution record ids/counts match upstream evidence
- execution records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not execution request records
- execution records contain safe ids and hashes only
- preflight summary and Markdown redaction checks pass

Fail criteria:

- execution request or summary schema is invalid
- request status is not `written`
- materializer status changes from `not_run`
- Phase 1 ran, training was admitted, or `DatasetConfirmation` changed
- dry-run failed or strict mode rejects needs-review evidence
- source hashes or ids mismatch
- execution records include excluded, blocked, or needs-review records
- execution records include unsafe values or paths
- preflight redaction fails

## Step 19: Property Quarantine Materializer

The property quarantine materializer consumes a preflight-checked execution
request and writes candidate-only quarantine records, a safe summary, and
redacted evidence. It is the first property-path step that can write candidate
materialized records, but those records remain quarantined. It does not create
training artifacts, admit training data, run Phase 1, or modify
`DatasetConfirmation`.

References:

- `docs/custom-corpus-property-quarantine-materializer.md`
- `docs/custom-corpus-property-materializer-execution-preflight.md`
- `docs/evidence/templates/custom-corpus-property-quarantine-materializer-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_quarantine_materializer \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --execution-request /path/outside/git/property_materializer_execution_request.json \
  --execution-request-summary /path/outside/git/property_materializer_execution_request_summary.json \
  --execution-preflight-summary /path/outside/git/property_materializer_execution_preflight_summary.json \
  --output-dir /tmp/property-quarantine-materializer \
  --quarantine-run-id property-quarantine-materializer-<date> \
  --created-by operator-redacted \
  --confirm-quarantine-materialization
```

Pass criteria:

- explicit quarantine materialization confirmation flag is present
- execution preflight status is `passed`, or `needs_review` is explicitly
  allowed
- execution request status is `written`
- execution mode is `request_only`
- materializer status before this step is `not_run`
- Phase 1 remains `not_run`
- training admitted remains false
- `DatasetConfirmation` remains unchanged
- all source hashes match actual input files
- candidate records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not quarantined candidates
- output directory is clean
- candidate artifact, summary, and evidence redaction checks pass

Fail criteria:

- confirmation flag is missing
- execution preflight failed or has errors
- execution preflight is `needs_review` without explicit allowance
- source hashes or ids mismatch
- execution request is not written or is not request-only
- Phase 1 ran, training was admitted, or `DatasetConfirmation` changed
- candidate records derive from excluded, blocked, or needs-review records
- no candidate records are produced
- output directory is not clean
- quarantine materializer redaction fails

## Step 20: Property Quarantine Candidate Preflight

The property quarantine candidate preflight reads candidate-only quarantine
records, the quarantine materializer summary, the execution preflight summary,
and upstream property governance evidence. It checks whether quarantined
candidate records remain internally consistent and safe before any future
training admission request. It is not training admission and produces no
training artifact.

References:

- `docs/custom-corpus-property-quarantine-candidate-preflight.md`
- `docs/custom-corpus-property-quarantine-materializer.md`
- `docs/evidence/templates/custom-corpus-property-quarantine-candidate-preflight-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_quarantine_candidate_preflight \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --execution-request /path/outside/git/property_materializer_execution_request.json \
  --execution-request-summary /path/outside/git/property_materializer_execution_request_summary.json \
  --execution-preflight-summary /path/outside/git/property_materializer_execution_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --quarantine-materializer-summary /path/outside/git/property_quarantine_materializer_summary.json \
  --output-summary /tmp/property-quarantine-candidate-preflight-summary.json \
  --output-markdown /tmp/property-quarantine-candidate-preflight-summary.md
```

Pass criteria:

- quarantine candidate artifact and summary schemas validate
- quarantine materializer status is `written`, or `needs_review` is allowed
- candidate materialization mode is `candidate_quarantine`
- candidate records are present and count/id fields match
- candidate records have quarantine boundary labels
- execution preflight status is `passed`, or `needs_review` is allowed
- all source hashes match actual input files
- candidate records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not candidate records
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

Fail criteria:

- quarantine candidate artifact or summary schema is invalid
- quarantine materializer failed or has errors
- needs-review evidence appears while strict no-needs-review mode is requested
- source hashes or ids mismatch
- candidate/materialization/execution record counts or ids mismatch
- candidate records derive from excluded, blocked, or needs-review records
- candidate artifact claims training admission, Phase 1 execution, or
  `DatasetConfirmation` mutation
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted evidence
- preflight redaction fails

## Step 24: Property Training Admission Request Draft

The property training admission request draft builder reads request preflight,
request plan, training admission readiness, quarantine candidate preflight,
and quarantine candidate record artifacts. It can write a reviewable training
admission request draft after explicit operator confirmation. It is not
execution: no training admission is executed, no training data is admitted,
and no training artifact is produced.

References:

- `docs/custom-corpus-property-training-admission-request-draft.md`
- `docs/custom-corpus-property-training-admission-request-preflight.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-request-draft-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_draft \
  --training-admission-request-plan /path/outside/git/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /path/outside/git/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /path/outside/git/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --output-dir /path/outside/git/property-training-admission-request-draft \
  --request-draft-id property-training-admission-request-draft-001 \
  --created-by operator-redacted \
  --confirm-training-admission-request-draft-output
```

Pass criteria:

- explicit draft-output confirmation is present
- request preflight status is `passed`
- request plan status is `planned`
- readiness status is `ready`
- planned candidate ids match quarantine candidate ids
- source hashes and ids match across plan/preflight/readiness/quarantine
  artifacts
- excluded, blocked, and needs-review records are not drafted
- output directory is clean
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- draft, summary, and Markdown redaction checks pass

Needs-review criteria:

- request preflight is `partial`
- `--allow-preflight-partial` is explicitly set
- no hard consistency check failed

Fail criteria:

- confirmation is missing
- request preflight is blocked
- request preflight is partial without explicit allowance
- request plan, readiness, or quarantine evidence is blocked
- source hashes or ids mismatch
- planned candidate ids are missing or below threshold
- planned candidate records derive from excluded, blocked, or needs-review
  records
- output directory is non-empty
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted artifacts
- draft redaction fails

## Step 25: Property Training Admission Request Draft Precheck

The property training admission request draft package precheck reads the
training admission request draft, draft summary, request plan, request
preflight, training admission readiness summary, quarantine candidate
preflight summary, and quarantine candidate records. It validates package
consistency before any future training admission execution. It is not
execution: no training admission is executed, no training data is admitted,
and no training artifact is produced.

References:

- `docs/custom-corpus-property-training-admission-request-draft-precheck.md`
- `docs/custom-corpus-property-training-admission-request-draft.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-request-draft-precheck-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck \
  --training-admission-request-draft /path/outside/git/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /path/outside/git/property_training_admission_request_draft_summary.json \
  --training-admission-request-plan /path/outside/git/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /path/outside/git/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /path/outside/git/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --output-summary /tmp/property-training-admission-request-draft-precheck-summary.json \
  --output-markdown /tmp/property-training-admission-request-draft-precheck-summary.md
```

Pass criteria:

- draft and draft summary schemas validate
- draft status is `written`
- request plan status is `planned`
- request preflight status is `passed`
- readiness status is `ready`
- planned candidate ids match draft records and quarantine candidate records
- source hashes and ids match across draft, summary, plan, preflight,
  readiness, and quarantine artifacts
- excluded, blocked, and needs-review records are not draft records
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

Needs-review criteria:

- no hard consistency check failed
- draft or upstream evidence carries allowed needs-review or partial status
- `--allow-draft-needs-review` is explicitly set

Fail criteria:

- draft, summary, request plan, preflight, readiness, or quarantine schema is
  invalid
- draft, request preflight, request plan, or readiness is blocked
- source hashes or ids mismatch
- draft record counts or ids mismatch
- planned candidate records derive from excluded, blocked, or needs-review
  records
- draft package claims training admission, Phase 1 execution, or
  `DatasetConfirmation` mutation
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted evidence
- precheck redaction fails

## Step 26: Property Training Admission Execution Request

The property training admission execution request builder reads the training
admission request draft, draft summary, draft package precheck, request plan,
request preflight, training admission readiness summary, quarantine candidate
preflight summary, and quarantine candidate records. It can write a reviewable
execution request after explicit operator confirmation. It is not execution:
no training admission is executed, no training data is admitted, and no
training artifact is produced.

References:

- `docs/custom-corpus-property-training-admission-execution-request.md`
- `docs/custom-corpus-property-training-admission-request-draft-precheck.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-execution-request-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_execution_request \
  --training-admission-request-draft /path/outside/git/property_training_admission_request.draft.json \
  --training-admission-request-draft-summary /path/outside/git/property_training_admission_request_draft_summary.json \
  --training-admission-request-draft-precheck /path/outside/git/property_training_admission_request_draft_precheck_summary.json \
  --training-admission-request-plan /path/outside/git/property_training_admission_request_plan_summary.json \
  --training-admission-request-preflight /path/outside/git/property_training_admission_request_preflight_summary.json \
  --training-admission-readiness-summary /path/outside/git/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --output-dir /path/outside/git/property-training-admission-execution-request \
  --execution-request-id property-training-admission-execution-request-001 \
  --created-by operator-redacted \
  --confirm-training-admission-execution-request-output
```

Pass criteria:

- explicit execution-request-output confirmation is present
- draft package precheck status is `passed`
- draft status is `written`
- request plan status is `planned`
- request preflight status is `passed`
- readiness status is `ready`
- planned candidate ids match draft records and quarantine candidate records
- source hashes and ids match across draft, precheck, plan, preflight,
  readiness, and quarantine artifacts
- excluded, blocked, and needs-review records are not requested
- output directory is clean
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- request, summary, and Markdown redaction checks pass

Needs-review criteria:

- draft precheck is `needs_review`
- `--allow-draft-precheck-needs-review` is explicitly set
- no hard consistency check failed

Fail criteria:

- confirmation is missing
- draft precheck is blocked
- draft precheck is needs-review without explicit allowance
- draft, request plan, request preflight, readiness, or quarantine evidence is
  blocked
- source hashes or ids mismatch
- draft/planned/execution record ids or counts mismatch
- planned candidate records derive from excluded, blocked, or needs-review
  records
- output directory is non-empty
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted artifacts
- request redaction fails

## Step 21: Property Training Admission Readiness

The property training admission readiness planner reads quarantine candidate
preflight evidence, candidate-only quarantine records, the quarantine
materializer summary, and upstream property governance artifacts. It reports
whether quarantined candidate records are ready to be considered by a future
training admission request. It is not training admission and produces no
training artifact.

References:

- `docs/custom-corpus-property-training-admission-readiness.md`
- `docs/custom-corpus-property-quarantine-candidate-preflight.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-readiness-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_readiness \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --execution-request /path/outside/git/property_materializer_execution_request.json \
  --execution-request-summary /path/outside/git/property_materializer_execution_request_summary.json \
  --execution-preflight-summary /path/outside/git/property_materializer_execution_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --quarantine-materializer-summary /path/outside/git/property_quarantine_materializer_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --output-summary /tmp/property-training-admission-readiness-summary.json \
  --output-markdown /tmp/property-training-admission-readiness-summary.md
```

Pass criteria:

- quarantine candidate preflight schema validates
- quarantine candidate preflight status is `passed`
- quarantine materializer status is `written`
- candidate record count is positive and meets the configured minimum
- candidate/materialization/execution record counts and ids match
- all source hashes match actual input files and upstream summaries
- candidate records derive only from admitted, accepted materialization records
- excluded, blocked, and needs-review records are not readiness candidates
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

Partial criteria:

- no hard consistency check failed
- quarantine candidate preflight or upstream quarantine evidence carries
  allowed `needs_review` status

Fail criteria:

- quarantine candidate preflight failed or has errors
- `needs_review` evidence appears while strict passed-preflight mode is
  requested
- source hashes or ids mismatch
- candidate/materialization/execution record counts or ids mismatch
- candidate records derive from excluded, blocked, or needs-review records
- candidate artifact claims training admission, Phase 1 execution, or
  `DatasetConfirmation` mutation
- minimum candidate record count is not met
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted evidence
- readiness redaction fails

## Step 22: Property Training Admission Request Planning

The property training admission request planner reads training admission
readiness evidence, candidate-only quarantine records, quarantine preflight
evidence, and upstream property governance artifacts. It reports how a future
training admission request could be organized. It is request planning only:
no training admission request is generated, no training action is created,
and no training data is admitted.

References:

- `docs/custom-corpus-property-training-admission-request-planner.md`
- `docs/custom-corpus-property-training-admission-readiness.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-request-plan-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_planner \
  --manifest /path/outside/git/custom-corpus-manifest.json \
  --dry-run-report /path/outside/git/dry_run_report.json \
  --review-manifest /path/outside/git/custom-corpus-review.json \
  --admission-request /path/outside/git/custom-corpus-admission-request.json \
  --formal-package-validation /path/outside/git/custom_corpus_admission_package_validation.json \
  --property-package-binding-summary /path/outside/git/property_package_binding_summary.json \
  --materialization-plan /path/outside/git/custom_corpus_materialization.draft.json \
  --materialization-plan-preflight-summary /path/outside/git/property_materialization_plan_preflight_summary.json \
  --offline-planner-output /path/outside/git/offline_materialization_planner_output.json \
  --property-planner-summary /path/outside/git/property_materialization_planner_summary.json \
  --materialization-dry-run-report /path/outside/git/property_materialization_dry_run_report.json \
  --execution-request /path/outside/git/property_materializer_execution_request.json \
  --execution-request-summary /path/outside/git/property_materializer_execution_request_summary.json \
  --execution-preflight-summary /path/outside/git/property_materializer_execution_preflight_summary.json \
  --quarantine-candidate-records /path/outside/git/property_quarantine_candidate_records.json \
  --quarantine-materializer-summary /path/outside/git/property_quarantine_materializer_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --training-admission-readiness-summary /path/outside/git/property_training_admission_readiness_summary.json \
  --output-summary /tmp/property-training-admission-request-plan-summary.json \
  --output-markdown /tmp/property-training-admission-request-plan-summary.md
```

Pass criteria:

- training admission readiness schema validates
- readiness status is `ready`
- planned candidate ids match quarantined candidate ids
- candidate/materialization/execution record counts and ids match
- source hashes and ids match local input files and upstream summaries
- planned candidates derive only from admitted, accepted records
- excluded, blocked, and needs-review records are not planned as candidates
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

Partial criteria:

- no hard consistency check failed
- training admission readiness status is `partial` and strict ready mode is
  not requested

Fail criteria:

- training admission readiness failed or has errors
- `partial` readiness appears while strict ready mode is requested
- source hashes or ids mismatch
- candidate/materialization/execution record counts or ids mismatch
- planned candidate records derive from excluded, blocked, or needs-review
  records
- minimum planned candidate count is not met
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted evidence
- request-plan redaction fails

## Step 23: Property Training Admission Request Preflight

The property training admission request preflight reads the request plan,
training admission readiness summary, and quarantine candidate preflight
summary. It validates schema, status, SHA, id, and candidate eligibility
consistency before any future training admission request execution layer can
be introduced. It is preflight only: no training admission is executed, no
training request is generated, and no training data is created.

References:

- `docs/custom-corpus-property-training-admission-request-preflight.md`
- `docs/custom-corpus-property-training-admission-request-planner.md`
- `docs/evidence/templates/custom-corpus-property-training-admission-request-preflight-evidence-template.md`

Example command:

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_training_admission_request_preflight \
  --training-admission-request-plan /path/outside/git/property_training_admission_request_plan_summary.json \
  --training-admission-readiness-summary /path/outside/git/property_training_admission_readiness_summary.json \
  --quarantine-candidate-preflight-summary /path/outside/git/property_quarantine_candidate_preflight_summary.json \
  --output-summary /tmp/property-training-admission-request-preflight-summary.json \
  --output-markdown /tmp/property-training-admission-request-preflight-summary.md
```

Pass criteria:

- request plan schema validates
- readiness status is `ready`
- quarantine candidate preflight status is `passed`
- planned candidate ids match quarantined candidate ids
- source hashes and ids match across the plan/readiness/preflight summaries
- excluded, blocked, and needs-review records are not planned as candidates
- no training data is admitted
- no training or candidate CSV/JSONL/Parquet/LMDB artifact is created
- Phase 1 remains `not_run`
- `DatasetConfirmation` remains unchanged
- summary and Markdown redaction checks pass

Partial criteria:

- no hard consistency check failed
- request plan/readiness/preflight evidence carries allowed partial or
  needs-review status

Fail criteria:

- request plan, readiness, or quarantine candidate preflight schema is invalid
- readiness is `blocked`
- quarantine candidate preflight failed
- source hashes or ids mismatch
- candidate ids or counts mismatch
- planned candidate records derive from excluded, blocked, or needs-review
  records
- candidate artifact claims training admission, Phase 1 execution, or
  `DatasetConfirmation` mutation
- raw text, private paths, token-like values, PDF names, or
  CSV/JSONL/Parquet/LMDB paths appear in emitted evidence
- preflight redaction fails

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
- property admission readiness planner
- property admission request planner
- property admission draft builder
- property admission draft package precheck
- property-aware package binding runner
- admission request schema and validator
- admission package binding validator
- property materialization plan draft builder
- property materialization plan preflight
- property-aware offline materialization planner runner
- property materialization dry-run runner
- property materializer execution request builder
- property materializer execution request preflight
- property quarantine materializer
- property quarantine candidate preflight
- property training admission readiness planner
- property training admission request planner
- property training admission request preflight
- property training admission request draft builder
- property training admission request draft precheck
- property training admission execution request builder
- materialization plan schema and validator
- offline materialization planner
- redacted summary/evidence templates

## Current Non-Goals

- no training dataset materialization
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
