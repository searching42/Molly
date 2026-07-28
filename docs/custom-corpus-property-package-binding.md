# Custom Corpus Property Package Binding

The property-aware package binding runner executes formal admission package
binding validation for the property-candidate path. It reads the standard
package inputs plus the property draft package precheck summary, gates formal
binding on that precheck evidence, calls the existing
`custom_corpus_admission_package.py` validator, and writes both the formal
package validation summary and a property-aware wrapper summary.

The runner executes formal package binding validation. It does not materialize
data, create materialization plans, create dataset candidate/training CSVs, run
Phase 1, or modify `DatasetConfirmation`. A passed package binding summary is
necessary but not sufficient for materialization.

## Relationship To Package Precheck

The previous property-specific boundary is documented in:

```text
docs/custom-corpus-property-admission-draft-package-precheck.md
```

The precheck verifies that the admission draft and upstream property summaries
are consistent enough to attempt formal package binding. This runner performs
the formal package binding step and preserves the precheck link in a wrapper
summary.

## Relationship To Formal Package Binding

The underlying formal validator is documented in:

```text
docs/custom-corpus-admission-package-binding.md
```

This runner calls the existing package validator rather than duplicating its
binding logic. The formal output remains
`custom_corpus_admission_package_validation.v1`.

## Inputs

The runner reads:

- `custom_corpus_manifest.v1`
- `custom_corpus_dry_run.v1`
- `custom_corpus_review.v1`
- `custom_corpus_admission.v1`
- `custom_corpus_property_admission_draft_package_precheck.v1`

The admission request may be a reviewed draft such as
`custom_corpus_admission.draft.json`.

## Explicit Confirmation Requirement

The CLI requires:

```text
--confirm-formal-package-binding
```

Without this flag the runner returns a failed wrapper summary and does not call
the formal package validator.

## Precheck Gating

By default, only `precheck_status=passed` is allowed. A failed precheck always
blocks formal package binding. A `needs_review` precheck blocks unless
`--allow-precheck-needs-review` is supplied, in which case the wrapper status
is `needs_review` if the formal validator passes.

The runner also checks local source hashes, ids, dry-run boundary fields,
admit/exclude counts, and blocked record ids before calling formal package
binding. Local failures block the formal validator.

## Output Artifacts

The runner creates a clean run directory:

```text
<output-dir>/<binding-run-id>/
  custom_corpus_admission_package_validation.json
  property_package_binding_summary.json
  redacted_property_package_binding_evidence.md
```

The run directory must be absent or empty. This PR does not implement
overwrite.

## After Package Binding: Materialization Plan Draft

The next property-candidate boundary is documented in:

```text
docs/custom-corpus-property-materialization-plan-draft.md
```

Future draft evidence should use:

```text
docs/evidence/templates/custom-corpus-property-materialization-plan-draft-evidence-template.md
```

Property-aware package binding runs formal package validation. The downstream
materialization plan draft builder can create a reviewable
`custom_corpus_materialization.v1` draft from package-validated property
admission records. That draft is still not materialization, and the offline
materialization planner and future materializer remain separate.

## Status Meanings

`passed` means the property precheck passed, local checks passed, and formal
package validation returned `validation_status=passed`.

`needs_review` means local checks passed and formal package validation passed,
but a `needs_review` property precheck was explicitly allowed.

`failed` means confirmation was missing, property precheck gating failed,
local consistency checks failed, formal package validation failed, or redaction
failed.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_property_package_binding \
  --manifest docs/examples/custom-corpus-manifest.example.json \
  --dry-run-report /tmp/custom-corpus-dry-run-report.json \
  --review-manifest docs/examples/custom-corpus-property-review-manifest.example.json \
  --admission-request /tmp/custom-corpus-property-admission-draft/property-admission-draft-example-001/custom_corpus_admission.draft.json \
  --property-precheck-summary /tmp/custom-corpus-property-admission-draft-package-precheck-summary.json \
  --output-dir /tmp/custom-corpus-property-package-binding \
  --binding-run-id property-package-binding-example-001 \
  --confirm-formal-package-binding
```

## Redaction Behavior

Before writing wrapper summaries or Markdown evidence, the runner serializes
planned outputs and scans for private path and credential markers. If unsafe
material is detected, it fails closed with:

```text
property_package_binding_redaction_failed
```

Unsafe Markdown is not written.

## Boundaries

- The runner executes formal package binding validation.
- The runner does not materialize data.
- The runner does not create materialization plans.
- The runner does not create dataset candidate/training CSVs.
- The runner does not run Phase 1.
- The runner does not modify `DatasetConfirmation`.
- A passed package binding summary is necessary but not sufficient for
  materialization.
