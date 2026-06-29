# Custom Corpus Materialization Planner

The offline materialization planner reads a validated
`custom_corpus_materialization.v1` plan and produces a safe execution summary
for a future materializer. It describes what would be planned, but it does not
create any materialized records or dataset files.

The planner output is necessary but not sufficient for future materialization.
A future materializer must still be implemented separately.

## Relationship To The Plan Schema

The plan schema is documented in:

```text
docs/custom-corpus-materialization-schema.md
```

Plan validation checks structure, source hash binding, explicit operator
confirmation, candidate-only intent, record selection, and the dry-run
boundary. The planner consumes that validated plan and emits a concise,
redacted summary of intended future outputs.

## Planner Input

The CLI requires one local JSON plan:

```text
custom_corpus_materialization.v1
```

The planner does not read raw PDFs, ParsedDocument files, MinerU bundles,
corpus workflow outputs, or package source artifacts. It uses the plan as the
only input and relies on the plan validator for structural safety.

## Scientific Property Scope

The planner does not define scientific property extraction scope. It only
consumes materialization records already selected by the materialization plan.
The only property-like field currently carried through this layer is
`field_name`.

The pre-review property candidate layer is documented in:

```text
docs/custom-corpus-property-candidate-schema.md
```

The property candidate schema defines the pre-review candidate representation.
Future extraction and trainability decision implementations should define how
numeric scientific properties are discovered, normalized, reviewed, and deemed
trainable. Those decisions belong outside the planner and must not be inferred
from planner output.

The property candidate planner is upstream of human review:

```text
docs/custom-corpus-property-candidate-planner.md
```

The property candidate review queue builder is also upstream of human review:

```text
docs/custom-corpus-property-candidate-review-queue.md
```

The property review binding validator is upstream of admission:

```text
docs/custom-corpus-property-review-binding.md
```

The property admission readiness planner is upstream of explicit admission:

```text
docs/custom-corpus-property-admission-readiness.md
```

The property admission request planner is also upstream of actual admission
request creation:

```text
docs/custom-corpus-property-admission-request-planner.md
```

The property admission draft builder is upstream of package binding:

```text
docs/custom-corpus-property-admission-draft-builder.md
```

The property admission draft package precheck is also upstream of formal
package binding:

```text
docs/custom-corpus-property-admission-draft-package-precheck.md
```

It does not create package validation artifacts and is not materialization
planning.

Do not confuse property candidate review planning or review queue preparation
with materialization planning. This materialization planner is downstream of
human review, property review binding, readiness, request planning when
applicable, explicit admission draft review, draft package precheck when used,
formal package validation, and the materialization plan.

## Planner Outputs

The planner can write:

- a safe JSON planner summary
- a safe Markdown planner summary

These are planner summaries only. They are not materialized candidate records,
candidate CSVs, training CSVs, provenance bindings, or rollback manifests.

## Planner Summary Schema

The JSON summary uses:

```text
custom_corpus_materialization_planner.v1
```

It includes:

- planner status: `planned` or `blocked`
- safe materialization plan basename and SHA-256
- plan, run, corpus, dry-run, review manifest, and admission request ids
- dataset target label
- materialization mode and decision
- package validation and admission decision fields
- dry-run Phase 1, DatasetConfirmation, and training-admission boundary fields
- candidate and excluded record counts
- candidate and excluded materialization record ids
- planned output labels
- rollback labels
- blocking reasons, warnings, and redaction status

It does not include normalized value summaries, provenance summaries, raw
record values, raw PDF paths, raw text, ParsedDocument text, MinerU bundle
paths, tokens, Authorization headers, cookies, or private paths.

## Planned Output Labels

The planner may list future output labels:

- `materialization_summary.json`
- `materialized_records.jsonl`
- `materialized_records.csv`
- `provenance_bindings.jsonl`
- `rollback_manifest.json`
- `redacted_evidence_summary.md`

These are labels only. The planner must not create these files unless the user
explicitly requested the planner's own JSON or Markdown summary path.

## Rollback Labels

The planner may list rollback labels:

- `rollback_manifest.json`
- `delete_generated_candidate_artifacts_only`
- `do_not_delete_source_pdfs`
- `do_not_delete_external_original_corpora`

These are labels only. No rollback manifest or deletion operation is created
or executed by the planner.

## Redaction And Fail-Closed Behavior

Before writing or printing planner summaries, the planner scans the serialized
summary for forbidden private path or credential markers. If unsafe material is
detected, it returns a minimal blocked summary with:

```json
{
  "schema_version": "custom_corpus_materialization_planner.v1",
  "planner_status": "blocked",
  "blocking_reasons": ["planner_summary_redaction_failed"],
  "redaction_status": "failed"
}
```

The planner must not write unsafe summary content.

## CLI Usage

```bash
PYTHONPATH=src python -m ai4s_agent.custom_corpus_materialization_planner \
  --materialization-plan docs/examples/custom-corpus-materialization-plan.example.json \
  --output-summary /tmp/custom-corpus-materialization-planner-summary.json \
  --output-markdown /tmp/custom-corpus-materialization-planner-summary.md
```

Return codes:

- `0` when a valid plan produces a safe planner summary, whether status is
  `planned` or `blocked`
- `1` when the plan is invalid or planner summary redaction fails

## Boundaries

- The planner does not implement materialization.
- The planner does not create candidate artifacts.
- The planner does not create candidate/training CSVs.
- The planner does not admit training data.
- The planner does not run Phase 1.
- The planner does not modify `DatasetConfirmation`.
- The planner output is necessary but not sufficient for future
  materialization.
