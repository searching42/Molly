# OLED Supplementary-Source Intake

## Purpose

This module is the controlled handoff after the
[supplementary-source recovery planner](oled-supplementary-source-recovery.md).
It binds a human-approved, already-local supplementary PDF to every item in an
existing recovery plan and writes a redacted, content-bound intake artifact.

It does not discover a supplementary URL, download a file, follow a redirect,
call MinerU or an LLM, parse scientific content, regenerate candidates, stage
evidence, create gold records, or write a dataset.

## Human Intake Manifest

The operator creates a local JSON manifest. It must bind exactly to the source
recovery plan's paper and four digests, explicitly decide every recovery item,
and set `intake_confirmed` to `true`.

```json
{
  "schema_version": "oled_supplementary_source_intake_manifest.v1",
  "paper_id": "paper016",
  "source_request_digest": "<request-digest>",
  "source_mapping_result_digest": "<mapping-result-digest>",
  "source_context_digest": "<context-digest>",
  "recovery_plan_digest": "<recovery-plan-digest>",
  "intake_confirmed": true,
  "sources": [
    {
      "source_id": "paper016-si-v1",
      "local_pdf_path": "<operator-local-path>/paper016_si.pdf",
      "expected_pdf_sha256": "",
      "provenance_category": "publisher-supplied",
      "access_policy": "reviewer-approved-local-copy",
      "provenance_note": "Human reviewer supplied the local supplementary file."
    }
  ],
  "decisions": [
    {
      "recovery_item_id": "supplementary-recovery:<item-id>",
      "decision": "approved",
      "source_id": "paper016-si-v1",
      "reviewed_by": "reviewer-01",
      "reviewed_at": "2026-07-13T08:00:00Z"
    }
  ]
}
```

`local_pdf_path` is an operator-local input only. It must not be committed or
copied into the output artifact. The manifest accepts `approved`, `deferred`,
and `rejected` decisions:

- `approved` requires one declared `source_id`.
- `deferred` and `rejected` must not bind a source and require a review note.
- Every recovery item must have exactly one decision; unknown or duplicate item
  IDs fail closed.
- A single approved local PDF may be explicitly bound to multiple recovery
  items. Each recovery item selects at most one source in one intake artifact.

For an approved PDF, the source intake validates a regular non-symlink `.pdf`
file, a non-empty bounded size, `%PDF-` header bytes, an EOF marker near the
end of the file, and its SHA-256. An optional `expected_pdf_sha256` must match
the calculated hash. The resulting artifact keeps the safe `source_id`, hash,
byte size, provenance/access metadata, and human decision metadata; redaction
here means it never serializes the source path or PDF bytes.

The emitted `application/pdf` content type is an envelope classification based
on these bytes, not a semantic PDF validation. Intake requires an OS
`O_NOFOLLOW` safeguard and fails closed on platforms that cannot provide it.

This is a PDF-envelope check, not a scientific-content parser. It deliberately
does not count pages or extract text/tables/images. Page-count verification and
actual parsing belong to the later, separately gated parser preflight.

## Manual versus Explicit Recovery Targets

The intake artifact preserves the recovery-plan target exactly:

- An approved `explicit_reference_found` table/figure item becomes
  `eligible_for_targeted_source_parse` using the already-established target
  locator.
- An approved `manual_locator_required` item remains manual and becomes only
  `eligible_for_manual_source_review`.

In particular, this module never invents a table/figure number, upgrades a
bare or generic citation to an explicit target, or changes its source anchors.

## CLI

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_supplementary_source_intake \
  --recovery-artifact runs/<run_id>/review/oled_supplementary_evidence_recovery.json \
  --intake-manifest <operator-local-path>/supplementary_source_intake_manifest.json \
  --output runs/<run_id>/review/oled_supplementary_source_intake.json
```

The output records the original request, mapping-result, context, and recovery
plan digests plus a new deterministic intake-plan digest. It remains
review-only, offline-only, and non-executable. All execution-side-effect flags
stay false, including network access, downloading, MinerU/LLM calls, PDF
content parsing, candidate regeneration, evidence staging, gold creation, and
dataset writing.

This is an operator-run local CLI. Only run a human-reviewed manifest from a
trusted local environment; it is not an API for arbitrary remote path access.
CLI status output reports only the output basename, never an operator-local
source path or output directory.

## Boundary After Intake

An intake artifact is not permission to parse automatically. The next phase
must separately verify that the current local PDF still hashes to the intake
artifact's recorded SHA-256, then obtain any required parse confirmation before
running a parser and regenerating review candidates. Network acquisition,
redirect/provenance verification, and source discovery are outside this module.
