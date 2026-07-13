# OLED Supplementary Parser Preflight

## Purpose

This gate follows [local supplementary-source intake](oled-supplementary-source-intake.md)
and precedes any parser execution. It rebinds an operator-local PDF path,
verifies that its current bytes are still the source approved by intake,
validates its page count, and emits a redacted plan for a later parser run.

It does not invoke MinerU, a generic document parser, an LLM, or a network
service. It does not extract text, tables, images, or formulas; regenerate
candidates; stage evidence; create gold records; or write a dataset.

## Inputs and Human Manifest

The preflight consumes both the supplementary recovery artifact and the
source-intake artifact. The recovery artifact makes it possible to verify that
the intake item still matches the original recovery status, target kind, and
locator. This rejects a self-consistent modified intake artifact.

Because the intake artifact deliberately omits local paths, the operator
provides a local parser-preflight manifest:

```json
{
  "schema_version": "oled_supplementary_parser_preflight_manifest.v1",
  "paper_id": "paper016",
  "source_request_digest": "<request-digest>",
  "source_mapping_result_digest": "<mapping-result-digest>",
  "source_context_digest": "<context-digest>",
  "recovery_plan_digest": "<recovery-plan-digest>",
  "intake_plan_digest": "<intake-plan-digest>",
  "parse_confirmed": true,
  "reviewed_by": "reviewer-02",
  "reviewed_at": "2026-07-13T09:00:00Z",
  "selected_recovery_item_ids": ["supplementary-recovery:<explicit-item-id>"],
  "sources": [
    {
      "source_id": "paper016-si-v1",
      "local_pdf_path": "<operator-local-path>/paper016_si.pdf"
    }
  ]
}
```

`parse_confirmed` must be true. Every selected item must be an already approved
`explicit_reference_found` table or figure with intake eligibility
`eligible_for_targeted_source_parse`. Manual, deferred, rejected, unknown, and
duplicate items are rejected. The local source mappings must cover exactly the
selected item sources, with no unused path accepted.

The manifest is operator-local input. Do not commit local paths or raw PDFs.

## Revalidation and Page Counts

For every source, the preflight:

- opens a regular, non-symlink PDF through one `O_NOFOLLOW` file descriptor;
- validates its PDF envelope, page count (using `pdfplumber` without
  text/table/image extraction), and SHA-256 through that same descriptor;
- checks descriptor metadata for changes while performing those checks; and
- compares that same-descriptor hash and byte size with the approved intake
  envelope.

Replacing the pathname while the descriptor is open is rejected rather than
allowing a page count from one PDF to be paired with another PDF's hash.

`pdfplumber` is required. Its absence is fail-closed; the preflight does not
invent a page count or fall back to MinerU. The output retains only source ID,
hash, byte size, page count, target kind/locator, and a deterministic digest.
It contains no local path, PDF bytes, raw text, table content, or parser output.

## Locator Semantics

`Supplementary Table S1` and `Supplementary Fig. S1` are locators, not PDF
page numbers. No trusted page range exists in the intake chain, so each item
uses `parse_scope = full_source_then_locator_review`. This gate never fabricates
`start_page`, `end_page`, a table index, or a manual locator.

## CLI

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_supplementary_parser_preflight \
  --recovery-artifact runs/<run_id>/review/oled_supplementary_evidence_recovery.json \
  --source-intake-artifact runs/<run_id>/review/oled_supplementary_source_intake.json \
  --parse-manifest <operator-local-path>/supplementary_parser_preflight_manifest.json \
  --output runs/<run_id>/review/oled_supplementary_parser_preflight.json
```

The output path must differ from all JSON inputs and every declared local PDF.
A collision fails before PDF inspection or writing, protecting the input
artifacts and source PDF from replacement.

## Boundary After Preflight

A successful preflight is not parser execution. A later, separately confirmed
stage may parse the full approved source, verify the locator against parsed
output, and then route its evidence through the existing review workflow.
