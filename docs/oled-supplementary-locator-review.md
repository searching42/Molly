# OLED supplementary locator review MVP

This step consumes a successful, content-bound supplementary MinerU execution
artifact and the corresponding normalized `ParsedDocument` JSON files. It
locates explicitly requested supplementary tables and produces JSON and
Markdown packets for human review.

It is deliberately an offline review boundary. It does not read PDFs, call
MinerU, use a network or LLM, regenerate candidates, stage reviewed evidence,
create gold records, or write a dataset. Device-only records remain excluded.

## Exact input binding

The operator supplies a local manifest:

```json
{
  "schema_version": "oled_supplementary_locator_manifest.v1",
  "run_id": "supp-mineru-run-001",
  "paper_id": "paper016",
  "execution_artifact_sha256": "sha256:<exact file hash>",
  "execution_artifact_digest": "sha256:<canonical artifact digest>",
  "sources": [
    {
      "source_id": "supp-source-001",
      "parsed_document_json": "/operator/local/parsed_document.json"
    }
  ]
}
```

The manifest must cover exactly the sources in the successful execution
artifact. Each parsed JSON file is opened with `O_NOFOLLOW` as a stable regular
file. Its byte size and SHA-256 must match the execution artifact's
`parsed_document_json` output binding. Its page count and parser backend must
also match the execution evidence. `ParsedDocument.paper_id` is parser-local
and may reflect the isolated execution snapshot filename, so it is not used as
the source identity; the exact output-byte binding is authoritative.

Local paths are never copied into the generated packets or printed by the CLI.
Both output files must be fresh and must not collide with any input.

## Locator rule

The MVP resolves table targets only. It normalizes Unicode, HTML markup, and
whitespace, then requires the caption to begin with one of these exact forms:

- `Table S1`
- `Supplementary Table S1`
- `Supporting Information Table S1`

The locator token has an exact alphanumeric boundary, so `S1` cannot match
`S10`. A table-of-contents entry whose caption is `Table of Contents` cannot
match merely because one of its rows mentions `Supplementary Table S1`.
Captions that begin with a locator range or list, such as `Table S1-S3`,
`Table S1/S2`, or `Table S1 and S2`, are not singleton matches and remain
unresolved. Normal descriptive text and `Table S1 (continued)` remain valid
singleton-caption forms.

Resolution is fail closed:

- one exact caption match: `exact_match`, with the table content included;
- zero exact matches: `not_found`;
- multiple exact matches: `ambiguous`, with no table selected;
- figure target: `unsupported_target_kind`;
- unrecognized locator syntax: `unsupported_locator_format`.

For an exact match, caption, headers, row values, footnotes, page, and source
bounding box are copied from `ParsedDocument`. Cell values remain strings and
are not numerically normalized, so reported precision such as `0.030` is
preserved. Packet size and table dimensions are bounded.

## Generate the packet

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_supplementary_locator_review \
  --execution-artifact /operator/local/supplementary_mineru_execution.json \
  --locator-manifest /operator/local/locator_manifest.json \
  --output-json /operator/local/supplementary_locator_review.json \
  --output-markdown /operator/local/supplementary_locator_review.md
```

The JSON artifact binds the exact execution and manifest bytes, every parsed
document hash, each selected table-content digest, and a canonical digest of
the complete packet. The Markdown file presents the same scientific content
for manual inspection.

## Human review

For every `exact_match` item:

1. Open the original supplementary PDF independently.
2. Confirm the requested locator, page, and full caption.
3. Compare every header, row value, and footnote with the source, including
   units, qualifiers, signs, inequality symbols, and trailing zeros.
4. Confirm that the selected table is the scientific table rather than a table
   of contents or cross-reference.
5. Record the decision separately. Generated entries intentionally remain
   `pending`; this MVP does not apply decisions or admit records.

Any unresolved item requires manual source location before later candidate
regeneration can be considered. Candidate regeneration and adjudication are a
separate follow-up step.
