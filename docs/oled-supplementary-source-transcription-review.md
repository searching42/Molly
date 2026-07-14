# OLED supplementary source-transcription review

PR-J is an offline, exact-bound human attestation boundary for one located
supplementary table. Its narrow claim is:

> The structured table shown in the review packet is visually equivalent to
> the same bounded table in the authoritative source PDF under one explicit,
> versioned visual-equivalence contract.

It does not decide whether the paper is scientifically correct. In particular,
it must keep source fidelity separate from semantic mapping, ontology coverage,
material identity, and downstream dataset admission.

## Two different meanings of evidence equality

PR-J uses two deliberately separate equality mechanisms.

### Machine-verifiable exact binding

`exact-bound` refers only to immutable evidence and artifact identity. The
packet and adjudication must bind and replay the complete upstream chain:

- the exact PR-G scoped candidate request;
- the original external response manifest and its PR-H validation artifact;
- the exact PR-I semantic-review packet, human decision manifest, and
  adjudication artifact;
- the exact authoritative supplementary PDF bytes; and
- the exact full-page image bytes reviewed by the human.

Every file is bound by its file SHA-256. Canonical artifacts also retain and
replay their canonical digests. Run, paper, scope, source, table, table-content,
review-item, cell, producer, and human-decision bindings must agree throughout
the chain. A missing, substituted, stale, duplicated, or self-consistently
rewritten upstream artifact fails closed.

The PDF remains authoritative. The page image is a review projection of those
exact PDF bytes, not a replacement source.

### Human-attested visual equivalence

Human acceptance is not byte equality between a typeset PDF and JSON. It is
visual equivalence under a versioned contract, initially:

```text
oled_supplementary_source_transcription_visual_equivalence.v1
```

Version 1 permits only representation changes that do not change scientific
content:

- typesetting line breaks may be collapsed or moved;
- non-semantic whitespace may be normalized; and
- visually equivalent subscript or superscript typography may be represented
  by the contract's equivalent LaTeX or HTML notation.

A rectangular parsed row also requires a non-empty internal key for every
column. Version 1 therefore keeps an exact positional parser key such as
`column_1` in a separate, machine-readable header binding when it is a
candidate placeholder for a visually blank source header. The review table
renders that source-header candidate as an empty cell; it never presents
`column_1` as source text. The binding can be accepted only when the reviewer
confirms that the corresponding source header is visibly blank. If the source
actually prints `column_1` or any other text there, `headers_check` must be
`mismatch`. Every non-placeholder header candidate must remain byte-for-byte
equal to its parser key.

Version 1 does **not** permit a change to any of the following:

- scientific symbols, Greek letters, arrows, or the object to which a
  subscript or superscript is attached;
- units or whether a unit is explicitly present in the source;
- positive or negative signs;
- digits, decimal points, scientific-notation exponents, or trailing zeros;
- header, row, column, or cell order and association;
- subject labels;
- footnote markers, footnote text, or marker-to-header association; or
- visible table boundaries, continuations, rows, columns, or cells.

The packet must name the visual-equivalence contract version. A later contract
change creates a new review boundary and cannot reinterpret an earlier human
decision silently.

## Authoritative PDF and page-asset chain

The page asset must be rendered from the exact bound PDF using Poppler as a
full-page, 200 dpi, RGB PNG. The artifact records at least:

- source PDF SHA-256;
- declared PDF page number and its indexing convention;
- `pdfinfo` and `pdftoppm` identity, common Poppler version, and executable
  SHA-256 values;
- a digest binding the declared Poppler runtime trust boundary and the two
  pinned front-end executable evidence records;
- 200 dpi RGB full-page render profile;
- rendered PNG dimensions and SHA-256; and
- the source table's page and `source_bbox` provenance.

The runner never searches the inherited `PATH`. It uses `/usr/bin` by default,
or an operator-supplied absolute `--poppler-bin-dir` as an explicit local trust
root. Both tools must be native ELF or Mach-O executables in that same pinned
directory, report the same Poppler version, remain unchanged across execution,
and have their exact bytes hashed into the evidence. Execution uses private
copies made directly from the pinned executable descriptors, so later
replacement of the operator-supplied source executable paths cannot affect
those copies. Version 1 does not claim to resist a same-UID adversary that can
concurrently replace files inside the process-owned private temporary
directory. Likewise, the verified PDF copy is unlinked and supplied to every
Poppler call through its still-open file descriptor; the renderer never
reopens the source by pathname. The subprocess
receives a minimal environment, no stdin, bounded logs, a private working
directory, CPU/file-size limits, an address-space limit where the host supports
it, and a killable process group.

This is explicitly an operator-trusted dynamic-runtime boundary, not a claim
that every shared library in the Poppler dependency closure was hashed. The
artifact records `dynamic_library_closure_bound=false` and names that trust
model. On system installations, OS package integrity remains part of the host
trust boundary; on a custom installation, the operator is responsible for the
selected runtime directory and its linked libraries. A future stronger profile
may instead bind a container/image digest or a complete copied dependency
closure, but it must use a new evidence contract rather than silently upgrading
version 1.

PNG acceptance requires a complete CRC-valid decode as 8-bit RGB. The decoded
dimensions must agree with the bound page MediaBox, page rotation, and 200 dpi
rendering (allowing only the renderer's one-pixel rounding alternatives).
Symlink output, malformed or truncated PNG, an alpha/palette/grayscale image,
an unsafe pixel count, or a cropped-size image fails closed.

The packet and decision bind the exact PNG bytes that the validated Markdown
references. Any alternate preview, recompression, resize, or derived annotation
is non-authoritative unless separately identified and must not replace the
bound image. The render command verifies the complete asset bundle immediately
before publishing Markdown, and adjudication rerenders the PDF and verifies the
bundle again. Version 1 assumes no adversarial concurrent filesystem mutation
while the human has the Markdown open; a stronger claim about the exact pixels
displayed at an instant would require a separately attested viewer or an
embedded immutable review artifact.

In version 1, `source_bbox` is provenance and locator metadata only. It must not
be used to crop the review asset. The reviewer sees the complete page so that
the caption, footnotes, page anchor, surrounding table boundary, and possible
continuation cues remain visible. A cropped-table-only packet is invalid for
PR-J v1.

## One table, one review item, seven checks

PR-J presents one review item per bound table, not one decision per cell. The
single table item contains exactly these seven component checks:

| Component | What the reviewer attests |
| --- | --- |
| `page_anchor_check` | The full-page asset shows the expected page and the same captioned table selected by the upstream locator chain. |
| `caption_check` | The complete caption is represented, subject only to permitted line-break and whitespace normalization. |
| `headers_check` | Every visible header candidate is present in the same order, with unchanged scientific symbols, units, sub/superscripts, and footnote markers; any separately displayed positional parser placeholder corresponds to a visibly blank source header. |
| `row_structure_check` | Every visible row and subject label is present in the same order, with the same header-to-row and subject-to-value associations. |
| `cell_literals_check` | Every visible cell literal is present at the same row and column, including signs, digits, decimal precision, and trailing zeros. |
| `footnotes_check` | Every table footnote is complete and each marker remains associated with the correct header or table element. |
| `table_extent_check` | The bounded table includes all visible rows, columns, cells, boundaries, and continuation content; no visible part of this table is omitted. |

Each component result is exactly one of:

```text
verified_equivalent
mismatch
not_checked
```

The seven results are structured evidence for one table-level decision. They
must not become seven independent admission decisions, and they must never be
expanded into 49 repetitive human clicks for the paper016 canary.

## Table-level decisions and compatibility rules

Exactly four table-level decisions are allowed.

### `accept_bounded_source_transcription`

Use only when all seven components are `verified_equivalent`. Neither
`mismatch` nor `not_checked` is compatible with acceptance. This decision
attests bounded source transcription under the named visual-equivalence
contract; it does not validate scientific truth.

### `needs_reparse`

Use when the correct source page and table are reviewable but the structured
representation differs from the source. `page_anchor_check` must be
`verified_equivalent`, and at least one of the remaining six components must be
`mismatch`. A bounded explanatory note is required.

There is no inline correction syntax. A caption, header, row, cell, trailing
zero, unit, footnote, or table-extent mismatch invalidates the parsed-table
content binding. Correction must restart the chain at locator review and then
regenerate PR-G, PR-H, and PR-I before a new PR-J packet can be reviewed:

```text
locator -> PR-G -> PR-H -> PR-I -> PR-J
```

No artifact based on the old table-content digest may remain eligible.

### `needs_source_check`

Use when the reviewer cannot determine equivalence from the bound source and
full-page asset, for example because the relevant source content is illegible,
ambiguous, missing, or cannot be checked. At least one component must be
`not_checked`, no component may be `mismatch`, and a bounded explanatory note
is required. A known transcription mismatch belongs to `needs_reparse`, not
this decision.

This outcome remains unresolved and grants no downstream eligibility. If a new
or better source PDF is supplied, its changed bytes require a newly bound
source-recovery and downstream artifact chain.

### `reject_scope`

Use when the located page/table is the wrong target, is outside the admitted
scope, or should be terminally rejected rather than reparsed. A bounded
explanatory note is required. A wrong page or wrong table normally appears as a
`page_anchor_check` mismatch; remaining components may be `not_checked` after
that terminal finding. Rejection cannot be converted into an accepted table by
editing the decision manifest.

Unknown decisions, unknown components, duplicate or missing components, stale
item IDs or digests, incomplete decision coverage, incompatible component
results, unsafe reviewer text, or a changed packet fail closed.

## Claim boundary after acceptance

An accepted PR-J adjudication may establish only source-PDF byte verification,
page-asset binding, the seven completed visual checks, and bounded table
transcription validation. It may not establish or change:

- property mapping or semantic-note decisions made in PR-I;
- physical correctness, author intent, or cross-paper comparability;
- ontology membership or ontology extensions;
- material or molecular identity, structure, role, or SMILES;
- schema-candidate materialization or automatic merging;
- reviewed-evidence or gold-candidate staging;
- direct or device-only admission; or
- curated-dataset or training-data writing.

Accordingly, broad flags such as `scientific_content_validated`,
`physical_semantics_validated`, `material_identity_resolved`,
`ontology_extensions_applied`, `schema_candidates_created`,
`gold_records_created`, and `dataset_written` remain false. The PDF remains
authoritative, and PR-J makes no claim that the whole paper or supplementary
information has been exhaustively reviewed.

Confirmed PR-I ontology-review outcomes remain outside every materialization
path. PR-J acceptance cannot turn them into known properties.

The adjudication status is `ready_for_later_identity_review` only when the
accepted-table intersection contains at least one PR-I later-eligible cell. An
accepted transcription with an empty identity-eligible roster remains
`review_complete_no_eligible_scopes`; downstream code must not infer readiness
from table acceptance alone.

## paper016 Supplementary Table S1 canary

The real canary must satisfy all of the following before
`accept_bounded_source_transcription` is valid.

### Exact evidence and page anchor

- paper ID is `paper016` and the run/scope identities match the complete
  PR-G/PR-H/PR-I chain;
- authoritative PDF SHA-256 is
  `sha256:b1d775a3eb59969ed170a81ea5e72d40a1c87833d1370a369807c9bb30d6f59b`;
- the reviewed asset is a Poppler 200 dpi RGB full-page PNG rendered from that
  exact PDF, with its exact PNG SHA-256 recorded;
- the visible printed page anchor is `S38`;
- the bound table is `table_p38_0178`; and
- the page shows Supplementary Table S1 rather than a cropped or substituted
  table image.

### Caption and headers

- the caption completely reports the TD-DFT summary for TDBA-based materials
  and includes `B3LYP/6-31G(d,p)`;
- there are eight ordered header positions: a visibly blank subject column,
  HOMO, LUMO, HOMO-to-LUMO gap, S1, T1, Delta-EST, and oscillator strength;
- the internal parser key `column_1` is shown only in the separate binding for
  the blank first header and is not represented as source-visible text;
- every explicit `eV` unit remains present; and
- the `a` and `b` superscript markers remain attached to Delta-EST and
  oscillator strength respectively.

### Rows, cells, precision, and footnotes

- there are seven ordered material-label rows and no omitted or invented row;
- the data region contains 56 visible row cells: seven subject labels and 49
  numeric cells;
- every numeric string remains in its source row and column;
- representative precision-sensitive literals include `-1.70`, `-5.50`,
  `3.80`, `3.30`, and `2.80` with two decimal places, plus `0.1280` with four
  decimal places;
- footnote `a` completely states that Delta-EST equals S1 minus T1;
- footnote `b` completely identifies oscillator strength; and
- the full-page view shows no omitted continuation, row, column, or table
  footnote for this bounded table.

### Downstream state

The accepted PR-I adjudication has 35 known-property cells eligible only for a
later materialization review and 14 cells still pending ontology review. PR-J
must preserve that split exactly:

```text
35 PR-I eligible known-property cells
14 ontology-pending cells
```

PR-J acceptance validates the transcription of all 49 numeric source cells but
does not make the 14 ontology-pending cells eligible. The unusual reported
HOMO/LUMO ordering remains governed by the separate PR-I semantic decision; a
source-faithful transcription is not an independent assertion of physical
correctness.

Later materialization remains an intersection, not a consequence of PR-J
alone:

```text
PR-I eligible and semantically resolved known cells
AND PR-J bounded transcription accepted
AND later material-identity resolution accepted
```

## CLI workflow

The examples below use an explicit trusted Poppler directory. Omit
`--poppler-bin-dir` only when both native tools are installed in `/usr/bin`.
The packet command requires a fresh asset directory. The render command requires
that directory to be named `assets` and to be a sibling of the Markdown output,
so its validated relative image references cannot silently point elsewhere.

Generate one exact-bound table review packet:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_source_transcription_review packet \
  --request-artifact /operator/local/supplementary_scoped_candidate_request.json \
  --response-manifest /operator/local/response_manifest.json \
  --response-artifact /operator/local/validated_response.json \
  --semantic-review-packet /operator/local/semantic_review_packet.json \
  --semantic-decision-manifest /operator/local/semantic_review_decisions.json \
  --semantic-adjudication /operator/local/semantic_adjudication.json \
  --source-pdf /operator/local/supplementary_information.pdf \
  --poppler-bin-dir /operator/trusted/poppler/bin \
  --asset-dir /operator/local/assets \
  --output /operator/local/source_transcription_review_packet.json
```

Render the human-readable packet:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_source_transcription_review render \
  --review-packet /operator/local/source_transcription_review_packet.json \
  --asset-dir /operator/local/assets \
  --output-markdown /operator/local/source_transcription_review_packet.md
```

Apply the one-table decision while replaying the complete chain:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_source_transcription_review adjudicate \
  --request-artifact /operator/local/supplementary_scoped_candidate_request.json \
  --response-manifest /operator/local/response_manifest.json \
  --response-artifact /operator/local/validated_response.json \
  --semantic-review-packet /operator/local/semantic_review_packet.json \
  --semantic-decision-manifest /operator/local/semantic_review_decisions.json \
  --semantic-adjudication /operator/local/semantic_adjudication.json \
  --source-pdf /operator/local/supplementary_information.pdf \
  --poppler-bin-dir /operator/trusted/poppler/bin \
  --review-packet /operator/local/source_transcription_review_packet.json \
  --decision-manifest /operator/local/source_transcription_decisions.json \
  --asset-dir /operator/local/assets \
  --output /operator/local/source_transcription_adjudication.json
```

All commands remain offline. Packet and adjudication call only the explicitly
pinned Poppler toolchain; they must not call an LLM, MinerU, network service, or
executable correction script. Publication retains the repository's safe-input,
fresh-output, no-collision, redacted-CLI, and fail-closed conventions.
