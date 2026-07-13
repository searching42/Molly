# OLED exact-bound supplementary candidate response MVP

This stage validates a separately supplied semantic-proposal response against
one PR-G supplementary scoped candidate request. It is an offline response
validation boundary only. It does not call an LLM, adjudicate scientific
meaning, create schema candidates, resolve material identity, or admit data.

## Inputs and exact binding

The file entry point requires:

- the complete PR-G scoped candidate request artifact; and
- an external response manifest that declares the exact request-file SHA-256
  and canonical `request_digest`.

Both files are read as stable regular files with `O_NOFOLLOW`, bounded byte
sizes, duplicate-key rejection, and non-finite JSON rejection. The output
records both the exact response-manifest SHA-256 and an order-normalized
canonical digest that can be replayed from the validated artifact.

The validator requires exact run and paper identities, exact scope coverage,
and repeated source-review, source-PDF, ParsedDocument, table-ID, and
table-content bindings for every scope. A missing, duplicate, or unknown scope
fails closed.

## Narrow table contract

Version 1 intentionally supports only a rectangular table with:

- non-empty, unique headers;
- every row containing exactly the declared headers;
- the first column used as the reported subject column; and
- at least one non-numeric subject label.

This prevents a response from selecting a numeric property column as the
subject and thereby omitting cells. Tables outside this narrow shape require a
later explicit subject-column review rather than a guessed mapping.

The numeric-cell roster is derived from the bound request, never trusted from
response counts. Every numeric-bearing cell outside the subject column must
have exactly one disposition, with exact scope, table, zero-based row and
column indexes, header, cell string, and same-row subject string.

Strict scalar cells preserve `reported_value_text` character for character and
recompute `reported_decimal_places`. Non-scalar numeric forms such as ranges,
comparators, or uncertainty expressions remain covered but must use
`needs_source_check:unsupported_numeric_form`; they are not silently omitted or
flattened into a scalar. Unicode minus is preserved in the reported literal but
normalized only for numeric/unit validation, including in scientific notation.

## Dispositions

Exactly four disposition kinds are allowed:

- `propose_known_property`;
- `needs_ontology_review`;
- `needs_source_check`; and
- `exclude_from_dataset`.

A known-property proposal may use only a property from the pinned PR-G
ontology. Its target layer must be allowed both by the request scope and by the
ontology property. The reported header is preserved, a trailing parenthesized
or bracketed source unit (including a trailing footnote marker), or a supported
bare unit suffix, must match the response literally and normalize to the pinned
canonical unit. A known-property proposal without an explicit parseable source
header unit fails closed; the response must use a source-check disposition
rather than infer a unit from outside the bound table.

Properties with required photophysical comparison context must include every
required context field explicitly. Unreported fields use explicit `null`;
omission is not treated as evidence of absence. The context is still a
proposal pending review.

An unsupported property cannot be assigned an invented property ID. It must
use `needs_ontology_review`, which records only the reported label, proposed
dataset layer, reported unit, and a bounded reason. PR-H never edits the
ontology or creates an ontology extension.

Device and measurement layers are not permitted in known or ontology-review
proposals. Headers matching pinned device/measurement properties or the
versioned obvious-device label set (for example EQE, current efficiency, power
efficiency, turn-on voltage, and EL peak) must use
`exclude_from_dataset:device_only`. Other device-only content must be caught in
human review; no exclusion is ever converted into admission to the current
molecule/interaction dataset.

## Semantic notes and authorship

Every PR-G semantic note is copied exactly. A non-empty note must remain
`unresolved`; response validation cannot claim that it has been corrected or
scientifically settled. All valid outputs remain ready only for human semantic
review.

The response manifest records whether it was human-authored or externally
LLM-assisted. LLM-assisted responses require provider, immutable
`model_snapshot_id`, prompt-contract version, exact `prompt_sha256`, and
timezone-aware production timestamp provenance. `prompt_sha256` is a producer
declaration: this validator checks its shape and binds it into the response
digests but does not read or verify separate prompt bytes. Audit timestamps must
obey request generation <= response production <= validation artifact
generation. The validator separately records that it made no network or LLM
call, avoiding the false implication that no LLM participated upstream.

A minimal response-manifest envelope is:

```json
{
  "schema_version": "oled_supplementary_scoped_candidate_response_manifest.v1",
  "run_id": "<exact request run_id>",
  "paper_id": "<exact request paper_id>",
  "request_artifact_sha256": "sha256:<exact request-file hash>",
  "request_digest": "sha256:<exact request canonical digest>",
  "producer": {
    "kind": "external_llm_assisted",
    "provider_id": "<provider>",
    "model_snapshot_id": "<immutable provider model snapshot>",
    "prompt_contract_version": "oled-supplementary-response.v1",
    "prompt_sha256": "sha256:<exact prompt hash>",
    "produced_at": "<ISO-8601 timestamp with timezone>"
  },
  "response_complete": true,
  "scope_results": ["<one complete exact-bound result per request scope>"]
}
```

Response-authored notes reject control characters, URL-like text, absolute and
high-confidence relative file references, credential-like text, code fences,
and high-confidence executable shell/Python forms. Literal source headers,
cells, and subject labels are not interpreted by this filter; they are
protected by exact request equality instead.

## Boundary flags

The validated artifact sets response structure, request byte/content binding,
scope coverage, cell coverage, and reported-literal preservation to true. It
keeps all scientific and downstream claims false, including:

- table transcription and exhaustiveness validation;
- scientific-content and physical-semantic validation;
- semantic-note resolution and human semantic adjudication;
- ontology application and schema-candidate creation;
- automatic merge and reviewed-evidence staging;
- direct or device-only admission;
- gold creation; and
- dataset writing.

`schema_mapping_proposed=true` only means that the external response contains
one or more structurally valid known-property proposals. It does not mean that
the mapping has been adjudicated.

## paper016 canary

The paper016-shaped canary has one 7-row, 8-column table. PR-H independently
derives 49 numeric cells and validates one disposition for each:

- 35 pinned-ontology proposals for reported HOMO, LUMO, S1, T1, and Delta-EST
  columns; and
- 14 ontology-review dispositions for the reported HOMO-LUMO gap and
  oscillator-strength columns.

The canary preserves values such as `2.80`, `3.30`, `-1.70`, `-5.50`, and
`0.1280`. The HOMO/LUMO labels and values remain as reported, while the scope
continues to carry the unresolved semantic note. This proves response binding
and completeness, not the correctness of the reported column semantics.

## Run the validator

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_supplementary_scoped_candidate_response \
  --request-artifact /operator/local/supplementary_scoped_candidate_request.json \
  --response-manifest /operator/local/supplementary_scoped_candidate_response_manifest.json \
  --output /operator/local/supplementary_scoped_candidate_response.json
```

The output must be fresh and cannot overwrite either input. Publication pins a
non-symlink parent directory by file descriptor, writes and fsyncs a same-dir
temporary file, links it exclusively through that directory descriptor, and
fails if the parent path changes before commit. CLI output contains only
redacted status and counts; it does not print paths, source cells, semantic
notes, credentials, or model identifiers.

## Next boundary

PR-I should generate a compact human semantic-review packet and exact-bound
adjudication artifact. It must consume and exact-bind both the original PR-G
request and the PR-H artifact (and the response manifest when authorship details
are shown), because PR-H output intentionally does not duplicate the full
caption, footnotes, nonnumeric cells, source locator, or source identity needed
for scientific review. It may group repeated cells by column/property mapping
to reduce reviewer burden, but it must preserve cell-level coverage and keep
unresolved source checks and ontology reviews outside schema materialization.
