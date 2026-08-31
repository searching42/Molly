# Molly Core v2 CORE-04 CanonicalDocument and deterministic parser router

Status: `PASS — Full CI and CodeQL complete`

This report records the CORE-04 implementation only.  CORE-05 and all later
milestones have not started.

```text
repository: searching42/Molly
base: origin/main@925fb97b909f2090899493e8c495ecff43c860c0
branch: codex/molly-core-v2-core-04-canonical-document
implementation commit: d5be2ba554853266fc3ea5ce3803b735e5da6a0a
report commit: 8be0bfa6c2d2990f7f5517c46b09a31ad6bfd0c5
Draft PR: #69 (https://github.com/searching42/Molly/pull/69), remains Draft
```

The base is the merged CORE-03A main line.  The accepted CORE-03 commit
`08c92386c6d2f8bfacbb3492eca72210213d30dc` and the CORE-03A executable/test
commit `ac6026afe2b6265930314498e4efcbb7d4722e4d` are both ancestors of this
branch.  The delta from the CORE-03A executable/test commit to the accepted
CORE-03 commit is documentation-only (`docs/v2/reports/CORE-03.md`).

Readiness remains the frozen pre-implementation state: C0-C7 are `PASS`,
`core_goal_mode_ready=true`, and `core_cutover_ready=false`.  The immutable
v1 tag `molly-v1-pre-core-v2-20260829` and branch `legacy/molly-v1` both still
resolve to `ae7892dbf8a6bfe85dd909056eadc2afecc40d9d`.

## Scope and architectural boundary

CORE-04 adds a new production namespace under `src/molly/documents/`.  Its
dependency direction is:

```text
molly.documents -> molly.core
```

The implementation has no import dependency on `ai4s_agent`,
`prototypes.core_v2_contract_spike`, acquisition/network clients, LLM
providers, subprocesses, OLED, BR1, or remote compute.  The parser core is
source-neutral and does not carry acquisition occurrence fields such as URL,
access class, license, credentials, cache identity, acquisition ID, or
retrieval time.  CORE-03 remains responsible for acquisition provenance;
CORE-04 receives an exact immutable source artifact through the host-owned
`DocumentService` boundary.

## CanonicalDocument

`CanonicalDocument` is a frozen, source-neutral normalized representation of
one exact input artifact.  It includes:

```text
schema name/version
source artifact ID and media type/family
parser name/version and parser configuration digest
language and title/identifiers
sections and typed text blocks
tables, cells, and figures
references
parser quality/status and bounded warnings/counts
```

Canonical bytes use `molly.core.ids.canonical_json_bytes`: compact sorted-key
JSON, UTF-8, deterministic separators, and rejection of non-finite numbers.
The output artifact identity is exactly:

```text
artifact_id = sha256(canonical UTF-8 JSON bytes)
```

Object IDs and source locators are deterministic hashes/structural positions,
not UUIDs or process-local counters.  The closed `SourceLocator` vocabulary is
`XML_ELEMENT`, `HTML_ELEMENT`, `PDF_PAGE`, `PDF_REGION`, and
`MINERU_ELEMENT`.  Canonical blocks are limited to `TITLE`, `ABSTRACT`,
`HEADING`, `PARAGRAPH`, `LIST_ITEM`, `CAPTION`, and `OTHER_TEXT`.

Tables preserve deterministic row/column positions and cell spans.  Figures
and references retain structural locators and bounded intrinsic identifiers.
Round-trip validation rejects unknown kinds, dangling IDs, duplicate IDs,
invalid source locators, inconsistent counts, and oversized canonical output.

## Deterministic parser router

`DocumentParserConfig` is frozen and server-owned.  Its canonical digest binds
parser versions, parser priority, normalization policy, safety limits, and
PDF fallback policy.  `ToolSpec.execution_config_digest` binds this digest to
the `document_parse` ToolSpec; model input cannot choose a parser, backend,
path, URL, or configuration.

The closed parser registry and router support:

```text
JATS/XML       bounded safe ElementTree parsing and JATS structure extraction
generic XML    the same bounded safe XML normalization without JATS assumptions
HTML           bounded HTMLParser tree; script/style/noscript content skipped
PDF            optional lazy pdfplumber text extraction
MinerU         host-supplied backend seam, with a deterministic offline fallback
```

XML parsing rejects `DOCTYPE`, `ENTITY`, external/XInclude constructs, and
malformed input before producing a document.  XML/HTML source size, node,
depth, text, collection, table, and reference limits are enforced.  PDF text
is an optional parser; a low-quality or unavailable PDF can use the explicit
host-configured MinerU fallback, otherwise the result is bounded
`PARSER_UNAVAILABLE` with no output artifact.  The fallback is a test seam and
does not claim real MinerU execution.

## DocumentService and AgentLoop integration

`DocumentService` verifies that the declared `sha256:<digest>` source exists
in the host ArtifactStore, rechecks the bytes and metadata, routes the exact
bytes, and returns a bounded `ToolResult` plus one `ArtifactDraft` for the
canonical JSON.  It does not publish artifacts or write lineage; AgentLoop
retains publication, ledger, and lineage authority.

The `document_parse` ToolSpec is a PURE, empty-argument operation.  It
requires exactly one declared input artifact and rejects zero or multiple
inputs.  A successful result exposes only a bounded summary and the output
artifact ID; the full canonical document is the immutable ArtifactStore
content.  An unavailable parser produces no draft and no successful
publication.

The offline CORE-03 integration regression acquires a synthetic full-text
response through the existing mocked acquisition boundary, publishes its
content artifact, invokes `document_parse`, and verifies the resulting
canonical artifact and `DERIVED_FROM` relation.  Public/private and access
metadata remain occurrence-bound to acquisition and do not enter the
digest-keyed canonical document.

## Fixtures and golden evidence

The existing repository-safe fixtures were reused without adding copyrighted
full text:

```text
tests/fixtures/v2/synthetic/minimal.jats.xml
tests/fixtures/v2/synthetic/minimal.html
tests/fixtures/v2/expected/minimal.jats.canonical.json
tests/fixtures/v2/expected/minimal.html.canonical.json
```

Fixture source SHA-256 values are:

```text
minimal.jats.xml  784735519f4012f07a0a42a103aeecc51fa7b48e0ec89b072c6bfb2ffaafd544
minimal.html      04675438ee754a6e6d0ffc5e8be7eb14989743f6b1197a2f58c65b295e8da053
```

The canonical output SHA-256 values are:

```text
minimal.jats.xml  41f196ee28ddd0bf2e3bf5c7a9683321769aff31440fdccdb926084f0fbce6e6
minimal.html      d47679f041751b1662663d6cc14d12ff971e88932194c686af9cba5972afac40
```

The default parser configuration digest is:

```text
9518cf2b842e8b354a2d345785adcc3b48a22b805ba16aeb341488cb222f32bb
```

## Tests and verification

The dedicated suite is:

```text
tests/molly/test_core04_documents.py
```

It covers deterministic JATS/XML/HTML structure, tables/cells/spans,
figures/references, source locators, canonical round trips and golden bytes,
parser-config binding, XML/HTML/PDF safety limits, malformed and hostile XML,
optional PDF extraction and offline MinerU fallback, service exact-source
verification, AgentLoop publication/lineage, CORE-03 offline integration,
closed tool input, parser-unavailability behavior, and import boundaries.

Completed local evidence before PR creation:

```text
CORE-04 focused suite: PASS
CORE-01/CORE-02/CORE-03/C4/readiness/fixtures/privacy/legacy regression: 180 passed
python -m compileall -q src tests prototypes: PASS
git diff --check: PASS
uv lock --check: PASS (185 packages resolved)
PR Fast: PASS — 1,640 passed, 5,664 deselected in 18:43
```

The final GitHub PR checks and Full CI were run against exact HEAD
`8be0bfa6c2d2990f7f5517c46b09a31ad6bfd0c5`:

```text
PR checks workflow 33353482217: compile and diff PASS; pytest (PR fast) PASS
Full CI workflow 33353499475: PASS
  compile and shard policy: PASS
  weighted shard 0: PASS (13m12s)
  weighted shard 1: PASS (12m25s)
  weighted shard 2: PASS (15m50s)
  weighted shard 3: PASS (22m06s)
CodeQL workflow 33353481171: PASS
  actions, javascript-typescript, python, and aggregate CodeQL checks: PASS
```

No live network, credentialed provider, real MinerU, GPU, remote compute,
fresh BR1, or experimental scientific validation was run or claimed.

## CORE-05 boundary and remaining risks

This milestone does not implement CanonicalDocument consumers, acquisition
providers, OLED extraction, MinerU production integration, BR1, LLM/provider
integration, UI/API, observability, remote compute, or error-propagation
research.  CORE-05 may begin only in a separate Goal after this Draft PR is
reviewed and the CORE-04 Full CI/CodeQL evidence is complete.  A real MinerU
backend and production PDF quality policy remain future work; PDFs without a
usable configured parser correctly fail closed.

The pre-CORE-02 remediation and CORE-01/02 execution contracts are preserved.
The v1 freeze remains the rollback reference.  B2, B3, and B4 remain
`PENDING`, and `core_cutover_ready=false`.

## Evidence classification

```text
IMPLEMENTED:
  CanonicalDocument, SourceLocator, parser quality model, closed parser
  registry/router, DocumentService, document_parse ToolSpec, bounded parsers,
  deterministic fixtures and production tests.

TESTED:
  focused CORE-04 tests, 180-test CORE-00-to-CORE-03 regression, compileall,
  diff check, uv lock check, and PR Fast as recorded above.

VALIDATED:
  exact immutable source verification, deterministic canonical identities,
  parser safety/limits, offline CORE-03 integration, AgentLoop publication and
  lineage behavior, and import/security boundaries.

BLOCKED/PENDING:
  live/real MinerU, network, GPU, remote, BR1, and scientific acceptance are
  outside this milestone.
```

## Final acceptance state

```json
{
  "CORE-04": "PASS",
  "core_goal_mode_ready": true,
  "core_cutover_ready": false,
  "B2": "PENDING",
  "B3": "PENDING",
  "B4": "PENDING",
  "CORE-05": "NOT_STARTED"
}
```

CORE-05 has not started.  This report does not change readiness C0-C7 or
authorize CORE-08 cutover.

## Subsequent CORE-05 macro checkpoint note

The statement above is historical evidence for the CORE-04 report commit.
CORE-05 is a separate Goal/checkpoint under the same accelerated scientific
intake macro PR #69, with the branch name retained for continuity. The
CORE-05 checkpoint consumes the accepted `CanonicalDocument` and
`SourceLocator` contracts recorded here; it does not rewrite or extend the
CORE-04 parser authority. Its separate evidence is recorded in
`docs/v2/reports/CORE-05.md`.
