# Molly Core v2 CORE-05 OLED scientific evidence and reviewed dataset

Status: `IMPLEMENTED — final macro CI evidence pending`

```text
repository: searching42/Molly
base: origin/main@925fb97b909f2090899493e8c495ecff43c860c0
branch: codex/molly-core-v2-core-04-canonical-document
pre-CORE-05 HEAD: c4d4ce20cbf97378ffa911bbd72d8893bb547fbf
Draft PR: #69 (https://github.com/searching42/Molly/pull/69), remains Draft
```

CORE-05 is the second checkpoint of the accelerated CORE-04 + CORE-05
scientific-intake macro. It does not start CORE-06 and does not change the
frozen readiness gates. C0-C7 remain `PASS`, `core_goal_mode_ready=true`,
`core_cutover_ready=false`, and B2/B3/B4 remain `PENDING`.

## Scope and CanonicalDocument boundary

The production dependency direction is:

```text
molly.evidence / molly.domains.oled / molly.llm -> molly.core
molly.evidence -> molly.documents.CanonicalDocument + SourceLocator
```

Candidate extraction accepts only a validated CORE-04 `CanonicalDocument`. It
never imports or accepts legacy `ParsedDocument`, raw PDF/HTML/XML objects,
MinerU objects, acquisition provider state, or run/controller state.
`SourceLocator` is retained on every candidate and every mapped scientific
field. No new controller or workflow engine was added.

## Candidate evidence and packets

`molly.evidence.candidates` implements immutable `EvidenceCandidate` and
`EvidenceCandidateBundle` records. Candidate IDs are SHA-256-derived from the
canonical document identity, candidate type, source locators, bounded source
payload, and extractor configuration. They contain no UUID, timestamp, run
ID, or filesystem path. The closed candidate types are `TEXT_EVIDENCE`,
`TABLE_ROW`, `TABLE_CELL_GROUP`, and `CAPTION_EVIDENCE`.

Table and text heuristics produce bounded field hints only; they do not make a
scientific assertion. `EvidencePacket` is one deterministic, bounded mapping
input. Its digest binds candidate IDs, locators, source text/table context,
and mapping schema version. It contains no repository dump, credentials,
private endpoint, or local agent context.

## Structured mapping contract

`FrozenOledMappingRequest` binds the request schema, candidate bundle artifact
ID, packet IDs and digests, OLED mapping schema digest, prompt-template
digest, logical provider profile reference, model identifier/version, and
mapping configuration digest. Its digest is the SHA-256 of exact canonical
UTF-8 JSON bytes.

`OledMappingResult` accepts a provider response only when its request digest
matches that exact request and every non-null mapped field has an explicit
candidate/source-locator evidence reference. The provider is a data
transformer, not an authority. `ScriptedMappingProvider` supplies the
offline contract runner keyed by exact request digest.

An optional `OpenAICompatibleStructuredProvider` exists only behind a
server-owned `StructuredProviderProfile` and injected transport. No live
provider, credentialed request, or network call was used; without an injected
transport the adapter reports `LIVE_STRUCTURED_MAPPING_PROVIDER_DEFERRED`.
Secrets are accepted only as a transient host header and are not placed in
model payloads or artifacts.

## OLED schema, identity, units, and conditions

`molly.domains.oled` defines a bounded `OledRecord` containing explicit
molecule identity, property, measurement condition, evidence references,
claim level, and validation status. Exact SMILES/InChIKey values may be
marked `RESOLVED` with an `EXACT_STRING` basis; a name alone remains
unresolved. No RDKit or other chemistry dependency is required by Core and
no chemical equivalence is inferred.

The initial property vocabulary is deliberately closed to `PLQY`. Explicit
fraction and percent values normalize deterministically to fraction while
retaining original value/unit. Missing, unsupported, non-finite, or
out-of-range values remain unresolved and block reviewed export. Conditions
are explicit and part of the comparison key, so the same molecule/property in
different conditions is not an exact duplicate.

Claim levels are closed to `SOURCE_REPORTED`, `DERIVED_NORMALIZATION`, and
`SYNTHETIC_CONTRACT_ONLY`. The synthetic fixture uses only
`SYNTHETIC_CONTRACT_ONLY`.

## Deterministic validation and duplicate/conflict handling

`OledValidationReport` reuses Core `ValidationResult` with only the approved
`ARTIFACT`, `RELATION`, and `BUNDLE` scopes and `PASS`, `FAIL`, and `REVIEW`
statuses. Validation bytes use the fixed sentinel validation timestamp so a
fixed mapping/configuration produces stable report bytes. No
error-propagation ontology was introduced.

The duplicate comparison key includes exact identity, property, normalized
unit, and condition. Duplicate groups are retained and classified as
`PRIMARY`, `CONSISTENT_DUPLICATE_CANDIDATE`, `CONFLICT_CANDIDATE`, or
`CONFLICTING_DUPLICATE_CANDIDATE`; nothing is silently merged or removed.
The fixture expects CCO 0.65/0.66 to remain consistent and CCN 0.58/0.91 to
remain a review-blocking conflict. Optional leakage checks report explicit
identity/source/group collisions but do not implement dataset splitting.

## Review bundle and dataset export gate

`ReviewBundleBuilder` creates an immutable exact-input review artifact binding
CanonicalDocument IDs, candidate bundle IDs, mapping/validation artifacts,
validated records, validation results, duplicate groups, blocking issues, and
a deterministic summary. The bundle does not contain a review decision. The
host constructs the existing Core `ReviewRecord`; no model tool can fabricate
`APPROVED`.

`DatasetExporter` requires `ReviewRecord.decision == APPROVED`, exact
`ReviewRecord.assert_matches` against the review bundle ID and SHA-256, no
structural blocking issue, and PASS validation status for every exported
record. It emits fixed-column LF-terminated UTF-8 CSV and canonical UTF-8
JSON. Rows are sorted by record ID; no run ID, timestamp, random UUID, or
`exported_at` enters exported content. The same exact bundle/review/config
produces byte-identical JSON and CSV.

## Tool boundary and offline fixture

```text
oled_extract_evidence         PURE
oled_contextual_map           NETWORK_READ (injected provider only)
oled_validate_records         PURE
oled_prepare_review_bundle    PURE
oled_export_reviewed_dataset  PURE
```

Their model argument schemas are empty and declared artifact inputs are
checked by the host executor. Each tool returns only a small summary in
`ToolResult`; all scientific content is an `ArtifactDraft`. AgentLoop remains
the only execution/publication authority.

The public-safe derived fixture is
`tests/fixtures/v2/synthetic/minimal.oled.jats.xml`, bound by
`docs/v2/fixtures/CORE05_OLED_EVIDENCE_FIXTURE_MANIFEST.json`, alongside the
existing reviewed synthetic `docs/v2/fixtures/oled_gold_fixture.json`.
The derived source SHA-256 is
`54517557dc88991a40c7d73f95422dbcbe47a82da7ba77f7296955bde6b4e9e8`; the
CORE-04 router produces CanonicalDocument artifact
`sha256:567fd2d70499b6aef4d07b7afe50a249e5948ae57c45f084bcd39213ff32e7ac`.

## Implementation and evidence

```text
src/molly/evidence/{candidates,packets,mapping,validation,review,dataset,tools,errors}.py
src/molly/domains/{__init__.py,oled/{__init__,identity,units,schema,extraction,validators}.py}
src/molly/llm/{__init__,profiles,structured_output}.py
tests/molly/test_core05_oled_evidence.py
tests/fixtures/v2/synthetic/minimal.oled.jats.xml
docs/v2/fixtures/CORE05_OLED_EVIDENCE_FIXTURE_MANIFEST.json
```

The focused suite passes 13 tests, including deterministic candidate IDs,
evidence/locator tamper rejection, exact mapping request binding,
identity/unit/condition behavior, duplicate/conflict validation, exact
ReviewRecord gating, deterministic JSON/CSV output, review artifact reload,
an AgentLoop chain to review bundle, the full offline CORE-04-to-CORE-05
path, injected-provider secret isolation, and import boundaries. `compileall`
and `git diff --check` also pass. Final PR Fast, CodeQL, and one GitHub Full
CI run are recorded only after the complete macro executable/test batch is
pushed.

## Evidence classification and limitations

```text
IMPLEMENTED:
  deterministic candidates/packets, digest-bound structured mapping,
  conservative OLED schema, unit/condition handling, duplicate/conflict and
  leakage checks, exact review bundle, host ReviewRecord gate, deterministic
  JSON/CSV export, closed tools, synthetic fixture and focused tests.

TESTED:
  CORE-05 focused tests, CORE-04 router-derived fixture, compileall, and diff
  check as recorded above.

VALIDATED LOCALLY:
  exact canonical-document/source-locator binding, evidence binding,
  deterministic offline end-to-end behavior, review/export fail-closed rules,
  and import/secret isolation boundaries.

NOT CLAIMED / PENDING:
  live LLM/provider mapping, live literature acquisition, experimental or
  computational OLED validity, RDKit canonical chemistry, CORE-06 applicability
  preflight, Uni-Mol, REINVENT4, Top-N, GPU, remote, BR1, and cutover.
```

Final status changes to `PASS` only after the final coherent macro HEAD
passes focused regressions, PR Fast, CodeQL, and Full CI. CORE-06 remains
unopened.
