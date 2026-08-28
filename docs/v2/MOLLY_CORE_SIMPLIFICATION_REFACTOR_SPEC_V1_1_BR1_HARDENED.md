# Molly Core v2 Simplification Refactor Specification

Status: `OWNER_REVIEWED_DRAFT_V1_2`
Repository: `searching42/Molly`
Audit baseline candidate: `main@4352f137db3976cff31bf6cb30f543caa38f8013`
Research-error-propagation dependency: `NOT_REQUIRED_FOR_CORE_REFACTOR`
BR1 policy: `OPTIONAL_INSTALL_BUT_MANDATORY_CUTOVER_PARITY`
Implementation authorization: gated by `C0-C7`; default cutover additionally gated by `B0-B4` and explicit Owner approval.

## 0. Purpose

This specification defines a research-direction-neutral simplification of Molly. It exists to reduce control-plane complexity while preserving scientific provenance, reproducibility, review boundaries, and the already-validated BR1 scientific capability.

The refactor MUST NOT assume that long-horizon scientific error propagation is the final research direction. Error-propagation-specific infrastructure is out of scope until separately approved.

The implementation objective is a small scientific workflow core with one execution path and explicit capability boundaries.

## 1. Core target architecture

The target execution spine is:

```text
RunRequest
  -> AgentLoop / RunEngine
  -> ToolRegistry + ToolPolicy
  -> Scientific tool
  -> ArtifactStore + RunLedger + ArtifactLineage
  -> ValidationResult / ReviewRecord
```

The scientific default path is:

```text
Literature metadata search
  -> full-text resolution and compliant acquisition
  -> structured full text first
  -> CanonicalDocument
  -> OLED evidence extraction and contextual mapping
  -> digest-bound human review
  -> dataset export
  -> optional BR1 inverse-design plugin
```

Core v2 MUST have exactly one authoritative execution path. UI, conversation, observability, remote compute, BR1, MinerU, and provider-specific adapters are projections or plugins around the core, not alternate authorities.

## 2. Mandatory simplification decisions

Core v2 MUST NOT migrate the following v1 control-plane abstractions as runtime authorities:

- Permission
- Authorization
- StartIntent
- independent Controller state machine
- independent Replanner authority
- Autonomy L1/L2
- AuthorityRelation / AuthoritySet
- AutonomyLease
- EvidenceGrant usage/admission chains
- autonomous failure-recovery successor authority
- multi-layer publication/adoption/reconciliation authority

These are preserved only in the frozen v1 line where needed for historical evidence and rollback.

The replacement authority model is intentionally small:

```text
RunRequest + closed ToolPolicy + exact ApprovalRecord
```

A `RunRequest` defines the goal, bounded inputs, profile, and run budget.
A `ToolPolicy` defines a closed set of allowed tools and side-effect classes.
An `ApprovalRecord` binds a human decision to one exact concrete tool-call digest when human approval is required.

Unknown tools, unknown policy values, stale approvals, or mismatched digests MUST fail closed.

## 3. Core data model

### 3.1 RunLedger

RunLedger is the append-only factual execution record. It MUST record at least:

- run_id
- step_id
- event type
- tool name and version
- logical provider/model profile where applicable
- input artifact ids
- output artifact ids
- status
- timestamps
- prompt/config digests where applicable
- relevant seed metadata where available

RunLedger MUST NOT be treated as a mutable Controller state machine.

### 3.2 ArtifactStore

Artifacts MUST be immutable or content-addressed after publication. Each artifact MUST carry:

- artifact_id
- SHA-256
- media type
- schema/version when applicable
- producer step
- input artifact references
- provenance/source metadata
- creation time

Scientific records MUST remain traceable to source evidence.

### 3.3 ArtifactLineage

Core v2 keeps lightweight dependency relations only:

- `CONSUMED_BY`
- `PRODUCED_BY`
- `DERIVED_FROM`
- `SUPPORTED_BY`

This is provenance/dependency lineage, not a causal-error graph.

### 3.4 Validation and review

Core validation scope is limited to:

- `ARTIFACT`: one object is structurally/scientifically valid in isolation
- `RELATION`: a binding between objects is valid
- `BUNDLE`: a local set of related objects is internally consistent

`ReviewRecord` MUST bind a human scientific decision to an exact artifact digest.

Core v2 MUST NOT introduce research-specific `ErrorInstance`, `InterventionSpec`, `PairedRunGroup`, `PropagationOutcome`, or descendant counterfactual replay.

## 4. Agent and tool execution

There is one `AgentLoop` / `RunEngine`.

A normal turn is:

```text
build bounded context
-> model proposes structured action
-> ToolRegistry resolves a known ToolSpec
-> ToolPolicy checks the action
-> optional exact approval check
-> tool executes
-> output schema validates
-> artifacts publish
-> RunLedger appends observation
-> continue, stop, or request review
```

Planning and replanning MAY exist as structured model outputs inside the same loop. They MUST NOT become separate execution authorities.

### 4.1 ToolRegistry

Each ToolSpec SHOULD define:

- name
- input schema
- output schema
- risk/side-effect class
- execution backend
- timeout/resource envelope
- human-approval requirement

The model MUST NOT provide physical credentials, arbitrary shell, arbitrary SSH targets, arbitrary filesystem paths, or arbitrary publisher endpoints unless a separately approved tool contract explicitly allows a bounded value.

## 5. Literature acquisition

The current PDF/conversation/gate-coupled literature intake is to be rewritten.

The acquisition subsystem SHOULD be separated into interfaces such as:

```text
MetadataProvider
FullTextResolver
FullTextFetcher
AcquisitionCache
AcquisitionScheduler
```

Initial provider work should prefer legitimate APIs, TDM endpoints, and open-access sources. Provider-specific rate limits, redirect limits, content-size limits, content-type checks, provenance, license metadata, and SSRF protections MUST be explicit.

The implementation MUST NOT add CAPTCHA bypass, residential proxy rotation, fingerprint evasion, or other access-control bypass behavior.

## 6. Document parsing

All downstream extraction MUST consume a source-neutral `CanonicalDocument` rather than MinerU-specific bytes.

Preferred routing:

```text
JATS/XML -> publisher XML -> HTML -> lightweight PDF text -> MinerU PDF fallback
```

`CanonicalDocument` SHOULD represent sections, blocks, tables, figures/references where available, plus stable typed source locators.

MinerU is optional and PDF-fallback-only. Minimal Core installation and structured-text routes MUST work without MinerU.

## 7. OLED scientific layer

OLED remains the first supported scientific domain but Core MUST NOT claim universal domain semantics.

The refactor SHOULD preserve and simplify:

- OLED field/schema definitions
- condition-aware molecular/property identity
- deterministic evidence-candidate extraction
- source/evidence locators
- contextual structured LLM mapping
- unit/duplicate/conflict/leakage validators
- digest-bound dataset review
- CSV/Parquet or equivalent deterministic export

Deterministic extraction and validation SHOULD precede LLM semantic mapping when possible.

## 8. LLM providers

Provider abstraction is retained but simplified.

Core SHOULD preserve:

- server-owned logical provider/model profiles
- structured output validation
- bounded SSE transport where used
- request/response/config digests
- explicit timeout/error taxonomy

Secrets MUST NOT enter prompts, artifacts, public evidence, or RunLedger payloads.

## 9. Observability

OpenTelemetry and LangSmith are optional observer-only exporters.

Authoritative data comes from RunLedger and ArtifactStore. Exporter failure or absence MUST NOT alter execution outcomes or authoritative artifact bytes.

## 10. Conversation, API, and UI

Conversation is not an execution authority in Core v2. It MAY later map user intent into a RunRequest.

Core acceptance MUST be possible through CLI or a minimal API without UI.

UI migration is deferred until the core contracts stabilize.

## 11. BR1 preservation and cutover contract

BR1 is optional for minimal installation but mandatory before repository default cutover.

The v1 verified BR1 reference path is preserved as historical evidence. The v2 BR1 plugin MUST re-establish functional/contract parity for:

```text
reviewed/current-run dataset
-> applicability preflight
-> fresh Uni-Mol training
-> model packaging
-> real REINVENT4 generation
-> generation packaging
-> current-model prediction
-> deterministic candidate evaluation
-> verified Computational Top-N projection
```

BR1 parity MUST verify:

- no historical model is reused as fresh training
- prediction is bound to the current-run trained model
- generation/prediction/evaluation are bound to current-run artifacts
- stale or foreign artifacts fail closed
- scientific claim boundaries remain explicit
- terminal projection replay is deterministic for frozen terminal inputs

Fresh stochastic runs are NOT required to reproduce identical candidate molecules or identical scores across v1 and v2. Parity is defined by scientific stages, contracts, freshness, lineage, terminal evaluation semantics, and representative successful runtime evidence.

### 11.1 Minimal remote-compute requirements

If representative BR1 requires remote GPU execution, v2 MUST retain only the minimal reliable compute contract:

- server-owned logical compute profile
- durable JobHandle
- idempotent submit
- restart-safe inspect
- restart-safe collect/adopt
- no duplicate dispatch after control-plane restart
- no stale/foreign output adoption

The v1 remote authority graph and lease system MUST NOT be reintroduced.

### 11.2 BR1 cutover gates

- `B0`: frozen v1 BR1 remains runnable/inspectable from immutable reference and rollback instructions exist
- `B1`: v2 BR1 contract fixtures and acceptance runner are frozen
- `B2`: fresh-real v2 BR1 acceptance succeeds
- `B3`: representative remote restart/inspect/collect canary succeeds when remote compute is used
- `B4`: Owner reviews and approves the BR1 parity report and default cutover

`CORE-08` MUST NOT execute unless B0-B4 are PASS and the Owner explicitly approves cutover.

## 12. Package and plugin boundaries

Target package shape may evolve during CORE-00/C2, but dependency direction MUST preserve a small core. A representative structure is:

```text
src/molly/
  core/
  llm/
  acquisition/
  documents/
  evidence/
  domains/oled/
  observability/
  cli.py
plugins/
  br1_inverse_design/
  remote_compute/
```

Core MUST NOT import from the frozen `ai4s_agent` package after migration closure.

Optional dependencies SHOULD be separated at least into core/minimal, PDF/MinerU, observability, BR1, and remote-compute groups where practical.

## 13. Migration policy

Use `docs/v2/MOLLY_CORE_MODULE_DISPOSITION_MATRIX_V1_1_BR1_HARDENED.csv` as the high-coverage migration inventory.

Allowed disposition decisions are:

- KEEP
- MIGRATE
- SIMPLIFY
- REWRITE
- PLUGIN
- DEFER
- ARCHIVE
- DELETE_FROM_V2

Before implementation beyond CORE-00, the matrix MUST be reconciled against the exact implementation HEAD at file level. Unclassified required modules are a blocker.

Do not preserve an old abstraction merely to keep historical tests green. Freeze legacy tests with v1 and write v2 tests from approved v2 requirements.

## 14. Readiness gates C0-C7

Production refactor implementation is authorized only when `docs/v2/readiness/core_refactor_readiness.json` truthfully records all C0-C7 as PASS and `core_goal_mode_ready=true`.

- `C0`: Owner-approved core scope/spec is frozen in repository
- `C1`: immutable v1 freeze reference, legacy branch/tag strategy, rollback/evidence inventory are recorded
- `C2`: exact-HEAD per-file module migration audit is complete
- `C3`: acquisition/network/credential/license/private-public security boundaries are frozen
- `C4`: minimal core contract spike and authority model are accepted
- `C5`: package/dependency/test/CI boundaries are frozen
- `C6`: representative literature/OLED fixtures and BR1 parity fixtures/runners are frozen
- `C7`: final Goal-mode execution contract, stop conditions, and repository instruction chain are frozen

No gate may be marked PASS without concrete evidence. Codex MUST NOT infer approval from intent.

## 15. Implementation queue

Recommended sequence:

- `CORE-00`: documentation-only freeze, exact-HEAD audit, readiness work, branch/tag/rollback planning
- `CORE-01`: RunLedger, ArtifactStore, ArtifactLineage, ValidationResult, ReviewRecord
- `CORE-02`: RunRequest, ToolRegistry, ToolPolicy, ApprovalRecord, single AgentLoop/RunEngine
- `CORE-03`: literature metadata/full-text acquisition, cache, provenance, security controls
- `CORE-04`: CanonicalDocument and parser router; MinerU optional fallback
- `CORE-05`: OLED extraction, contextual mapping, validators, human-reviewed dataset export
- `CORE-06A`: BR1 scientific plugin migration
- `CORE-06B`: minimal compute backend required by representative BR1
- `CORE-06C`: BR1 contract-parity, fresh-real acceptance, restart evidence, parity report
- `CORE-07`: minimal CLI/API and observer-only observability
- `CORE-08`: default entrypoint cutover and removal/archival of obsolete v1 runtime paths

CORE-08 is separately gated by B0-B4 and explicit Owner approval.

## 16. Codex Goal-mode rules

Codex MUST:

1. inspect repository state before modifying code
2. preserve v1 evidence and rollbackability
3. update readiness only from real evidence
4. prefer deletion/simplification over compatibility wrappers when v2 requirements permit
5. keep one execution authority
6. keep credentials/configuration server-owned
7. add targeted tests for each migrated contract
8. keep BR1 cutover blocked until B0-B4
9. report blockers rather than fabricating evidence

Codex MUST NOT:

- modify `main` directly
- force-push or move immutable freeze references
- merge its own refactor PRs without explicit Owner direction
- claim error-propagation research support
- implement InterventionEngine, ErrorInstance, PairedRunGroup, descendant counterfactual replay, or propagation statistics
- restore Permission/Authorization/StartIntent or the v1 autonomy/authority stack under new names
- add arbitrary shell/SSH/filesystem/network authority to the model
- silently weaken BR1 freshness, current-model binding, or scientific claim boundaries
- delete the legacy BR1 path before v2 BR1 parity is accepted

## 17. Stop conditions

Codex MUST stop and report when:

- C0-C7 are not all PASS before production refactor work
- required exact-HEAD module disposition is missing or ambiguous
- a security boundary is unresolved
- a required private credential/resource/fixture is unavailable
- BR1 fresh-real or remote canary evidence cannot be produced
- a proposed simplification would weaken a frozen scientific or safety invariant
- Owner approval is required for cutover

Partial completion with an explicit blocker report is preferred over speculative compatibility code.

## 18. Acceptance principles

Core v2 is acceptable only if representative tests demonstrate:

- one execution path
- no v2 Permission/Authorization/StartIntent/autonomy authority imports
- immutable/digest-bound artifacts and review
- reproducible run inspection from RunLedger/lineage
- structured full-text parsing can run without MinerU
- compliant acquisition cache/provenance works on representative sources
- OLED evidence can be traced to source locators
- exporter failure is non-fatal
- minimal installation does not require BR1 or MinerU
- BR1 remains available through the frozen v1 reference until v2 parity passes
- default cutover occurs only after B0-B4 and Owner approval

## 19. Non-goals for the current refactor

Current Core v2 does not aim to:

- become a universal production Agent platform
- support all scientific domains
- maintain full v1 API compatibility
- implement multi-agent teams
- implement autonomous long-running authority leases
- implement research-grade error propagation experiments
- automate wet-lab execution
- provide unrestricted shell or browser authority
- ship a complete UI before core acceptance

## 20. Owner review checklist

Before enabling production Goal-mode work, verify:

- [ ] core scope is approved
- [ ] v1 freeze/rollback reference exists
- [ ] exact-HEAD migration matrix is complete
- [ ] acquisition security policy is frozen
- [ ] minimal core contract is accepted
- [ ] CI/dependency boundaries are frozen
- [ ] representative fixtures are frozen
- [ ] Goal-mode contract is frozen
- [ ] readiness C0-C7 are evidence-backed PASS

Before CORE-08, additionally verify:

- [ ] B0-B4 are evidence-backed PASS
- [ ] fresh-real BR1 parity report is accepted
- [ ] Owner explicitly approves default cutover
