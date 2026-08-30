# Molly Core v2 Codex Goal execution contract

Status: `FROZEN_C7_READINESS_CONTRACT`

Date recorded: 2026-08-30 (Asia/Shanghai)

This repository-local contract is bound to the user-provided
`<USER_PROVIDED_LAUNCHER>` and the explicit Owner authorization carried by that
Goal. It records readiness authorization for later production implementation;
it does not itself start a production milestone.

## Repository identity and frozen references

```yaml
repository: searching42/Molly
review_branch: codex/molly-core-v2-launcher-1-2
draft_pr: 64
v1_freeze_commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9d
v1_freeze_tag: molly-v1-pre-core-v2-20260829
legacy_branch: legacy/molly-v1
```

The freeze tag and legacy branch both point to the exact v1 freeze commit and
were verified locally and at the repository remote. The tag is immutable; it
must never be force-moved. Rollback and evidence details are recorded in
[`V1_ROLLBACK_AND_EVIDENCE_INVENTORY.md`](V1_ROLLBACK_AND_EVIDENCE_INVENTORY.md).

## Authoritative inputs and digests

The readiness source of truth is
[`core_refactor_readiness.json`](readiness/core_refactor_readiness.json). The
following references are frozen for the C0-C7 state:

| Evidence | SHA-256 or binding |
| --- | --- |
| Core simplification spec | `0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7` |
| Reconciled module matrix | `d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4` |
| Matrix digest at Owner approval | `2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d` |
| Exact file disposition inventory | `6dc12b0a6d430e9fe6a31c38c3bfde2a12443dbe0dbdafb760859bd596ab83b3` |
| C2 audit report | `af615f2b54287546435e5918344a077df61ca00439812feb439025f7386fc4e8` |
| Acquisition security contract | `69fdb6f100e6b06a2a6d2998e521babaebc96d9f1e69ea0bb67d793801d019d6` |
| C4 contract-spike report | `554c4510d951997ed11db2faacce3286efe24812ee7ee345eec1d366e4578efa` |
| Dependency/CI boundary contract | `d1d73b8e495dc8d36bc6ef7609a10dd36d75c2d472c34a110ef8c460f5b059cf` |
| Repaired current-repository lock | `f204dc52afd4d2b50e58651e75bb75a8f6fa0a0192d9f17e03c79891122b30c4` |
| Literature fixture manifest | `626792c1be7d00eb6da884e6cc100cf4e404a5bf0a924e1c2f2dc51deb992a03` |
| OLED fixture manifest | `265c01225893a95e9eca034fb9fa6fb7a09ea28a281e4602a58de952006e8512` |
| BR1 parity manifest | `66734335c2707712049ef977f23f14eee9bdae35201471de4e34ebff844ff99c8` |
| Split README | `eb9d2512a662174d5955625e94806aad35f516e367cc0150403e99219a57b40a` |
| Roadmap | `06ca47cefd64a1ba7bad210b7d5bf324de5bfb92044a8c384e8b13ce4956194b` |
| v1 BR1 acceptance README | `784976fdc739c266c1af47570e943ddd8b59a0e1341d78fb19ccc0fe273874c4` |
| v1 BR1 acceptance manifest | `fdbad5cccbd096bba390a5926671ecdab864d50877c24cf1d3eb66acd9c38bd2` |

The scope approval in
[`CORE_V2_SCOPE_APPROVAL.md`](decisions/CORE_V2_SCOPE_APPROVAL.md) binds the
Owner decision to the exact spec digest and records both matrix digests. C2 is
a factual path/inventory reconciliation under that approval, not a new
architectural decision.

## Readiness authorization

The machine-readable manifest must remain the sole readiness truth:

```text
core_goal_mode_ready = true  iff C0-C7 are all PASS
core_cutover_ready   = true  iff C0-C7 and B0-B4 are all PASS
                         and a future Owner decision explicitly approves cutover
```

The final C0-C7 state authorizes a later, separate Goal to implement CORE-01
through CORE-07 production milestones. This closure Goal does not start any of
them. C0-C7 PASS does not authorize CORE-08. CORE-08 requires B0-B4 PASS plus
explicit future Owner cutover approval.

The approved scope explicitly keeps
`error_propagation_required=false` and
`error_propagation_implementation_authorized=false`. The following remain
outside this contract: `ErrorInstance`, `InterventionSpec`, `PairedRunGroup`,
`PropagationAnalyzer`, descendant counterfactual replay, propagation
benchmarks/statistics, and experimental scientific claims.

## Core and dependency boundaries

The approved future execution shape is:

```text
RunRequest -> AgentLoop / RunEngine -> ToolRegistry + ToolPolicy
-> scientific tool -> ArtifactStore + RunLedger + ArtifactLineage
-> ValidationResult / ReviewRecord
```

The future v2 package must not import `ai4s_agent`, recreate the v1 Controller,
replanner, authority graph, or recovery state machine, or expose credentials,
arbitrary hosts, paths, URLs, shell, or SSH to a model. `core/minimal` excludes
MinerU, RDKit, Uni-Mol, REINVENT4, OpenTelemetry exporters, LangSmith, remote
SSH, and UI-only dependencies. The detailed boundary is frozen in
[`PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md`](contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md).

The acquisition boundary is frozen in
[`ACQUISITION_SECURITY_AND_PROVENANCE.md`](contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md).
It permits only closed, server-configured provider routes and legal/access
status with provenance; it does not authorize a production fetcher here.

## Frozen fixtures and BR1 claim boundary

[`literature_fixture_manifest.json`](fixtures/literature_fixture_manifest.json)
freezes synthetic JATS/XML and HTML parser inputs. No redistributable PDF was
available at C6, so none is included. The OLED manifest reuses an existing
explicitly synthetic source and labels all values as fixture-only; it contains
identity, value, unit, condition, locator, duplicate, and conflict coverage.

[`br1_parity_manifest.json`](fixtures/br1_parity_manifest.json) freezes the
required BR1 scientific stages and lineage invariants. It is contract-only:
there was no fresh-real BR1, GPU, remote, or network-live run in this Goal.
B1 is contract-level PASS; B2 and B3 remain pending until their exact canaries
run.

## Implementation queue (not started by this Goal)

Each item is a separate Goal and independent milestone commit with targeted
tests and evidence:

1. CORE-01: bounded core schemas, ArtifactStore, RunLedger, and lineage.
2. CORE-02: one AgentLoop/RunEngine with ToolRegistry, ToolPolicy, and exact
   approvals.
3. CORE-03: conservative acquisition providers, cache, provenance, and
   security tests.
4. CORE-04: CanonicalDocument and XML/JATS/HTML-first parser contracts, with
   optional PDF/MinerU boundaries.
5. CORE-05: OLED evidence, validation, review, and leakage contracts.
6. CORE-06: optional BR1 and remote-compute plugin work under the frozen parity
   and restart-canary contracts.
7. CORE-07: package, observability, read-only API/CLI, and acceptance closure.
8. CORE-08: default cutover only after B0-B4 and explicit Owner approval.

No item in this queue is executed by the readiness-closure Goal.

## B0-B4 cutover rules

| Gate | Current state | Rule |
| --- | --- | --- |
| B0 | `PASS` | Immutable v1 freeze, rollback commands, and existing v1 evidence inventory are inspectable. |
| B1 | `PASS` (contract-only) | BR1 parity stages/invariants and offline contract are frozen; this is not fresh-real parity. |
| B2 | `PENDING` | Requires fresh Uni-Mol training, real REINVENT4 generation, current-run prediction/evaluation, and exact evidence. |
| B3 | `PENDING` | Requires the remote-restart canary with durable, idempotent, credential-safe evidence. |
| B4 | `PENDING` | Requires explicit future Owner approval for default cutover. |

Therefore `core_cutover_ready=false`, legacy v1 remains intact, and the
immutable freeze reference remains the rollback line.

## Stop conditions

Stop and record a blocker rather than guessing if any of the following occurs:

- `main` differs from the frozen candidate commit;
- the freeze tag or `legacy/molly-v1` exists at a different commit;
- required evidence is private, unavailable, or cannot be inspected;
- a literature license/access route cannot be verified;
- lock regeneration creates unexplained source or dependency drift;
- the exact file inventory contains an unresolved required file;
- a security boundary remains ambiguous;
- the contract spike requires redesigning the approved architecture;
- a conclusion would require claiming a fresh-real BR1, GPU, remote, network-live,
  or experimental run that did not execute; or
- the requested work would require CORE-01+ production implementation in this
  closure Goal, CORE-08 cutover, history rewrite, force-push, freeze-tag move,
  legacy deletion, or error-propagation implementation.

## Instruction hierarchy and Git discipline

The user-provided Goal and its explicit Owner authorization define this run's
scope. Checked-in repository policy, especially [`SECURITY.md`](../../SECURITY.md),
`docs/development-guidance.md`, and [`docs/roadmap.md`](../../docs/roadmap.md),
defines public repository and project conventions. Applicable non-repository
execution instructions were considered during the run as execution context
only; they are not repository authority and are not reproduced here. Tool and
sandbox safety constraints remain applicable.

Keep PR #64 Draft, continue the reviewed branch, use independent milestone
commits, run targeted checks during implementation, and do not merge. Before
any later production push, run the repository's applicable compile, diff,
targeted, and PR Fast checks. Prefer GitHub Full CI as the authoritative full
suite when the final review HEAD is ready.

## Final readiness decision

The closure report is
[`CORE-00-READINESS-CLOSURE.md`](reports/CORE-00-READINESS-CLOSURE.md).
It records per-gate evidence, commit binding, remaining risk, B0-B4 state, and
the verification results. The intended final decision is:

```yaml
owner_decision: APPROVED_FOR_CORE_IMPLEMENTATION_NOT_CUTOVER
core_goal_mode_ready: true
core_cutover_ready: false
```
