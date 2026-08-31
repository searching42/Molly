# Molly documentation map

`docs/v2/` is the current documentation authority for Molly Core v2. The
repository roadmap records active scope and status; contracts and reports
provide the inspectable evidence for each milestone.

## Current Core v2 documents

- [Core v2 roadmap](roadmap.md)
- [Readiness manifest](v2/readiness/core_refactor_readiness.json)
- [Execution contract](v2/CODEX_GOAL_EXECUTION_CONTRACT.md)
- [Core v2 simplification specification](v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md)
- [CORE-08 cutover report](v2/reports/CORE-08.md)
- [B4 cutover approval](v2/decisions/CORE_V2_CUTOVER_APPROVAL.md)
- [Package/dependency boundary](v2/contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md)
- [Acquisition security boundary](v2/contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md)

The `v2` source and tests define the current contracts for the AgentLoop,
immutable artifacts, append-only run records, acquisition, documents,
scientific evidence, optional BR1 plugins, runtime inspection, and observer-
only telemetry.

## Current operator guidance

Use the root [README](../README.md) for installation and safe offline
inspection. Host-specific runtime, provider, credential, and compute profiles
must remain outside the repository. The [security policy](../SECURITY.md)
defines the public repository boundary.

## Historical material

Files outside `docs/v2/` that describe the earlier Harness, Controller,
Autonomy, queued workflows, or pre-Core-v2 acceptance are retained as
historical evidence and are not current architecture authority. This includes
the older literature, remote-worker, session, and milestone guides and the
historical material under `docs/evidence/`.

Historical evidence must not be rewritten to look like Core v2 evidence. Use
the v2 reports and manifests when making current readiness or cutover claims.

## Documentation rules

- Prefer repository-relative links and keep referenced paths valid.
- Use logical resource IDs and synthetic/public-safe examples.
- Never publish credentials, private source material, user-home paths,
  concrete host identities, or raw runtime bundles.
- Mark snapshots and old implementation contracts as historical/non-normative.
- Update `roadmap.md` when current scope or execution order changes.
