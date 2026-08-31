# Molly Core v2 default cutover approval

Decision: `APPROVED`

B4: `PASS`

Default cutover: `AUTHORIZED`

This repository decision records the explicit Owner approval supplied for
CORE-08. It authorizes the transition of the default package and runtime on
`main` to Molly Core v2, removal of obsolete v1 runtime code from the current
tree, and permanent retention of the immutable v1 rollback branch and tag.

## Approval applies to

- accepted CORE-06 BR1 parity evidence;
- accepted CORE-07 runtime, CLI, and observer-only observability surface;
- transition of the main/default package and runtime to Molly Core v2;
- archival/removal of obsolete v1 runtime from `main`;
- permanent retention of `legacy/molly-v1` and
  `molly-v1-pre-core-v2-20260829` at the frozen v1 commit.

## Explicitly not required before cutover

- real paper → reviewed dataset → BR1 integrated acceptance;
- a mandatory MinerU production route;
- live LangSmith or OpenTelemetry acceptance;
- UI migration;
- a general HTTP API.

This approval does not authorize experimental scientific claims, unrelated new
scientific features, error-propagation research implementation, or movement
of the immutable v1 references. Real literature/data-mining integration and
alternative structured-source pipelines remain post-cutover work.

## Binding evidence

The approval is conditioned on the existing C0–C7 readiness evidence and
the following immutable rollback references:

```yaml
v1_freeze_commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9
v1_freeze_tag: molly-v1-pre-core-v2-20260829
legacy_branch: legacy/molly-v1
readiness_manifest: docs/v2/readiness/core_refactor_readiness.json
```
