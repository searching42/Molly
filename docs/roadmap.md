# Molly public roadmap

> Status: Core v2 is the default mainline runtime.
>
> The current status in this file applies to the repository tree on `main`;
> older milestone documents are historical evidence.

## Current status

Molly Core v2 milestones `CORE-00` through `CORE-08` are complete for their
defined contracts. The readiness manifest records:

```text
C0-C7 = PASS
B0-B4 = PASS
core_goal_mode_ready = true
core_cutover_ready = true
```

The default package and console entrypoint are Molly Core v2. The v1 runtime
is frozen for rollback/history at the immutable references recorded in the
[cutover report](v2/reports/CORE-08.md).

## Core v2 surface

The active execution spine is:

```text
RunRequest → AgentLoop → ToolRegistry/ToolPolicy
    → bounded scientific tools/plugins
    → ArtifactStore/RunLedger/ArtifactLineage
    → ValidationResult/ReviewRecord
```

Current contracts cover immutable scientific content, append-only execution
records, bounded provenance, compliant literature acquisition, deterministic
document parsing, reviewed OLED evidence, optional BR1 inverse design,
runtime/CLI inspection, observer-only observability, and a loopback-only
operator UI. BR1 request parameters are extracted through a selected,
server-configured structured LLM; there is no rule-based parsing fallback or
per-run budget control.

MinerU is an optional PDF fallback. BR1 and remote compute are optional plugin
surfaces. Telemetry is observer-only and cannot advance scientific state.

## Post-cutover work

The following work is intentionally post-cutover and must receive its own
scope and evidence:

- validate a real literature/data-mining pipeline and its review boundary;
- evaluate structured acquisition paths that may replace or bypass MinerU;
- extend scientific domains and acceptance datasets;
- extend the loopback-only UI or add a general HTTP API only if a separate
  authority review approves it;
- pursue research trajectory or error-propagation work only under a separately
  approved research proposal.

None of these items is claimed complete by the Core v2 cutover.

## Evidence semantics

Implementation, focused tests, representative runtime validation, and reviewed
scientific acceptance are distinct claims. A fixture or schema does not prove a
fresh-real run. Computational BR1 results remain computational-only claims.

## Historical context

Pre-Core-v2 Harness/Controller/Autonomy documents and acceptance records remain
under `docs/` for audit context. They are not active roadmap items and must not
be used as the default runtime instructions for current main.
