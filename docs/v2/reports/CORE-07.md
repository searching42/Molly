# CORE-07 — Runtime Surface and Observer-Only Observability

Status: implemented and verified on the final executable/test commit. CORE-08 was not started.

## Integration and scope

- Repository: `searching42/Molly`
- Branch: `codex/molly-core-v2-core-07-runtime-surface`
- Draft PR: [#71](https://github.com/searching42/Molly/pull/71)
- Base commit: `04d0e614713797abe047510df8488ad173af9572`
- Final executable/test commit: `b531a7a1bffc940452016d3c27022942e0302b5b`
- Final report commit: documentation-only commit after the final executable/test validation
- PR #70 is present in the base branch.

The readiness preflight remained valid: C0–C7 are `PASS`, `core_goal_mode_ready` is `true`, B0–B3 are `PASS`, B4 is `PENDING_OWNER_APPROVAL`, and `core_cutover_ready` is `false`. CORE-07 did not modify the readiness manifest or promote any cutover gate.

## Read-only inspection

`src/molly/core/inspection.py` adds a deterministic `RunInspector` projection for runs and artifacts. It validates the append-only ledger chain, referenced immutable artifact identities and digests, bounded lineage relations, request/profile bindings, materialized tool-call bindings, approvals, result-data digests, and lifecycle integrity before returning projections. Inspection is read-only and does not repair, append, publish, or execute anything. The projection uses a stable canonical JSON representation and digest and applies an explicit safe metadata allowlist.

The inspector supports artifact projections including producer, consumer, derived-from, and support relations. Shared content identities are not treated as proof of a current-run occurrence; run-scoped lineage metadata and exact ledger bindings remain authoritative.

## Runtime profile and service

`src/molly/runtime/` adds:

- closed, server-owned `RuntimeProfile` and `RuntimeProfileRegistry` objects;
- deterministic profile/config digests without model-selected imports, paths, hosts, or credentials;
- `RuntimeService` operations for start/resume, run/artifact inspection, exact approval recording, digest-bound review recording, and observation;
- exact persisted request and materialized-call resume behavior.

No second execution authority was introduced. `RunLedger`, `ArtifactStore`, and `ArtifactLineage` remain the factual execution, immutable-content, and provenance authorities. There is no default production profile: an unavailable or unregistered profile fails closed.

## Minimal CLI

`src/molly/cli.py` provides an `argparse` surface for:

```text
molly run start|resume
molly inspect run|artifact
molly approve
molly review
molly observe
```

The CLI emits deterministic JSON or bounded human-readable output and sanitizes user-facing errors. The console entry point is `molly = "molly.cli:main"`. A general HTTP/API surface remains deferred (`API=DEFERRED_THIN_WRAPPER`); no conversation or UI migration was included.

## Observer-only observability

`src/molly/observability/` adds deterministic `RunTrace`, `TraceSpan`, and `TraceEvent` models plus a ledger/lineage projector. The projection is allowlisted and excludes hidden chain-of-thought, prompts, credentials, raw unbounded result data, and private paths. The JSON exporter is dependency-free. OpenTelemetry and LangSmith exporters are lazy optional adapters; unavailable or failed exporters do not mutate Core state. Before/after authoritative snapshots detect observer-side mutation or corruption.

No live OTel collector or LangSmith account was contacted. Offline fake-client tests cover the optional adapters and exporter failure isolation. No CORE-06 training, generation, remote-compute, or scientific side effect was rerun.

## CORE-05/CORE-06 inspection coverage

Focused inspection tests exercise generic projections of CORE-05 fixture/evidence manifests and CORE-06 acceptance evidence without importing domain execution authority or rerunning scientific workflows. These tests verify that inspection remains a read-only projection rather than a new scientific or scheduling layer.

## Verification evidence

All entries below refer to the final executable/test commit `b531a7a1bffc940452016d3c27022942e0302b5b` unless explicitly marked report-only:

| Check | Result | Evidence |
| --- | --- | --- |
| CORE-07 focused tests | PASS, 15 passed | `tests/molly/test_core07_inspection.py`, `test_core07_cli.py`, `test_core07_observability.py` |
| CORE-01/02/05/06/privacy regressions | PASS, 113 passed | targeted regression selection |
| Compile policy | PASS | `python -m compileall -q src tests prototypes` |
| Whitespace policy | PASS | `git diff --check` |
| Lock consistency | PASS | `uv lock --check` |
| PR Fast | PASS | workflow `33383922141`; compile job `99462152196`; pytest job `99462203767` |
| CodeQL | PASS | workflow `33383920164`; Actions `99462149265`; JS/TS `99462149533`; Python `99462149379`; aggregate check `99462282450` |
| Full CI | PASS | run `33384575242`; compile/shard policy `99464187289`; weighted shards 0/1/2/3: `99464226562`, `99464226642`, `99464226608`, `99464226652` |

The final Full CI run passed all four weighted shards and the compile/shard policy on the exact executable/test commit. The later report-only commit did not change executable code, tests, or evidence identity and therefore was not used as a replacement Full CI test HEAD.

## Frozen v1 references

Both immutable rollback references remain unchanged and resolve to `ae7892dbf8a6bfe85dd909056eadc2afecc40d9`:

```text
tag:    molly-v1-pre-core-v2-20260829
branch: legacy/molly-v1
```

Legacy v1 code was not modified by CORE-07.

## Readiness and remaining limits

```json
{
  "C0": "PASS",
  "C1": "PASS",
  "C2": "PASS",
  "C3": "PASS",
  "C4": "PASS",
  "C5": "PASS",
  "C6": "PASS",
  "C7": "PASS",
  "B0": "PASS",
  "B1": "PASS",
  "B2": "PASS",
  "B3": "PASS",
  "B4": "PENDING_OWNER_APPROVAL",
  "core_goal_mode_ready": true,
  "core_cutover_ready": false,
  "owner_decision": "APPROVED_FOR_CORE_IMPLEMENTATION_NOT_CUTOVER"
}
```

The quoted JSON keys above are intended as the readiness summary; the repository readiness manifest remains the machine-readable source of truth. B4 still requires an explicit future Owner cutover decision. CORE-08, default cutover, conversation/UI migration, and production scientific workflow changes remain out of scope.

## CORE-08 recommendation

Do not start CORE-08 from this task automatically. First obtain the explicit B4 Owner cutover approval, then review the runtime/inspection/observability evidence and define CORE-08 as a separate Goal with its own preflight and rollback checks.
