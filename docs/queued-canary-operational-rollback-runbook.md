# Queued Canary Operational Rollback Runbook

This runbook defines the one-step operational rollback drill for the
feature-flagged queued execute canary on `/api/run-plan/execute`.

The rollback changes routing for new requests only. It does not consume,
cancel, retry, delete, or otherwise mutate existing queue jobs or leases.

Queued execution remains feature-flagged. Synchronous execution remains the
default. Default migration remains blocked.

This runbook names the rollback owner, rollback trigger conditions, the exact
flag-off action, the sync-compatible response evidence to collect, and the
conditions for re-enable versus continued sync-only operation.

## Rollback Owner

- Primary rollback owner: the on-call operator or release owner responsible for
  queued-canary rollout decisions.
- Secondary reviewer: the engineering owner responsible for queued-canary
  rollout policy and evidence review.

## Rollback Trigger Conditions

Trigger rollback if any of the following occur on an allowlisted queued-canary
chain:

- response compatibility diverges from the synchronous path
- `execution_backend="queued_canary"` appears on a request that should have
  fallen back to sync
- `queue_summary` is missing or inconsistent on a queued-canary response
- a failed queued-canary execution is misreported as success
- an old queued job, retry child, or stale queue record is consumed by a new
  request
- telemetry or log evidence becomes inconsistent with the queued-canary
  rollout policy
- rollback evidence itself no longer passes

## One-Step Rollback Action

1. Set AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY=false.
2. Or remove/disable the equivalent app-config override that enables queued
   canary routing.
3. Submit a fresh independent `POST /api/run-plan/execute` request for an
   allowlisted chain.
4. Verify the response is sync-compatible.
5. Verify `execution_backend` is absent.
6. Verify `queue_summary` is absent.
7. Verify the new sync run does not create a per-run queue directory under
   `.ai4s_internal/run_plan_queues/<project_id>/<run_id>`.
8. Inspect pre-existing queue jobs and leases without mutating them.

Rollback must be one step: disable the canary flag and re-issue a new request.

## Expected Response Behavior After Rollback

After rollback:

- new `/api/run-plan/execute` requests use the synchronous path
- the response remains sync-compatible
- `execution_backend` is absent
- `queue_summary` is absent
- existing synchronous response fields remain available for the caller

## Expected Queue Behavior After Rollback

After rollback:

- existing jobs and leases remain unchanged
- existing failed source jobs remain inspectable
- existing queued retry children remain queued and inspectable
- existing terminal or stale lease records remain available for inspection
- rollback does not create a new run-plan queue for the fresh sync request
- rollback does not mutate queue state from earlier queued-canary runs

Rollback affects new route selection only. It does not mutate existing queue
state.

## Evidence To Collect

Collect the following evidence for the rollback drill:

- the pre-rollback queued-canary response
- the post-rollback sync-compatible response
- proof that `execution_backend` is absent after rollback
- proof that `queue_summary` is absent after rollback
- normalized before/after snapshots of existing queue jobs and leases
- proof that the fresh sync request did not create a new queue directory
- log evidence showing the sync path after rollback
- if present, failed queued-canary response and telemetry evidence that remain
  available for later inspection

## Re-Enable Conditions

The canary may be re-enabled only when:

- the immediate incident or mismatch is understood
- rollback evidence shows that sync routing is stable
- existing queue jobs and leases remain unchanged
- the root cause is fixed or isolated well enough for another bounded canary
  attempt
- the responsible operator explicitly chooses to re-enable the flag for fresh
  independent runs

Re-enable means setting `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY=true` or
restoring the equivalent app-config override for a new request. Re-enable does
not clean up old queue state.

## Continued Sync-Only Conditions

Continue sync-only routing when:

- queued-canary response parity is not restored
- queue immutability cannot be demonstrated
- failed queued-canary runs are not reliably inspectable
- rollback evidence is incomplete
- production-sized scientific workloads are still not proven
- remote workers or SQLite decisions are still unresolved in ways that block a
  broader rollout

## Explicit Non-Steps

Do not delete queue files as part of rollback.

Do not modify `worker_queue.json` manually.

Do not cancel all queued jobs.

Do not retry failed jobs automatically.

Do not change the allowlist as part of rollback.

Do not change `/api/run-plan/resume`.

Do not perform a SQLite migration during rollback.

Do not enable remote workers as part of rollback.

## Scope Boundary

This runbook proves only that disabling
`AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` returns new requests to the
synchronous path without mutating existing queue state.

It does not prove production readiness.
It does not complete default migration.
It does not introduce a rollback API route.
It does not introduce a retry API route.
