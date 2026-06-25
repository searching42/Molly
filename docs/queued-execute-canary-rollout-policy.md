# Queued Execute Canary Rollout Policy

This document defines the rollout policy and decision matrix for the
feature-flagged `/api/run-plan/execute` queued canary. It is a policy and test
guard document only. It does not expand the allowlist, change execution
semantics, enable remote workers, migrate storage, or make queued execution the
default path.

## Current State

`/api/run-plan/execute` remains synchronous by default.
`AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` is a master switch for the queued
canary path, not a default migration switch.

When the flag is on, the route still requires the low-risk task-chain allowlist
to pass. The current allowlist is:

- `inspect_dataset`
- `clean_dataset`
- `check_trainability`
- `run_baseline`
- `render_report`

Only an all-allowlisted task chain may return:

- `execution_backend="queued_canary"`
- `queue_summary`

If the chain contains a non-allowlisted task, the route falls back to the
synchronous executor path. The fallback response remains sync-compatible and
does not include `execution_backend` or `queue_summary`. The route records:

- `sync_fallback_not_allowlisted`
- the disallowed task ids

Current boundaries:

- `/api/run-plan/resume` is unchanged.
- remote worker execution is not enabled.
- SQLite storage is not enabled.
- queued execution is not the default route behavior.
- `train_model` remains excluded.
- generation remains excluded.
- literature/mining remains excluded.

## Decision Matrix

| Condition | Expected backend | Response shape | Action |
| --- | --- | --- | --- |
| flag off | sync | sync-compatible | normal sync execution |
| flag on + allowlisted chain | queued_canary | includes `queue_summary` | canary execution |
| flag on + non-allowlisted chain | sync fallback | sync-compatible | log fallback and execute sync |
| queued canary fails with executor failed dict | queued_canary | includes `queue_summary` + failed execution | return `ok=false`; keep flag reversible |
| queued canary not terminal | queued_canary | includes `queue_summary` | return `ok=false`; do not treat as success |
| queue has old unrelated job | queued_canary for new job only | includes new `queued_job_id` | old job remains queued |
| sync fallback fails | sync fallback | sync-compatible error | no queue files created |
| rollback needed | sync | sync-compatible | disable flag |

## Green Criteria Before Expanding Allowlist

Response parity:

- Sync fallback responses do not contain queued-only fields.
- Queued canary responses explicitly contain backend and summary fields.
- Existing clients that consume sync responses are not broken.

WAITING_USER parity:

- Sync and queued canary agree on status.
- Sync and queued canary agree on the waiting task.
- Sync and queued canary agree on required gates.
- Artifact references required for resume are equivalent.

Artifact registry parity:

- An allowlisted queued canary run registers the same logical artifacts as the
  synchronous path for the same task chain.
- Missing or differently named artifacts block allowlist expansion.
- PR #125 adds first artifact registry parity fixture for an existing
  allowlisted chain. The fixture compares logical artifact ids and file existence
  across sync and queued canary runs.
- Exact paths/hashes are not required unless artifacts are deterministic and
  run-id-independent.
- This does not expand the allowlist.

Failure classification parity:

- An executor failed dict is not treated as success.
- Route logs include backend and failed task/status.
- The response keeps useful status and error fields for review.
- PR #126 adds failure classification parity fixture for an existing
  allowlisted queued execute chain. The fixture compares sync versus queued
  canary failed status, failed task, and useful error message fields.
- Exact error strings are not required to match because queue wrapping may add
  context.
- Queued executor failed dict must not be treated as success.
- This does not expand the allowlist.

Queue safety:

- No old job is consumed.
- No orphan queued jobs are left after invalid target selectors.
- No queue files are created for sync fallback.

Rollback evidence:

- Disabling `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` immediately returns to
  the sync path.
- Sync rollback does not touch existing queued jobs.
- Rollback does not require any database or storage migration.

No hidden scope expansion:

- No remote worker.
- No SQLite migration.
- No default-route migration.
- No heavy adapters.

## Red Conditions / Must Disable Canary

Disable `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` if any of these occur:

- Queued canary response diverges from sync response for an allowlisted chain.
- Queued canary consumes the wrong job.
- Old queued jobs are modified unexpectedly.
- A queue job remains stuck running without recovery evidence.
- WAITING_USER gate metadata differs from sync.
- Artifact registry misses expected artifacts.
- An executor failed dict is returned as success.
- Rollback test fails.
- Queue files appear for sync fallback.
- Any non-allowlisted task reaches the queued backend.

## Allowlist Expansion Rules

Allowlist expansion must happen one task chain at a time.

Each new task chain must have:

- sync fixture
- queued canary fixture
- response parity test
- artifact registry parity test
- failure behavior test
- rollback test

train_model remains excluded until model artifact, WAITING_USER, and gate
decision semantics are separately proven.

generation remains excluded until external/heavy generation adapter contracts
are hardened.

literature/mining remains excluded until acquisition, parsing, provenance,
permission, and audit contracts are hardened.

No task enters the allowlist by name coincidence. It must be explicitly listed
and tested.

## Exit Criteria Before Default Migration

Default migration is not allowed until:

- all allowlisted chains have green parity tests.
- rollback policy is documented and tested.
- queue recovery semantics are proven.
- observability is sufficient for backend, job id, run id, status, and failure
  reason.
- storage migration decision is made.
- remote worker decision is made.
- default-route canary is stable across repeated runs.
- `/api/run-plan/resume` remains unaffected or has its own migration policy.

## Current Recommendation

Keep the queued canary feature-flagged. Keep the allowlist conservative.

The next engineering PR should add parity fixtures for one additional
low-risk allowlisted chain or add artifact registry parity tests for the
existing allowlisted chains.

Do not move `train_model`, generation, literature, or mining tasks into the
queued canary yet.
