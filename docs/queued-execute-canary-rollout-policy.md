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
- PR #130 adds a second allowlisted chain parity fixture.
- The fixture covers `render_report` or the actual second all-allowlisted
  chain selected by the current planner. Today that second chain is
  `check_trainability`, because the current `render_report` planner expansion
  reaches non-allowlisted tasks and therefore remains sync-only.
- The second allowlisted chain fixture compares sync vs queued canary artifact
  registry and failure classification for the same real chain.
- This still does not expand the allowlist.
- This still does not justify default migration by itself.

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
- PR #127 adds repeated-run stability coverage for existing allowlisted queued
  execute chains. Repeated queued canary runs must isolate queue state by project_id/run_id,
  preserve stable response shape, and preserve stable logical artifact ids.
- PR #128 adds queue recovery and stale lease coverage for queued execute
  canary.
- Stale running jobs must not be mistaken for the target job.
- Target-job selection must remain valid after stale lease recovery.
- PR #131 adds cancellation coverage for queued execute canary.
- Cancelled queued jobs must not be mistaken for the target job.
- Sync fallback must not process or mutate cancelled queued jobs.
- If explicit retry/requeue production semantics are not implemented yet,
  default migration remains blocked until retry behavior is defined and tested.
- This does not expand the allowlist.

Rollback evidence:

- Disabling `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` immediately returns to
  the sync path.
- Sync rollback does not touch existing queued jobs.
- Rollback to sync must not touch existing queued jobs from other runs.
- Sync fallback must not process or mutate queued jobs.
- Rollback does not require any database or storage migration.

No hidden scope expansion:

- No remote worker.
- No SQLite migration.
- This does not enable remote workers or SQLite.
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

## Production-Sized Fixture Boundary

Current parity fixtures use small deterministic datasets. They are enough for
route, queue, and control-plane confidence, but they are not enough for
production-sized scientific workload confidence. In short: small deterministic
datasets are not enough for production-sized scientific workload confidence.

PR #132 only documents and guards the production-sized boundary. It does not
claim production readiness, and it does not make queued execution default. A
production-sized fixture must be separately defined before default migration.

Minimum production-sized fixture expectations:

- dataset size target:
  - not necessarily full production data in CI
  - but representative of larger row counts, wider property schema, and
    realistic missingness
- runtime budget:
  - must stay within CI constraints
  - larger fixture profiles may need a nightly or offline lane rather than the
    default presubmit suite
- artifact expectations:
  - same logical artifact ids as sync path
  - no missing registry artifacts
  - artifact file existence
  - failure classification remains comparable between sync and queued canary
- queue expectations:
  - target-job safety
  - no orphan jobs
  - stale or cancelled old jobs are not consumed
- rollback expectations:
  - flag off returns sync-compatible response
  - sync path does not create queue files

Current decision:

- small deterministic datasets remain the current confidence boundary
- a production-sized or nightly boundary fixture policy still needs to exist
- Default migration remains blocked

## Optional Nightly Production-Sized Fixture Lane Design

### Purpose

The nightly lane is intended to validate queued canary behavior on larger,
more realistic scientific datasets.

- It is not part of the default presubmit suite.
- It is not enabled by this PR.
- It must remain optional until runtime, data, and storage constraints are
  understood.

### Dataset profile

The future nightly fixture should represent at least:

- larger row count than small deterministic CI fixtures
- wider property schema
- realistic missingness
- mixed valid and invalid SMILES rows
- `split_group` coverage
- representative OLED properties if available
- no private or proprietary data committed to the repo
- documented fixture provenance

### Execution scope

The future nightly lane should stay constrained to current allowlisted task
chains only:

- `inspect_dataset`
- `clean_dataset`
- `check_trainability`
- `run_baseline`

Additional scope constraints:

- `train_model` remains excluded.
- generation remains excluded.
- literature/mining remains excluded.
- `render_report` remains sync-only if planner expansion reaches
  non-allowlisted tasks.
- No allowlist expansion is part of the nightly lane design.

### Required parity checks

Any future nightly lane must check:

- sync vs queued canary response compatibility
- logical artifact ids match
- artifact files exist
- failure classification remains comparable
- queued job final status is stable
- old queued jobs are not consumed
- stale/cancelled jobs are not consumed
- rollback to sync remains safe

### Runtime and CI budget

Nightly lane constraints remain explicit:

- default CI must remain lightweight and deterministic
- nightly lane must have explicit runtime budget
- large fixture may need opt-in label, manual dispatch, or scheduled nightly
  workflow
- failing nightly lane should block default migration, but should not
  necessarily block ordinary PRs until policy is finalized
- no nightly workflow is enabled in this PR

### Storage and observability requirements

The future nightly lane will need:

- artifact size expectations
- retention policy
- log volume expectations
- queue file size expectations
- telemetry markers from the observability checklist
- no centralized telemetry sink yet
- no production dashboard yet
- no alerting yet

### Current decision

Current decision: nightly production-sized fixture lane is designed but not
enabled.

- Queued canary remains feature-flagged.
- Allowlist remains conservative.
- Default migration remains blocked.

## Telemetry and Observability Checklist

This checklist documents the minimum observability surface expected before the
queued canary can move toward a broader rollout. It does not mean production
telemetry is implemented today.

### Required backend markers

Queued canary observability must distinguish:

- sync path
- queued_canary path
- sync_fallback_not_allowlisted
- failed queued canary
- rollback to sync

Current existing coverage already references:

- `RunPlan execution backend: sync`
- `RunPlan execution backend: queued_canary`
- `RunPlan execution backend: sync_fallback_not_allowlisted`
- `execution_backend="queued_canary"`
- `queue_summary`

### Required identity fields

Queued canary observability must be able to locate:

- `project_id`
- `run_id`
- `job_id`
- `lease_id`
- `worker_id`
- `execution_backend`
- `queued_job_id`
- `final_job.status`
- `final_lease.status`

### Required execution state fields

Queued canary observability must surface:

- execution status
- executed tasks
- failed task
- waiting task
- required gates
- error message or error type
- cancellation status
- stale lease or recovery events
- queue final state

### Required safety evidence

Queued canary observability must prove:

- old job not consumed
- cancelled job not consumed
- stale job not mistaken for target job
- sync fallback does not create queue files
- sync fallback does not mutate queue jobs
- non-allowlisted tasks do not reach queued backend
- rollback to sync does not touch existing queued jobs

### Still missing / not production-grade

Current gaps remain explicit:

- no centralized telemetry sink
- no dashboard
- no alerting
- no SLO or SLA policy
- no production correlation id policy
- no remote worker metrics
- no SQLite or storage migration metrics
- no production-sized or nightly telemetry fixture
- no default migration readiness from telemetry alone

### Current decision

Current decision: telemetry checklist is documented but not fully implemented
as production observability.

- Keep queued canary feature-flagged.
- Keep allowlist conservative.
- Default migration remains blocked until telemetry/observability is
  implemented and tested.

## Default Migration Readiness Checklist

### 1. Current Green Coverage

Current green coverage already exists for:

- response compatibility and rollback evidence
- low-risk allowlist enforcement
- artifact registry parity
- second allowlisted chain parity
- failure classification parity
- repeated-run stability
- cancellation coverage
- stale lease and queue recovery coverage
- target-job safety
- sync fallback compatibility
- retry remains documented as future production work
- observability checklist documented

### 2. Still Blocking Default Migration

The following items still block default migration:

- the allowlist still covers only a small set of low-risk chains
- production-sized datasets are not yet proven
- long-running or heavy adapters are not yet proven
- explicit retry/requeue production semantics are not yet defined
- `train_model` remains excluded
- generation remains excluded
- literature/mining remains excluded
- the remote worker contract is not defined
- the SQLite/storage migration decision is not complete
- queue observability is not yet production-grade
- no centralized telemetry sink, dashboard, or alerting exists yet
- operational rollback and alerting policy is not yet complete
- `/api/run-plan/resume` does not have a default-migration strategy

### 3. Required Before Default Migration

The following are required before default migration:

- at least two allowlisted task chains with artifact parity coverage
- failure parity coverage for success, failure, and partial failure cases
- repeated-run stability coverage across broader run counts
- queue recovery coverage for stale running, cancellation, and retry paths
- production-sized fixture policy defined, potentially via nightly or offline
  coverage that still respects CI constraints
- explicit retry/requeue production semantics defined and tested, if queued
  execution is to move beyond current canary scope
- storage backend decision completed
- remote worker decision completed
- sync fallback and queued canary telemetry both defined and reviewable
- telemetry and observability implemented beyond documentation-only checklist
- default migration must have a one-step rollback path
- all non-allowlisted tasks must continue to force sync fallback
- owner, rollout, and rollback steps must be explicit in the docs

### 4. Explicit Current Decision

Current decision: do not make queued execution default.

- Keep `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` feature-flagged.
- Keep the low-risk allowlist conservative.
- second allowlisted chain parity fixture
- queue cancellation/retry fixture
- production-sized fixture
- observability checklist

## Current Recommendation

Keep the queued canary feature-flagged. Keep the allowlist conservative.

PR #130 adds a second allowlisted chain parity fixture. PR #131 adds
cancellation coverage plus explicit documentation that retry/requeue
production semantics remain future work. PR #132 documents and guards the
production-sized fixture boundary without claiming production-sized proof. The
PR #133 documents and guards the telemetry/observability checklist without
claiming production-grade telemetry. The next engineering PR should implement
minimal structured telemetry fields for queued canary, or design an optional
nightly production-sized fixture lane. The default-migration readiness
checklist is now documented, but this is still not enough to justify default
migration.

Do not move `train_model`, generation, literature, or mining tasks into the
queued canary yet.
