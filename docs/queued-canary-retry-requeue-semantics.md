# Queued Canary Retry And Requeue Semantics

This document defines conservative retry/requeue semantics for the local
JSON-backed queued execute canary. PR #138 now implements the smallest allowed
subset of that contract: atomic one-shot retry-child creation for eligible
failed local queue jobs plus an allowlisted queued-canary helper.

This PR does not implement a public API route.

It does not add `WorkerQueue.retry`, `WorkerQueue.requeue`, automatic retry,
timers, or any change to `/api/run-plan/execute` or `/api/run-plan/resume`.

The contract here is specific to the local queued execute canary. It does not
change existing non-queue failed-stage retry routes elsewhere in the
application.

## Scope And Non-Goals

This document exists to keep future retry behavior narrow and reviewable.

It does not:

- implement a public retry or requeue API
- add automatic retry
- add a public API route
- change the current `attempts` field meaning
- change stale lease recovery behavior
- expand the queued-canary allowlist
- move `train_model`, generation, literature, mining, or unknown task chains
  into queued retry scope
- make queued execution default

Default migration remains blocked after this PR.

## Required Terminology

### Lease attempt

A lease attempt is one acquisition of a queued job by a worker.

- The current `WorkerQueue.attempts` field counts acquisitions.
- Reacquisition after stale lease recovery increments this field.
- A lease attempt is not an explicit retry count.
- The current `attempts` field must keep this meaning.

### Stale lease recovery

Stale lease recovery is existing queue behavior.

- An expired lease is marked `stale`.
- The same `job_id` is returned to queued state.
- Stale recovery keeps the same `job_id`.
- Stale recovery does not create a new retry job.
- Stale recovery does not represent an operator-requested retry.
- Target-job isolation must continue to hold after stale recovery.

### Explicit retry

Explicit retry is a future, deliberate request to rerun a terminal failed job.

- Explicit retry creates a new `job_id`.
- The original failed job remains immutable.
- Explicit retry must not mutate the original terminal job back to queued.
- Explicit retry is never automatic in the initial queued-canary design.

### Explicit requeue

In the future queued-canary contract, explicit requeue means placing a newly
accepted retry child job into queued state. It must not mean flipping the
original terminal job back to queued.

### Rerun / new execution

A changed `RunPlan`, changed input artifacts, changed task options, changed
snapshot, or changed scientific intent is a new execution, not a retry.

- A changed snapshot or payload requires a new execution rather than retry.
- A changed scientific intent requires a newly reviewed execution request.
- `WAITING_USER` uses the existing gate/resume path, not retry.

## Conservative V1 Retry Policy

### Eligibility

Only a terminal failed job may be considered for explicit retry.

The following are not eligible:

- a succeeded job is not retryable
- a cancelled job is not retryable
- a queued job is not retryable
- a running job is not retryable
- a job with `cancellation_requested` is not retryable
- `WAITING_USER` is not a retry condition
- validation failures are not retryable without a corrected new request
- permission failures are not retryable without a corrected new request
- malformed-payload failures are not retryable without a corrected new request
- policy-denial failures are not retryable without a corrected new request

Retry remains restricted to the current queued-canary allowlist.

No literature/mining, generation, `train_model`, or unknown task chain becomes retryable through this policy.

### Identity And Lineage

A future explicit retry job should carry stable lineage fields such as:

- `retry_of_job_id`
- `retry_root_job_id`
- `retry_index`
- `retry_request_id`
- `retry_reason`
- `retry_requested_by`
- `original_project_id`
- `original_run_id`

PR #138 fixes the first implementation shape:

- a new `job_id` is required for every explicit retry
- the original failed job remains immutable
- `project_id` and `run_id` cannot change within a retry
- the serialized `RunPlan` task envelope and input snapshot must be unchanged
- a changed snapshot or payload requires a new execution rather than retry

### Idempotency And Limits

A retry request must have a stable `retry_request_id`.

- retry_request_id provides idempotency
- repeating the same `retry_request_id` must not create duplicate retry jobs
- at most one active queued/running retry may exist for the same source job
- the initial canary implementation should permit at most one explicit retry
- broader retry limits, backoff, and automation remain future policy work

### Execution And Artifact Rules

- no automatic retry loop
- no exponential backoff in the initial implementation
- no retry from inside `WorkerQueuePoller`
- stale recovery must not consume the explicit retry allowance
- retry must not overwrite or erase the original job or lease history
- partial scientific outputs from a failed attempt must not be treated as
  successful final artifacts
- artifact-output isolation must be proven before explicit retry is enabled for
  production scientific adapters

PR #138 implements only queue-control-level retry behavior plus the
`enqueue_queued_canary_retry(...)` helper for the current allowlisted
queued-canary `run_plan_execute` envelope. It does not expose a route, add
automatic retry, or widen queued-canary scope.

### Cancellation

- cancelling a queued job remains terminal cancellation
- cancelling a running job continues to set `cancellation_requested`
- a cancelled job cannot be converted into a retry job
- a user who wants to run cancelled work again must submit a newly reviewed
  execution request

### Audit And Observability Contract

Future audit/telemetry events should include:

- `retry_requested`
- `retry_accepted`
- `retry_rejected`
- `retry_job_queued`
- `retry_job_succeeded`
- `retry_job_failed`
- `retry_job_cancelled`

Required future correlation fields:

- `project_id`
- `run_id`
- `source_job_id`
- `retry_job_id`
- `retry_root_job_id`
- `retry_index`
- `retry_request_id`
- `actor`
- `reason`
- `final status`

These events are not implemented in PR #138.

### Rollback And Default-Route Boundaries

- disabling `AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY` must continue to route
  new executions to sync
- disabling the canary must not mutate existing queue jobs
- no retry worker, timer, background scheduler, or scheduled workflow is
  introduced
- default-route migration remains blocked

## Initial Implementation Boundary For PR #138

PR #138 remains narrow and conservative.

Implemented in PR #138:

- the source job is terminal failed
- the retry request is idempotent by `retry_request_id`
- the original failed job remains immutable
- the retry creates a new child job with a new `job_id`
- lineage fields are preserved
- the child starts with reset runtime state and a copied task envelope
- the queued-canary helper rejects non-allowlisted, malformed, or mismatched
  run-plan queue jobs
- no automatic retry is introduced
- no `WAITING_USER` path is rewritten into retry
- no cancellation path is reinterpreted as retry

It must not:

- change `WorkerQueue.attempts` semantics
- treat stale lease recovery as explicit retry
- requeue the original failed job
- add automatic retry
- make cancelled jobs retryable
- make `WAITING_USER` use retry instead of resume
- make queued execution default

## Current Implementation Boundary After PR #138

PR #138 implements:

- `WorkerQueue.enqueue_retry_of_failed_job(...)` as an atomic low-level
  queue-control mutation under the existing queue lock
- strict one-shot child creation with `retry_request_id` idempotency
- original failed-job immutability
- `enqueue_queued_canary_retry(...)` as a higher-level validator for the
  existing allowlisted queued-canary `run_plan_execute` task envelope
- deterministic test coverage proving the retry child can be targeted and
  processed by the existing queue/poller infrastructure

PR #138 still does not implement:

- a Flask route or public retry/requeue API
- automatic retry, timers, or a scheduler
- retry initiation from `WorkerQueuePoller`
- allowlist expansion
- `train_model`, generation, literature, mining, or unknown queued-canary retry
- remote workers
- SQLite queue storage
- default-route migration
