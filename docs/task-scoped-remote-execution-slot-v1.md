# Task-Scoped Remote Execution Slot v1

Status: PR-BN compatibility-preserving generalization of the existing remote
execution lifecycle.

## Problem and invariant

The merged lifecycle owns one fixed `remote-execution` directory under a run.
That makes a second remote task collide with the first immutable request and
share mutable transport, StageState, output, and recovery state. A mixed or
multi-remote plan therefore cannot safely reuse the run-scoped slot.

PR-BN makes the lifecycle operate on a pinned slot root while retaining one
implementation of request preparation, approval, dispatch, refresh,
committed-output verification, publication, Registry mutation, cancellation,
inspection, and recovery.

```text
legacy call with no slot identity
  -> runs/<run_id>/remote-execution/              (unchanged)

Controller task attempt
  -> runs/<run_id>/remote-executions/<slot_id>/
```

No Controller call is allowed to fall back from a requested task slot to the
legacy directory.

## Slot identity and binding

`slot_id` is a deterministic validated logical identifier derived from:

```text
controller_execution_id
planned_task_index
task_id
attempt
```

The slot publishes an immutable server-owned binding before a remote request.
Its semantic digest directly covers:

- project/run/controller execution and exact planned task/index/attempt;
- task-authority, remote AuthoritySet, and Controller execution digests;
- exact input artifact IDs/content digests and transfer-manifest digest;
- remote task type and output-contract digest; and
- the remote request ID/digest once prepared.

The immutable Controller execution digest transitively binds the start intent,
authorization, RunPlan, Controller policy, dispatch intent, compiled options,
logical connection, execution profile, configured resources, and full remote
authority roster. The lifecycle compares the complete direct binding; the
Controller separately re-verifies all transitive sources before using it.

Hosts, paths, SSH aliases, commands, environment data, credentials, raw
payloads, and mutable job state are excluded from the public binding. Private
connection resolution remains owned by existing server stores.

A caller must supply the expected slot binding to every slot-aware lifecycle
operation. The lifecycle exact-reads and compares it before touching mutable
transport or output state. A mismatched task, index, attempt, authority,
manifest, request, controller, or output contract is a conflict, not a new
request.

## Directory and Registry isolation

Every slot owns separate immutable/mutable files, locks, staging, output,
publication, StageState, and recovery material beneath its pinned root.
Published Registry paths are derived from that exact root and cannot use a
hard-coded legacy prefix. Logical plan artifacts may be adopted only through
an explicit Controller binding that maps a verified publication output to the
RunPlan output ID; same ID/same bytes is replay, while same ID/different bytes
is a conflict.

Slot traversal uses the existing dirfd/no-follow regular-file protections,
bounded JSON reads, private modes, inter-process file locks, fsync discipline,
and no-replace immutable publication. Slot IDs are validated before any path
operation; separators, traversal, aliases, and symlinks fail closed.

## Lifecycle and remote approval

```text
EMPTY
  -> REQUEST_PREPARED
  -> APPROVED | REJECTED
  -> SUBMITTED/RUNNING
  -> RECOVERY_REQUIRED | CANCELLED | FAILED | SUCCEEDED
  -> immutable output publication + Registry group + success StageState
```

The exact existing lifecycle remains authoritative for the transitions. The
Controller adds orchestration, not duplicate transition code.

The transfer manifest and output contract are server-derived from the exact
authorized task, registered input artifacts, selected execution profile, and
fixed remote task protocol. The public Controller request never accepts a
manifest, input/output path, connection, resources, adapter, command,
environment, or worker override.

Approval is immutable, positive, and slot-specific. The lifecycle approval
binds the exact request digest, slot binding, trusted actor, and bounded note.
The Controller's immutable request checkpoint separately binds the canonical
client request ID to that approval operation. Approval is rechecked
immediately before dispatch. A Gate decision, plan authorization, start
intent, approval for another attempt, or legacy run-scoped approval cannot
substitute.

## Crash, retry, concurrency, and recovery

All operations hold the exact slot's process lock for their existing critical
section. Concurrent operations on the same slot serialize; different slots do
not share request/job/output state. Immutable IDs are semantic and exclude
wall-clock or trace values.

Same request bytes in the same empty/prepared slot replay exactly. Different
bytes conflict. A crash is reconciled from immutable request/approval,
pending-dispatch anchor, remote job identity, committed-output claim,
publication, Registry, and StageState in that authority order. Mutable
transport status is observational when stronger evidence exists. Recovery is
explicit and bounded to one lifecycle call; the Controller never polls or
automatically increments the attempt.

A later attempt receives a new slot and cannot overwrite or reinterpret the
previous attempt. Attempt creation is a Controller recovery decision with an
immutable receipt and current authority re-verification.

## Legacy compatibility

The existing public service methods and remote lifecycle HTTP routes omit a
slot and continue to resolve exactly `runs/<run_id>/remote-execution`. Their
request/publication schemas, directory bytes, status precedence, and worker
protocol remain unchanged. New slot-aware methods are additive wrappers over
the same internal lifecycle implementation.

Compatibility tests freeze the legacy path and behavior, prove isolation of
two task slots in one run, reject cross-slot approval/job/output reuse, and
exercise crash/re-entry and concurrent advance behavior. There is no automatic
migration or aliasing of an existing legacy slot into a Controller execution.
