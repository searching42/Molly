# Scientific Agent Autonomy Lease v1

This note freezes the implementation contract for `M3.5-AUT-LEASE`.  The
lease is a server-owned runtime eligibility layer over an already verified
`AutonomyGrant`; it is not a capability, an execution request, a scheduler, or
a new Controller.

## Identity and authority relation

`AutonomyLeaseV1` is identified by the digest of its semantic material.  Its
semantic bindings include the project, exact grant ID/digest, authority epoch,
validity interval, finite budgets, and the server policy ID/digest.  A lease is
published only after the current server grant has been re-read and its digest,
project, run lineage, and trusted actor source have been verified.

The lease may only narrow the grant.  Its task, effect, parameter, resource,
and external-I/O authority is empty by construction; those capabilities remain
owned by the existing Permission -> Authorization -> StartIntent -> Controller
chain.  A new grant epoch may receive a new lease after explicit server-owned
authority issuance.  L2 authority reuse and failure-recovery successors reuse
the predecessor grant/lease and do not renew it.

## Accounting units

Two independent receipt kinds are supported:

- `ACTIVE_EXECUTION`: server-side Controller/Executor work, measured with an
  injected monotonic clock around the bounded effect section.
- `REMOTE_RUNTIME`: trusted server-reported remote/GPU runtime, supplied only
  through the lease service seam; it is never derived from user/provider text.

Human/Gate/Evidence waiting, user review latency, process downtime, ordinary
conversation idle, and provider-unavailable idle are not active execution.
Execution Agent provider calls are not charged in v1; the lease is checked
before the call and rechecked immediately before Controller application.

The Controller reserves a server-derived finite per-operation upper bound
before an automatic effect.  It commits the measured monotonic duration after
the effect, or retains the reservation as reconciliation-required when the
effect outcome is unknown.  A reservation is never taken from a caller's
budget argument.

## Atomic reservation key

The aggregate critical section is the project-scoped lease lock for:

```text
(project_id, lease_id, lease_digest, grant_id, grant_digest, authority_epoch)
```

Within that process-safe lock the service verifies the current lease, rebuilds
committed receipts and unresolved reservations, checks the requested finite
reservation against the remaining budget, and publishes one immutable ordinal
reservation.  Controller effect application runs while the existing
execution-wide Controller lock is held; the lease check therefore cannot be a
check-then-release TOCTOU gate.  Controller operations immediately publish a
separate immutable `STARTED` checkpoint before entering the adapter boundary.

## Crash windows and replay

- Reservation before effect: the explicit server restart/reconciliation
  entrypoint rereads the exact operation using its durable checkpoints.  A
  reservation with no `STARTED` checkpoint can be marked `NOT_STARTED` and
  released; an already `STARTED` operation is retained until its Controller
  outcome is known.  `COMMITTED` commits one receipt; `UNKNOWN_EFFECT` retains
  the reservation and returns `AUTONOMY_LEASE_RECONCILIATION_REQUIRED`.
  Unknown effects are never automatically rerun.
- Effect before usage receipt: Controller replay re-reads the immutable effect
  evidence and commits the same operation-bound usage receipt exactly once.
- Usage receipt before session projection: budget evidence is rebuilt from
  receipts and reservation checkpoints, never from the Conversation projection.
- Same operation and same bytes replay the existing receipt.  Same operation
  with different usage/binding bytes is an immutable conflict.

## Runtime insertion points

The lease service is wired into the server-owned Harness Controller.  Every
automatic local, deterministic fastpath, remote-control, Execution Agent v2,
and trusted failure-recovery successor effect reaches the same Controller
boundary.  L1 Conversation checks the current lease before provider calls and
the Controller checks/reserves again immediately before effect dispatch.
L2 `SUBSET + NONE` remains insufficient without that current lease and budget
check.  Explicit Gate, remote approval, cancel, and recovery authority routes
remain human authority paths and do not consume autonomous lease budget.

## Fail-closed reason codes

The lease service uses only these server-owned codes:

```text
AUTONOMY_LEASE_UNAVAILABLE
AUTONOMY_LEASE_NOT_YET_VALID
AUTONOMY_LEASE_EXPIRED
AUTONOMY_ACTIVE_BUDGET_EXHAUSTED
AUTONOMY_REMOTE_BUDGET_EXHAUSTED
AUTONOMY_LEASE_STALE
AUTONOMY_LEASE_CONFLICT
AUTONOMY_LEASE_RECONCILIATION_REQUIRED
```

No LLM/provider response, natural-language message, clock argument, or budget
argument can create, extend, or alter a lease.

## Explicit v1 limitations

This is implementation/test evidence only (`I/T/—`).  There is no scheduler,
daemon, heartbeat, automatic renewal, representative/live acceptance, new GPU
canary, or BR2 real acceptance.  Remote runtime accounting is available only
when a trusted remote lifecycle supplies a server-derived runtime interval;
the local Controller polling/transport latency is not mislabeled as GPU runtime.
