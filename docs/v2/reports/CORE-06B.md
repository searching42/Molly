# CORE-06B — minimal durable compute backend

Status: PASS for the local durable backend contract checkpoint. CORE-06C
fresh-real acceptance is still required for B2, and the final B3 decision is
deferred until the representative BR1 path has run on the selected backend.

## Scope

CORE-06B adds the optional `molly.plugins.remote_compute` package with a
small server-owned interface:

```text
submit(task, idempotency_key) → JobHandle
inspect(JobHandle)             → JobStatus
collect(JobHandle)             → ArtifactBundle
```

`LocalComputeBackend` and `RemoteComputeBackend` share a filesystem-backed
durable implementation. The remote class is a host adapter seam; it does not
contain an SSH implementation or expose a model-selected host, executable,
path, or credential. A server-owned runner is the only place where actual
local or remote execution can be attached.

The durable state binds profile digest, task digest, idempotency key, input
artifact IDs, and execution-config digest. Reusing an exact idempotency key
returns the persisted handle without dispatching another job. A changed task
or profile fails closed. Inspection is read-only. Collection verifies the
exact job identity, output manifest, artifact identities, sizes, and content
before returning an `ArtifactBundle`. A failed or interrupted job is not
implicitly rerun.

The BR1 compute runtime now carries the safe, secret-free `JobHandle` in its
stage metadata so acceptance evidence can bind each scientific occurrence to
the durable compute identity.

## Files

- `src/molly/plugins/remote_compute/models.py`: profile, JobHandle, status,
  output and bundle contracts.
- `src/molly/plugins/remote_compute/backend.py`: atomic durable submit,
  read-only inspect, verified collect, local/remote backend names.
- `src/molly/plugins/remote_compute/errors.py` and `__init__.py`.
- `tests/molly/test_core06_remote_compute.py`: idempotency, restart,
  tampering, foreign handles, secret-free durable state, and no implicit
  retry coverage.
- `src/molly/plugins/br1_inverse_design/runtime.py`: safe JobHandle evidence
  projection for compute-backed BR1 stages.

## Evidence

The focused CORE-06B command passed:

```text
PYTHONPATH=src PYTHONHASHSEED=0 python -m pytest -q \
  tests/molly/test_core06_remote_compute.py
6 passed
```

`compileall` and `git diff --check` passed for the checkpoint. Per the macro
workflow, Full CI and final CodeQL/PR checks are deferred until CORE-06C is
complete.

## Limitations and next checkpoint

The backend currently executes a supplied server-owned runner synchronously;
durable state and explicit no-replay behavior make the crash boundary
inspectable, but asynchronous scheduling is outside CORE-06. A real remote
adapter must be supplied by the acceptance environment. CORE-06C must use the
actual BR1 plugin path, perform fresh training/generation/prediction, and
record B2/B3 evidence without publishing private hosts, paths, or secrets.
CORE-07 is not started.
