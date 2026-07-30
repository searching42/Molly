# Stage 6B: Governed Remote Execution Lifecycle

Stage 6B turns the Stage 6A resource and transfer contracts into a controlled
remote lifecycle. It does not add a generic SSH executor, a browser-provided
command, a Slurm adapter, or a second scientific task queue.

## Authority boundary

The authoritative facts remain Molly's existing control plane:

- the run `StageState`;
- the immutable execution request and approval;
- the immutable publication record;
- the project `Artifact Registry`;
- a parent Session's existing child-run reconciliation and Action recovery.

`remote-execution/state.json` is transport telemetry. It records the last
observed remote status, remote job identity, and bounded error code, but it
cannot publish a scientific result. A successful terminal state is committed
only after the immutable publication and every output byte pass exact replay
and the complete Artifact Registry group is registered.

The existing control-plane event projector therefore needs no remote-specific
state machine. It observes the same child `StageState`, Gate, Action, and
Artifact facts that it already projects. A Session adopts a completed remote
child through the existing reconciliation/recovery path.

## Immutable execution request and Gate

`molly_remote_execution_request.v1` binds:

- project, run, task, and request identities;
- exact Connection Profile ID and digest;
- exact repository-owned Execution Profile ID and digest;
- complete Stage 6A Transfer Manifest and digest;
- GPU, CPU-thread, and wall-time limits;
- the fixed output contract;
- a canonical request SHA-256.

Requested resources must remain within the Execution Profile ceilings and its
CPU/GPU policy. An imported or fully re-signed request is independently
validated by Pydantic and by `verify_transfer_manifest_binding()`.

The approval endpoint requires the exact request SHA-256. The resulting
`molly_remote_execution_approval.v1` record binds that digest, actor, note, and
approval time. A changed input, connection, profile, resource limit, task, or
output contract therefore invalidates the approval.

Immediately before publishing the approval, Molly:

1. reloads the current Connection and Execution Profiles;
2. exact-verifies their digests against the request;
3. performs a fresh read-only capability probe;
4. requires the expected hostname and all required capabilities;
5. reopens and re-hashes the complete staged input roster.

An old settings-page probe is never sufficient for submission.

## Artifact-based input staging

The API does not accept an arbitrary local input path. It accepts an exact map
from every Transfer Manifest relative path to an Artifact Registry ID. Molly
then:

1. resolves only run-relative registered paths;
2. opens each path component with no-follow semantics;
3. verifies size and SHA-256 against the Transfer Manifest;
4. copies bytes into `remote-execution/inputs/` through private immutable
   staging;
5. rejects missing, extra, duplicate, special, or symlink entries;
6. repeats exact verification immediately before dispatch.

The fixed remote protocol stages the immutable request and approval, transfers
only that roster, and requires the remote worker to return the exact input
manifest digest before `execute` is invoked.

## Fixed transport protocol

The only allowed remote program is `molly-worker`. SSH targets are validated
Connection Profile aliases, and strict host-key checking is always enabled.
Commands are constructed as argument vectors; no Shell command is accepted or
interpolated.

```text
molly-worker stage --json
molly-worker stage-input --request-id <safe-id> --path <safe-relative-path> --size <bytes> --sha256 <digest> --json
molly-worker verify-inputs --request-id <safe-id> --json
molly-worker execute --json
molly-worker status --request-id <safe-id> --json
molly-worker cancel --request-id <safe-id> --json
molly-worker fetch-output --request-id <safe-id> --path <safe-relative-path> --size <bytes> --sha256 <digest>
```

Request, approval, and digest bindings travel as JSON on stdin. The remote
`stage-input` command receives only the declared raw input bytes, while
`fetch-output` emits only the publication-bound raw output bytes on stdout. The
deployment supplies task-specific adapters behind this protocol (for example
REINVENT4, MinerU, or UniMol); the browser and LLM cannot select an executable,
environment activation, command argument, output path, or Shell fragment.

Transport stderr is never persisted or returned. Non-zero exits, malformed or
oversized JSON, identity mismatches, and unavailable connections are reduced to
bounded local error codes.

## Monitor, cancel, and recovery

`refresh` reads one structured remote observation. A network failure does not
guess the remote outcome; the child run becomes paused with
`RECOVERY_REQUIRED` transport telemetry and must use the explicit recovery
endpoint.

Cancellation is deliberately two-phase:

```text
CANCEL_REQUESTED
        ↓ remote confirms termination
CANCELLED
```

If the remote worker still reports `RUNNING`, Molly preserves
`CANCEL_REQUESTED`. If the connection is lost, Molly records
`RECOVERY_REQUIRED`; it never assumes that the process stopped or that resource
charges ended. Recovery calls the same fixed status protocol and only adopts a
remote terminal fact that binds the exact request.

## Verified result import

A successful observation must carry
`molly_remote_execution_publication.v1`, binding:

- request and approval digests;
- input manifest digest;
- output contract;
- a unique, deterministically sorted artifact ID/path roster;
- size and SHA-256 for every output;
- the publication digest.

Molly downloads only this roster into private run staging. Relative paths are
validated, parent directories use no-follow creation, and each downloaded file
is checked before and after publication. Missing, extra, modified, or symlinked
outputs fail closed.

The complete Artifact Registry group is inserted atomically before the
publication record becomes visible. A publication write failure compensates by
removing only the exact group inserted by that attempt. Every later GET exact-
replays the publication and local output bytes, so post-success tampering is
detected.

## Local API

```text
POST /api/projects/<project_id>/remote-executions
GET  /api/projects/<project_id>/remote-executions/<run_id>
POST /api/projects/<project_id>/remote-executions/<run_id>/approve
POST /api/projects/<project_id>/remote-executions/<run_id>/refresh
POST /api/projects/<project_id>/remote-executions/<run_id>/cancel
POST /api/projects/<project_id>/remote-executions/<run_id>/recover
```

Mutating routes inherit Molly's localhost session-token, Origin, Host, and
remote-address protections. GET inspection is read-only and does not create a
project, run, or remote-execution directory.

Preparation and approval are protected by an in-process lock plus a per-run
`flock`. Concurrent duplicate preparation returns the same immutable request;
concurrent duplicate approval dispatches at most once. If an approval record is
present but dispatch outcome was not committed, Molly requires recovery rather
than dispatching a second potentially billable job.

## Explicit non-goals

- arbitrary Shell or a generic SSH command service;
- commands generated by an LLM;
- automatic fallback to another server;
- optimistic cancellation;
- trusting remote paths or an unbound result manifest;
- Slurm in the first controlled transport;
- using transport telemetry as scientific truth.
