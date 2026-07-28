# Stage 6A: Resource Profiles And Transfer Contracts

Stage 6A adds configuration and preflight contracts for remote compute. It does
not dispatch, monitor, cancel, recover, or import a scientific remote job. Those
operations remain Stage 6B work and must continue through Molly's existing
Action, Session, Gate, immutable request, and publication control plane.

## Authority boundary

The following artifacts have different authority:

- A **Connection Profile** is private, user-local connection metadata. It says
  how Molly identifies an SSH endpoint, not what arbitrary command may run.
- An **Execution Profile** is a repository-owned allowlisted task contract. It
  fixes the worker entrypoint, task type, environment, resource ceiling, input
  contract, output contract, device policy, and process policy.
- A **Capability Probe** is read-only telemetry. It never authorizes execution,
  and a later submission must run its own lightweight preflight.
- A **Transfer Manifest** is a content-bound candidate input contract. It is not
  an approval or a transfer receipt.

The LLM and browser cannot provide a shell command, Python executable, Conda
activation command, scheduler script, or arbitrary worker entrypoint.

## User-level storage

Connection profiles are stored outside the Molly repository:

```text
macOS:   ~/Library/Application Support/Molly/connections.json
Linux:   ~/.config/molly/connections.json
Windows: %APPDATA%\Molly\connections.json
```

`platformdirs` selects the platform path. `MOLLY_CONFIG_DIR` or the
`user_config_dir` application argument can override it for tests and managed
deployments. The directory is mode `0700`; profile and probe files are written
through a mode `0600` temporary file, `fsync`, atomic `os.replace`, and parent
directory `fsync`.

SSH passwords, private keys, passphrases, tokens, and command templates are
rejected. Authentication remains in `ssh-agent`, the OS keychain, and
`~/.ssh/config`. `known_hosts_path` is an optional path reference, not copied
key material. Capability probing always enables strict host-key checking.

The legacy `workers/remote_workers.json` file is import-only. New and updated
legacy `/api/workers` calls are mapped into the same user-level Connection
Profile store, so they do not create a second worker registry.

## Settings API

```text
GET    /api/settings/compute
PUT    /api/settings/compute/connections/<connection_id>
DELETE /api/settings/compute/connections/<connection_id>
POST   /api/settings/compute/connections/<connection_id>/probe
```

`PUT` is a complete replacement for one connection. The URL identity must match
the body identity. The existing localhost write protections apply to every
mutating endpoint.

The UI Settings dialog can create and delete connections, run a read-only
probe, display the last probe status, and inspect the fixed execution profiles.

## Fixed execution profiles

The first allowlist contains:

| Profile | Task type | Device policy |
| --- | --- | --- |
| `reinvent4-cpu-v1` | molecular generation | CPU-only, nice 19, one thread |
| `mineru-v1` | document parsing | bounded resources, GPU allowed |
| `unimol-train-v1` | model training | bounded resources, GPU required |

Every profile uses the fixed logical entrypoint:

```text
molly-worker execute
```

Stage 6A does not invoke it. Stage 6B must pass an immutable request to that
entrypoint and bind the exact Connection Profile digest, Execution Profile
digest, transfer manifest digest, and approval digest.

The Stage 6A resolver maps the existing compatibility identifiers containing
`gpu_worker_main` and `compute_worker_main` to private Connection Profiles plus
`reinvent4-cpu-v1`. Those stable strings are policy/replay keys rather than live
hostnames; the mapping itself contains no host alias or path. The current
task-specific adapter and its historical immutable-receipt verifier remain
unchanged. The compatibility constants cannot be removed without preserving
exact replay of existing publications.

## Capability probe

The only probe command shape is:

```text
ssh [fixed safety options] <ssh-config-alias> -- molly-worker probe --json
```

The alias is validated as a single subprocess argument. The UI cannot append
arguments. Probe stderr is never persisted or returned. The bounded JSON result
records:

- expected and observed hostname status;
- verified capability labels;
- structured details such as CPU count, GPU identity, driver, runtime, toolkit,
  PyTorch CUDA build, and cuDNN version;
- check time and Connection Profile digest.

An observed hostname mismatch fails closed. Probe results are telemetry and do
not make a connection trusted forever.

## Transfer manifest

The builder pins the staging root with `O_DIRECTORY | O_NOFOLLOW`, opens every
path component relative to directory descriptors, rejects symlinks and special
files, and requires descriptors to cover the complete staging roster. Each file
is read from one stable descriptor and is bound by:

- run-relative path;
- purpose;
- media type;
- byte size;
- SHA-256.

The manifest additionally binds the Connection Profile digest, Execution
Profile digest, logical target purpose, total size, roster digest, and manifest
digest. `TransferManifest` independently validates canonical relative paths,
unique deterministic ordering, non-negative bounded sizes, digest syntax,
purpose/media syntax, and the repository-owned Execution Profile allowlist even
when all digests were recomputed by an importer. Stage 6B must additionally call
`verify_transfer_manifest_binding()` to bind an imported manifest to the exact
current Connection and Execution Profile digests.

The builder pins every directory and regular file descriptor in the complete
input tree before hashing. All descriptors remain open until the manifest is
constructed. Immediately before return, Molly rescans the roster, compares the
inode, mode, size, mtime, and ctime of every named directory and file against
the pinned descriptors, and re-hashes every file. Replacing an earlier file
while a later file is read therefore fails closed.

Stage 6B must still re-open and re-verify these exact bytes immediately before
upload; a committed manifest does not eliminate later filesystem changes.

## Explicitly deferred

- arbitrary SSH or shell execution;
- Slurm submission;
- scientific job dispatch;
- approval and cancellation lifecycle;
- remote output download and verified Artifact Registry import;
- automatic fallback between connections.
