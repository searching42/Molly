# Private remote worker setup

Molly addresses remote resources through logical IDs and SSH config aliases.
The repository must not contain a real hostname, IP address, username, home
directory, scheduler account, or environment path.

## 1. Define an SSH alias

Add the real endpoint only to `~/.ssh/config`:

```sshconfig
Host molly-gpu-main
    HostName gpu-host.example.internal
    User your-account
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Verify it outside Molly:

```bash
ssh molly-gpu-main -- hostname -s
```

Use `ssh-agent` or the operating-system Keychain for key material. Never put a
private key, password, bearer token, or `ProxyCommand` in a Molly JSON file.

## 2. Install the fixed worker protocol

Build an `ai4s-agent` wheel from the same reviewed commit as the control plane,
copy that wheel to the remote machine, and install it into a dedicated virtual
environment. The package exposes the fixed `molly-worker` console entrypoint;
do not replace it with a probe-only success shim.

```bash
python -m build
python3 -m venv /srv/molly/worker-venv
/srv/molly/worker-venv/bin/python -m pip install /path/to/ai4s_agent-0.1.0-py3-none-any.whl
```

The non-interactive SSH PATH must resolve `molly-worker`. Install a fixed
wrapper or symlink in a system PATH directory such as `/usr/local/bin`, or
configure the remote account's non-interactive PATH explicitly. Verify the
same command shape Molly will use:

```bash
ssh molly-gpu-main -- command -v molly-worker
ssh molly-gpu-main -- molly-worker probe --json
```

The worker reads a mode-`0600` private config from
`~/.config/Molly/worker.json` by default. Paths stay on the remote machine and
must never be committed:

```json
{
  "schema_version": "molly_worker_config.v1",
  "root": "/srv/molly/worker-state",
  "reinvent4_repository": "/srv/molly/repositories/reinvent4",
  "reinvent4_python": "/srv/molly/envs/reinvent4/bin/python",
  "unimol_repository": "/srv/molly/repositories/unimol",
  "unimol_python": "/srv/molly/envs/unimol/bin/python"
}
```

`probe` is read-only and reports a workload only when its configured repository
exists and its configured interpreter can import the expected distribution.
The initial adapters implement `reinvent4-cpu-v1` and `unimol-train-v1` only;
they never accept a browser-provided command, interpreter, working directory,
or output path. The Uni-Mol v1 publication contains one model, so the initial
adapter accepts `kfold=1` and fails closed on multi-model configurations.

Worker state is private, request-scoped, content-bound, and stored below the
configured root. Keep that root outside source checkouts and back it with
enough space for the bounded output contracts.

Each adapter runs in a dedicated process group. Walltime expiry and explicit
cancellation send `SIGTERM`, wait a fixed grace interval, escalate to
`SIGKILL`, and confirm that the group (including ordinary descendants) has
exited before publishing a terminal state.

## 3. Register the connection

The Settings page provides a guided connection form:

1. choose the GPU or CPU resource role; Molly assigns the stable logical ID;
2. enter the name after `Host` in `~/.ssh/config`;
3. run `ssh <alias> -- hostname -s` and enter the exact short hostname;
4. enter an absolute run directory on the remote machine;
5. select only workloads whose worker environments are installed remotely.

The dedicated `known_hosts` path is under **Advanced security options**.
Literature-only profiles may leave it blank and use SSH's normal host-key
resolution. Profiles selected for REINVENT4 or Uni-Mol execution must point to
a local, pinned `known_hosts` file; otherwise model dispatch is rejected before
transport. Saving also starts the bounded read-only worker probe. The UI reports full readiness
only when every declared capability is present in the probe's verified
capabilities; otherwise it lists the missing labels. This comparison never
rewrites the declaration, and neither a saved profile nor a successful probe
becomes execution authority.

The page writes a private `connections.json`. A minimal logical profile is
equivalent to:

```json
{
  "connection_id": "gpu-worker-main",
  "transport": "ssh",
  "ssh_host_alias": "molly-gpu-main",
  "expected_hostname": "gpu-host-short-name",
  "remote_root": "/private/path/chosen/by/the/user",
  "known_hosts_path": "/private/local/path/to/known_hosts",
  "scheduler": "direct",
  "declared_capabilities": ["gpu", "unimol"]
}
```

This file is local-only. Public plans, documentation, and repository-owned
execution profiles should refer only to `gpu-worker-main`. Run the capability
probe after saving; submission still performs a fresh, read-only preflight.

## 4. Register runtime environments

Remote repository and interpreter paths live in private
`environments.json`, not source code. After saving the connection in Settings,
open **运行环境路径（REINVENT4 / Uni-Mol）** and save one logical environment
per installed backend:

1. choose a stable ID such as `unimol-default` or `reinvent4-default`;
2. select the connection resource created in step 2;
3. enter the absolute remote repository root;
4. enter the absolute remote Python interpreter;
5. optionally enter the Conda environment name.

The form is equivalent to this private record:

```json
{
  "schema_version": "molly_environment_profiles.v1",
  "environments": [
    {
      "environment_id": "unimol-default",
      "connection_id": "gpu-worker-main",
      "repository_root": "/private/remote/path/to/unimol",
      "python_path": "/private/remote/environment/bin/python",
      "conda_environment": "unimol"
    }
  ]
}
```

For the legacy REINVENT4 adapter, select `environment_profile_id` in the task
payload or set:

```bash
export MOLLY_REINVENT4_ENVIRONMENT_ID=reinvent4-default
```

The Uni-Mol training interface resolves the same field. Its optional launcher
default is:

```bash
export MOLLY_UNIMOL_ENVIRONMENT_ID=unimol-default
```

The model-training UI selects the environment explicitly, so these environment
variables are normally unnecessary for browser-driven runs.

### REINVENT4 config template contract

The browser accepts a local REINVENT4 sampling **template**, not a reusable
effective config. Keep the complete model and sampling settings in that file and
bind these four values exactly where REINVENT4 expects them:

```toml
run_type = "sampling"
device = "cpu"
seed = {{molly_seed}}
json_out_config = "{{molly_output_csv}}.{{molly_design_request_id}}.{{molly_design_request_sha256}}.json"

[parameters]
output_file = "{{molly_output_csv}}"
# retain the rest of the validated REINVENT4 parameters here
```

Required placeholders are `{{molly_output_csv}}`,
`{{molly_design_request_id}}`, `{{molly_seed}}`, and
`{{molly_design_request_sha256}}`. Molly renders them only after allocating a
fresh attempt ID and a run-owned remote directory. It then freezes the rendered
bytes locally, transfers that already-open inode, imports the raw CSV bytes into
the same local attempt, and binds the request, effective config, raw output, and
profile digests in the generation publication. The UI does not accept or reuse
a fixed remote config/output path.

The OLED inverse-design execution policies resolve the following logical
environment IDs from this same private file:

```text
reinvent4-gpu-main
reinvent4-compute-main
```

Their linked connection profiles are `gpu-worker-main` and
`compute-worker-main`. The resulting publication stores only those logical
IDs, the connection/environment profile digests, and the pinned known-hosts
digest. SSH aliases, hostnames, repository roots, interpreter paths, and the
local known-hosts path are used in memory for transport and are not written to
the scientific publication.

Stage 6B requests still bind repository-owned execution profiles, immutable
input manifests, approvals, and output contracts. Private connection and
environment records only resolve how that fixed contract reaches a machine.

## 5. Replay retired publications

Retired v1 publications are never rewritten and retired transport profiles
cannot start new work. If exact replay is required, place the original static
transport contract in the private `legacy_transport_profiles.json`:

```json
{
  "schema_version": "molly_legacy_transport_profiles.v1",
  "profiles": [
    {
      "legacy_profile_id": "retired-profile-id-from-the-publication",
      "ssh_target": "private-ssh-alias-used-at-publication-time",
      "expected_hostname": "private-hostname-used-at-publication-time",
      "repository_root": "/private/historical/repository/path",
      "python_path": "/private/historical/environment/bin/python",
      "host_key_policy": "strict_pinned_known_hosts",
      "config_renderer": "reinvent4_v1"
    }
  ]
}
```

This file is verifier-only. It reconstructs the historical content-bound
contract locally without reintroducing private values into source code.

## 6. Scheduler information

The first private connection format supports direct execution. Do not add
Slurm/PBS partitions, accounts, reservations, or arbitrary submit command
templates to repository files. A future scheduler adapter should store those
values in user-level configuration and expose only a logical scheduler profile
ID to execution contracts.

## 7. Safe evidence and support bundles

Before publishing evidence:

- omit SSH aliases when they reveal an internal naming scheme;
- omit usernames, absolute paths, scheduler accounts, and IPs;
- include artifact SHA-256 values and contract IDs instead of paths;
- never attach `connections.json`, `environments.json`, `llm_profiles.json`,
  `capability_probes.json`, `legacy_transport_profiles.json`, known-hosts
  files, lock files, or `secrets/`.
