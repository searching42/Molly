# Local deployment and private configuration

Molly is a local single-user web application. Repository files contain only
logical resource identifiers and public execution contracts. Hostnames,
usernames, remote paths, environment paths, and credentials belong to the
user-level configuration directory.

## Workspace

Choose the workspace at launch time instead of editing a source path:

```bash
export MOLLY_WORKSPACE="$HOME/Molly"
export PYTHONPATH="$MOLLY_WORKSPACE/src"
python -m flask --app 'ai4s_agent.app:create_app' run \
  --host 127.0.0.1 --port 8792
```

`MOLLY_WORKSPACE` may point to another local disk. Do not commit its resolved
absolute value. Runtime state remains under the selected workspace
(`projects/`, `runs/`, conversations, and artifacts) and is ignored by Git.

## User-level Molly directory

Molly uses `platformdirs` and `MOLLY_CONFIG_DIR`:

```text
macOS:   ~/Library/Application Support/Molly/
Linux:   ~/.config/molly/
Windows: %APPDATA%\Molly\
```

For an explicit location:

```bash
export MOLLY_CONFIG_DIR="$HOME/.config/molly"
```

The private layout is:

```text
Molly/
├── connections.json
├── environments.json
├── llm_profiles.json
├── capability_probes.json
├── legacy_transport_profiles.json
└── secrets/
```

Files containing private configuration are created with mode `0600`; the
directory is mode `0700` on POSIX systems. They must not be copied into the
repository. `connection_profiles.json` from an earlier Stage 6A installation
is migrated to `connections.json` and replaced by a metadata-free tombstone.
Molly's recursive ignore rules cover this complete layout even when an
explicit `MOLLY_CONFIG_DIR` is accidentally placed below a repository.

## LLM provider

Configure the provider from the first-run dialog or Settings. Molly stores
non-secret provider metadata in `llm_profiles.json`. The API key source is
explicitly selected from environment, system keyring, compatibility file, or
read-only auto discovery. API keys are never returned by the settings API or
written to browser persistence.

Recommended sources are the system Keychain/keyring, a launcher-provided
environment variable, and only then the compatibility secret file. The legacy
workspace file `.ai4s/llm_provider.json` is ignored and should be removed after
confirming the user-level profile works.

For the recommended macOS setup, select **系统 Keychain / keyring** in Settings,
enter the key once, and save. If the deployment is launched from a terminal or
service manager, an environment variable is also supported:

```bash
export MOLLY_LLM_API_KEY='replace-with-key-from-a-secure-source'
python -m flask --app 'ai4s_agent.app:create_app' run \
  --host 127.0.0.1 --port 8792
```

In Settings, select **环境变量**, keep the variable name as
`MOLLY_LLM_API_KEY`, and still save the endpoint and model name. The server must
be started by a process that inherits the variable; setting it in an unrelated
terminal after the server starts has no effect. Avoid placing the literal key
in repository files, committed `.env` files, or reusable shell history.

A configured non-loopback LLM endpoint is never used implicitly by an API
request. The caller must set `external_llm_approved=true` for the specific
request before Molly sends request data to that endpoint. Supplying
`llm_provider: null` explicitly selects the deterministic local rule-based
planner even when a user-level provider is configured.

## Backup and export

Treat the user-level Molly directory and local scientific workspace as private
data. Do not include them in public bug reports or source archives. Export
scientific artifacts through Molly's content-bound publication mechanisms,
not by copying an entire workspace.

## Git-history note

Removing a value from the current tree does not remove it from previous Git
objects. Repository administrators should audit all refs before a public
release. If historical host or user information must be erased, schedule a
separate coordinated `git filter-repo` rewrite, rotate affected credentials
first, force-push all rewritten refs, and require every clone and fork to be
replaced. A normal feature PR intentionally does not rewrite shared history.
