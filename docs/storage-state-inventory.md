# Mutable JSON State Inventory

This inventory supports HARDEN-008. It documents current JSON/JSONL state
ownership and read-modify-write behavior before adding a storage consistency
checker or considering SQLite migration.

Scope:

- Project-scoped state under `workspace/projects/<project_id>/...`
- Legacy run state under `runs/<run_id>/...` where compatibility routes still
  read or write it
- Agent memory state under `workspace/memory/<project_id>/...`
- Asset metadata under `workspace/projects/<project_id>/assets/...`

This document is intentionally descriptive. It does not change storage
behavior.

## Classification Terms

- `locked RMW`: read-modify-write is serialized by `json_rmw_lock.py` using an
  in-process lock plus same-directory `.lock` file with `fcntl` when available.
- `append-only`: writers append records instead of rewriting previous records.
- `immutable artifact`: writer creates a versioned or exclusive file and should
  not later mutate it.
- `currently unlocked RMW`: code reads existing JSON, mutates it, then writes it
  atomically, but without a cross-process RMW lock.
- `atomic overwrite`: code writes the whole JSON file with same-directory temp
  file plus `os.replace`; this avoids partial writes but does not prevent lost
  updates when multiple writers perform RMW concurrently.
- `SQLite migration candidate`: state that would benefit from transactional
  updates, indexes, leases, or concurrent writers.

## Summary Table

| State file | Primary owner | Current classification | SQLite candidate | Notes |
| --- | --- | --- | --- | --- |
| `job_state.json` | `JobManager`, `project_scoped_jobs`, `local_worker_runner` | currently unlocked RMW / atomic overwrite | yes | Transitions rewrite the full job object and history. Important for worker leases, cancellation, terminal state, and active job listing. |
| `background_job_state.json` | `JobManager`, `project_scoped_jobs` | currently unlocked RMW / atomic overwrite | yes | Checkpoint append rewrites full state. Budget counters and resume checkpoint need transactional updates once real workers run concurrently. |
| `plan.json` | `Orchestrator`, `project_plan_guard` | atomic create/overwrite | maybe | Mostly create-once plan state. Low concurrency risk, but useful to index with run metadata if job/run state moves to SQLite. |
| `run_plan.json` | legacy status readers | owner unclear / compatibility read | no until owner exists | `routes/run_control.py` checks it as a compatibility fallback, but current RunPlan execution does not appear to use it as authoritative persisted state. Treat as a contract cleanup item. |
| `stage.json` | `RunPlanExecutor`, job routes | currently unlocked RMW / atomic overwrite | maybe | Stores current execution stage, snapshot, artifacts, and history. Atomic writes prevent torn files, but concurrent executor/resume paths can overwrite history without a lock. |
| `artifact_registry.json` | `ProjectStorage.register_artifact_path` | locked RMW for project-scoped hot path | maybe | Project-scoped writes are patched by `install_json_rmw_locks()`. Legacy/manual test writes are not locked. Checker should validate referenced paths exist. |
| `gate_decisions.json` | `ProjectStorage.append_gate_decision`, `Orchestrator.approve_gate` | mixed: locked RMW for project-scoped path, currently unlocked RMW for legacy path | yes | Approval records are ordering-sensitive and audit-relevant. Legacy route path still reads/mutates/writes through `ArtifactStore`. |
| `asset_promotion_records.json` | `ProjectStorage.append_asset_promotion_record` | locked RMW | maybe | Append-like JSON list, locked in project storage. Could remain file-based if checker validates schema and references. |
| `permission_grants.json` | `ServerPermissionStore` | currently unlocked RMW / atomic overwrite | yes | Grant create/revoke reads all grants, mutates list, and writes it back. Expiry/revoke/scope semantics make this a strong transactional-state candidate. |
| `permission_audit.jsonl` | `ServerPermissionStore.audit_decision` | append-only | maybe | Append-only audit log. Keep file-based short term; SQLite may help querying and integrity indexes later. |
| `project_memory_records.json` | `ProjectMemory` | currently unlocked RMW / atomic overwrite | maybe | Save/update/delete reads full records list and rewrites it. Could stay JSON for MVP but needs lock or transaction if multiple agents edit memory. |
| `memory_manifest.json` | `ProjectMemory` | currently unlocked RMW / atomic overwrite | maybe | Collect/confirm memory entries read and rewrite manifest. Needs consistency checks against referenced run artifacts. |
| `project_memory_policy.json` | `ProjectMemory` | atomic overwrite | no | Single boolean-style policy document. Low value for SQLite unless memory records move too. |
| `<run_id>_memory.json` | `ProjectMemory.collect_run_artifacts` | derived artifact / atomic overwrite | no | Per-run collection snapshot. Re-running the same collection overwrites the same filename, so checker should treat it as a derived artifact, not authoritative mutable state. |
| `asset_manifest.json` | `ProjectStorage`, upload asset writer | mixed: immutable for uploads, atomic overwrite for model promotion | maybe | Upload manifests are versioned candidate artifacts. Model promotion rewrites manifest status from candidate to confirmed. |
| `upload_record.json` | upload asset writer | immutable artifact | no | Written once inside versioned upload asset directory after data file creation. |
| uploaded data file | upload asset writer | immutable artifact | no | Created with `xb` inside a versioned directory; legacy compatibility copy also uses exclusive create unless file already exists. |
| `promoted_model_asset.json` | `ProjectStorage.promote_model_asset` | immutable artifact per promotion | no | Written inside model asset version directory. Promotion record list is the mutable index. |
| `model_registration_record.json` | `ProjectStorage.register_model_asset` | immutable artifact per model version | no | Written once in versioned model asset directory. |
| `project.json` | project routes | atomic create/overwrite | maybe | Project metadata is low volume but useful for future project index/session model. |
| worker registry JSON | `remote_worker.py` | atomic overwrite | yes | Remote worker registration/state needs leases, heartbeats, and queryable worker status if remote execution expands. |
| `worker_queue.json` | `JsonWorkerQueueStore` | locked RMW / atomic overwrite | yes | HARDEN-012 skeleton queue state. Jobs are acquired deterministically and mutated under `.worker_queue.lock`. |
| `worker_leases.json` | `JsonWorkerQueueStore` | locked RMW / atomic overwrite | yes | HARDEN-012 skeleton lease state. Active, stale, completed, and failed leases are mutated with the queue lock. |

## Locked RMW

The following hot paths are currently protected by `json_rmw_lock.py` after the
route extension installer runs:

- `artifact_registry.json`
- `gate_decisions.json`
- `asset_promotion_records.json`

Implementation notes:

- Locking wraps `ProjectStorage.register_artifact_path`,
  `ProjectStorage.append_gate_decision`, and
  `ProjectStorage.append_asset_promotion_record`.
- The lock uses a process-local `threading.RLock` plus a same-directory
  `.<filename>.lock` file with `fcntl.flock` on POSIX.
- Writes still go through `ProjectStorage._write_json()`, which uses atomic
  replacement.
- The legacy `Orchestrator` path uses `ArtifactStore` and is not covered by the
  project-scoped lock wrapper.

Follow-up:

- HARDEN-009 should check that `.lock` files are ignored by readers and that
  locked JSON files remain valid after interrupted writes.
- If legacy routes remain write-capable, either migrate them through
  `ProjectStorage` or mark them explicitly as single-process compatibility
  only.

## Append-Only

Append-only files:

- `permission_audit.jsonl`
- `job_log.jsonl`

Notes:

- Permission audit records are appended directly with `path.open("a")`.
- Project job logs append per event. Legacy `JobManager.save_job_log()` can also
  rewrite `job_log.jsonl` after merging in-memory entries, so log behavior is
  mostly append-only but not purely append-only in all paths.

Follow-up:

- HARDEN-009 should validate every JSONL line independently and report malformed
  records without failing the entire file.
- Future SQLite migration can index audit/log records, but the JSONL files are
  acceptable as durable append logs for the localhost MVP.

## Currently Unlocked RMW

These files use atomic writes but still have lost-update risk when two writers
perform read-modify-write concurrently:

- `job_state.json`
- `background_job_state.json`
- `stage.json`
- `permission_grants.json`
- `project_memory_records.json`
- `memory_manifest.json`
- legacy `gate_decisions.json` written through `Orchestrator`

Why these matter:

- `job_state.json` will become the worker lease, heartbeat, cancellation, and
  terminal-state source until a dedicated worker store exists.
- `background_job_state.json` carries checkpoint and budget counters.
- `stage.json` carries execution snapshot approval state and history.
- `permission_grants.json` carries active/revoked/expired grants and must not
  lose revoke/create updates.
- `project_memory_records.json` and `memory_manifest.json` can be edited by
  agent actions and user-approved memory writes.

Follow-up:

- For PR #58, the consistency checker should flag malformed schema, missing
  expected identity fields, and dangling artifact references.
- For later hardening, add locks or move the strongest candidates to SQLite
  before enabling multiple API/worker processes.

## Immutable Or Versioned Artifacts

These files are better kept file-based:

- uploaded data files under versioned upload asset directories
- `upload_record.json`
- upload `asset_manifest.json`
- `model_registration_record.json`
- `promoted_model_asset.json`
- adapter outputs such as `adapter_result.json`, reports, diagnostics, and
  extracted datasets

Notes:

- Upload data files are created with exclusive create (`xb`) inside a versioned
  directory.
- Versioned model and upload assets are content/provenance artifacts. The
  mutable index is the registry or promotion record, not the artifact payload.
- Model `asset_manifest.json` is an exception: promotion rewrites its status
  from candidate to confirmed.

Follow-up:

- HARDEN-009 should validate that every manifest points to existing files and
  that immutable artifacts are not referenced outside their project/version
  directory.

## SQLite Migration Candidates

Highest value candidates:

- `job_state.json`
- `background_job_state.json`
- `permission_grants.json`
- gate decision index from `gate_decisions.json`
- worker registry / lease state
- `worker_queue.json`
- `worker_leases.json`

Medium value candidates:

- `stage.json`
- `project_memory_records.json`
- `memory_manifest.json`
- `permission_audit.jsonl` as an indexed audit table
- `artifact_registry.json` as an index while keeping artifact files on disk

Low value candidates:

- immutable uploaded assets and reports
- versioned asset records
- project memory policy
- per-run memory snapshots

Suggested boundary:

- Keep raw artifacts, reports, uploaded datasets, parser outputs, model
  directories, and manifests as files.
- Move control-plane state that needs transactions, leases, indexes, or
  concurrent mutation into SQLite only after HARDEN-009 provides a checker for
  the current JSON layout.

## HARDEN-009 Checker Targets

The storage consistency checker should initially validate:

- `stage.json` schema and whether referenced artifacts exist.
- `artifact_registry.json` keys and relative paths.
- `gate_decisions.json` ordering, required fields, and snapshot hash fields
  where present.
- `asset_promotion_records.json` references to promoted assets and actor fields.
- `permission_grants.json` grant ids, active/revoked/expired semantics, and
  malformed scope fields.
- `permission_audit.jsonl` line-by-line JSON validity and reason/action fields.
- project memory record schema and disabled/enabled state.
- `memory_manifest.json` references to run artifacts.
- asset manifests, upload records, and promoted model records under versioned
  asset directories.
- job/background state status transitions, leases, heartbeats, cancellation
  fields, and references to missing runs.
- worker queue job ordering, active lease references, stale lease recovery, and
  cancellation flags.

## Open Questions

- Should legacy `Orchestrator` writes be migrated to `ProjectStorage` or
  explicitly frozen as compatibility-only?
- Should `run_plan.json` be removed from status checks, or should a clear owner
  start writing it as an execution plan snapshot?
- Should model promotion update `asset_manifest.json`, or should confirmed
  status move to `promoted_model_asset.json` plus promotion records only?
- Should permission grants move before worker state, given production profile
  now relies on grants for privileged writes?
