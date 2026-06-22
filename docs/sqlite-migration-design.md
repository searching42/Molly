# HARDEN-010: SQLite Migration Design Note

## Status

Design proposal — not yet implemented.  Evaluate after HARDEN-009 (storage
consistency checker) is merged and passing against the e2e workflow test.

## Current State

Mutable project state is stored as per-project JSON files.  The authoritative
inventory is `docs/storage-state-inventory.md`.  Below is a summary for the
SQLite migration conversation:

| File | Locking | Notes |
|------|---------|-------|
| `runs/<run_id>/stage.json` | currently unlocked RMW / atomic overwrite | Concurrent executor/resume paths can overwrite history without a lock. |
| `runs/<run_id>/artifact_registry.json` | locked RMW for project-scoped hot path | Legacy/manual writes are not locked. |
| `runs/<run_id>/gate_decisions.json` | mixed: locked RMW for project-scoped path, unlocked RMW for legacy path | Approval records are ordering-sensitive and audit-relevant. |
| `runs/<run_id>/execution_confirmations.json` | locked RMW | Appended through `ProjectStorage.append_execution_confirmation`. |
| `runs/<run_id>/job_state.json` | currently unlocked RMW / atomic overwrite | Transitions rewrite the full job object and history. |
| `runs/<run_id>/background_job_state.json` | currently unlocked RMW / atomic overwrite | Checkpoint append rewrites full state. |
| `runs/<run_id>/plan.json` | atomic create/overwrite | Mostly create-once plan state, low concurrency risk. |
| `permissions/permission_grants.json` | currently unlocked RMW / atomic overwrite | Create/revoke reads all grants, mutates, and rewrites. |
| `permissions/permission_audit.jsonl` | append-only | Append-only JSONL audit log, no RMW. |
| `../memory/<project_id>/project_memory_records.json` | currently unlocked RMW / atomic overwrite | Save/update/delete reads full records and rewrites. |
| `../memory/<project_id>/memory_manifest.json` | currently unlocked RMW / atomic overwrite | Collect/confirm rewrite the manifest. |
| `../memory/<project_id>/project_memory_policy.json` | atomic overwrite | Single boolean policy, low value for SQLite unless memory moves too. |
| `asset_manifest.json` | mixed: immutable for uploads, atomic overwrite for model promotion | Promotion rewrites status from candidate to confirmed. |
| `asset_promotion_records.json` | locked RMW | Append-like JSON list, locked in project storage. |

Lock categories (from `docs/storage-state-inventory.md`):

- **locked RMW**: `json_rmw_lock.py` (`fcntl.flock` + per-thread `RLock`).
- **currently unlocked RMW / atomic overwrite**: code reads, mutates, writes
  with a same-directory temp file for atomicity, but without per-file locks.
- **mixed**: locked on the project-scoped path, unlocked on the legacy path.
- **append-only**: atomic per-line append, no full-file RMW.
- **atomic create/overwrite**: write-once or full replace via temp file.

## Motivation For Migration

1. **Unlocked RMW is the norm, not the exception**: Only `artifact_registry`,
   `asset_promotion_records`, and `execution_confirmations` are locked.
   `stage.json`, `job_state.json`, `gate_decisions.json` (legacy path),
   `permission_grants.json`, `project_memory_records.json`, and
   `memory_manifest.json` all use unlocked RMW with atomic file overwrite.
   Concurrent requests can silently drop writes.

2. **Whole-file replacement**: Every append or update rewrites the entire
   file. For `project_memory_records.json` and `permission_grants.json` this
   grows linearly with project history.

3. **No transactional semantics**: A `promote_registered_model_asset` call
   writes three files (promoted_model_asset.json, asset_manifest.json,
   asset_promotion_records.json). If the process crashes between writes,
   the asset is partially promoted.

4. **Poor query support**: Listing runs by status, finding all WAITING_USER
   runs for a project, aggregating job metrics — all require
   `rglob` + `json.loads` across the filesystem.

5. **Cross-project isolation**: Project directories already provide
   filesystem-level isolation; a single SQLite database per project
   preserves this without introducing cross-project contention.

## Proposed Architecture

### Per-project SQLite database

```
projects/<project_id>/state.db  (SQLite, WAL mode)
```

### Schema — tables that move into SQLite

```sql
-- Run lifecycle
CREATE TABLE runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    stage          TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'pending',
    next_stage     TEXT DEFAULT NULL,
    started_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    ended_at       TEXT DEFAULT NULL,
    error_json     TEXT DEFAULT NULL,
    details_json   TEXT DEFAULT NULL,
    executed_tasks_json TEXT DEFAULT '[]'
);

-- Per-task stage history (currently stage.json details)
CREATE TABLE stage_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    adapter        TEXT DEFAULT '',
    started_at     TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Artifact registry (currently artifact_registry.json)
CREATE TABLE artifacts (
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    artifact_id    TEXT NOT NULL,
    relative_path  TEXT NOT NULL,
    content_hash   TEXT DEFAULT '',
    size_bytes     INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, artifact_id)
);

-- Gate decisions (currently gate_decisions.json)
CREATE TABLE gate_decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    gate           TEXT NOT NULL,
    approved       INTEGER NOT NULL DEFAULT 0,  -- boolean
    actor          TEXT NOT NULL DEFAULT '',
    note           TEXT NOT NULL DEFAULT '',
    approved_at    TEXT NOT NULL
);

-- Execution confirmations (currently execution_confirmations.json)
CREATE TABLE execution_confirmations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    task_id        TEXT NOT NULL,
    adapter        TEXT NOT NULL DEFAULT '',
    snapshot_id    TEXT NOT NULL,
    snapshot_hash  TEXT NOT NULL,
    actor          TEXT NOT NULL,
    confirmed_at   TEXT NOT NULL,
    note           TEXT NOT NULL DEFAULT '',
    approved_gates_json TEXT NOT NULL DEFAULT '[]',
    confirmation_type TEXT NOT NULL DEFAULT 'execute_ready_resume'
);

-- Permission grants (currently permission_grants.json)
CREATE TABLE permission_grants (
    grant_id       TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    run_id         TEXT NOT NULL DEFAULT '',
    action         TEXT NOT NULL,
    actor          TEXT NOT NULL,
    actor_source   TEXT NOT NULL DEFAULT '',
    reason         TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL DEFAULT '',
    active         INTEGER NOT NULL DEFAULT 1,
    revoked_at     TEXT DEFAULT NULL,
    revoked_by     TEXT DEFAULT NULL,
    revoked_by_source TEXT DEFAULT NULL,
    revoke_reason  TEXT DEFAULT NULL
);

-- Permission audit (currently permission_audit.jsonl)
CREATE TABLE permission_audit (
    decision_id    TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    run_id         TEXT NOT NULL DEFAULT '',
    action         TEXT NOT NULL,
    allowed        INTEGER NOT NULL,
    reason         TEXT NOT NULL,
    actor          TEXT NOT NULL DEFAULT '',
    actor_source   TEXT NOT NULL DEFAULT '',
    grant_id       TEXT NOT NULL DEFAULT '',
    legacy_client_flag INTEGER NOT NULL DEFAULT 0,
    decided_at     TEXT NOT NULL
);

-- Job state (currently job_state.json)
CREATE TABLE jobs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    attempt        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    state_json     TEXT NOT NULL DEFAULT '{}'
);

-- Project memory records (currently project_memory_records.json)
CREATE TABLE memory_records (
    record_id      TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    category       TEXT NOT NULL,
    summary        TEXT NOT NULL,
    value_json     TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    source_hashes_json TEXT NOT NULL DEFAULT '[]',
    decision       TEXT NOT NULL DEFAULT '',
    confirmed_by   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    disabled       INTEGER NOT NULL DEFAULT 0
);

-- Storage policy (currently project_memory_policy.json)
CREATE TABLE storage_policy (
    project_id     TEXT PRIMARY KEY,
    memory_enabled INTEGER NOT NULL DEFAULT 1
);
```

### What stays as files

- **Immutable artifacts**: Generated reports, CSVs, plots, model files under
  `runs/<run_id>/`. These are content-addressed, versioned assets that should
  remain directly inspectable on the filesystem.
- **Uploaded datasets**: Under `uploads/<upload_id>/`. Versioned and
  immutable.  Filesystem storage with manifest is appropriate.
- **Promoted model assets**: Under `assets/models/`. Versioned directories
  with model weights, manifest, and calibration files. Must remain directly
  loadable by inference code.
- **Run logs**: `job_log.jsonl` stays as append-only JSONL. Logs are
  unbounded and sequential; SQLite BLOB storage adds overhead with no benefit.

## Migration Strategy

### Phase 1: Dual-write (non-breaking)

1. Add `ProjectSQLiteStore(project_id)` class that manages the per-project
   `state.db`.
2. Every write method in `ProjectStorage` writes to both JSON files AND
   SQLite tables.
3. Every read method reads from JSON files (backward-compatible).
4. Add a `--storage-backend=sqlite` flag to opt into SQLite reads.
5. Run in dual-write mode for a release cycle; CI runs both paths.

### Phase 2: SQLite primary (breaking for new installs)

1. Switch reads to SQLite by default.
2. JSON files become read-only backup, written but never read.
3. New installs use SQLite only (no JSON files created).

### Phase 3: JSON removal (breaking)

1. Stop writing JSON files.
2. Existing JSON files remain as historical artifacts but are no longer
   updated.
3. Migration script converts existing JSON state to SQLite on first access.

## Risks

1. **WAL file growth**: SQLite WAL mode requires periodic checkpointing.
   The default auto-checkpoint (1000 pages) should be sufficient for
   single-user workloads, but long-running multi-user deployments may need
   explicit `PRAGMA wal_checkpoint`.

2. **Concurrent access**: SQLite supports multiple readers / single writer.
   For a local single-user app this is sufficient. For multi-user
   deployment, a connection pool or mutex per database is needed.

3. **Filesystem vs SQLite for artifacts**: Generated artifacts (reports,
   models) remain on the filesystem. The state database only stores
   *metadata about* artifacts, not the artifacts themselves.

4. **Rollback**: During Phase 1, JSON files remain authoritative. Rollback
   means deleting the SQLite database and continuing with JSON-only.

## Open Questions

1. Should `memory/` (cross-project) also move to SQLite, or stay
   per-project?
2. Should asset promotion records (currently
   `asset_promotion_records.json`) move into SQLite or stay as an append-only
   audit trail?
3. Should the migration use an ORM (SQLAlchemy) or raw `sqlite3` module?
   Recommend raw `sqlite3` for zero-dependency simplicity.

## References

- `docs/post-open-hardening.md` — HARDEN-010 acceptance criteria
- `src/ai4s_agent/storage.py` — current JSON storage
- `src/ai4s_agent/json_rmw_lock.py` — current locking mechanism
- `src/ai4s_agent/storage_consistency.py` — integrity checker
- SQLite WAL mode: https://www.sqlite.org/wal.html
