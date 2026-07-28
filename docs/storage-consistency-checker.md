# Storage Consistency Checker

PR #58 adds a read-only checker for AI4S workspace JSON/JSONL state. It reports
structural problems without rewriting files, adding locks, or migrating storage.

Python API:

```python
from ai4s_agent.storage_consistency import check_workspace_storage

report = check_workspace_storage("/path/to/workspace", legacy_runs_dir="/path/to/runs")
payload = report.to_dict()
```

CLI:

```bash
python -m ai4s_agent.storage_consistency /path/to/workspace
python -m ai4s_agent.storage_consistency /path/to/workspace --legacy-runs-dir /path/to/runs
```

The command prints JSON and exits with:

- `0` when no errors are found
- `1` when one or more errors are found

Report shape:

```json
{
  "ok": true,
  "errors": [],
  "warnings": [],
  "checked_files": [],
  "summary": {
    "checked_files": 0,
    "errors": 0,
    "warnings": 0
  }
}
```

Initial coverage:

- JSON parse and object-root checks for known mutable files
- `artifact_registry.json` relative path containment and existence
- `gate_decisions.json` required fields, unknown gates, duplicate gates, and
  ordering regression
- `permission_grants.json` required fields, duplicate grant ids, revoked grant
  metadata, and `expires_at` parsing
- `permission_audit.jsonl` line-by-line JSON and required audit fields
- `project_memory_records.json` schema validation
- `memory_manifest.json` missing artifact path checks
- `asset_manifest.json` schema validation
- `stage.json`, `job_state.json`, and `background_job_state.json` basic schema
  and run/status checks
- `worker_queue.json` and `worker_leases.json` list/object record shape,
  required fields, timestamps, status values, cancellation flags, TTL values,
  duplicate ids, and lease references to queued jobs

Out of scope for PR #58:

- Repairing malformed files
- Adding new locks
- Changing legacy write paths
- SQLite migration
