from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ai4s_agent.gatekeeper import GATE_SEQUENCE
from ai4s_agent.schemas import AssetManifest, BackgroundJobState, ProjectMemoryRecord, RunStatus, StageState


@dataclass(frozen=True)
class StorageConsistencyIssue:
    code: str
    message: str
    path: str
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
            "details": dict(self.details),
        }


@dataclass
class StorageConsistencyReport:
    ok: bool = True
    errors: list[StorageConsistencyIssue] = field(default_factory=list)
    warnings: list[StorageConsistencyIssue] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, path: Path, **details: Any) -> None:
        self.ok = False
        self.errors.append(StorageConsistencyIssue(code=code, message=message, path=str(path), details=details))

    def add_warning(self, code: str, message: str, path: Path, **details: Any) -> None:
        self.warnings.append(StorageConsistencyIssue(code=code, message=message, path=str(path), severity="warning", details=details))

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checked_files": list(self.checked_files),
            "summary": dict(self.summary),
        }

    def finalize(self) -> None:
        self.ok = not self.errors
        self.summary = {
            "checked_files": len(self.checked_files),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


def check_workspace_storage(workspace_dir: str | Path, *, legacy_runs_dir: str | Path | None = None) -> StorageConsistencyReport:
    workspace = Path(workspace_dir).expanduser().resolve()
    report = StorageConsistencyReport()
    _check_project_run_files(report, workspace / "projects")
    _check_permission_files(report, workspace / "projects")
    _check_memory_files(report, workspace / "memory")
    _check_asset_files(report, workspace / "projects")
    _check_worker_queue_files(report, workspace)
    if legacy_runs_dir is not None:
        _check_legacy_run_files(report, Path(legacy_runs_dir).expanduser().resolve())
    report.finalize()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AI4S agent workspace storage consistency.")
    parser.add_argument("workspace_dir", help="AI4S workspace directory to inspect")
    parser.add_argument("--legacy-runs-dir", default="", help="Optional legacy runs directory to inspect")
    args = parser.parse_args(argv)
    report = check_workspace_storage(
        args.workspace_dir,
        legacy_runs_dir=args.legacy_runs_dir or None,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


def _check_project_run_files(report: StorageConsistencyReport, projects_root: Path) -> None:
    if not projects_root.exists():
        return
    for run_dir in sorted(projects_root.glob("*/runs/*")):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        _check_stage_state(report, run_dir / "stage.json")
        _check_artifact_registry(report, run_dir / "artifact_registry.json", run_dir)
        _check_gate_decisions(report, run_dir / "gate_decisions.json")
        _check_job_state(report, run_dir / "job_state.json", run_id=run_id)
        _check_background_job_state(report, run_dir / "background_job_state.json")


def _check_legacy_run_files(report: StorageConsistencyReport, runs_root: Path) -> None:
    if not runs_root.exists():
        return
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        _check_artifact_registry(report, run_dir / "artifact_registry.json", run_dir)
        _check_gate_decisions(report, run_dir / "gate_decisions.json")
        _check_stage_state(report, run_dir / "stage.json")
        _check_job_state(report, run_dir / "job_state.json", run_id=run_dir.name)
        _check_background_job_state(report, run_dir / "background_job_state.json")


def _check_permission_files(report: StorageConsistencyReport, projects_root: Path) -> None:
    if not projects_root.exists():
        return
    for perm_dir in sorted(projects_root.glob("*/permissions")):
        if not perm_dir.is_dir():
            continue
        _check_permission_grants(report, perm_dir / "permission_grants.json")
        _check_permission_audit(report, perm_dir / "permission_audit.jsonl")


def _check_memory_files(report: StorageConsistencyReport, memory_root: Path) -> None:
    if not memory_root.exists():
        return
    for project_dir in sorted(memory_root.iterdir()):
        if not project_dir.is_dir():
            continue
        _check_project_memory_records(report, project_dir / "project_memory_records.json")
        _check_memory_manifest(report, project_dir / "memory_manifest.json")


def _check_asset_files(report: StorageConsistencyReport, projects_root: Path) -> None:
    if not projects_root.exists():
        return
    for path in sorted(projects_root.glob("*/assets/**/asset_manifest.json")):
        _check_asset_manifest(report, path)


def _check_worker_queue_files(report: StorageConsistencyReport, workspace: Path) -> None:
    if not workspace.exists():
        return
    roots = {path.parent for path in workspace.rglob("worker_queue.json")}
    roots.update(path.parent for path in workspace.rglob("worker_leases.json"))
    for root in sorted(roots):
        job_ids = _check_worker_queue(report, root / "worker_queue.json")
        _check_worker_leases(report, root / "worker_leases.json", job_ids=job_ids)


def _check_stage_state(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    try:
        StageState.model_validate(payload)
    except ValueError as exc:
        report.add_error("invalid_stage_state", "stage.json does not match StageState schema", path, error=str(exc))


def _check_artifact_registry(report: StorageConsistencyReport, path: Path, run_dir: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        report.add_error("artifact_registry_invalid", "artifact_registry.json artifacts must be an object", path)
        return
    for artifact_id, raw_relative in artifacts.items():
        relative = str(raw_relative or "").strip()
        if not relative:
            report.add_error("artifact_path_missing", "artifact registry path is empty", path, artifact_id=str(artifact_id))
            continue
        target = (run_dir / relative).resolve()
        if not _is_relative_to(target, run_dir.resolve()):
            report.add_error("artifact_path_escape", "artifact registry path escapes run directory", path, artifact_id=str(artifact_id), relative_path=relative)
            continue
        if not target.exists():
            report.add_error("artifact_missing", "artifact registry path does not exist", path, artifact_id=str(artifact_id), relative_path=relative)


def _check_gate_decisions(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        report.add_error("gate_decisions_invalid", "gate_decisions.json decisions must be a list", path)
        return
    gate_order = {gate.value: index for index, gate in enumerate(GATE_SEQUENCE)}
    previous_index = -1
    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            report.add_error("gate_decision_invalid", "gate decision must be an object", path, index=index)
            continue
        for field_name in ("gate", "approved", "actor", "approved_at"):
            if field_name not in decision or decision.get(field_name) in {"", None}:
                report.add_error("gate_decision_missing_field", f"gate decision missing {field_name}", path, index=index, field=field_name)
        gate = str(decision.get("gate") or "")
        if gate and gate not in gate_order:
            report.add_error("gate_decision_unknown_gate", "gate decision contains unknown gate", path, index=index, gate=gate)
            continue
        current_index = gate_order.get(gate, previous_index)
        if gate in seen:
            report.add_warning("gate_decision_duplicate_gate", "gate decision contains duplicate gate", path, index=index, gate=gate)
        if current_index < previous_index:
            report.add_error("gate_decision_out_of_order", "gate decision order regressed", path, index=index, gate=gate)
        previous_index = max(previous_index, current_index)
        if gate:
            seen.add(gate)


def _check_job_state(report: StorageConsistencyReport, path: Path, *, run_id: str) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    if str(payload.get("run_id") or "") != run_id:
        report.add_error("job_state_run_mismatch", "job_state.json run_id does not match directory", path, expected_run_id=run_id, actual_run_id=str(payload.get("run_id") or ""))
    status = str(payload.get("status") or "")
    if status not in {item.value for item in RunStatus}:
        report.add_error("job_state_invalid_status", "job_state.json status is not a known RunStatus", path, status=status)


def _check_background_job_state(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    try:
        BackgroundJobState.model_validate(payload)
    except ValueError as exc:
        report.add_error("invalid_background_job_state", "background_job_state.json does not match BackgroundJobState schema", path, error=str(exc))


def _check_permission_grants(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    grants = payload.get("grants")
    if not isinstance(grants, list):
        report.add_error("permission_grants_invalid", "permission_grants.json grants must be a list", path)
        return
    seen: set[str] = set()
    for index, grant in enumerate(grants):
        if not isinstance(grant, dict):
            report.add_error("permission_grant_invalid", "permission grant must be an object", path, index=index)
            continue
        for field_name in ("grant_id", "action", "actor", "active"):
            if field_name not in grant or grant.get(field_name) in {"", None}:
                report.add_error("permission_grant_missing_field", f"permission grant missing {field_name}", path, index=index, field=field_name)
        grant_id = str(grant.get("grant_id") or "")
        if grant_id in seen:
            report.add_error("permission_grant_duplicate_id", "permission grant id is duplicated", path, index=index, grant_id=grant_id)
        if grant_id:
            seen.add(grant_id)
        if grant.get("active") is False:
            for field_name in ("revoked_at", "revoked_by"):
                if not str(grant.get(field_name) or "").strip():
                    report.add_error("permission_grant_revoked_missing_field", f"revoked grant missing {field_name}", path, index=index, grant_id=grant_id, field=field_name)
        expires_at = str(grant.get("expires_at") or "").strip()
        if expires_at and _parse_timezone_aware_iso(expires_at) is None:
            report.add_error("permission_grant_invalid_expires_at", "permission grant expires_at must be ISO timestamp with timezone", path, index=index, grant_id=grant_id, expires_at=expires_at)


def _check_permission_audit(report: StorageConsistencyReport, path: Path) -> None:
    if not path.exists():
        return
    report.checked_files.append(str(path))
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            report.add_error("invalid_jsonl", "permission audit line is not valid JSON", path, line=line_no, error=str(exc))
            continue
        if not isinstance(payload, dict):
            report.add_error("invalid_jsonl", "permission audit line must be an object", path, line=line_no)
            continue
        for field_name in ("action", "reason", "allowed"):
            if field_name not in payload or payload.get(field_name) in {"", None}:
                report.add_error("permission_audit_missing_field", f"permission audit missing {field_name}", path, line=line_no, field=field_name)


def _check_project_memory_records(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    records = payload.get("records")
    if not isinstance(records, list):
        report.add_error("project_memory_records_invalid", "project_memory_records.json records must be a list", path)
        return
    for index, record in enumerate(records):
        try:
            ProjectMemoryRecord.model_validate(record)
        except ValueError as exc:
            report.add_error("invalid_project_memory_record", "project memory record does not match schema", path, index=index, error=str(exc))


def _check_memory_manifest(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        report.add_error("memory_manifest_invalid", "memory_manifest.json artifacts must be a list", path)
        return
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            report.add_error("memory_manifest_invalid_artifact", "memory manifest artifact must be an object", path, index=index)
            continue
        artifact_path = str(item.get("artifact_path") or "").strip()
        if artifact_path and not Path(artifact_path).expanduser().exists():
            report.add_error("memory_artifact_missing", "memory manifest artifact path does not exist", path, index=index, artifact_path=artifact_path)


def _check_asset_manifest(report: StorageConsistencyReport, path: Path) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    try:
        AssetManifest.model_validate(payload)
    except ValueError as exc:
        report.add_error("invalid_asset_manifest", "asset_manifest.json does not match AssetManifest schema", path, error=str(exc))


def _check_worker_queue(report: StorageConsistencyReport, path: Path) -> set[str] | None:
    payload = _read_json_object(report, path)
    if payload is None:
        return None
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        report.add_error("worker_queue_jobs_invalid", "worker_queue.json jobs must be a list", path)
        return set()
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            report.add_error("worker_queue_job_invalid", "worker queue job must be an object", path, index=index)
            continue
        for field_name in ("job_id", "project_id", "run_id", "task", "status", "created_at", "updated_at", "cancellation_requested"):
            if _missing_required(job, field_name):
                report.add_error("worker_queue_job_missing_field", f"worker queue job missing {field_name}", path, index=index, field=field_name)
        job_id = str(job.get("job_id") or "")
        if job_id in seen:
            report.add_error("worker_queue_job_duplicate_id", "worker queue job id is duplicated", path, index=index, job_id=job_id)
        if job_id:
            seen.add(job_id)
        if not isinstance(job.get("task"), dict):
            report.add_error("worker_queue_job_invalid_task", "worker queue job task must be an object", path, index=index, job_id=job_id)
        status = str(job.get("status") or "")
        if status and status not in {"queued", "running", "cancelled", "succeeded", "failed"}:
            report.add_error("worker_queue_job_invalid_status", "worker queue job status is invalid", path, index=index, job_id=job_id, status=status)
        if not isinstance(job.get("cancellation_requested"), bool):
            report.add_error("worker_queue_job_invalid_cancellation", "worker queue job cancellation_requested must be boolean", path, index=index, job_id=job_id)
        for field_name in ("created_at", "updated_at", "heartbeat_at", "stale_recovered_at"):
            timestamp = str(job.get(field_name) or "").strip()
            if timestamp and _parse_timezone_aware_iso(timestamp) is None:
                report.add_error("worker_queue_job_invalid_timestamp", f"worker queue job {field_name} must be ISO timestamp with timezone", path, index=index, job_id=job_id, field=field_name, value=timestamp)
        attempts = job.get("attempts")
        if attempts is not None and (not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0):
            report.add_error("worker_queue_job_invalid_attempts", "worker queue job attempts must be a non-negative integer", path, index=index, job_id=job_id)
    return seen


def _check_worker_leases(report: StorageConsistencyReport, path: Path, *, job_ids: set[str] | None) -> None:
    payload = _read_json_object(report, path)
    if payload is None:
        return
    leases = payload.get("leases")
    if not isinstance(leases, list):
        report.add_error("worker_leases_invalid", "worker_leases.json leases must be a list", path)
        return
    seen: set[str] = set()
    for index, lease in enumerate(leases):
        if not isinstance(lease, dict):
            report.add_error("worker_lease_invalid", "worker lease must be an object", path, index=index)
            continue
        for field_name in ("lease_id", "job_id", "worker_id", "status", "acquired_at", "heartbeat_at", "expires_at", "ttl_sec"):
            if _missing_required(lease, field_name):
                report.add_error("worker_lease_missing_field", f"worker lease missing {field_name}", path, index=index, field=field_name)
        lease_id = str(lease.get("lease_id") or "")
        if lease_id in seen:
            report.add_error("worker_lease_duplicate_id", "worker lease id is duplicated", path, index=index, lease_id=lease_id)
        if lease_id:
            seen.add(lease_id)
        status = str(lease.get("status") or "")
        if status and status not in {"active", "completed", "failed", "stale"}:
            report.add_error("worker_lease_invalid_status", "worker lease status is invalid", path, index=index, lease_id=lease_id, status=status)
        for field_name in ("acquired_at", "heartbeat_at", "expires_at", "completed_at", "stale_at"):
            timestamp = str(lease.get(field_name) or "").strip()
            if timestamp and _parse_timezone_aware_iso(timestamp) is None:
                report.add_error("worker_lease_invalid_timestamp", f"worker lease {field_name} must be ISO timestamp with timezone", path, index=index, lease_id=lease_id, field=field_name, value=timestamp)
        ttl_sec = lease.get("ttl_sec")
        if not isinstance(ttl_sec, int) or isinstance(ttl_sec, bool) or ttl_sec <= 0:
            report.add_error("worker_lease_invalid_ttl", "worker lease ttl_sec must be a positive integer", path, index=index, lease_id=lease_id)
        job_id = str(lease.get("job_id") or "")
        if job_ids is not None and job_id and job_id not in job_ids:
            report.add_error("worker_lease_dangling_job", "worker lease references a missing queue job", path, index=index, lease_id=lease_id, job_id=job_id)


def _read_json_object(report: StorageConsistencyReport, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report.checked_files.append(str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add_error("invalid_json", "file is not valid JSON", path, error=str(exc))
        return None
    if not isinstance(payload, dict):
        report.add_error("invalid_json", "file JSON root must be an object", path)
        return None
    return payload


def _missing_required(payload: dict[str, Any], field_name: str) -> bool:
    if field_name not in payload:
        return True
    value = payload.get(field_name)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_timezone_aware_iso(value: str) -> datetime | None:
    clean = value.strip()
    if clean.endswith("Z"):
        clean = f"{clean[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
