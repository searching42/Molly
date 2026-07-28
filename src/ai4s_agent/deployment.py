from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai4s_agent.memory import PermissionPolicy, ProjectMemory
from ai4s_agent.schemas import MultiUserBoundaryCheck, MultiUserDeploymentReadiness, ProjectMemoryRecord
from ai4s_agent.storage import ProjectStorage


def assess_multi_user_deployment(
    *,
    workspace_dir: Path,
    runs_dir: Path | None = None,
) -> MultiUserDeploymentReadiness:
    """Assess whether the current local workspace has multi-user safety boundaries."""
    workspace = Path(workspace_dir).resolve()
    storage = ProjectStorage(workspace)
    checks = [
        _permission_actor_boundary_check(),
        _project_memory_boundary_check(workspace),
        _audit_actor_boundary_check(storage),
    ]
    status = "blocked" if any(check.status == "fail" for check in checks) else "ready"
    if runs_dir is not None:
        checks.append(_runs_dir_boundary_check(workspace, Path(runs_dir)))
        status = "blocked" if any(check.status == "fail" for check in checks) else "ready"
    return MultiUserDeploymentReadiness(status=status, checks=checks, executable=False)


def _permission_actor_boundary_check() -> MultiUserBoundaryCheck:
    policy = PermissionPolicy(require_actor_for_project_approved=True)
    confirm_missing_actor = policy.decide("train_model", confirmed=True)
    project_missing_actor = policy.decide("predict_candidates", project_approved=True)
    project_with_actor = policy.decide("predict_candidates", project_approved=True, actor="readiness-check")
    ok = (
        not confirm_missing_actor.allowed
        and confirm_missing_actor.reason == "CONFIRMATION_ACTOR_REQUIRED"
        and not project_missing_actor.allowed
        and project_missing_actor.reason == "PROJECT_APPROVAL_ACTOR_REQUIRED"
        and project_with_actor.allowed
    )
    return MultiUserBoundaryCheck(
        name="permission_actor_boundary",
        status="pass" if ok else "fail",
        message=(
            "Strict permission policy requires actor attribution for confirmed and project-approved actions."
            if ok
            else "Strict permission policy did not enforce actor attribution for all gated actions."
        ),
        evidence={
            "confirm_missing_actor_reason": confirm_missing_actor.reason,
            "project_missing_actor_reason": project_missing_actor.reason,
            "project_with_actor_allowed": project_with_actor.allowed,
            "checked_actions": ["train_model", "predict_candidates"],
        },
    )


def _project_memory_boundary_check(workspace: Path) -> MultiUserBoundaryCheck:
    memory = ProjectMemory(workspace_dir=workspace)
    path_guard_ok = False
    sensitive_guard_ok = False
    try:
        memory.list_project_records("../escape")
    except ValueError:
        path_guard_ok = True
    try:
        ProjectMemoryRecord(
            record_id="readiness-secret",
            category="remote_host",
            summary="Do not store api_key=sk-test in memory.",
            value={},
            decision="reject_sensitive_memory",
        )
    except (ValidationError, ValueError):
        sensitive_guard_ok = True
    ok = path_guard_ok and sensitive_guard_ok
    return MultiUserBoundaryCheck(
        name="project_memory_boundary",
        status="pass" if ok else "fail",
        message=(
            "Project memory rejects path traversal and sensitive/raw-data payloads."
            if ok
            else "Project memory boundaries are not strict enough for multi-user deployment."
        ),
        evidence={
            "path_traversal_rejected": path_guard_ok,
            "sensitive_memory_rejected": sensitive_guard_ok,
        },
    )


def _audit_actor_boundary_check(storage: ProjectStorage) -> MultiUserBoundaryCheck:
    projects_root = storage.projects_root
    missing_actor_refs: list[str] = []
    scanned_records = 0
    if projects_root.exists():
        for path in sorted(projects_root.rglob("gate_decisions.json")):
            for index, record in enumerate(_records_from_json(path, "decisions")):
                if not bool(record.get("approved")):
                    continue
                scanned_records += 1
                if not _record_actor(record):
                    missing_actor_refs.append(f"{_safe_relative(projects_root, path)}#decisions[{index}]")
        for path in sorted(projects_root.rglob("asset_promotion_records.json")):
            for index, record in enumerate(_records_from_json(path, "records")):
                scanned_records += 1
                if not _record_actor(record):
                    missing_actor_refs.append(f"{_safe_relative(projects_root, path)}#records[{index}]")
        for path in sorted(projects_root.rglob("model_registration_record.json")):
            record = _json_object(path)
            if not record:
                continue
            scanned_records += 1
            if not _record_actor(record):
                missing_actor_refs.append(str(_safe_relative(projects_root, path)))
    ok = not missing_actor_refs
    return MultiUserBoundaryCheck(
        name="audit_actor_boundary",
        status="pass" if ok else "fail",
        message=(
            "Existing approval and promotion audit records include actor attribution."
            if ok
            else "Some approval or promotion audit records lack actor attribution."
        ),
        evidence={
            "scanned_records": scanned_records,
            "missing_actor_count": len(missing_actor_refs),
            "missing_actor_records": missing_actor_refs[:50],
        },
    )


def _runs_dir_boundary_check(workspace: Path, runs_dir: Path) -> MultiUserBoundaryCheck:
    runs = runs_dir.resolve()
    inside_workspace = runs.is_relative_to(workspace)
    return MultiUserBoundaryCheck(
        name="runs_dir_boundary",
        status="pass" if inside_workspace else "fail",
        message=(
            "Runs directory is inside the configured workspace."
            if inside_workspace
            else "Runs directory is outside the configured workspace."
        ),
        evidence={"workspace_dir": str(workspace), "runs_dir": str(runs)},
    )


def _records_from_json(path: Path, key: str) -> list[dict[str, Any]]:
    loaded = _json_object(path)
    records = loaded.get(key, []) if isinstance(loaded, dict) else []
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record_actor(record: dict[str, Any]) -> str:
    return str(record.get("actor") or record.get("approved_by") or record.get("user_id") or "").strip()


def _safe_relative(base: Path, path: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path
