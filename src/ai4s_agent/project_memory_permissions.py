from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request
from pydantic import ValidationError

from ai4s_agent.memory import PermissionLevel, PermissionPolicy, ProjectMemory
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.server_permissions import ServerPermissionStore, decide_server_permission


MEMORY_WRITE_ACTION = "project_memory_write"


def install_project_memory_permission_routes() -> None:
    """Protect project memory mutations with server-side permission grants."""

    import ai4s_agent.api as api_module

    original_register_routes = api_module.register_routes
    if getattr(original_register_routes, "_project_memory_permissions", False):
        return

    def register_routes_with_project_memory_permissions(app: Any, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
        original_register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        workspace = api_module._workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        project_memory = ProjectMemory(workspace_dir=workspace)
        store = ServerPermissionStore(workspace_dir=workspace)
        policy = PermissionPolicy()
        policy.set_policy(MEMORY_WRITE_ACTION, PermissionLevel.PROJECT_APPROVED)
        allow_legacy = _config_bool(app.config.get("AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS", False), default=False)
        app.view_functions["create_project_memory_record"] = _create_memory_record_view(project_memory=project_memory, store=store, policy=policy, allow_legacy_client_flags=allow_legacy)
        app.view_functions["update_project_memory_record"] = _update_memory_record_view(project_memory=project_memory, store=store, policy=policy, allow_legacy_client_flags=allow_legacy)
        app.view_functions["delete_project_memory_record"] = _delete_memory_record_view(project_memory=project_memory, store=store, policy=policy, allow_legacy_client_flags=allow_legacy)
        app.view_functions["set_project_memory_enabled"] = _set_memory_enabled_view(project_memory=project_memory, store=store, policy=policy, allow_legacy_client_flags=allow_legacy)

    register_routes_with_project_memory_permissions._project_memory_permissions = True  # type: ignore[attr-defined]
    api_module.register_routes = register_routes_with_project_memory_permissions  # type: ignore[method-assign]


def _create_memory_record_view(*, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy, allow_legacy_client_flags: bool):
    def create_project_memory_record(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(store, policy, project_id, payload=payload, allow_legacy_client_flags=allow_legacy_client_flags)
        if error_response is not None:
            return error_response
        try:
            record_payload = _strip_permission_fields(payload)
            record = ProjectMemoryRecord.model_validate(record_payload)
            saved = project_memory.save_project_record(str(project_id or "").strip(), record)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc), "permission": decision}), 400
        return jsonify({"ok": True, "record": saved.model_dump(mode="json"), "permission": decision})

    return create_project_memory_record


def _update_memory_record_view(*, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy, allow_legacy_client_flags: bool):
    def update_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(store, policy, project_id, payload=payload, record_id=record_id, allow_legacy_client_flags=allow_legacy_client_flags)
        if error_response is not None:
            return error_response
        try:
            updated = project_memory.update_project_record(str(project_id or "").strip(), record_id, _strip_permission_fields(payload))
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc), "permission": decision}), 400
        if updated is None:
            return jsonify({"ok": False, "error": "memory record not found", "permission": decision}), 404
        return jsonify({"ok": True, "record": updated.model_dump(mode="json"), "permission": decision})

    return update_project_memory_record


def _delete_memory_record_view(*, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy, allow_legacy_client_flags: bool):
    def delete_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(store, policy, project_id, payload=payload, record_id=record_id, allow_legacy_client_flags=allow_legacy_client_flags)
        if error_response is not None:
            return error_response
        try:
            deleted = project_memory.delete_project_record(str(project_id or "").strip(), record_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "permission": decision}), 400
        return jsonify({"ok": True, "deleted": deleted, "permission": decision})

    return delete_project_memory_record


def _set_memory_enabled_view(*, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy, allow_legacy_client_flags: bool):
    def set_project_memory_enabled(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if not isinstance(payload.get("enabled"), bool):
            return jsonify({"ok": False, "error": "enabled boolean required"}), 400
        decision, error_response = _authorize_memory_write(store, policy, project_id, payload=payload, allow_legacy_client_flags=allow_legacy_client_flags)
        if error_response is not None:
            return error_response
        enabled = payload["enabled"]
        try:
            project_memory.set_project_memory_enabled(str(project_id or "").strip(), enabled)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "permission": decision}), 400
        return jsonify({"ok": True, "project_id": project_id, "enabled": enabled, "permission": decision})

    return set_project_memory_enabled


def _authorize_memory_write(
    store: ServerPermissionStore,
    policy: PermissionPolicy,
    project_id: str,
    *,
    payload: dict[str, Any],
    record_id: str = "",
    allow_legacy_client_flags: bool,
) -> tuple[dict[str, Any], Any | None]:
    clean_project = str(project_id or "").strip()
    actor = str(payload.get("actor") or payload.get("confirmed_by") or request.headers.get("X-Actor") or "").strip()
    legacy_project_approved = _as_bool(payload.get("project_approved")) or _as_bool(request.headers.get("X-Project-Approved"))
    try:
        decision = decide_server_permission(
            store,
            policy,
            MEMORY_WRITE_ACTION,
            project_id=clean_project,
            actor=actor,
            legacy_project_approved=legacy_project_approved,
            allow_legacy_client_flags=allow_legacy_client_flags,
        )
    except ValueError as exc:
        return {}, (jsonify({"ok": False, "error": str(exc)}), 400)
    if not decision["allowed"]:
        return decision, (
            jsonify(
                {
                    "ok": False,
                    "error": "server permission grant required for project memory write",
                    "record_id": str(record_id or payload.get("record_id") or "").strip(),
                    "permission": decision,
                }
            ),
            403,
        )
    return decision, None


def _strip_permission_fields(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"actor", "project_approved", "confirmed", "approved_by"}
    return {str(key): value for key, value in payload.items() if str(key) not in blocked}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _config_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value != 0
    clean = str(value).strip().lower()
    if clean in {"false", "0", "no", "n", "off"}:
        return False
    if clean in {"true", "1", "yes", "y", "on"}:
        return True
    return default
