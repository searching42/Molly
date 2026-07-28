from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flask import jsonify, request
from pydantic import ValidationError

from ai4s_agent.actor_identity import resolve_actor
from ai4s_agent.memory import PermissionLevel, PermissionPolicy, ProjectMemory
from ai4s_agent.profiles import legacy_client_permission_flags_enabled
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.server_permissions import ServerPermissionStore, decide_server_permission

if TYPE_CHECKING:
    from ai4s_agent.api_route_extensions import RouteExtensionContext


MEMORY_WRITE_ACTION = "project_memory_write"


def install_project_memory_permission_routes() -> None:
    """Project memory permission routes are installed by the explicit hook."""


def apply_project_memory_permission_routes(context: "RouteExtensionContext") -> None:
    """Protect project memory mutations with server-side permission grants."""

    import ai4s_agent.api as api_module

    workspace = api_module._workspace_from_config(
        base_runs_dir=context.base_runs_dir,
        workspace_dir=context.workspace_dir,
    )
    project_memory = ProjectMemory(workspace_dir=workspace)
    store = ServerPermissionStore(workspace_dir=workspace)
    policy = PermissionPolicy()
    policy.set_policy(MEMORY_WRITE_ACTION, PermissionLevel.PROJECT_APPROVED)
    view_kwargs = {
        "app": context.app,
        "project_memory": project_memory,
        "store": store,
        "policy": policy,
    }
    context.route_overrides.apply_route_override(
        context.app,
        extension_id="project_memory_permission_routes",
        endpoint="create_project_memory_record",
        view_func=_create_memory_record_view(**view_kwargs),
    )
    context.route_overrides.apply_route_override(
        context.app,
        extension_id="project_memory_permission_routes",
        endpoint="update_project_memory_record",
        view_func=_update_memory_record_view(**view_kwargs),
    )
    context.route_overrides.apply_route_override(
        context.app,
        extension_id="project_memory_permission_routes",
        endpoint="delete_project_memory_record",
        view_func=_delete_memory_record_view(**view_kwargs),
    )
    context.route_overrides.apply_route_override(
        context.app,
        extension_id="project_memory_permission_routes",
        endpoint="set_project_memory_enabled",
        view_func=_set_memory_enabled_view(**view_kwargs),
    )


def _create_memory_record_view(*, app: Any, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy):
    def create_project_memory_record(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(app, store, policy, project_id, payload=payload)
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


def _update_memory_record_view(*, app: Any, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy):
    def update_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(app, store, policy, project_id, payload=payload, record_id=record_id)
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


def _delete_memory_record_view(*, app: Any, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy):
    def delete_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        decision, error_response = _authorize_memory_write(app, store, policy, project_id, payload=payload, record_id=record_id)
        if error_response is not None:
            return error_response
        try:
            deleted = project_memory.delete_project_record(str(project_id or "").strip(), record_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "permission": decision}), 400
        return jsonify({"ok": True, "deleted": deleted, "permission": decision})

    return delete_project_memory_record


def _set_memory_enabled_view(*, app: Any, project_memory: ProjectMemory, store: ServerPermissionStore, policy: PermissionPolicy):
    def set_project_memory_enabled(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if not isinstance(payload.get("enabled"), bool):
            return jsonify({"ok": False, "error": "enabled boolean required"}), 400
        decision, error_response = _authorize_memory_write(app, store, policy, project_id, payload=payload)
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
    app: Any,
    store: ServerPermissionStore,
    policy: PermissionPolicy,
    project_id: str,
    *,
    payload: dict[str, Any],
    record_id: str = "",
) -> tuple[dict[str, Any], Any | None]:
    clean_project = str(project_id or "").strip()
    actor_context = resolve_actor(request)
    allow_legacy_client_flags = legacy_client_permission_flags_enabled(
        app,
        "AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS",
        default=False,
    )
    legacy_project_approved = _as_bool(payload.get("project_approved")) or _as_bool(request.headers.get("X-Project-Approved"))
    try:
        decision = decide_server_permission(
            store,
            policy,
            MEMORY_WRITE_ACTION,
            project_id=clean_project,
            actor=actor_context.actor,
            actor_source=actor_context.source,
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
