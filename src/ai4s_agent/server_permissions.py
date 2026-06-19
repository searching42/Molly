from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from flask import jsonify, request

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.memory import PermissionLevel, PermissionPolicy
from ai4s_agent.storage import ProjectStorage


class ServerPermissionStore:
    """Durable server-side permission grants and audit records."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.projects = ProjectStorage(workspace_dir=self.workspace_dir)

    def create_grant(self, project_id: str, action: str, *, actor: str, reason: str = "", run_id: str = "") -> dict[str, Any]:
        project = _clean_segment(project_id, "project_id")
        clean_action = _clean_action(action)
        clean_actor = str(actor or "").strip()
        if not clean_actor:
            raise ValueError("actor required for permission grant")
        grant = {
            "grant_id": f"grant-{clean_action}-{uuid.uuid4().hex[:12]}",
            "project_id": project,
            "run_id": str(run_id or "").strip(),
            "action": clean_action,
            "actor": clean_actor,
            "reason": str(reason or "").strip(),
            "created_at": now_iso(),
            "active": True,
        }
        grants = self.list_grants(project)
        grants.append(grant)
        write_json(self._grants_path(project), {"project_id": project, "updated_at": now_iso(), "grants": grants})
        self.audit_decision(project, action=clean_action, run_id=grant["run_id"], actor=clean_actor, allowed=True, reason="SERVER_GRANT_CREATED", grant_id=grant["grant_id"])
        return grant

    def list_grants(self, project_id: str, *, action: str = "") -> list[dict[str, Any]]:
        project = _clean_segment(project_id, "project_id")
        path = self._grants_path(project)
        if not path.exists():
            return []
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        grants = loaded.get("grants", []) if isinstance(loaded, dict) else []
        clean_action = _clean_action(action) if action else ""
        result: list[dict[str, Any]] = []
        for item in grants:
            if not isinstance(item, dict):
                continue
            record = {str(key): value for key, value in item.items()}
            if clean_action and str(record.get("action") or "") != clean_action:
                continue
            result.append(record)
        return result

    def find_grant(self, project_id: str, action: str, *, run_id: str = "") -> dict[str, Any] | None:
        clean_run = str(run_id or "").strip()
        for grant in reversed(self.list_grants(project_id, action=action)):
            if not bool(grant.get("active", True)):
                continue
            grant_run = str(grant.get("run_id") or "").strip()
            if grant_run and grant_run != clean_run:
                continue
            return grant
        return None

    def audit_decision(
        self,
        project_id: str,
        *,
        action: str,
        allowed: bool,
        reason: str,
        run_id: str = "",
        actor: str = "",
        grant_id: str = "",
        legacy_client_flag: bool = False,
    ) -> dict[str, Any]:
        project = _clean_segment(project_id, "project_id")
        record = {
            "decision_id": f"perm-{uuid.uuid4().hex[:12]}",
            "project_id": project,
            "run_id": str(run_id or "").strip(),
            "action": _clean_action(action),
            "allowed": bool(allowed),
            "reason": str(reason or "").strip(),
            "actor": str(actor or "").strip(),
            "grant_id": str(grant_id or "").strip(),
            "legacy_client_flag": bool(legacy_client_flag),
            "decided_at": now_iso(),
        }
        path = self._audit_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_audit(self, project_id: str) -> list[dict[str, Any]]:
        project = _clean_segment(project_id, "project_id")
        path = self._audit_path(project)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                records.append({str(key): value for key, value in loaded.items()})
        return records

    def _permission_dir(self, project_id: str) -> Path:
        project_dir = self.projects.project_dir(project_id)
        path = (project_dir / "permissions").resolve()
        if not path.is_relative_to(project_dir.resolve()):
            raise ValueError("permission path escapes project directory")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _grants_path(self, project_id: str) -> Path:
        return self._permission_dir(project_id) / "permission_grants.json"

    def _audit_path(self, project_id: str) -> Path:
        return self._permission_dir(project_id) / "permission_audit.jsonl"


def decide_server_permission(
    store: ServerPermissionStore,
    policy: PermissionPolicy,
    action: str,
    *,
    project_id: str,
    run_id: str = "",
    actor: str = "",
    confirmed: bool = False,
    legacy_project_approved: bool = False,
    allow_legacy_client_flags: bool = True,
) -> dict[str, Any]:
    clean_action = _clean_action(action)
    level = policy.resolve(clean_action)
    clean_actor = str(actor or "").strip()
    grant = store.find_grant(project_id, clean_action, run_id=run_id)
    allowed = False
    reason = ""
    grant_id = ""
    legacy = False
    if level == PermissionLevel.AUTO:
        allowed = True
        reason = "AUTO_ALLOWED"
    elif level == PermissionLevel.PROJECT_APPROVED:
        if grant:
            allowed = True
            reason = "SERVER_GRANT"
            grant_id = str(grant.get("grant_id") or "")
        elif legacy_project_approved and allow_legacy_client_flags:
            allowed = True
            reason = "LEGACY_CLIENT_PROJECT_APPROVED"
            legacy = True
        else:
            reason = "SERVER_GRANT_REQUIRED"
    else:
        if confirmed and clean_actor:
            allowed = True
            reason = "CONFIRMED"
        elif confirmed and not clean_actor:
            reason = "CONFIRMATION_ACTOR_REQUIRED"
        else:
            reason = "CONFIRMATION_REQUIRED"
    audit = store.audit_decision(
        project_id,
        action=clean_action,
        run_id=run_id,
        actor=clean_actor,
        allowed=allowed,
        reason=reason,
        grant_id=grant_id,
        legacy_client_flag=legacy,
    )
    return {
        "action": clean_action,
        "level": level.value,
        "allowed": allowed,
        "reason": reason,
        "project_id": project_id,
        "run_id": run_id,
        "actor": clean_actor,
        "grant_id": grant_id,
        "server_authorized": bool(grant_id or level == PermissionLevel.AUTO),
        "legacy_client_flag": legacy,
        "audit": audit,
    }


def install_server_permission_routes() -> None:
    import ai4s_agent.api as api_module

    original_register_routes = api_module.register_routes
    if getattr(original_register_routes, "_server_permission_routes", False):
        return

    def register_routes_with_server_permissions(app: Any, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
        original_register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        workspace = api_module._workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        store = ServerPermissionStore(workspace_dir=workspace)
        _add_permission_routes(app, store=store)

    register_routes_with_server_permissions._server_permission_routes = True  # type: ignore[attr-defined]
    api_module.register_routes = register_routes_with_server_permissions  # type: ignore[method-assign]


def _add_permission_routes(app: Any, *, store: ServerPermissionStore) -> None:
    app.add_url_rule(
        "/api/projects/<project_id>/permissions/grants",
        endpoint="create_permission_grant",
        view_func=_create_permission_grant_view(store=store),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/permissions/grants",
        endpoint="list_permission_grants",
        view_func=_list_permission_grants_view(store=store),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/permissions/audit",
        endpoint="list_permission_audit",
        view_func=_list_permission_audit_view(store=store),
        methods=["GET"],
    )


def _create_permission_grant_view(*, store: ServerPermissionStore):
    def create_permission_grant(project_id: str):
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action") or "").strip()
        actor = str(payload.get("actor") or payload.get("approved_by") or "").strip()
        confirmed = payload.get("confirmed") is True
        if not action:
            return jsonify({"ok": False, "error": "action required"}), 400
        if not confirmed or not actor:
            return jsonify({"ok": False, "error": "server permission grant requires confirmed=true and actor"}), 403
        try:
            grant = store.create_grant(project_id, action, actor=actor, reason=str(payload.get("reason") or ""), run_id=str(payload.get("run_id") or ""))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "grant": grant})

    return create_permission_grant


def _list_permission_grants_view(*, store: ServerPermissionStore):
    def list_permission_grants(project_id: str):
        try:
            grants = store.list_grants(project_id, action=str(request.args.get("action") or ""))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "project_id": project_id, "grants": grants})

    return list_permission_grants


def _list_permission_audit_view(*, store: ServerPermissionStore):
    def list_permission_audit(project_id: str):
        try:
            audit = store.read_audit(project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "project_id": project_id, "audit": audit})

    return list_permission_audit


def _clean_action(action: str) -> str:
    clean = str(action or "").strip()
    if not clean or any(ch in clean for ch in "/\\"):
        raise ValueError("permission action must be a safe action name")
    return clean


def _clean_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean in {".", ".."} or any(ch in clean for ch in "/\\"):
        raise ValueError(f"{label} must be a single safe path segment")
    return clean
