from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from flask import jsonify, request

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.memory import PermissionLevel, PermissionPolicy
from ai4s_agent.storage import ProjectStorage

if TYPE_CHECKING:
    from ai4s_agent.api_route_extensions import RouteExtensionContext


class ServerPermissionStore:
    """Durable server-side permission grants and audit records."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.projects = ProjectStorage(workspace_dir=self.workspace_dir)

    def create_grant(self, project_id: str, action: str, *, actor: str, reason: str = "", run_id: str = "", expires_at: str = "") -> dict[str, Any]:
        project = _clean_segment(project_id, "project_id")
        clean_action = _clean_action(action)
        clean_actor = str(actor or "").strip()
        if not clean_actor:
            raise ValueError("actor required for permission grant")
        clean_expires_at = _normalize_future_expires_at(expires_at)
        grant = {
            "grant_id": f"grant-{clean_action}-{uuid.uuid4().hex[:12]}",
            "project_id": project,
            "run_id": str(run_id or "").strip(),
            "action": clean_action,
            "actor": clean_actor,
            "reason": str(reason or "").strip(),
            "created_at": now_iso(),
            "expires_at": clean_expires_at,
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
            if _grant_is_expired(grant):
                continue
            return grant
        return None

    def revoke_grant(self, project_id: str, grant_id: str, *, revoked_by: str, revoke_reason: str = "") -> dict[str, Any]:
        project = _clean_segment(project_id, "project_id")
        clean_grant_id = str(grant_id or "").strip()
        if not clean_grant_id:
            raise ValueError("grant_id required")
        clean_actor = str(revoked_by or "").strip()
        if not clean_actor:
            raise ValueError("revoked_by required for permission revoke")
        grants = self.list_grants(project)
        changed: dict[str, Any] | None = None
        updated: list[dict[str, Any]] = []
        for item in grants:
            if str(item.get("grant_id") or "") == clean_grant_id:
                changed = dict(item)
                changed["active"] = False
                changed["revoked_at"] = now_iso()
                changed["revoked_by"] = clean_actor
                changed["revoke_reason"] = str(revoke_reason or "").strip()
                updated.append(changed)
            else:
                updated.append(item)
        if changed is None:
            raise ValueError(f"grant not found: {clean_grant_id}")
        write_json(self._grants_path(project), {"project_id": project, "updated_at": now_iso(), "grants": updated})
        self.audit_decision(
            project,
            action=str(changed.get("action") or ""),
            run_id=str(changed.get("run_id") or ""),
            actor=clean_actor,
            allowed=False,
            reason="SERVER_GRANT_REVOKED",
            grant_id=clean_grant_id,
        )
        return changed

    def find_expired_grant(self, project_id: str, action: str, *, run_id: str = "") -> dict[str, Any] | None:
        clean_run = str(run_id or "").strip()
        for grant in reversed(self.list_grants(project_id, action=action)):
            if not bool(grant.get("active", True)):
                continue
            grant_run = str(grant.get("run_id") or "").strip()
            if grant_run and grant_run != clean_run:
                continue
            if _grant_is_expired(grant):
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
    expired_grant = None if grant else store.find_expired_grant(project_id, clean_action, run_id=run_id)
    revoked_grant = _find_revoked_grant(store, project_id, clean_action, run_id=run_id)
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
        elif revoked_grant:
            reason = "REVOKED_GRANT"
            grant_id = str(revoked_grant.get("grant_id") or "")
        elif expired_grant:
            reason = "EXPIRED_GRANT"
            grant_id = str(expired_grant.get("grant_id") or "")
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
        "server_authorized": bool(level == PermissionLevel.AUTO or (allowed and grant_id)),
        "legacy_client_flag": legacy,
        "audit": audit,
    }


def install_server_permission_routes() -> None:
    """Server permission routes are installed by the explicit route hook."""


def apply_server_permission_routes(context: "RouteExtensionContext") -> None:
    import ai4s_agent.api as api_module

    workspace = api_module._workspace_from_config(
        base_runs_dir=context.base_runs_dir,
        workspace_dir=context.workspace_dir,
    )
    store = ServerPermissionStore(workspace_dir=workspace)
    _add_permission_routes(context, store=store)


def _add_permission_routes(
    context: "RouteExtensionContext",
    *,
    store: ServerPermissionStore,
) -> None:
    context.route_overrides.apply_new_route(
        context.app,
        extension_id="server_permission_routes",
        endpoint="create_permission_grant",
        rule="/api/projects/<project_id>/permissions/grants",
        view_func=_create_permission_grant_view(store=store),
        methods=("POST",),
    )
    context.route_overrides.apply_new_route(
        context.app,
        extension_id="server_permission_routes",
        endpoint="list_permission_grants",
        rule="/api/projects/<project_id>/permissions/grants",
        view_func=_list_permission_grants_view(store=store),
        methods=("GET",),
    )
    context.route_overrides.apply_new_route(
        context.app,
        extension_id="server_permission_routes",
        endpoint="list_permission_audit",
        rule="/api/projects/<project_id>/permissions/audit",
        view_func=_list_permission_audit_view(store=store),
        methods=("GET",),
    )
    context.route_overrides.apply_new_route(
        context.app,
        extension_id="server_permission_routes",
        endpoint="revoke_permission_grant",
        rule="/api/projects/<project_id>/permissions/grants/<grant_id>",
        view_func=_revoke_permission_grant_view(store=store),
        methods=("DELETE",),
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
            grant = store.create_grant(
                project_id,
                action,
                actor=actor,
                reason=str(payload.get("reason") or ""),
                run_id=str(payload.get("run_id") or ""),
                expires_at=str(payload.get("expires_at") or ""),
            )
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


def _revoke_permission_grant_view(*, store: ServerPermissionStore):
    def revoke_permission_grant(project_id: str, grant_id: str):
        actor = str(request.headers.get("X-Actor") or "").strip()
        revoked_by = str(request.form.get("actor") or request.form.get("revoked_by") or actor).strip()
        if not revoked_by:
            payload = request.get_json(silent=True) or {}
            revoked_by = str(payload.get("actor") or payload.get("revoked_by") or "").strip()
        if not revoked_by:
            return jsonify({"ok": False, "error": "revoked_by or actor required in request"}), 403
        revoke_reason = str(request.form.get("revoke_reason") or "")
        if not revoke_reason:
            payload = request.get_json(silent=True) or {}
            revoke_reason = str(payload.get("revoke_reason") or "").strip()
        try:
            grant = store.revoke_grant(
                project_id,
                grant_id,
                revoked_by=revoked_by,
                revoke_reason=revoke_reason,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "grant": grant})

    return revoke_permission_grant


def _find_revoked_grant(
    store: ServerPermissionStore,
    project_id: str,
    action: str,
    *,
    run_id: str = "",
) -> dict[str, Any] | None:
    """Return the most recently revoked matching grant, or None.

    Only considers revoked grants that are *not* shadowed by a newer active
    grant for the same scope.  Once an active grant is encountered while
    walking the list oldest-to-newest, earlier revoked grants are ignored.
    """
    clean_run = str(run_id or "").strip()
    found: dict[str, Any] | None = None
    for item in store.list_grants(project_id, action=action):
        grant_run = str(item.get("run_id") or "").strip()
        run_mismatch = bool(grant_run and grant_run != clean_run)
        if run_mismatch:
            continue
        if bool(item.get("active", True)):
            # A newer active grant overrides any earlier revoked grant.
            found = None
            continue
        if item.get("revoked_at"):
            found = item
    return found


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


def _normalize_future_expires_at(raw: str) -> str:
    clean = str(raw or "").strip()
    if not clean:
        return ""
    parsed = _parse_expires_at(clean)
    if parsed <= datetime.now(timezone.utc):
        raise ValueError("expires_at must be in the future")
    return parsed.isoformat().replace("+00:00", "Z")


def _grant_is_expired(grant: dict[str, Any]) -> bool:
    raw = str(grant.get("expires_at") or "").strip()
    if not raw:
        return False
    try:
        expires_at = _parse_expires_at(raw)
    except ValueError:
        return True
    return expires_at <= datetime.now(timezone.utc)


def _parse_expires_at(raw: str) -> datetime:
    clean = str(raw or "").strip()
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("expires_at must be an ISO timestamp with timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("expires_at must include timezone")
    return parsed.astimezone(timezone.utc)
