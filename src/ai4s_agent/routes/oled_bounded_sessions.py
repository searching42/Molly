"""PR-AW API and page routes for bounded OLED discovery sessions."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from ai4s_agent.oled_bounded_discovery_session import (
    create_oled_bounded_discovery_session,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    build_oled_bounded_discovery_session_view,
    validated_oled_bounded_project_id,
)
from ai4s_agent.storage import ProjectStorage


def register_oled_bounded_session_routes(
    app: Flask,
    *,
    projects: ProjectStorage,
    actions: OledBoundedDiscoverySessionActionService,
) -> None:
    @app.get("/oled-bounded-sessions")
    def oled_bounded_sessions_page():
        return render_template("oled_bounded_sessions.html")

    @app.get("/api/projects/<project_id>/oled-bounded-sessions")
    def list_oled_bounded_sessions(project_id: str):
        try:
            clean_project = validated_oled_bounded_project_id(project_id)
            root = projects.project_dir(clean_project) / "bounded-discovery-sessions"
            session_ids = (
                sorted(
                    child.name
                    for child in root.iterdir()
                    if child.is_dir()
                    and not child.is_symlink()
                    and child.name.startswith("oled-bounded-session-")
                )
                if root.is_dir()
                else []
            )
            sessions = [
                build_oled_bounded_discovery_session_view(
                    storage=projects,
                    project_id=clean_project,
                    session_id=session_id,
                )
                for session_id in session_ids
            ]
            return jsonify(
                {"ok": True, "project_id": clean_project, "sessions": sessions}
            )
        except (OSError, ValueError) as exc:
            return _error(exc)

    @app.post("/api/projects/<project_id>/oled-bounded-sessions")
    def create_oled_bounded_session(project_id: str):
        try:
            clean_project = validated_oled_bounded_project_id(project_id)
            payload = _json_object()
            spec = payload.get("session_spec")
            if not isinstance(spec, dict):
                raise ValueError("session_spec object required")
            created = create_oled_bounded_discovery_session(
                storage=projects,
                project_id=clean_project,
                session_spec=spec,
            )
            view = build_oled_bounded_discovery_session_view(
                storage=projects,
                project_id=clean_project,
                session_id=created.session_id,
            )
            return jsonify({"ok": True, "session": view}), 201
        except (OSError, ValueError) as exc:
            return _error(exc)

    @app.get(
        "/api/projects/<project_id>/oled-bounded-sessions/<session_id>"
    )
    def inspect_oled_bounded_session(project_id: str, session_id: str):
        try:
            clean_project = validated_oled_bounded_project_id(project_id)
            view = build_oled_bounded_discovery_session_view(
                storage=projects,
                project_id=clean_project,
                session_id=session_id,
            )
            return jsonify({"ok": True, "session": view})
        except (OSError, ValueError) as exc:
            return _error(exc)

    @app.post(
        "/api/projects/<project_id>/oled-bounded-sessions/<session_id>/actions/advance"
    )
    def advance_oled_bounded_session(project_id: str, session_id: str):
        try:
            payload = _json_object()
            action = actions.enqueue_advance(
                project_id=project_id,
                session_id=session_id,
                expected_revision=_revision(payload),
            )
            return jsonify({"ok": True, "action": action}), 202
        except (OSError, ValueError) as exc:
            return _error(exc)

    @app.post(
        "/api/projects/<project_id>/oled-bounded-sessions/<session_id>/actions/approve"
    )
    def approve_oled_bounded_session(project_id: str, session_id: str):
        try:
            payload = _json_object()
            action = actions.enqueue_approval(
                project_id=project_id,
                session_id=session_id,
                expected_revision=_revision(payload),
                actor=str(payload.get("actor") or ""),
                note=str(payload.get("note") or ""),
            )
            return jsonify({"ok": True, "action": action}), 202
        except (OSError, ValueError) as exc:
            return _error(exc)

    @app.get(
        "/api/projects/<project_id>/oled-bounded-session-actions/<action_id>"
    )
    def inspect_oled_bounded_session_action(project_id: str, action_id: str):
        try:
            action = actions.get_action(project_id=project_id, action_id=action_id)
            return jsonify({"ok": True, "action": action})
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValueError) as exc:
            return _error(exc)


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    return {str(key): value for key, value in payload.items()}


def _revision(payload: dict[str, Any]) -> int:
    value = payload.get("expected_revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected_revision must be a non-negative integer")
    return value


def _error(exc: Exception):
    message = str(exc) or exc.__class__.__name__
    conflict_markers = (
        "revision conflict",
        "already has an active",
        "requires gate approval",
        "not waiting for approval",
        "terminal session",
    )
    status = 409 if any(marker in message for marker in conflict_markers) else 400
    if "unavailable" in message:
        status = 404
    return jsonify({"ok": False, "error": message}), status


__all__ = ["register_oled_bounded_session_routes"]
