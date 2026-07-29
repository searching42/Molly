"""PR-AW API and page routes for bounded OLED discovery sessions."""

from __future__ import annotations

import re
from typing import Any

from flask import Flask, after_this_request, jsonify, render_template, request

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
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    _verified_oled_scientific_agent_failure_attribution,
)
from ai4s_agent.oled_scientific_agent_trajectory_inspection import (
    InspectionLimitError,
    InspectionRequestError,
    build_oled_scientific_agent_trajectory_inspection,
    parse_inspection_filters,
)
from ai4s_agent.storage import ProjectStorage


_INSPECTION_PUBLICATION_IDS = {
    "trajectory_publication_id": re.compile(
        r"^scientific-agent-trajectory-publication:[0-9a-f]{64}$"
    ),
    "audit_publication_id": re.compile(
        r"^scientific-agent-trajectory-audit-publication:[0-9a-f]{64}$"
    ),
    "attribution_publication_id": re.compile(
        r"^scientific-agent-failure-attribution-publication:[0-9a-f]{64}$"
    ),
}
_INSPECTION_REQUIRED = frozenset(_INSPECTION_PUBLICATION_IDS)


def register_oled_bounded_session_routes(
    app: Flask,
    *,
    projects: ProjectStorage,
    actions: OledBoundedDiscoverySessionActionService,
) -> None:
    @app.get("/oled-bounded-sessions")
    def oled_bounded_sessions_page():
        @after_this_request
        def prevent_local_token_cache(response):
            response.headers["Cache-Control"] = "no-store"
            return response

        return render_template(
            "oled_bounded_sessions.html",
            local_session_token=app.config["MOLLY_LOCAL_SESSION_TOKEN"],
        )

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

    @app.get(
        "/api/projects/<project_id>/oled-bounded-sessions/<session_id>/trajectory-inspect"
    )
    def inspect_oled_bounded_session_trajectory(project_id: str, session_id: str):
        """Return one ephemeral view built inside the three-source verifier seam."""

        @after_this_request
        def prevent_inspection_cache(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

        try:
            clean_project = validated_oled_bounded_project_id(project_id)
            if not re.fullmatch(r"oled-bounded-session-[0-9a-f]{64}", session_id):
                raise InspectionRequestError("session_id is invalid")
            values = request.args.to_dict(flat=False)
            if any(len(items) != 1 for items in values.values()):
                raise InspectionRequestError("query parameters must be unique")
            missing = _INSPECTION_REQUIRED - set(values)
            if missing:
                raise InspectionRequestError("observer publication IDs are required")
            publication_ids = {
                name: str(values.pop(name)[0]) for name in sorted(_INSPECTION_REQUIRED)
            }
            for name, value in publication_ids.items():
                if not _INSPECTION_PUBLICATION_IDS[name].fullmatch(value):
                    raise InspectionRequestError("observer publication ID is invalid")
            filters = parse_inspection_filters(
                {name: str(items[0]) for name, items in values.items()}
            )
            project_dir = projects.projects_root / clean_project
            trajectory_dir = (
                project_dir
                / "trajectory-projections"
                / publication_ids["trajectory_publication_id"]
            )
            audit_dir = (
                project_dir
                / "trajectory-audits"
                / publication_ids["audit_publication_id"]
            )
            attribution_dir = (
                project_dir
                / "trajectory-failure-attributions"
                / publication_ids["attribution_publication_id"]
            )
            if not all(
                path.is_dir() and not path.is_symlink()
                for path in (trajectory_dir, audit_dir, attribution_dir)
            ):
                return _inspection_error(
                    "observer_publication_unavailable",
                    "The requested observer publication is unavailable.",
                    404,
                )
            with _verified_oled_scientific_agent_failure_attribution(
                storage=projects,
                project_id=clean_project,
                session_id=session_id,
                actions_root=actions.actions_root,
                trajectory_publication_dir=trajectory_dir,
                audit_publication_dir=audit_dir,
                attribution_publication_dir=attribution_dir,
            ) as bound:
                payload = build_oled_scientific_agent_trajectory_inspection(
                    project_id=clean_project,
                    session_id=session_id,
                    bound=bound,
                    filters=filters,
                )
            return jsonify(payload)
        except InspectionLimitError:
            return _inspection_error(
                "inspection_response_limit_exceeded",
                "Inspection response limit exceeds the allowed maximum.",
                400,
            )
        except InspectionRequestError:
            return _inspection_error(
                "invalid_inspection_request",
                "Inspection identifiers or filters are invalid.",
                400,
            )
        except (OSError, ValueError) as exc:
            app.logger.warning("trajectory inspection verification failed", exc_info=True)
            message = str(exc).lower()
            mismatch = any(
                marker in message
                for marker in (
                    "chain mismatch",
                    "binding mismatch",
                    "session mismatch",
                    "identity mismatch",
                )
            )
            return _inspection_error(
                "observer_publication_chain_mismatch"
                if mismatch
                else "observer_publication_integrity_failure",
                "The observer publication chain could not be verified.",
                409,
            )

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

    @app.post(
        "/api/projects/<project_id>/oled-bounded-session-actions/"
        "<action_id>/recover"
    )
    def recover_oled_bounded_session_action(project_id: str, action_id: str):
        try:
            action = actions.recover_interrupted_action(
                project_id=project_id,
                action_id=action_id,
            )
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
        "not interrupted and recoverable",
        "still owned by a live worker",
        "not backed by a completed publication",
    )
    status = 409 if any(marker in message for marker in conflict_markers) else 400
    if "unavailable" in message:
        status = 404
    return jsonify({"ok": False, "error": message}), status


def _inspection_error(code: str, message: str, status: int):
    return jsonify({"ok": False, "error_code": code, "error": message}), status


__all__ = ["register_oled_bounded_session_routes"]
