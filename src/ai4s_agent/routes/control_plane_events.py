"""Read-only snapshot and SSE routes for control-plane event projections."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context

from ai4s_agent._utils import now_iso
from ai4s_agent.control_plane_events import ControlPlaneEventProjector


_POLL_SECONDS = 0.75


def register_control_plane_event_routes(
    app: Flask,
    *,
    projector: ControlPlaneEventProjector,
) -> None:
    base = (
        "/api/projects/<project_id>/oled-bounded-sessions/"
        "<session_id>/event-projection"
    )

    @app.get(base)
    def inspect_control_plane_event_projection(project_id: str, session_id: str):
        try:
            projection = projector.project(
                project_id=project_id,
                session_id=session_id,
                after_event_id=_cursor(),
            )
            response = jsonify({"ok": True, **projection})
            response.headers["Cache-Control"] = "no-store"
            return response
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get(base + "/events")
    def stream_control_plane_events(project_id: str, session_id: str):
        try:
            after = _cursor()
            initial = projector.project(
                project_id=project_id,
                session_id=session_id,
                after_event_id=after,
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        once = str(request.args.get("once") or "").strip().lower() in {"1", "true", "yes"}

        @stream_with_context
        def generate() -> Iterator[str]:
            nonlocal initial
            cursor = after
            yield "retry: 1500\n\n"
            yield _sse(
                event="snapshot",
                data={
                    "snapshot": initial["snapshot"],
                    "cursor": initial["cursor"],
                    "authority": initial["authority"],
                },
            )
            while True:
                for event in initial["durable_events"]:
                    cursor = int(event["event_id"])
                    yield _sse(
                        event=str(event["event_type"]),
                        data=event,
                        event_id=cursor,
                    )
                yield _sse(
                    event="heartbeat",
                    data={
                        "schema_version": "control_plane_ephemeral_delta.v1",
                        "event_type": "heartbeat",
                        "durable": False,
                        "observed_at": now_iso(),
                    },
                )
                if once:
                    return
                time.sleep(_POLL_SECONDS)
                try:
                    initial = projector.project(
                        project_id=project_id,
                        session_id=session_id,
                        after_event_id=cursor,
                    )
                except Exception:
                    yield _sse(
                        event="observer.error",
                        data={
                            "schema_version": "control_plane_ephemeral_delta.v1",
                            "event_type": "observer.error",
                            "durable": False,
                            "message": "event projection is temporarily unavailable",
                        },
                    )
                    return

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Accel-Buffering"] = "no"
        return response


def _cursor() -> int:
    raw: Any = request.headers.get("Last-Event-ID")
    if raw is None or str(raw).strip() == "":
        raw = request.args.get("after", "0")
    clean = str(raw).strip()
    if not clean.isdigit():
        raise ValueError("Last-Event-ID must be a non-negative integer")
    return int(clean)


def _sse(*, event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines.extend(f"data: {line}" for line in encoded.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


__all__ = ["register_control_plane_event_routes"]
