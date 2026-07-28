from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.remote_execution_lifecycle import RemoteExecutionLifecycleService


def register_remote_execution_routes(
    app: Flask, *, executions: RemoteExecutionLifecycleService
) -> None:
    @app.post("/api/projects/<project_id>/remote-executions")
    def prepare_remote_execution(project_id: str):
        try:
            payload = _payload()
            result = executions.prepare(
                project_id=project_id,
                run_id=str(payload.get("run_id") or ""),
                task_id=str(payload.get("task_id") or ""),
                transfer_manifest=_object(payload, "transfer_manifest"),
                requested_resources=_object(payload, "requested_resources"),
                input_artifacts=_object(payload, "input_artifacts"),
            )
            return jsonify({"ok": True, "remote_execution": result}), 201
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)

    @app.get("/api/projects/<project_id>/remote-executions/<run_id>")
    def inspect_remote_execution(project_id: str, run_id: str):
        try:
            result = executions.inspect(project_id=project_id, run_id=run_id)
            return jsonify({"ok": True, "remote_execution": result})
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)

    @app.post("/api/projects/<project_id>/remote-executions/<run_id>/approve")
    def approve_remote_execution(project_id: str, run_id: str):
        try:
            payload = _payload()
            result = executions.approve(
                project_id=project_id,
                run_id=run_id,
                request_sha256=str(payload.get("request_sha256") or ""),
                actor=str(payload.get("actor") or ""),
                note=str(payload.get("note") or ""),
            )
            return jsonify({"ok": True, "remote_execution": result}), 202
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)

    @app.post("/api/projects/<project_id>/remote-executions/<run_id>/refresh")
    def refresh_remote_execution(project_id: str, run_id: str):
        try:
            result = executions.refresh(project_id=project_id, run_id=run_id)
            return jsonify({"ok": True, "remote_execution": result})
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)

    @app.post("/api/projects/<project_id>/remote-executions/<run_id>/cancel")
    def cancel_remote_execution(project_id: str, run_id: str):
        try:
            payload = _payload()
            result = executions.cancel(
                project_id=project_id,
                run_id=run_id,
                request_sha256=str(payload.get("request_sha256") or ""),
            )
            return jsonify({"ok": True, "remote_execution": result}), 202
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)

    @app.post("/api/projects/<project_id>/remote-executions/<run_id>/recover")
    def recover_remote_execution(project_id: str, run_id: str):
        try:
            result = executions.recover(project_id=project_id, run_id=run_id)
            return jsonify({"ok": True, "remote_execution": result})
        except (OSError, ValueError, ValidationError) as exc:
            return _error(exc)


def _payload() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return {str(key): item for key, item in value.items()}


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} object required")
    return value


def _error(exc: Exception):
    message = str(exc) or exc.__class__.__name__
    status = 409 if any(
        marker in message
        for marker in (
            "already progressed",
            "already differs",
            "does not require recovery",
            "changed after request",
            "conflicts with Artifact Registry",
        )
    ) else 400
    if "unavailable" in message:
        status = 404
    return jsonify({"ok": False, "error": message}), status


__all__ = ["register_remote_execution_routes"]
