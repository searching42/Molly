from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.deployment import assess_multi_user_deployment
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.remote_worker import RemoteWorkerRegistry
from ai4s_agent.resource_profiles import (
    CapabilityProbeService,
    ConnectionProfile,
    ResourceProfileStore,
)
from ai4s_agent.schemas import RemoteWorkerConfig, RemoteWorkerRequest


def register_worker_deployment_routes(
    app: Flask,
    *,
    workspace: Path,
    runs: Path,
    user_config_dir: Path | None = None,
    resource_profiles: ResourceProfileStore | None = None,
) -> ResourceProfileStore:
    resource_profiles = resource_profiles or ResourceProfileStore(
        workspace_dir=workspace, config_dir=user_config_dir
    )
    capability_probe = CapabilityProbeService(store=resource_profiles)
    remote_workers = RemoteWorkerRegistry(
        workspace_dir=workspace,
        resource_profiles=resource_profiles,
    )
    app.extensions["resource_profile_store"] = resource_profiles
    app.extensions["capability_probe_service"] = capability_probe

    @app.get("/api/settings/compute")
    def get_compute_settings():
        try:
            state = resource_profiles.public_state()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return jsonify({"ok": True, **state})

    @app.put("/api/settings/compute/connections/<connection_id>")
    def save_connection_profile(connection_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if payload.get("connection_id") not in {None, "", connection_id}:
            return jsonify({"ok": False, "error": "connection_id does not match URL"}), 400
        try:
            connection = resource_profiles.save_connection(
                ConnectionProfile.model_validate({**payload, "connection_id": connection_id})
            )
        except ValidationError as exc:
            return jsonify({"ok": False, "error": _safe_validation_error(exc)}), 400
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "connection": {
                    **connection.model_dump(mode="json"),
                    "connection_profile_digest": connection.digest(),
                },
            }
        )

    @app.delete("/api/settings/compute/connections/<connection_id>")
    def delete_connection_profile(connection_id: str):
        try:
            deleted = resource_profiles.delete_connection(connection_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "deleted": deleted})

    @app.post("/api/settings/compute/connections/<connection_id>/probe")
    def probe_connection_profile(connection_id: str):
        try:
            result = app.extensions["capability_probe_service"].probe(connection_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "probe": result.model_dump(mode="json")})

    @app.get("/api/workers")
    def list_remote_workers():
        include_disabled = _as_bool(request.args.get("include_disabled"))
        try:
            workers = remote_workers.list_workers(include_disabled=include_disabled)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return jsonify({"ok": True, "workers": [worker.model_dump(mode="json") for worker in workers]})

    @app.post("/api/workers")
    def save_remote_worker():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            worker = remote_workers.save_worker(RemoteWorkerConfig.model_validate(payload))
        except ValidationError as exc:
            return jsonify({"ok": False, "error": _safe_validation_error(exc)}), 400
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "worker": worker.model_dump(mode="json")})

    @app.post("/api/workers/assignment")
    def plan_remote_worker_assignment():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            assignment = remote_workers.plan_assignment(RemoteWorkerRequest.model_validate(payload))
        except ValidationError as exc:
            return jsonify({"ok": False, "error": _safe_validation_error(exc)}), 400
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "assignment": assignment.model_dump(mode="json")})

    @app.get("/api/deployment/multi-user-readiness")
    def multi_user_readiness():
        readiness = assess_multi_user_deployment(workspace_dir=workspace, runs_dir=runs)
        return jsonify({"ok": True, "readiness": readiness.model_dump(mode="json")})

    @app.get("/api/atomic-tasks")
    def list_atomic_tasks():
        registry = AtomicTaskRegistry()
        return jsonify(
            {
                "ok": True,
                "tasks": [task.model_dump(mode="json") for task in registry.list_tasks()],
            }
        )

    return resource_profiles


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}


def _safe_validation_error(error: ValidationError) -> str:
    details = error.errors(include_input=False, include_url=False)
    if not details:
        return "invalid request payload"
    first = details[0]
    location = ".".join(str(item) for item in first.get("loc", ())) or "payload"
    message = str(first.get("msg") or "is invalid")
    return f"invalid {location}: {message}"
