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
from ai4s_agent.runtime_environments import (
    RuntimeEnvironmentProfile,
    RuntimeEnvironmentStore,
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
    runtime_environments = RuntimeEnvironmentStore(config_dir=user_config_dir)
    remote_workers = RemoteWorkerRegistry(
        workspace_dir=workspace,
        resource_profiles=resource_profiles,
    )
    app.extensions["resource_profile_store"] = resource_profiles
    app.extensions["capability_probe_service"] = capability_probe
    app.extensions["runtime_environment_store"] = runtime_environments

    @app.get("/api/settings/compute")
    def get_compute_settings():
        try:
            state = resource_profiles.public_state()
            environments = runtime_environments.list_environments()
        except ValueError:
            app.logger.warning("compute settings verification failed", exc_info=True)
            return _compute_error(
                "compute_settings_unavailable",
                "Compute settings failed integrity verification.",
                409,
            )
        return jsonify(
            {
                "ok": True,
                **state,
                "environments": [
                    {
                        **environment.model_dump(mode="json"),
                        "environment_profile_digest": environment.digest(),
                    }
                    for environment in environments
                ],
            }
        )

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
        except ValidationError:
            return _compute_error(
                "connection_profile_invalid",
                "Connection profile fields are invalid.",
                400,
            )
        except ValueError:
            app.logger.warning("connection profile save failed", exc_info=True)
            return _compute_error(
                "connection_profile_unavailable",
                "The connection profile could not be saved.",
                400,
            )
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
        except ValueError:
            app.logger.warning("connection profile delete failed", exc_info=True)
            return _compute_error(
                "connection_profile_delete_failed",
                "The connection profile could not be deleted.",
                400,
            )
        return jsonify({"ok": True, "deleted": deleted})

    @app.put("/api/settings/compute/environments/<environment_id>")
    def save_runtime_environment_profile(environment_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if payload.get("environment_id") not in {None, "", environment_id}:
            return jsonify({"ok": False, "error": "environment_id does not match URL"}), 400
        try:
            environment = RuntimeEnvironmentProfile.model_validate(
                {**payload, "environment_id": environment_id}
            )
            connection = resource_profiles.get_connection(environment.connection_id)
            if not connection.enabled:
                raise ValueError("runtime environment connection profile is disabled")
            saved = runtime_environments.save_environment(environment)
        except ValidationError:
            return _compute_error(
                "runtime_environment_invalid",
                "Runtime environment fields are invalid.",
                400,
            )
        except ValueError:
            app.logger.warning("runtime environment save failed", exc_info=True)
            return _compute_error(
                "runtime_environment_unavailable",
                "The runtime environment could not be saved.",
                400,
            )
        return jsonify(
            {
                "ok": True,
                "environment": {
                    **saved.model_dump(mode="json"),
                    "environment_profile_digest": saved.digest(),
                },
            }
        )

    @app.delete("/api/settings/compute/environments/<environment_id>")
    def delete_runtime_environment_profile(environment_id: str):
        try:
            deleted = runtime_environments.delete_environment(environment_id)
        except ValueError:
            app.logger.warning("runtime environment delete failed", exc_info=True)
            return _compute_error(
                "runtime_environment_delete_failed",
                "The runtime environment could not be deleted.",
                400,
            )
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


def _compute_error(code: str, message: str, status: int):
    return jsonify({"ok": False, "error": message, "error_code": code}), status
