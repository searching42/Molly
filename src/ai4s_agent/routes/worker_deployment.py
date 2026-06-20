from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.deployment import assess_multi_user_deployment
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.remote_worker import RemoteWorkerRegistry
from ai4s_agent.schemas import RemoteWorkerConfig, RemoteWorkerRequest


def register_worker_deployment_routes(app: Flask, *, workspace: Path, runs: Path) -> None:
    remote_workers = RemoteWorkerRegistry(workspace_dir=workspace)

    @app.get("/api/workers")
    def list_remote_workers():
        include_disabled = _as_bool(request.args.get("include_disabled"))
        workers = remote_workers.list_workers(include_disabled=include_disabled)
        return jsonify({"ok": True, "workers": [worker.model_dump(mode="json") for worker in workers]})

    @app.post("/api/workers")
    def save_remote_worker():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            worker = remote_workers.save_worker(RemoteWorkerConfig.model_validate(payload))
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "worker": worker.model_dump(mode="json")})

    @app.post("/api/workers/assignment")
    def plan_remote_worker_assignment():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            assignment = remote_workers.plan_assignment(RemoteWorkerRequest.model_validate(payload))
        except (ValidationError, ValueError) as exc:
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}
