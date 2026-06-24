from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, current_app, jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import truthy
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.run_plan_queue_summary import build_run_plan_queue_execution_summary
from ai4s_agent.run_plan_task_runner import ExecutorFactory
from ai4s_agent.schemas import RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG = "AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"
EXECUTOR_FACTORY_CONFIG_KEY = "AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"


def register_internal_run_plan_queue_routes(app: Flask, *, projects: ProjectStorage) -> None:
    @app.post("/api/internal/run-plan/queue/execute")
    def internal_run_plan_queue_execute():
        if not internal_run_plan_queue_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal run-plan queue route disabled"}), 404
        try:
            payload = _request_json_object()
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            run_plan_payload = payload.get("run_plan")
            if not isinstance(run_plan_payload, dict):
                raise ValueError("run_plan object required")
            run_plan = RunPlan.model_validate(run_plan_payload)
            run_id = _safe_path_component(run_plan.run_id, "run_id")
            input_artifacts = _optional_object(payload.get("input_artifacts"), "input_artifacts")
            task_options = _optional_object(payload.get("task_options"), "task_options")
            if "queue_dir" in payload:
                raise ValueError("queue_dir is not accepted by the internal run-plan queue route")
            queue = WorkerQueue(JsonWorkerQueueStore(_queue_dir(projects, project_id, run_id)))
            summary = run_run_plan_via_local_queue(
                queue=queue,
                storage=projects,
                project_id=project_id,
                run_plan=run_plan,
                input_artifacts=input_artifacts,
                task_options=task_options,
                max_iterations=_max_iterations(payload.get("max_iterations")),
                executor_factory=_executor_factory(current_app),
            )
            return jsonify(summary), 200 if bool(summary.get("ok")) and bool(summary.get("terminal")) else 400
        except (OSError, ValidationError, ValueError) as exc:
            return jsonify(_error_summary(exc)), 400


def internal_run_plan_queue_route_enabled(app: Any) -> bool:
    if INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG in app.config:
        return truthy(app.config.get(INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG))
    env_value = os.environ.get(INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG)
    return truthy(env_value)


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _safe_path_component(value: object, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    if clean in {".", ".."} or "/" in clean or "\\" in clean:
        raise ValueError(f"{label} must be a safe path component")
    return clean


def _optional_object(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _max_iterations(value: object) -> int:
    if value is None:
        return 10
    if isinstance(value, bool):
        raise ValueError("max_iterations must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max_iterations must be positive")
    return parsed


def _queue_dir(projects: ProjectStorage, project_id: str, run_id: str) -> Path:
    base = (projects.workspace_dir / ".ai4s_internal" / "run_plan_queues").resolve()
    queue_dir = (base / project_id / run_id).resolve()
    if queue_dir != base and not queue_dir.is_relative_to(base):
        raise ValueError("internal queue path must stay under workspace")
    return queue_dir


def _executor_factory(app: Any) -> ExecutorFactory | None:
    factory = app.config.get(EXECUTOR_FACTORY_CONFIG_KEY)
    return factory if callable(factory) else None


def _error_summary(exc: BaseException) -> dict[str, Any]:
    return build_run_plan_queue_execution_summary(
        ok=False,
        terminal=False,
        error={
            "type": "validation_error",
            "message": str(exc).strip() or exc.__class__.__name__,
        },
    )
