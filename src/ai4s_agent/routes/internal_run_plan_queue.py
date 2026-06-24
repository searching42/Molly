from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, current_app, jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import now_iso, truthy
from ai4s_agent.actor_identity import ActorContext, resolve_actor
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
        payload: dict[str, Any] = {}
        actor = ActorContext(actor="", source="missing", required=True)
        project_id = ""
        run_id = ""
        try:
            payload = _request_json_object()
            project_id = _audit_string(payload.get("project_id"))
            run_plan_payload = payload.get("run_plan")
            if isinstance(run_plan_payload, dict):
                run_id = _audit_string(run_plan_payload.get("run_id"))
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            if not isinstance(run_plan_payload, dict):
                raise ValueError("run_plan object required")
            run_plan = RunPlan.model_validate(run_plan_payload)
            run_id = _safe_path_component(run_plan.run_id, "run_id")
            input_artifacts = _optional_object(payload.get("input_artifacts"), "input_artifacts")
            task_options = _optional_object(payload.get("task_options"), "task_options")
            if "queue_dir" in payload:
                raise ValueError("queue_dir is not accepted by the internal run-plan queue route")
            max_iterations = _max_iterations(payload.get("max_iterations"))
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome="requested",
                status_code=202,
                queued_job_id="",
                error=None,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            queue = WorkerQueue(JsonWorkerQueueStore(_queue_dir(projects, project_id, run_id)))
            summary = run_run_plan_via_local_queue(
                queue=queue,
                storage=projects,
                project_id=project_id,
                run_plan=run_plan,
                input_artifacts=input_artifacts,
                task_options=task_options,
                max_iterations=max_iterations,
                executor_factory=_executor_factory(current_app),
            )
            status_code = 200 if bool(summary.get("ok")) and bool(summary.get("terminal")) else 400
            outcome = "succeeded" if status_code == 200 else "failed"
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome=outcome,
                status_code=status_code,
                queued_job_id=str(summary.get("queued_job_id") or ""),
                error=_summary_error(summary),
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(summary), status_code
        except (OSError, ValidationError, ValueError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            summary = _error_summary(exc)
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome="validation_error",
                status_code=status_code,
                queued_job_id="",
                error=_summary_error_dict(exc),
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(summary), status_code


class _RouteRequestError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


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


def _audit_string(value: object) -> str:
    return str(value or "").strip()


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


def _append_audit_or_error(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    outcome: str,
    status_code: int,
    queued_job_id: str,
    error: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        _append_audit_event(
            projects,
            actor=actor,
            project_id=project_id,
            run_id=run_id,
            outcome=outcome,
            status_code=status_code,
            queued_job_id=queued_job_id,
            error=error,
        )
    except OSError as exc:
        return build_run_plan_queue_execution_summary(
            ok=False,
            terminal=False,
            error={
                "type": "audit_write_failed",
                "message": str(exc).strip() or exc.__class__.__name__,
            },
        )
    return None


def _append_audit_event(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    outcome: str,
    status_code: int,
    queued_job_id: str,
    error: dict[str, Any] | None,
) -> None:
    record = {
        "event": "internal_run_plan_queue_execute",
        "timestamp": now_iso(),
        "actor": actor.actor,
        "actor_source": actor.source,
        "project_id": project_id,
        "run_id": run_id,
        "route": "/api/internal/run-plan/queue/execute",
        "feature_flag_enabled": True,
        "outcome": outcome,
        "status_code": int(status_code),
        "queued_job_id": queued_job_id,
        "error": error,
    }
    path = _audit_path(projects)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _audit_path(projects: ProjectStorage) -> Path:
    base = (projects.workspace_dir / ".ai4s_internal" / "audit").resolve()
    path = (base / "internal_run_plan_queue_audit.jsonl").resolve()
    if path.parent != base:
        raise ValueError("internal run-plan queue audit path must stay under workspace")
    return path


def _summary_error(summary: dict[str, Any]) -> dict[str, Any] | None:
    raw_error = summary.get("error")
    if isinstance(raw_error, dict):
        return {
            "type": str(raw_error.get("type") or "execution_error"),
            "message": str(raw_error.get("message") or "execution failed"),
        }
    final_job = summary.get("final_job")
    if isinstance(final_job, dict):
        job_error = final_job.get("error")
        if isinstance(job_error, dict):
            message = str(job_error.get("reason") or job_error.get("message") or "").strip()
            if message:
                return {"type": "execution_error", "message": message}
    return None


def _summary_error_dict(exc: BaseException) -> dict[str, Any]:
    return {
        "type": "validation_error",
        "message": str(exc).strip() or exc.__class__.__name__,
    }


def _error_summary(exc: BaseException) -> dict[str, Any]:
    return build_run_plan_queue_execution_summary(
        ok=False,
        terminal=False,
        error=_summary_error_dict(exc),
    )
