from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.job_manager import JobManager
from ai4s_agent.planner import build_plan, diff_run_plans, expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage


def register_run_plan_routes(app: Flask, *, projects: ProjectStorage, jobs: JobManager) -> None:
    @app.post("/api/run-plan/expand")
    def expand_plan_preview():
        payload = request.get_json(silent=True) or {}
        try:
            run_plan = _expand_plan_from_payload(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "run_plan": run_plan.model_dump(mode="json")})

    @app.post("/api/run-plan/diff")
    def diff_plan_preview():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "preview").strip() or "preview"
        try:
            before = run_plan_from_payload(payload.get("before"), run_id=run_id)
            after = run_plan_from_payload(payload.get("after"), run_id=run_id)
            diff = diff_run_plans(before, after)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "before": before.model_dump(mode="json"),
                "after": after.model_dump(mode="json"),
                "diff": diff.model_dump(mode="json"),
            }
        )

    @app.post("/api/run-plan/regenerate")
    def regenerate_plan_preview():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        plan = build_plan(run_id=run_id, prompt=prompt)
        requested = payload.get("requested_tasks")
        if not isinstance(requested, list) or not requested:
            return jsonify({"ok": False, "error": "requested_tasks required"}), 400
        try:
            run_plan = expand_run_plan(
                run_id=run_id,
                requested_tasks=[str(task) for task in requested if str(task).strip()],
                available_artifacts=_string_list(payload.get("available_artifacts")),
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "plan": plan.model_dump(mode="json"),
                "run_plan": run_plan.model_dump(mode="json"),
            }
        )

    @app.post("/api/run-plan/execute")
    def execute_run_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        run_plan_payload = payload.get("run_plan")
        if not isinstance(run_plan_payload, dict):
            return jsonify({"ok": False, "error": "run_plan object required"}), 400
        input_artifacts = payload.get("input_artifacts", {})
        if input_artifacts is None:
            input_artifacts = {}
        if not isinstance(input_artifacts, dict):
            return jsonify({"ok": False, "error": "input_artifacts must be an object"}), 400
        task_options = payload.get("task_options", {})
        if task_options is None:
            task_options = {}
        if not isinstance(task_options, dict):
            return jsonify({"ok": False, "error": "task_options must be an object"}), 400
        run_plan: RunPlan | None = None
        try:
            run_plan = RunPlan.model_validate(run_plan_payload)
            jobs.add_log(run_plan.run_id, "INFO", "run_plan", "RunPlan execution started")
            execution = RunPlanExecutor(storage=projects).execute(
                project_id=project_id,
                run_plan=run_plan,
                input_artifacts={str(k): str(v) for k, v in input_artifacts.items()},
                task_options=_task_options(task_options),
            )
            _log_run_plan_execution_result(jobs, run_plan.run_id, execution)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            if run_plan is not None:
                jobs.add_log(run_plan.run_id, "ERROR", "run_plan", f"RunPlan execution failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution": execution})

    @app.post("/api/run-plan/resume")
    def resume_run_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        actor = str(payload.get("actor") or payload.get("approved_by") or "").strip()
        if not actor:
            return jsonify({"ok": False, "error": "actor required"}), 400
        run_plan_payload = payload.get("run_plan")
        if not isinstance(run_plan_payload, dict):
            return jsonify({"ok": False, "error": "run_plan object required"}), 400
        approved_gates = payload.get("approved_gates", [])
        if approved_gates is None:
            approved_gates = []
        if not isinstance(approved_gates, list):
            return jsonify({"ok": False, "error": "approved_gates must be a list"}), 400
        input_artifacts = payload.get("input_artifacts", {})
        if input_artifacts is None:
            input_artifacts = {}
        if not isinstance(input_artifacts, dict):
            return jsonify({"ok": False, "error": "input_artifacts must be an object"}), 400
        task_options = payload.get("task_options", {})
        if task_options is None:
            task_options = {}
        if not isinstance(task_options, dict):
            return jsonify({"ok": False, "error": "task_options must be an object"}), 400
        run_plan: RunPlan | None = None
        try:
            run_plan = RunPlan.model_validate(run_plan_payload)
            jobs.add_log(run_plan.run_id, "INFO", "run_plan", "RunPlan resume requested")
            execution = RunPlanExecutor(storage=projects).resume_after_gate(
                project_id=project_id,
                run_plan=run_plan,
                approved_gates=[str(gate) for gate in approved_gates],
                actor=actor,
                note=str(payload.get("note") or ""),
                input_artifacts={str(k): str(v) for k, v in input_artifacts.items()},
                task_options=_task_options(task_options),
            )
            _log_run_plan_execution_result(jobs, run_plan.run_id, execution)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            if run_plan is not None:
                jobs.add_log(run_plan.run_id, "ERROR", "run_plan", f"RunPlan resume failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution": execution})


def run_plan_from_payload(value: object, *, run_id: str) -> RunPlan:
    if not isinstance(value, dict):
        raise ValueError("before and after plan payloads are required")
    if "tasks" in value and "requested_tasks" in value:
        return RunPlan.model_validate(value | {"run_id": str(value.get("run_id") or run_id)})
    payload = dict(value)
    payload.setdefault("run_id", run_id)
    return _expand_plan_from_payload(payload)


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _task_options(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for task_id, options in value.items():
        if isinstance(options, dict):
            normalized[str(task_id)] = {str(key): option_value for key, option_value in options.items()}
    return normalized


def _log_run_plan_execution_result(jobs: JobManager, run_id: str, execution: dict[str, Any]) -> None:
    status = str(execution.get("status") or "")
    if status == RunStatus.WAITING_USER.value:
        task = str(execution.get("waiting_task") or execution.get("planned_task") or "").strip()
        suffix = f" at {task}" if task else ""
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan waiting for user{suffix}")
    elif status == RunStatus.FAILED.value:
        task = str(execution.get("failed_task") or "").strip()
        suffix = f" at {task}" if task else ""
        jobs.add_log(run_id, "ERROR", "run_plan", f"RunPlan execution failed{suffix}")
    elif status in {RunStatus.SUCCEEDED.value, RunStatus.DONE.value}:
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan execution completed: {status}")
    else:
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan execution status: {status or 'unknown'}")


def _expand_plan_from_payload(payload: dict) -> RunPlan:
    run_id = str(payload.get("run_id") or "preview").strip() or "preview"
    requested_tasks = _string_list(payload.get("requested_tasks"))
    if not requested_tasks:
        raise ValueError("requested_tasks required")
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=requested_tasks,
        available_artifacts=_string_list(payload.get("available_artifacts")),
    )
