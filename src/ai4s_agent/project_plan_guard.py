from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request

from ai4s_agent.planner import build_plan
from ai4s_agent.schemas import GateName, RunStatus


def install_project_plan_route_guard() -> None:
    """Route project plan creation through project-scoped storage."""

    import ai4s_agent.project_job_routes as routes

    if getattr(routes._create_plan_view, "_project_plan_guard", False):
        return
    guarded = _create_plan_view_with_project_plan_state
    guarded._project_plan_guard = True  # type: ignore[attr-defined]
    routes._create_plan_view = guarded  # type: ignore[assignment]


def _create_plan_view_with_project_plan_state(*, jobs: Any, orch: Any, projects: Any | None = None):
    def create_plan():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        try:
            if project_id:
                if projects is None:
                    return jsonify({"ok": False, "error": "project storage is not initialized"}), 500
                _validate_project_job_key(project_id, run_id)
                if jobs.get_project_job(project_id, run_id):
                    return jsonify({"ok": False, "error": f"job already active: {project_id}/{run_id}"}), 409
                project_status = read_project_plan_status(projects, project_id, run_id)
                if project_status.get("plan_exists") or project_status.get("gate_decisions"):
                    return jsonify({"ok": False, "error": f"project run already exists: {project_id}/{run_id}"}), 409
                status = start_project_plan(projects, project_id, run_id, prompt)
                try:
                    job = jobs.start_project_job(project_id, run_id, details={"gate": status.get("gate")})
                except Exception:
                    _rollback_project_plan_file(projects, project_id, run_id)
                    raise
                return jsonify({"ok": True, **status, "job": job, "job_key": job.get("job_key")})
            if jobs.get_job(run_id):
                return jsonify({"ok": False, "error": f"job already active: {run_id}"}), 409
            status = orch.start_run(run_id=run_id, prompt=prompt)
            jobs.start_job(run_id, details={"gate": status.get("gate")})
        except ValueError as exc:
            message = str(exc)
            status_code = 409 if "already active" in message or "already exists" in message else 400
            return jsonify({"ok": False, "error": message}), status_code
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **status})

    return create_plan


def start_project_plan(projects: Any, project_id: str, run_id: str, prompt: str) -> dict[str, str]:
    _validate_project_job_key(project_id, run_id)
    status = read_project_plan_status(projects, project_id, run_id)
    if status.get("plan_exists") or status.get("gate_decisions"):
        raise ValueError(f"project run already exists: {project_id}/{run_id}")
    plan = build_plan(run_id=run_id, prompt=prompt)
    projects._write_json(projects.run_dir(project_id, run_id), "plan.json", plan.model_dump())
    first_gate = GateName.TASK_PARSE
    return {
        "project_id": project_id,
        "run_id": run_id,
        "state": RunStatus.WAITING_USER.value,
        "gate": first_gate.value,
        "plan_scope": "project",
    }


def read_project_plan_status(projects: Any, project_id: str, run_id: str) -> dict[str, Any]:
    _validate_project_job_key(project_id, run_id)
    run_path = projects.run_dir(project_id, run_id)
    plan = projects._read_json(run_path, "plan.json")
    gate_decisions = projects._read_json(run_path, "gate_decisions.json").get("decisions", [])
    if not isinstance(gate_decisions, list):
        gate_decisions = []
    return {
        "project_id": project_id,
        "run_id": run_id,
        "plan_exists": bool(plan),
        "gate_decisions": [item for item in gate_decisions if isinstance(item, dict)],
        "plan_scope": "project",
    }


def _validate_project_job_key(project_id: str, run_id: str) -> None:
    _safe_segment(project_id, "project_id")
    _safe_segment(run_id, "run_id")


def _safe_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean in {".", ".."} or Path(clean).name != clean:
        raise ValueError(f"{label} must be a single safe path segment")
    return clean


def _rollback_project_plan_file(projects: Any, project_id: str, run_id: str) -> None:
    try:
        run_path = projects.run_dir(project_id, run_id)
        plan_path = (run_path / "plan.json").resolve()
        if plan_path.exists() and plan_path.is_relative_to(run_path.resolve()):
            plan_path.unlink()
    except Exception:
        return
