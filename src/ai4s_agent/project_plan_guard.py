from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.planner import build_plan
from ai4s_agent.schemas import GateName, RunStatus


def install_project_plan_route_guard() -> None:
    """Project plan guard helpers are called directly by project plan routes."""


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
