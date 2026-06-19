from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request

from ai4s_agent.job_manager import JobManager
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.project_plan_guard import read_project_plan_status, start_project_plan
from ai4s_agent.storage import ProjectStorage


def install_project_scoped_plan_routes() -> None:
    """Finalize project-scoped plan/status routes after job route migration."""

    import ai4s_agent.api as api_module

    original_register_routes = api_module.register_routes
    if getattr(original_register_routes, "_project_scoped_plan_routes", False):
        return

    def register_routes_with_project_plan_state(app: Any, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
        original_register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        runs = Path(base_runs_dir or api_module.DEFAULT_RUNS_DIR).resolve()
        workspace = api_module._workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        jobs = JobManager(runs_dir=runs)
        orch = Orchestrator(base_runs_dir=runs)
        projects = ProjectStorage(workspace_dir=workspace)
        app.view_functions["create_plan"] = _create_plan_view(jobs=jobs, orch=orch, projects=projects)
        app.add_url_rule(
            "/api/projects/<project_id>/runs/<run_id>/status",
            endpoint="project_run_status",
            view_func=_project_run_status_view(projects=projects),
            methods=["GET"],
        )

    register_routes_with_project_plan_state._project_scoped_plan_routes = True  # type: ignore[attr-defined]
    api_module.register_routes = register_routes_with_project_plan_state  # type: ignore[method-assign]


def _create_plan_view(*, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage):
    def create_plan():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        try:
            if project_id:
                if jobs.get_project_job(project_id, run_id):
                    return jsonify({"ok": False, "error": f"job already active: {project_id}/{run_id}"}), 409
                status = start_project_plan(projects, project_id, run_id, prompt)
                try:
                    job = jobs.start_project_job(project_id, run_id, details={"gate": status.get("gate")})
                except Exception:
                    _rollback_project_plan(projects, project_id, run_id)
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


def _project_run_status_view(*, projects: ProjectStorage):
    def project_run_status(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            status = read_project_plan_status(projects, clean_project_id, clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "status": status})

    return project_run_status


def _rollback_project_plan(projects: ProjectStorage, project_id: str, run_id: str) -> None:
    try:
        run_path = projects.run_dir(project_id, run_id)
        plan_path = (run_path / "plan.json").resolve()
        if plan_path.exists() and plan_path.is_relative_to(run_path.resolve()):
            plan_path.unlink()
    except Exception:
        return
