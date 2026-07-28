from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from flask import jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.job_manager import JobManager
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.project_plan_guard import read_project_plan_status
from ai4s_agent.schemas import BackgroundJobBudget, RunStatus, StageHistoryItem
from ai4s_agent.storage import ProjectStorage

if TYPE_CHECKING:
    from ai4s_agent.api_route_extensions import RouteExtensionContext


def install_project_scoped_job_routes() -> None:
    """Project job routes are installed by the explicit route hook."""


def apply_project_scoped_job_routes(context: "RouteExtensionContext") -> None:
    import ai4s_agent.api as api_module

    runs = Path(context.base_runs_dir or api_module.DEFAULT_RUNS_DIR).resolve()
    workspace = api_module._workspace_from_config(
        base_runs_dir=context.base_runs_dir,
        workspace_dir=context.workspace_dir,
    )
    jobs = JobManager(runs_dir=runs)
    orch = Orchestrator(base_runs_dir=runs)
    projects = ProjectStorage(workspace_dir=workspace)
    _replace_legacy_job_views(context, jobs=jobs, orch=orch, projects=projects)
    _add_project_job_routes(context, jobs=jobs)


def _replace_legacy_job_views(context: "RouteExtensionContext", *, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage) -> None:
    registry = context.route_overrides
    app = context.app
    extension_id = "project_scoped_job_routes"
    registry.apply_route_override(app, extension_id=extension_id, endpoint="run_logs", view_func=_run_logs_view(jobs=jobs))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="pause_run", view_func=_job_control_view(jobs=jobs, action="pause"))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="resume_run", view_func=_job_control_view(jobs=jobs, action="resume"))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="stop_run", view_func=_job_control_view(jobs=jobs, action="stop"))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="create_background_job", view_func=_create_background_job_view(jobs=jobs))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="get_background_job", view_func=_get_background_job_view(jobs=jobs))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="record_background_checkpoint", view_func=_record_background_checkpoint_view(jobs=jobs))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="background_resume_plan", view_func=_background_resume_plan_view(jobs=jobs))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="retry_run", view_func=_retry_run_view(jobs=jobs, orch=orch, projects=projects))
    registry.apply_route_override(app, extension_id=extension_id, endpoint="list_jobs", view_func=_list_jobs_view(jobs=jobs))


def _add_project_job_routes(context: "RouteExtensionContext", *, jobs: JobManager) -> None:
    registry = context.route_overrides
    app = context.app
    extension_id = "project_scoped_job_routes"
    registry.apply_new_route(
        app,
        extension_id=extension_id,
        endpoint="project_run_logs",
        rule="/api/projects/<project_id>/runs/<run_id>/logs",
        view_func=_project_run_logs_view(jobs=jobs),
        methods=("GET",),
    )
    registry.apply_new_route(
        app,
        extension_id=extension_id,
        endpoint="project_pause_run",
        rule="/api/projects/<project_id>/runs/<run_id>/pause",
        view_func=_project_job_control_view(jobs=jobs, action="pause"),
        methods=("POST",),
    )
    registry.apply_new_route(
        app,
        extension_id=extension_id,
        endpoint="project_resume_run",
        rule="/api/projects/<project_id>/runs/<run_id>/resume",
        view_func=_project_job_control_view(jobs=jobs, action="resume"),
        methods=("POST",),
    )
    registry.apply_new_route(
        app,
        extension_id=extension_id,
        endpoint="project_stop_run",
        rule="/api/projects/<project_id>/runs/<run_id>/stop",
        view_func=_project_job_control_view(jobs=jobs, action="stop"),
        methods=("POST",),
    )


def _run_logs_view(*, jobs: JobManager):
    def run_logs(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = _request_project_id()
        limit = int(request.args.get("limit", 50))
        if project_id:
            entries = jobs.get_project_logs(project_id, clean_run_id, limit=limit)
            return jsonify({"ok": True, "project_id": project_id, "run_id": clean_run_id, "job_key": {"project_id": project_id, "run_id": clean_run_id}, "logs": entries})
        entries = jobs.get_logs(clean_run_id, limit=limit)
        return jsonify({"ok": True, "run_id": clean_run_id, "logs": entries})

    return run_logs


def _project_run_logs_view(*, jobs: JobManager):
    def project_run_logs(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = _clean_run_id(run_id)
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            entries = jobs.get_project_logs(clean_project_id, clean_run_id, limit=int(request.args.get("limit", 50)))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "project_id": clean_project_id, "run_id": clean_run_id, "job_key": {"project_id": clean_project_id, "run_id": clean_run_id}, "logs": entries})

    return project_run_logs


def _job_control_view(*, jobs: JobManager, action: str):
    def control(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = _request_project_id()
        try:
            if project_id:
                job = _apply_project_job_action(jobs, project_id, clean_run_id, action)
            else:
                try:
                    job = _apply_legacy_job_action(jobs, clean_run_id, action)
                except KeyError:
                    project_id = _unique_project_for_active_run(jobs, clean_run_id)
                    job = _apply_project_job_action(jobs, project_id, clean_run_id, action)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "job": job})

    return control


def _project_job_control_view(*, jobs: JobManager, action: str):
    def control(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = _clean_run_id(run_id)
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            job = _apply_project_job_action(jobs, clean_project_id, clean_run_id, action)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "job": job})

    return control


def _create_background_job_view(*, jobs: JobManager):
    def create_background_job():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        run_id = str(payload.get("run_id") or "").strip()
        task_id = str(payload.get("task_id") or "").strip()
        budget_payload = payload.get("budget")
        details_payload = payload.get("details")
        if not run_id or not task_id:
            return jsonify({"ok": False, "error": "run_id and task_id required"}), 400
        if not isinstance(budget_payload, dict):
            return jsonify({"ok": False, "error": "budget object required"}), 400
        if details_payload is not None and not isinstance(details_payload, dict):
            return jsonify({"ok": False, "error": "details must be an object"}), 400
        try:
            budget = BackgroundJobBudget.model_validate(budget_payload)
            job = jobs.start_project_background_job(project_id, run_id, task_id=task_id, budget=budget, details=details_payload if isinstance(details_payload, dict) else None) if project_id else jobs.start_background_job(run_id, project_id=project_id, task_id=task_id, budget=budget, details=details_payload if isinstance(details_payload, dict) else None)
        except ValidationError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409 if "already active" in str(exc) else 400
        return jsonify({"ok": True, "job": job})

    return create_background_job


def _get_background_job_view(*, jobs: JobManager):
    def get_background_job(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = _request_project_id()
        try:
            if project_id:
                job = jobs.get_project_background_job(project_id, clean_run_id)
            else:
                job = jobs.get_background_job(clean_run_id)
                if job is None:
                    project_id = _unique_project_for_background_run(jobs, clean_run_id)
                    job = jobs.get_project_background_job(project_id, clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if job is None:
            return jsonify({"ok": False, "error": "no background job"}), 404
        return jsonify({"ok": True, "job": job})

    return get_background_job


def _record_background_checkpoint_view(*, jobs: JobManager):
    def record_background_checkpoint(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or request.args.get("project_id") or "").strip()
        stage = str(payload.get("stage") or "").strip()
        cursor = payload.get("cursor")
        artifact_refs = payload.get("artifact_refs")
        if not stage:
            return jsonify({"ok": False, "error": "stage required"}), 400
        if cursor is not None and not isinstance(cursor, dict):
            return jsonify({"ok": False, "error": "cursor must be an object"}), 400
        if artifact_refs is not None and not isinstance(artifact_refs, list):
            return jsonify({"ok": False, "error": "artifact_refs must be a list"}), 400
        kwargs = {"stage": stage, "cursor": cursor if isinstance(cursor, dict) else None, "completed_units": payload.get("completed_units", 0), "runtime_sec": payload.get("runtime_sec", 0), "cost_usd": payload.get("cost_usd", 0.0), "artifact_refs": [str(item) for item in artifact_refs] if isinstance(artifact_refs, list) else None}
        try:
            if project_id:
                checkpoint = jobs.record_project_background_checkpoint(project_id, clean_run_id, **kwargs)
            else:
                try:
                    checkpoint = jobs.record_background_checkpoint(clean_run_id, **kwargs)
                except KeyError:
                    project_id = _unique_project_for_background_run(jobs, clean_run_id)
                    checkpoint = jobs.record_project_background_checkpoint(project_id, clean_run_id, **kwargs)
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "checkpoint": checkpoint})

    return record_background_checkpoint


def _background_resume_plan_view(*, jobs: JobManager):
    def background_resume_plan(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = _request_project_id()
        try:
            if project_id:
                resume_plan = jobs.project_background_resume_plan(project_id, clean_run_id)
            else:
                try:
                    resume_plan = jobs.background_resume_plan(clean_run_id)
                except KeyError:
                    project_id = _unique_project_for_background_run(jobs, clean_run_id)
                    resume_plan = jobs.project_background_resume_plan(project_id, clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "resume_plan": resume_plan})

    return background_resume_plan


def _retry_run_view(*, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage):
    def retry_run(run_id: str):
        clean_run_id = _clean_run_id(run_id)
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        stage = str(payload.get("stage") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        state = None
        if project_id:
            try:
                project_status = read_project_plan_status(projects, project_id, clean_run_id)
                state = projects.read_stage_state(project_id, clean_run_id)
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            if not project_status.get("plan_exists") and state is None:
                return jsonify({"ok": False, "error": "no project run state found for run"}), 404
            if jobs.get_project_job(project_id, clean_run_id):
                return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
        else:
            status = orch.read_status(clean_run_id)
            if not status.get("plan_exists"):
                return jsonify({"ok": False, "error": "no plan found for run"}), 404
            if jobs.get_job(clean_run_id):
                return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
            return jsonify({"ok": False, "error": "project_id required for failed-stage retry"}), 400
        if state is None:
            return jsonify({"ok": False, "error": "no stage state found for run"}), 404
        if state.status != RunStatus.FAILED:
            return jsonify({"ok": False, "error": "latest stage has not failed"}), 409
        error = state.error if isinstance(state.error, dict) else {}
        retryable_stages = error.get("retryable_stages", [])
        if not isinstance(retryable_stages, list):
            retryable_stages = []
        requested_stage = stage or state.stage
        explicitly_retryable = requested_stage in {str(item) for item in retryable_stages}
        if requested_stage != state.stage and not explicitly_retryable:
            return jsonify({"ok": False, "error": "retry is limited to latest failed stage or explicitly retryable stage", "latest_failed_stage": state.stage}), 400
        if not bool(error.get("retryable")) and not explicitly_retryable:
            return jsonify({"ok": False, "error": "latest failed stage is not retryable"}), 409
        now = now_iso()
        details = dict(state.details)
        details["retry_requested_at"] = now
        details["retry_stage"] = requested_stage
        details["retry_count"] = int(details.get("retry_count") or 0) + 1
        state.status = RunStatus.PENDING
        state.started_at = now
        state.updated_at = now
        state.ended_at = None
        state.details = details
        state.history.append(StageHistoryItem(stage=requested_stage, status=RunStatus.PENDING, updated_at=now, note="retry requested"))
        projects.write_stage_state(project_id, clean_run_id, state)
        try:
            job = jobs.start_project_job(project_id, clean_run_id, details={"retry": True, "retry_stage": requested_stage, "project_id": project_id})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_project_log(project_id, clean_run_id, "INFO", "retry", f"Retry requested for stage: {requested_stage}")
        return jsonify({"ok": True, "project_id": project_id, "run_id": clean_run_id, "retry_stage": requested_stage, "job": job, "job_key": job.get("job_key")})

    return retry_run


def _list_jobs_view(*, jobs: JobManager):
    def list_jobs():
        project_id = _request_project_id()
        if project_id:
            return jsonify({"ok": True, "project_id": project_id, "jobs": jobs.list_project_jobs(project_id)})
        return jsonify({"ok": True, "jobs": jobs.list_jobs()})

    return list_jobs


def _unique_project_for_active_run(jobs: JobManager, run_id: str) -> str:
    matches = [str(job.get("project_id")) for job in jobs.list_project_jobs() if str(job.get("run_id") or "") == run_id]
    matches = [item for item in matches if item]
    if not matches:
        raise KeyError(f"no active project job: {run_id}")
    if len(set(matches)) > 1:
        raise ValueError("project_id required for ambiguous project-scoped job")
    return matches[0]


def _unique_project_for_background_run(jobs: JobManager, run_id: str) -> str:
    root = jobs.runs_dir / "projects"
    if not root.exists():
        raise KeyError(f"no background job: {run_id}")
    matches: list[str] = []
    for project_dir in sorted(root.iterdir()):
        if project_dir.is_dir() and (project_dir / "runs" / run_id / "background_job_state.json").exists():
            matches.append(project_dir.name)
    if not matches:
        raise KeyError(f"no background job: {run_id}")
    if len(matches) > 1:
        raise ValueError("project_id required for ambiguous project-scoped background job")
    return matches[0]


def _apply_project_job_action(jobs: JobManager, project_id: str, run_id: str, action: str) -> dict[str, Any]:
    if action == "pause":
        return jobs.pause_project_job(project_id, run_id)
    if action == "resume":
        return jobs.resume_project_job(project_id, run_id)
    if action == "stop":
        return jobs.stop_project_job(project_id, run_id)
    raise ValueError(f"unsupported job action: {action}")


def _apply_legacy_job_action(jobs: JobManager, run_id: str, action: str) -> dict[str, Any]:
    if action == "pause":
        return jobs.pause_job(run_id)
    if action == "resume":
        return jobs.resume_job(run_id)
    if action == "stop":
        return jobs.stop_job(run_id)
    raise ValueError(f"unsupported job action: {action}")


def _request_project_id() -> str:
    payload = request.get_json(silent=True) if request.method != "GET" else None
    if isinstance(payload, dict):
        value = payload.get("project_id")
        if value:
            return str(value).strip()
    return str(request.args.get("project_id") or "").strip()


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _clean_run_id(run_id: str) -> str:
    return str(run_id or "").strip()
