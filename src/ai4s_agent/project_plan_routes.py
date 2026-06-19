from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request

from ai4s_agent._utils import now_iso
from ai4s_agent.gatekeeper import GATE_SEQUENCE
from ai4s_agent.job_manager import JobManager
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.project_plan_guard import read_project_plan_status, start_project_plan
from ai4s_agent.schemas import GateDecision, GateName, RunStatus, StageHistoryItem
from ai4s_agent.storage import ProjectStorage


def install_project_scoped_plan_routes() -> None:
    """Finalize project-scoped plan/status/gate routes after job route migration."""

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
        app.view_functions["approve_gate"] = _approve_gate_view(jobs=jobs, orch=orch, projects=projects)
        app.view_functions["retry_run"] = _retry_run_view(jobs=jobs, orch=orch, projects=projects)
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


def _approve_gate_view(*, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage):
    def approve_gate():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        gate_raw = str(payload.get("gate") or "").strip()
        actor = str(payload.get("actor") or "").strip()
        note = str(payload.get("note") or "").strip()
        if not run_id or not gate_raw or not actor:
            return jsonify({"ok": False, "error": "run_id, gate, and actor required"}), 400
        try:
            gate = GateName(gate_raw)
        except ValueError:
            return jsonify({"ok": False, "error": f"unknown gate: {gate_raw}"}), 400
        try:
            if project_id:
                status = approve_project_gate(projects, project_id, run_id, gate, actor=actor, note=note)
                jobs.add_project_log(project_id, run_id, "INFO", "gate", f"Gate {gate_raw} approved by {actor}")
                return jsonify({"ok": True, **status})
            status = orch.approve_gate(run_id=run_id, gate=gate, actor=actor, note=note)
            jobs.add_log(run_id, "INFO", "gate", f"Gate {gate_raw} approved by {actor}")
            return jsonify({"ok": True, **status})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    return approve_gate


def _retry_run_view(*, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage):
    def retry_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        stage = str(payload.get("stage") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if project_id:
            try:
                project_status = read_project_plan_status(projects, project_id, clean_run_id)
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            if not project_status.get("plan_exists"):
                legacy_status = orch.read_status(clean_run_id)
                if not legacy_status.get("plan_exists"):
                    return jsonify({"ok": False, "error": "no project plan found for run"}), 404
            if jobs.get_project_job(project_id, clean_run_id):
                return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
        else:
            status = orch.read_status(clean_run_id)
            if not status.get("plan_exists"):
                return jsonify({"ok": False, "error": "no plan found for run"}), 404
            if jobs.get_job(clean_run_id):
                return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
            return jsonify({"ok": False, "error": "project_id required for failed-stage retry"}), 400
        try:
            state = projects.read_stage_state(project_id, clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
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


def approve_project_gate(projects: ProjectStorage, project_id: str, run_id: str, gate: GateName, *, actor: str, note: str = "") -> dict[str, Any]:
    status = read_project_plan_status(projects, project_id, run_id)
    if not status.get("plan_exists"):
        raise ValueError(f"no project plan found for run: {project_id}/{run_id}")
    expected = _next_project_gate(status.get("gate_decisions", []))
    if expected != gate:
        expected_value = expected.value if expected is not None else "none"
        raise ValueError(f"gate approval out of order: expected {expected_value}, got {gate.value}")
    decision = GateDecision(gate=gate, approved=True, actor=actor, note=note, approved_at=now_iso())
    projects.append_gate_decision(project_id, run_id, decision)
    updated = read_project_plan_status(projects, project_id, run_id)
    next_gate = _next_project_gate(updated.get("gate_decisions", []))
    state = RunStatus.WAITING_USER.value if next_gate is not None else RunStatus.SUCCEEDED.value
    return {
        "project_id": project_id,
        "run_id": run_id,
        "state": state,
        "gate": gate.value,
        "next_gate": next_gate.value if next_gate is not None else "",
        "approved": True,
        "plan_scope": "project",
    }


def _next_project_gate(decisions: object) -> GateName | None:
    approved: set[GateName] = set()
    if isinstance(decisions, list):
        for raw in decisions:
            if not isinstance(raw, dict) or not bool(raw.get("approved")):
                continue
            try:
                approved.add(GateName(str(raw.get("gate") or "")))
            except ValueError:
                continue
    for gate in GATE_SEQUENCE:
        if gate not in approved:
            return gate
    return None


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
