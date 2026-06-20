from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.job_manager import JobManager
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.schemas import BackgroundJobBudget, RunStatus, StageHistoryItem
from ai4s_agent.storage import ProjectStorage


def register_job_routes(app: Flask, *, jobs: JobManager, orch: Orchestrator, projects: ProjectStorage) -> None:
    @app.get("/api/runs/<run_id>/logs")
    def run_logs(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        limit = int(request.args.get("limit", 50))
        entries = jobs.get_logs(clean_run_id, limit=limit)
        return jsonify({"ok": True, "run_id": clean_run_id, "logs": entries})

    @app.post("/api/runs/<run_id>/pause")
    def pause_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.pause_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/runs/<run_id>/resume")
    def resume_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.resume_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/runs/<run_id>/stop")
    def stop_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.stop_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/background-jobs")
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
            job = jobs.start_background_job(
                run_id,
                project_id=project_id,
                task_id=task_id,
                budget=budget,
                details=details_payload if isinstance(details_payload, dict) else None,
            )
        except ValidationError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except ValueError as exc:
            status_code = 409 if "already active" in str(exc) else 400
            return jsonify({"ok": False, "error": str(exc)}), status_code
        return jsonify({"ok": True, "job": job})

    @app.get("/api/background-jobs/<run_id>")
    def get_background_job(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.get_background_job(clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if job is None:
            return jsonify({"ok": False, "error": "no background job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/background-jobs/<run_id>/checkpoints")
    def record_background_checkpoint(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        stage = str(payload.get("stage") or "").strip()
        cursor = payload.get("cursor")
        artifact_refs = payload.get("artifact_refs")
        if not stage:
            return jsonify({"ok": False, "error": "stage required"}), 400
        if cursor is not None and not isinstance(cursor, dict):
            return jsonify({"ok": False, "error": "cursor must be an object"}), 400
        if artifact_refs is not None and not isinstance(artifact_refs, list):
            return jsonify({"ok": False, "error": "artifact_refs must be a list"}), 400
        try:
            completed_units = payload.get("completed_units", 0)
            runtime_sec = payload.get("runtime_sec", 0)
            cost_usd = payload.get("cost_usd", 0.0)
            checkpoint = jobs.record_background_checkpoint(
                clean_run_id,
                stage=stage,
                cursor=cursor if isinstance(cursor, dict) else None,
                completed_units=completed_units,
                runtime_sec=runtime_sec,
                cost_usd=cost_usd,
                artifact_refs=[str(item) for item in artifact_refs] if isinstance(artifact_refs, list) else None,
            )
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "checkpoint": checkpoint})

    @app.get("/api/background-jobs/<run_id>/resume-plan")
    def background_resume_plan(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            resume_plan = jobs.background_resume_plan(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "resume_plan": resume_plan})

    @app.post("/api/runs/<run_id>/retry")
    def retry_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        stage = str(payload.get("stage") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        status = orch.read_status(clean_run_id)
        if not status.get("plan_exists"):
            return jsonify({"ok": False, "error": "no plan found for run"}), 404
        if jobs.get_job(clean_run_id):
            return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
        if not project_id:
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
            return jsonify(
                {
                    "ok": False,
                    "error": "retry is limited to latest failed stage or explicitly retryable stage",
                    "latest_failed_stage": state.stage,
                }
            ), 400
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
        state.history.append(
            StageHistoryItem(
                stage=requested_stage,
                status=RunStatus.PENDING,
                updated_at=now,
                note="retry requested",
            )
        )
        projects.write_stage_state(project_id, clean_run_id, state)
        try:
            job = jobs.start_job(
                clean_run_id,
                details={"retry": True, "retry_stage": requested_stage, "project_id": project_id},
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(clean_run_id, "INFO", "retry", f"Retry requested for stage: {requested_stage}")
        return jsonify({"ok": True, "run_id": clean_run_id, "retry_stage": requested_stage, "job": job})

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify({"ok": True, "jobs": jobs.list_jobs()})


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload
