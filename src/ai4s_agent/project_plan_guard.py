from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request


def install_project_plan_route_guard() -> None:
    """Harden `/api/plan` while Orchestrator plan state remains legacy run-scoped."""

    import ai4s_agent.project_job_routes as routes

    if getattr(routes._create_plan_view, "_project_plan_guard", False):
        return
    guarded = _create_plan_view_with_project_key_guard
    guarded._project_plan_guard = True  # type: ignore[attr-defined]
    routes._create_plan_view = guarded  # type: ignore[assignment]


def _create_plan_view_with_project_key_guard(*, jobs: Any, orch: Any):
    def create_plan():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        try:
            if project_id:
                _validate_project_job_key(project_id, run_id)
                if jobs.get_project_job(project_id, run_id):
                    return jsonify({"ok": False, "error": f"job already active: {project_id}/{run_id}"}), 409
                legacy_status = orch.read_status(run_id)
                if legacy_status.get("plan_exists") or legacy_status.get("gate_decisions"):
                    return jsonify(
                        {
                            "ok": False,
                            "error": "run_id already exists in legacy plan namespace; project-scoped plan state is tracked by OPEN-014",
                        }
                    ), 409
                status = orch.start_run(run_id=run_id, prompt=prompt)
                try:
                    job = jobs.start_project_job(project_id, run_id, details={"gate": status.get("gate")})
                except Exception:
                    _rollback_legacy_plan_file(orch, run_id)
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


def _validate_project_job_key(project_id: str, run_id: str) -> None:
    _safe_segment(project_id, "project_id")
    _safe_segment(run_id, "run_id")


def _safe_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean or clean in {".", ".."} or Path(clean).name != clean:
        raise ValueError(f"{label} must be a single safe path segment")
    return clean


def _rollback_legacy_plan_file(orch: Any, run_id: str) -> None:
    try:
        base_dir = Path(orch.store.base_dir).resolve()
        plan_path = (base_dir / run_id / "plan.json").resolve()
        if plan_path.exists() and plan_path.is_relative_to(base_dir):
            plan_path.unlink()
    except Exception:
        return
