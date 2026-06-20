from __future__ import annotations

from flask import Flask, jsonify, request

from ai4s_agent.job_manager import JobManager
from ai4s_agent.orchestrator import Orchestrator


def register_legacy_plan_routes(app: Flask, *, orch: Orchestrator, jobs: JobManager) -> None:
    @app.post("/api/plan")
    def create_plan():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        if jobs.get_job(run_id):
            return jsonify({"ok": False, "error": f"job already active: {run_id}"}), 409
        try:
            status = orch.start_run(run_id=run_id, prompt=prompt)
            jobs.start_job(run_id, details={"gate": status.get("gate")})
        except ValueError as exc:
            message = str(exc)
            status_code = 409 if "already active" in message or "already exists" in message else 400
            return jsonify({"ok": False, "error": message}), status_code
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **status})
