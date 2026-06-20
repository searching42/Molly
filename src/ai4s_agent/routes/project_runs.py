from __future__ import annotations

from flask import Flask, jsonify, request

from ai4s_agent.agents.observer import ObserverAgent
from ai4s_agent.agents.verifier import VerifierAgent
from ai4s_agent.job_manager import JobManager
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.ui_cards import build_report_preview, build_stage_timeline


def register_project_run_routes(app: Flask, *, projects: ProjectStorage, jobs: JobManager) -> None:
    @app.get("/api/projects/<project_id>/runs/<run_id>/stage-timeline")
    def stage_timeline(project_id: str, run_id: str):
        try:
            state = projects.read_stage_state(project_id, run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if state is None:
            return jsonify({"ok": False, "error": "no stage state found for run"}), 404
        return jsonify({"ok": True, "timeline": build_stage_timeline(state)})

    @app.get("/api/projects/<project_id>/runs/<run_id>/report-preview")
    def report_preview(project_id: str, run_id: str):
        artifact_id = str(request.args.get("artifact_id") or "").strip()
        if not artifact_id:
            return jsonify({"ok": False, "error": "artifact_id required"}), 400
        try:
            registry = projects.read_artifact_registry(project_id, run_id)
            relative_path = registry.get(artifact_id, "")
            if not relative_path:
                return jsonify({"ok": False, "error": "report artifact not found in registry"}), 404
            preview = build_report_preview(
                run_dir=projects.run_dir(project_id, run_id),
                artifact_id=artifact_id,
                relative_path=relative_path,
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "preview": preview})

    @app.post("/api/projects/<project_id>/runs/<run_id>/verify")
    def verify_project_run(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            observer = ObserverAgent(storage=projects, jobs=jobs)
            observation = observer.observe_run(clean_project_id, clean_run_id)
            verifier = VerifierAgent()
            report = verifier.verify(observation)
            report_json, report_md = verifier.write_reports(projects, clean_project_id, clean_run_id, report)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(clean_run_id, "INFO", "verifier", f"Verifier decision: {report.overall_decision}")
        return jsonify(
            {
                "ok": True,
                "report": report.model_dump(mode="json"),
                "outputs": {
                    "verification_report_json": str(report_json),
                    "verification_report_md": str(report_md),
                },
            }
        )
