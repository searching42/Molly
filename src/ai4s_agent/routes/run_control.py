from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request

import ai4s_agent.adapters as adapter_exports
from ai4s_agent._utils import strict_bool
from ai4s_agent.job_manager import JobManager
from ai4s_agent.memory import PermissionPolicy
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import GateName
from ai4s_agent.storage import ProjectStorage


def register_run_control_routes(
    app: Flask,
    *,
    orch: Orchestrator,
    jobs: JobManager,
    projects: ProjectStorage,
    permissions: PermissionPolicy,
) -> None:
    @app.post("/api/gates/approve")
    def approve_gate():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
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
            status = orch.approve_gate(run_id=run_id, gate=gate, actor=actor, note=note)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(run_id, "INFO", "gate", f"Gate {gate_raw} approved by {actor}")
        return jsonify({"ok": True, **status})

    @app.get("/api/runs/<run_id>")
    def run_status(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = str(request.args.get("project_id") or "").strip()
        try:
            status = _read_run_status(orch, projects, clean_run_id, project_id=project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        job = jobs.get_job(clean_run_id)
        return jsonify({"ok": True, "job": job, **status})

    @app.post("/api/adapters/execute")
    def execute_adapter():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        adapter_name = str(payload.get("adapter") or "").strip()
        adapter_payload = payload.get("payload")
        if not run_id or not adapter_name:
            return jsonify({"ok": False, "error": "run_id and adapter required"}), 400
        if not isinstance(adapter_payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        adapter = getattr(adapter_exports, adapter_name, None)
        if not callable(adapter):
            return jsonify({"ok": False, "error": f"unknown adapter: {adapter_name}"}), 400
        try:
            policy = _adapter_execution_policy(adapter_name, adapter_payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if policy is None:
            return jsonify({"ok": False, "error": f"adapter is not registered for direct execution: {adapter_name}"}), 400
        action, required_gates = policy
        project_id = str(payload.get("project_id") or adapter_payload.get("project_id") or "")
        actor = str(
            payload.get("actor")
            or payload.get("approved_by")
            or adapter_payload.get("actor")
            or adapter_payload.get("approved_by")
            or ""
        )
        decision = permissions.decide(
            action,
            project_id=project_id,
            run_id=run_id,
            project_approved=_as_bool(payload.get("project_approved"))
            or _as_bool(adapter_payload.get("project_approved")),
            confirmed=_as_bool(payload.get("confirmed")) or _as_bool(adapter_payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "adapter execution requires permission",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        if required_gates:
            return jsonify(
                {
                    "ok": False,
                    "error": "gated adapter execution requires run-plan snapshot approval",
                    "required_gates": required_gates,
                    "permission": decision.model_dump(mode="json"),
                }
            ), 400
        try:
            snapshot_required = _adapter_requires_snapshot_for_execute(adapter_name, adapter_payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if snapshot_required:
            return jsonify(
                {
                    "ok": False,
                    "error": "this adapter requires run-plan snapshot approval when execute=true",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 400
        try:
            status = _read_run_status(orch, projects, run_id, project_id=project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        missing_gates = [
            gate
            for gate in required_gates
            if not _gate_approved(status, gate)
        ]
        if missing_gates:
            return jsonify(
                {
                    "ok": False,
                    "error": "gate approval required before adapter execution",
                    "missing_gates": missing_gates,
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        jobs.add_log(run_id, "INFO", "adapter", f"Starting adapter: {adapter_name}")
        try:
            result = adapter(adapter_payload)
        except Exception as exc:
            jobs.add_log(run_id, "ERROR", "adapter", f"Adapter {adapter_name} raised: {exc}")
            return jsonify({"ok": False, "error": str(exc), "adapter": adapter_name}), 500
        status = str(result.get("status") or "") if isinstance(result, dict) else ""
        level = "INFO" if status in {"success", "planned"} else "ERROR"
        jobs.add_log(run_id, level, "adapter", f"Adapter {adapter_name} finished: {status or 'unknown'}")
        return jsonify({"ok": True, "adapter": adapter_name, "result": result})


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}


def _adapter_execution_policy(adapter_name: str, adapter_payload: dict) -> tuple[str, list[str]] | None:
    registry = AtomicTaskRegistry()
    by_adapter = {
        task.default_adapter: task
        for task in registry.list_tasks()
        if task.default_adapter
    }
    adapter_aliases = {
        "draft_cleaning_rules_adapter": "clean_dataset",
        "train_model_unimol_legacy_adapter": "train_model",
        "predict_candidates_domain_model_adapter": "predict_candidates",
        "predict_candidates_unimol_legacy_adapter": "predict_candidates",
        "parse_pdf_folder_mineru_adapter": "parse_document",
        "parse_document_pdfplumber_adapter": "parse_document",
        "parse_document_pymupdf_adapter": "parse_document",
        "parse_document_grobid_adapter": "parse_document",
    }
    task = by_adapter.get(adapter_name)
    if task is None and adapter_name in adapter_aliases:
        task = registry.get(adapter_aliases[adapter_name])
    if task is None:
        return None

    action = task.task_id
    if task.task_id == "generate_candidates":
        backend = str(adapter_payload.get("backend") or "deterministic_stub").strip().lower()
        try:
            count = int(adapter_payload.get("count") or adapter_payload.get("num_candidates") or 32)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation count must be a positive integer") from exc
        if count <= 0:
            raise ValueError("generation count must be a positive integer")
        if backend != "deterministic_stub" or count >= 128:
            action = "generate_candidates_expensive"
    return action, list(task.gates)


_CANNOT_DIRECT_EXECUTE = frozenset(
    {
        "predict_candidates_unimol_legacy_adapter",
        "predict_candidates_domain_model_adapter",
        "train_model_unimol_legacy_adapter",
    }
)


def _adapter_requires_snapshot_for_execute(adapter_name: str, adapter_payload: dict) -> bool:
    """Return True when a plan-capable adapter requests execute=true directly."""
    if adapter_name not in _CANNOT_DIRECT_EXECUTE:
        return False
    execute_raw = adapter_payload.get("execute")
    if execute_raw is None:
        return False
    return strict_bool(execute_raw, key="execute")


def _read_run_status(
    orch: Orchestrator,
    projects: ProjectStorage,
    run_id: str,
    *,
    project_id: str = "",
) -> dict[str, object]:
    legacy_status = dict(orch.read_status(run_id))
    clean_project_id = str(project_id or "").strip()
    if not clean_project_id:
        return {**legacy_status, "state_source": "legacy"}

    project_status = _read_project_run_status(projects, clean_project_id, run_id)
    if not project_status:
        return {
            **legacy_status,
            "project_id": clean_project_id,
            "state_source": "legacy",
        }
    return {
        **legacy_status,
        **project_status,
        "legacy_plan_exists": bool(legacy_status.get("plan_exists")),
    }


def _read_project_run_status(projects: ProjectStorage, project_id: str, run_id: str) -> dict[str, object]:
    run_path = _project_run_dir_if_exists(projects, project_id, run_id)
    if run_path is None:
        return {}

    stage = _read_json(run_path / "stage.json")
    gate_payload = _read_json(run_path / "gate_decisions.json")
    artifact_payload = _read_json(run_path / "artifact_registry.json")
    decisions = gate_payload.get("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
    artifacts = artifact_payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    status: dict[str, object] = {
        "run_id": run_id,
        "project_id": project_id,
        "state_source": "project",
        "plan_exists": bool(_read_json(run_path / "run_plan.json") or _read_json(run_path / "plan.json")),
        "gate_decisions": [decision for decision in decisions if isinstance(decision, dict)],
        "artifacts": {str(key): str(value) for key, value in artifacts.items()},
    }
    if stage:
        status["stage"] = stage
        status["stage_status"] = str(stage.get("status") or "")
    return status


def _project_run_dir_if_exists(projects: ProjectStorage, project_id: str, run_id: str) -> Path | None:
    clean_project_id = str(project_id or "").strip()
    clean_run_id = str(run_id or "").strip()
    if not clean_project_id or not clean_run_id:
        return None

    project_path = (projects.projects_root / clean_project_id).resolve()
    if not project_path.is_relative_to(projects.projects_root):
        raise ValueError("project_id escapes base directory")
    runs_base = (project_path / "runs").resolve()
    run_path = (runs_base / clean_run_id).resolve()
    if not run_path.is_relative_to(runs_base):
        raise ValueError("run_id escapes base directory")
    return run_path if run_path.exists() else None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _gate_approved(status: dict[str, object], gate: str) -> bool:
    decisions = status.get("gate_decisions", [])
    if not isinstance(decisions, list):
        return False
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("gate") or "") == gate and bool(decision.get("approved")):
            return True
    return False
