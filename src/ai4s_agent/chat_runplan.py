from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request
from pydantic import ValidationError

from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.chat_context import _PROJECT_STORAGES, infer_project_available_inputs
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import RunStatus, StageState
from ai4s_agent.ui_cards import build_stage_timeline


_CONTROLLED_EXECUTION_NOTICE = (
    "Preview only. Execute/resume must still go through the controlled RunPlan "
    "API with snapshot and gate confirmation."
)


def install_chat_run_plan_routes() -> None:
    """Install chat-to-RunPlan preview and feedback routes.

    The routes are attached after the standard API routes are registered.  They
    intentionally expose preview/feedback only; they do not bypass the existing
    `/api/run-plan/execute` or `/api/run-plan/resume` confirmation paths.
    """

    import ai4s_agent.api as api_module

    original_register_routes = api_module.register_routes
    if getattr(original_register_routes, "_chat_run_plan_routes", False):
        return

    def register_routes_with_chat_run_plan(app: Any, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
        original_register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        _register_chat_run_plan_routes(app)

    register_routes_with_chat_run_plan._chat_run_plan_routes = True  # type: ignore[attr-defined]
    api_module.register_routes = register_routes_with_chat_run_plan  # type: ignore[method-assign]


def _register_chat_run_plan_routes(app: Any) -> None:
    if "agent_conversation_run_plan_preview" not in app.view_functions:

        @app.post("/api/agent/conversation/run-plan-preview")
        def agent_conversation_run_plan_preview():
            try:
                payload = _request_json_object()
                run_id = str(payload.get("run_id") or "").strip()
                project_id = str(payload.get("project_id") or "").strip()
                if not run_id:
                    return jsonify({"ok": False, "error": "run_id required"}), 400
                modeling_payload = _modeling_payload_from_request(payload, run_id=run_id, project_id=project_id)
                available_artifacts = _available_artifacts(project_id=project_id, run_id=run_id, payload=payload)
                requested_tasks = _requested_tasks(payload=payload, modeling_payload=modeling_payload)
                run_plan = expand_run_plan(
                    run_id=run_id,
                    requested_tasks=requested_tasks,
                    available_artifacts=available_artifacts,
                )
                preview = _run_plan_preview_payload(run_plan)
            except (ValidationError, ValueError) as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            return jsonify(
                {
                    "ok": True,
                    "modeling_plan_payload": modeling_payload,
                    "run_plan": run_plan.model_dump(mode="json"),
                    "preview": preview,
                    "execution_control": {
                        "direct_execution": False,
                        "message": _CONTROLLED_EXECUTION_NOTICE,
                        "execute_endpoint": "/api/run-plan/execute",
                        "resume_endpoint": "/api/run-plan/resume",
                    },
                }
            )

    if "agent_conversation_execution_feedback" not in app.view_functions:

        @app.post("/api/agent/conversation/execution-feedback")
        def agent_conversation_execution_feedback():
            try:
                payload = _request_json_object()
                project_id = str(payload.get("project_id") or "").strip()
                run_id = str(payload.get("run_id") or "").strip()
                if not project_id or not run_id:
                    return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
                storage = _project_storage()
                state = storage.read_stage_state(project_id, run_id)
                registry = storage.read_artifact_registry(project_id, run_id)
                gate_decisions = storage.read_gate_decisions(project_id, run_id)
                confirmations = (
                    storage.read_execution_confirmations(project_id, run_id)
                    if hasattr(storage, "read_execution_confirmations")
                    else []
                )
                feedback = _execution_feedback_payload(
                    state=state,
                    artifact_registry=registry,
                    gate_decisions=gate_decisions,
                    execution_confirmations=confirmations,
                )
            except (ValidationError, ValueError, FileNotFoundError) as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            return jsonify({"ok": True, "feedback": feedback})


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _modeling_payload_from_request(payload: dict[str, Any], *, run_id: str, project_id: str) -> dict[str, Any]:
    modeling_payload = payload.get("modeling_plan_payload")
    if isinstance(modeling_payload, dict):
        return {str(key): value for key, value in modeling_payload.items()}
    messages = payload.get("messages")
    return ConversationAgent().prepare_modeling_plan_payload(
        run_id=run_id,
        project_id=project_id or None,
        messages=messages,
        available_inputs=payload.get("available_inputs"),
    )


def _available_artifacts(*, project_id: str, run_id: str, payload: dict[str, Any]) -> list[str]:
    raw = payload.get("available_artifacts")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    artifacts: list[str] = []
    if project_id and run_id:
        try:
            storage = _project_storage()
            registry = storage.read_artifact_registry(project_id, run_id)
        except (ValueError, FileNotFoundError):
            registry = {}
        artifacts.extend(str(item).strip() for item in registry if str(item).strip())
    for item in infer_project_available_inputs(project_id=project_id, run_id=run_id):
        clean = str(item).strip()
        if clean and clean not in artifacts:
            artifacts.append(clean)
    return artifacts


def _requested_tasks(*, payload: dict[str, Any], modeling_payload: dict[str, Any]) -> list[str]:
    for source in (payload.get("requested_tasks"), modeling_payload.get("requested_tasks")):
        if isinstance(source, list):
            tasks = [str(item).strip() for item in source if str(item).strip()]
            if tasks:
                return tasks
    goal = str(modeling_payload.get("goal") or "").lower()
    if any(term in goal for term in ("report", "top", "rank", "screen", "筛选", "排序", "报告")):
        return ["render_report"]
    if any(term in goal for term in ("generate", "candidate", "design", "生成", "候选", "设计")):
        return ["generate_candidates"]
    return ["train_model"]


def _run_plan_preview_payload(run_plan: Any) -> dict[str, Any]:
    registry = AtomicTaskRegistry()
    required_gates: list[str] = []
    task_summaries: list[dict[str, Any]] = []
    for task in run_plan.tasks:
        spec = registry.get(task.task_id)
        for gate in spec.gates:
            if gate not in required_gates:
                required_gates.append(gate)
        task_summaries.append(
            {
                "task_id": task.task_id,
                "risk_level": spec.risk_level.value,
                "gates": list(spec.gates),
                "depends_on": list(task.depends_on),
                "unresolved_requirements": list(task.unresolved_requirements),
            }
        )
    return {
        "status": "ready_for_controlled_execution" if not run_plan.missing_artifacts else "blocked_missing_artifacts",
        "requested_tasks": list(run_plan.requested_tasks),
        "task_count": len(run_plan.tasks),
        "required_gates": required_gates,
        "missing_artifacts": list(run_plan.missing_artifacts),
        "tasks": task_summaries,
        "next_actions": _preview_next_actions(required_gates=required_gates, missing_artifacts=list(run_plan.missing_artifacts)),
    }


def _preview_next_actions(*, required_gates: list[str], missing_artifacts: list[str]) -> list[str]:
    if missing_artifacts:
        return ["provide_missing_artifacts", "regenerate_run_plan_preview"]
    if required_gates:
        return ["review_run_plan_preview", "execute_to_snapshot", "confirm_required_gates_before_resume"]
    return ["review_run_plan_preview", "execute_controlled_run_plan"]


def _execution_feedback_payload(
    *,
    state: StageState | None,
    artifact_registry: dict[str, str],
    gate_decisions: list[dict[str, Any]],
    execution_confirmations: list[dict[str, Any]],
) -> dict[str, Any]:
    if state is None:
        return {
            "status": "NO_STAGE_STATE",
            "summary": "No RunPlan execution state has been recorded yet.",
            "artifact_registry": artifact_registry,
            "gate_decisions": gate_decisions,
            "execution_confirmations": execution_confirmations,
            "next_actions": ["generate_run_plan_preview", "execute_controlled_run_plan"],
        }
    timeline = build_stage_timeline(state)
    snapshot = state.details.get("execution_snapshot") if isinstance(state.details, dict) else None
    snapshot_summary = {}
    if isinstance(snapshot, dict):
        snapshot_summary = {
            "snapshot_id": str(snapshot.get("snapshot_id") or ""),
            "snapshot_hash": str(snapshot.get("snapshot_hash") or ""),
            "task_id": str(snapshot.get("task_id") or state.stage),
            "adapter": str(snapshot.get("adapter") or ""),
            "required_gates": list(state.details.get("required_gates") or []),
        }
    return {
        "status": state.status.value,
        "stage": state.stage,
        "next_stage": state.next_stage,
        "timeline": timeline,
        "execution_snapshot": snapshot_summary,
        "artifact_registry": artifact_registry,
        "gate_decisions": gate_decisions,
        "execution_confirmations": execution_confirmations,
        "next_actions": _feedback_next_actions(state),
    }


def _feedback_next_actions(state: StageState) -> list[str]:
    if state.status == RunStatus.WAITING_USER:
        return ["review_execution_snapshot", "confirm_required_gates", "resume_run_plan"]
    if state.status == RunStatus.SUCCEEDED:
        return ["review_artifacts", "render_report_preview", "continue_chat_with_results"]
    if state.status == RunStatus.FAILED:
        return ["inspect_error", "request_replan", "revise_run_plan"]
    return ["refresh_execution_feedback", "continue_monitoring"]


def _project_storage() -> Any:
    if not _PROJECT_STORAGES:
        raise ValueError("project storage is not initialized")
    return _PROJECT_STORAGES[-1]
