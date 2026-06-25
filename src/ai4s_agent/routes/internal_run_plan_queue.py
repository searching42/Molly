from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, current_app, jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import now_iso, truthy
from ai4s_agent.actor_identity import ActorContext, resolve_actor
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.memory import PermissionLevel, PermissionPolicy
from ai4s_agent.run_plan_queue_lifecycle import internal_run_plan_queue_dir, read_run_plan_queue_status
from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.run_plan_queue_summary import build_run_plan_queue_execution_summary
from ai4s_agent.run_plan_task_runner import ExecutorFactory
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import (
    RunPlanApplicationArtifactBundle,
    write_replan_application_artifacts,
)
from ai4s_agent.run_plan_replan_application_audit_memory import (
    REPLAN_APPLICATION_AUDIT_REF,
    append_replan_application_audit_record,
    save_replan_application_summary_to_memory,
)
from ai4s_agent.run_plan_resume_intent_validation import ResumeIntentValidationResult, validate_resume_intent
from ai4s_agent.run_plan_resume_intent_validation_audit_memory import (
    RESUME_INTENT_VALIDATION_AUDIT_REF,
    append_resume_intent_validation_audit_record,
    save_resume_intent_validation_summary_to_memory,
)
from ai4s_agent.run_plan_review_card import read_run_plan_review_card
from ai4s_agent.schemas import RunPlan, StageState
from ai4s_agent.server_permissions import ServerPermissionStore, decide_server_permission
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG = "AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"
INTERNAL_RESUME_INTENT_VALIDATION_ROUTE_FLAG = "AI4S_ENABLE_INTERNAL_RESUME_INTENT_VALIDATION_ROUTE"
INTERNAL_RESUME_INTENT_EXECUTE_ROUTE_FLAG = "AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE"
EXECUTOR_FACTORY_CONFIG_KEY = "AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"
INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION = "run_plan_queue_execute"
INTERNAL_RUN_PLAN_REPLAN_APPLY_PERMISSION_ACTION = "run_plan_replan_apply"
INTERNAL_RUN_PLAN_RESUME_INTENT_USE_PERMISSION_ACTION = "run_plan_resume_intent_use"
INTERNAL_RUN_PLAN_RESUME_EXECUTE_PERMISSION_ACTION = "run_plan_resume_execute"


def register_internal_run_plan_queue_routes(app: Flask, *, projects: ProjectStorage) -> None:
    @app.get("/api/internal/run-plan/queue/status")
    def internal_run_plan_queue_status():
        if not internal_run_plan_queue_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal run-plan queue route disabled"}), 404
        try:
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(request.args.get("project_id"), "project_id")
            run_id = _safe_path_component(request.args.get("run_id"), "run_id")
            permission = _decide_permission(projects, actor=actor, project_id=project_id, run_id=run_id)
            if not bool(permission.get("allowed")):
                return jsonify({
                    "ok": False,
                    "project_id": project_id,
                    "run_id": run_id,
                    "error": _permission_error_dict(permission),
                    "permission": _public_permission_decision(permission, project_id=project_id, run_id=run_id),
                }), 403
            queue = WorkerQueue(JsonWorkerQueueStore(_queue_dir(projects, project_id, run_id)))
            return jsonify({
                "ok": True,
                "project_id": project_id,
                "run_id": run_id,
                "status": read_run_plan_queue_status(queue),
                "permission": _public_permission_decision(permission, project_id=project_id, run_id=run_id),
            })
        except (OSError, ValidationError, ValueError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            return jsonify({
                "ok": False,
                "error": _summary_error_dict(exc),
            }), status_code

    @app.get("/api/internal/run-plan/review-card")
    def internal_run_plan_review_card():
        if not internal_run_plan_queue_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal run-plan queue route disabled"}), 404
        try:
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(request.args.get("project_id"), "project_id")
            run_id = _safe_path_component(request.args.get("run_id"), "run_id")
            permission = _decide_permission(projects, actor=actor, project_id=project_id, run_id=run_id)
            if not bool(permission.get("allowed")):
                return jsonify({
                    "ok": False,
                    "project_id": project_id,
                    "run_id": run_id,
                    "error": _permission_error_dict(permission),
                    "permission": _public_permission_decision(permission, project_id=project_id, run_id=run_id),
                }), 403
            card = read_run_plan_review_card(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
            )
            return jsonify({
                "ok": True,
                "project_id": project_id,
                "run_id": run_id,
                "card": card.model_dump(mode="json"),
                "permission": _public_permission_decision(permission, project_id=project_id, run_id=run_id),
            })
        except FileNotFoundError as exc:
            return jsonify({
                "ok": False,
                "error": {
                    "type": "not_found",
                    "message": str(exc).strip() or exc.__class__.__name__,
                },
            }), 404
        except (OSError, ValidationError, ValueError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            return jsonify({
                "ok": False,
                "error": _summary_error_dict(exc),
            }), status_code

    @app.post("/api/internal/run-plan/replan/apply-review")
    def internal_replan_application_apply_review():
        if not internal_run_plan_queue_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal run-plan queue route disabled"}), 404
        actor = ActorContext(actor="", source="missing", required=True)
        project_id = ""
        run_id = ""
        permission: dict[str, Any] | None = None
        bundle: RunPlanApplicationArtifactBundle | None = None
        try:
            payload = _request_json_object()
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            run_id = _safe_path_component(payload.get("run_id"), "run_id")
            application_request = ReplanApplicationRequest.model_validate(
                _replan_application_request_payload(payload, project_id=project_id, run_id=run_id)
            )
            permission = _decide_permission(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                action=INTERNAL_RUN_PLAN_REPLAN_APPLY_PERMISSION_ACTION,
            )
            if not bool(permission.get("allowed")):
                error = _permission_error_dict(permission)
                audit_error = _append_replan_application_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="replan_application_failed",
                    bundle=None,
                    error=error,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_replan_application_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    error=error,
                    permission=permission,
                )), 403
            audit_error = _append_replan_application_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="replan_application_requested",
                bundle=None,
                error=None,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            bundle = write_replan_application_artifacts(
                workspace_dir=projects.workspace_dir,
                request=application_request,
                actor=actor.actor,
                actor_source=actor.source,
                current_run_plan=_read_current_run_plan(projects, project_id=project_id, run_id=run_id),
                stage_state=_read_stage_state(projects, project_id=project_id, run_id=run_id),
            )
            memory_save = save_replan_application_summary_to_memory(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
                bundle=bundle,
                audit_refs=[REPLAN_APPLICATION_AUDIT_REF],
                confirmed_by=actor.actor,
            )
            audit_error = _append_replan_application_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="replan_application_completed",
                bundle=bundle,
                error=None,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(_replan_application_success_summary(
                bundle=bundle,
                memory_record=memory_save.record.model_dump(mode="json"),
                permission=permission,
            ))
        except (OSError, ValidationError, ValueError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            error = _summary_error_dict(exc)
            if actor.actor and project_id and run_id:
                audit_error = _append_replan_application_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="replan_application_failed",
                    bundle=bundle,
                    error=error,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
            return jsonify(_replan_application_error_summary(
                project_id=project_id,
                run_id=run_id,
                error=error,
                permission=permission,
            )), status_code

    @app.post("/api/internal/run-plan/resume-intent/validate")
    def internal_resume_intent_validate():
        if not internal_resume_intent_validation_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal resume intent validation route disabled"}), 404
        actor = ActorContext(actor="", source="missing", required=True)
        project_id = ""
        run_id = ""
        permission: dict[str, Any] | None = None
        try:
            payload = _request_json_object()
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            run_id = _safe_path_component(payload.get("run_id"), "run_id")
            approved_gates = (
                _optional_string_list(payload.get("approved_gates"), "approved_gates")
                if "approved_gates" in payload
                else None
            )
            permission = _decide_permission(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                action=INTERNAL_RUN_PLAN_RESUME_INTENT_USE_PERMISSION_ACTION,
            )
            if not bool(permission.get("allowed")):
                error = _permission_error_dict(permission)
                audit_error = _append_resume_intent_validation_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="resume_intent_validation_failed",
                    result=None,
                    error=error,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_resume_intent_validation_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    error=error,
                    permission=permission,
                )), 403
            audit_error = _append_resume_intent_validation_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="resume_intent_validation_requested",
                result=None,
                error=None,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            run_plan = _read_current_run_plan(projects, project_id=project_id, run_id=run_id)
            result = validate_resume_intent(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
                current_run_plan=run_plan,
                stage_state=_read_stage_state(projects, project_id=project_id, run_id=run_id),
                audit_records=_read_resume_intent_validation_audit_records(
                    projects,
                    project_id=project_id,
                    run_id=run_id,
                ),
                approved_gates=approved_gates,
            )
            memory_save = save_resume_intent_validation_summary_to_memory(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
                result=result,
                audit_refs=[RESUME_INTENT_VALIDATION_AUDIT_REF],
                confirmed_by=actor.actor,
            )
            audit_error = _append_resume_intent_validation_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="resume_intent_validation_completed",
                result=result,
                error=result.error,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(_resume_intent_validation_success_summary(
                result=result,
                memory_record=memory_save.record.model_dump(mode="json"),
                permission=permission,
            ))
        except (OSError, ValidationError, ValueError, FileNotFoundError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            error = _summary_error_dict(exc)
            if actor.actor and project_id and run_id:
                audit_error = _append_resume_intent_validation_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="resume_intent_validation_failed",
                    result=None,
                    error=error,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
            return jsonify(_resume_intent_validation_error_summary(
                project_id=project_id,
                run_id=run_id,
                error=error,
                permission=permission,
            )), status_code

    @app.post("/api/internal/run-plan/resume-intent/execute")
    def internal_resume_intent_execute():
        if not internal_resume_intent_execute_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal resume intent execute route disabled"}), 404
        actor = ActorContext(actor="", source="missing", required=True)
        project_id = ""
        run_id = ""
        permission: dict[str, Any] | None = None
        validation: ResumeIntentValidationResult | None = None
        try:
            payload = _request_json_object()
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            run_id = _safe_path_component(payload.get("run_id"), "run_id")
            approved_gates = _optional_string_list(payload.get("approved_gates"), "approved_gates")
            permission = _decide_permission(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                action=INTERNAL_RUN_PLAN_RESUME_EXECUTE_PERMISSION_ACTION,
            )
            if not bool(permission.get("allowed")):
                error = _permission_error_dict(permission)
                audit_error = _append_resume_intent_execute_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="run_plan_resume_failed",
                    result=None,
                    error=error,
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_resume_intent_execute_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    validation=None,
                    error=error,
                    permission=permission,
                )), 403
            run_plan = _read_current_run_plan(projects, project_id=project_id, run_id=run_id)
            validation = validate_resume_intent(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
                current_run_plan=run_plan,
                stage_state=_read_stage_state(projects, project_id=project_id, run_id=run_id),
                audit_records=_read_resume_intent_validation_audit_records(
                    projects,
                    project_id=project_id,
                    run_id=run_id,
                ),
                approved_gates=approved_gates,
            )
            if not validation.ok or validation.decision != "resume_eligible":
                error = {
                    "type": "resume_intent_not_eligible",
                    "message": f"resume intent validation decision must be resume_eligible, got {validation.decision}",
                }
                audit_error = _append_resume_intent_execute_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="run_plan_resume_failed",
                    result=validation,
                    error=error,
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_resume_intent_execute_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    validation=validation,
                    error=error,
                    permission=permission,
                )), 409
            audit_error = _append_resume_intent_execute_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="resume_intent_consumed",
                result=validation,
                error=None,
                permission=permission,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            try:
                execution = RunPlanExecutor(storage=projects).resume_after_gate(
                    project_id=project_id,
                    run_plan=run_plan,
                    approved_gates=validation.approved_gates,
                    actor=actor.actor,
                    note=str(payload.get("note") or ""),
                )
            except (ValidationError, ValueError, FileNotFoundError) as exc:
                error = _summary_error_dict(exc)
                audit_error = _append_resume_intent_execute_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="run_plan_resume_failed",
                    result=validation,
                    error=error,
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_resume_intent_execute_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    validation=validation,
                    error=error,
                    permission=permission,
                )), 400
            if _execution_failed(execution):
                error = _execution_error_dict(execution)
                audit_error = _append_resume_intent_execute_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="run_plan_resume_failed",
                    result=validation,
                    error=error,
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(_resume_intent_execute_error_summary(
                    project_id=project_id,
                    run_id=run_id,
                    validation=validation,
                    error=error,
                    permission=permission,
                    execution=execution,
                )), 500
            memory_save = save_resume_intent_validation_summary_to_memory(
                workspace_dir=projects.workspace_dir,
                project_id=project_id,
                run_id=run_id,
                result=validation,
                audit_refs=[RESUME_INTENT_VALIDATION_AUDIT_REF],
                confirmed_by=actor.actor,
            )
            audit_error = _append_resume_intent_execute_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                event="run_plan_resume_completed",
                result=validation,
                error=None,
                permission=permission,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(_resume_intent_execute_success_summary(
                result=validation,
                execution=execution,
                memory_record=memory_save.record.model_dump(mode="json"),
                permission=permission,
            ))
        except (OSError, ValidationError, ValueError, FileNotFoundError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            error = _summary_error_dict(exc)
            if actor.actor and project_id and run_id:
                audit_error = _append_resume_intent_execute_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    event="run_plan_resume_failed",
                    result=validation,
                    error=error,
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
            return jsonify(_resume_intent_execute_error_summary(
                project_id=project_id,
                run_id=run_id,
                validation=validation,
                error=error,
                permission=permission,
            )), status_code

    @app.post("/api/internal/run-plan/queue/execute")
    def internal_run_plan_queue_execute():
        if not internal_run_plan_queue_route_enabled(current_app):
            return jsonify({"ok": False, "error": "internal run-plan queue route disabled"}), 404
        payload: dict[str, Any] = {}
        actor = ActorContext(actor="", source="missing", required=True)
        project_id = ""
        run_id = ""
        try:
            payload = _request_json_object()
            project_id = _audit_string(payload.get("project_id"))
            run_plan_payload = payload.get("run_plan")
            if isinstance(run_plan_payload, dict):
                run_id = _audit_string(run_plan_payload.get("run_id"))
            actor = resolve_actor(request, required=True)
            if not actor.actor:
                raise _RouteRequestError("actor required", status_code=403)
            project_id = _safe_path_component(payload.get("project_id"), "project_id")
            if not isinstance(run_plan_payload, dict):
                raise ValueError("run_plan object required")
            run_plan = RunPlan.model_validate(run_plan_payload)
            run_id = _safe_path_component(run_plan.run_id, "run_id")
            input_artifacts = _optional_object(payload.get("input_artifacts"), "input_artifacts")
            task_options = _optional_object(payload.get("task_options"), "task_options")
            if "queue_dir" in payload:
                raise ValueError("queue_dir is not accepted by the internal run-plan queue route")
            max_iterations = _max_iterations(payload.get("max_iterations"))
            permission = _decide_permission(projects, actor=actor, project_id=project_id, run_id=run_id)
            if not bool(permission.get("allowed")):
                summary = _permission_denied_summary(permission)
                audit_error = _append_audit_or_error(
                    projects,
                    actor=actor,
                    project_id=project_id,
                    run_id=run_id,
                    outcome="permission_denied",
                    status_code=403,
                    queued_job_id="",
                    error=_permission_error_dict(permission),
                    permission=permission,
                )
                if audit_error is not None:
                    return jsonify(audit_error), 500
                return jsonify(summary), 403
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome="requested",
                status_code=202,
                queued_job_id="",
                error=None,
                permission=permission,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            queue = WorkerQueue(JsonWorkerQueueStore(_queue_dir(projects, project_id, run_id)))
            summary = run_run_plan_via_local_queue(
                queue=queue,
                storage=projects,
                project_id=project_id,
                run_plan=run_plan,
                input_artifacts=input_artifacts,
                task_options=task_options,
                max_iterations=max_iterations,
                require_empty_queue=False,
                target_project_id=project_id,
                target_run_id=run_id,
                executor_factory=_executor_factory(current_app),
            )
            status_code = 200 if bool(summary.get("ok")) and bool(summary.get("terminal")) else 400
            outcome = _terminal_audit_outcome(summary, status_code=status_code)
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome=outcome,
                status_code=status_code,
                queued_job_id=str(summary.get("queued_job_id") or ""),
                error=_summary_error(summary),
                permission=permission,
                waiting_metadata=_waiting_audit_metadata(summary),
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(summary), status_code
        except (OSError, ValidationError, ValueError) as exc:
            status_code = exc.status_code if isinstance(exc, _RouteRequestError) else 400
            summary = _error_summary(exc)
            audit_error = _append_audit_or_error(
                projects,
                actor=actor,
                project_id=project_id,
                run_id=run_id,
                outcome="validation_error",
                status_code=status_code,
                queued_job_id="",
                error=_summary_error_dict(exc),
                permission=None,
            )
            if audit_error is not None:
                return jsonify(audit_error), 500
            return jsonify(summary), status_code


class _RouteRequestError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RunPlanQueuePermissionPolicy(PermissionPolicy):
    def resolve(self, action: str) -> PermissionLevel:
        if str(action or "").strip() in {
            INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION,
            INTERNAL_RUN_PLAN_REPLAN_APPLY_PERMISSION_ACTION,
            INTERNAL_RUN_PLAN_RESUME_INTENT_USE_PERMISSION_ACTION,
            INTERNAL_RUN_PLAN_RESUME_EXECUTE_PERMISSION_ACTION,
        }:
            return PermissionLevel.PROJECT_APPROVED
        return super().resolve(action)


def internal_run_plan_queue_route_enabled(app: Any) -> bool:
    if INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG in app.config:
        return truthy(app.config.get(INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG))
    env_value = os.environ.get(INTERNAL_RUN_PLAN_QUEUE_ROUTE_FLAG)
    return truthy(env_value)


def internal_resume_intent_validation_route_enabled(app: Any) -> bool:
    if INTERNAL_RESUME_INTENT_VALIDATION_ROUTE_FLAG in app.config:
        return truthy(app.config.get(INTERNAL_RESUME_INTENT_VALIDATION_ROUTE_FLAG))
    env_value = os.environ.get(INTERNAL_RESUME_INTENT_VALIDATION_ROUTE_FLAG)
    return truthy(env_value)


def internal_resume_intent_execute_route_enabled(app: Any) -> bool:
    if INTERNAL_RESUME_INTENT_EXECUTE_ROUTE_FLAG in app.config:
        return truthy(app.config.get(INTERNAL_RESUME_INTENT_EXECUTE_ROUTE_FLAG))
    env_value = os.environ.get(INTERNAL_RESUME_INTENT_EXECUTE_ROUTE_FLAG)
    return truthy(env_value)


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _safe_path_component(value: object, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} required")
    if clean in {".", ".."} or "/" in clean or "\\" in clean:
        raise ValueError(f"{label} must be a safe path component")
    return clean


def _audit_string(value: object) -> str:
    return str(value or "").strip()


def _optional_object(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _optional_string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} must contain only strings")
        clean = item.strip()
        if not clean:
            raise ValueError(f"{label} must not contain empty strings")
        if clean in seen:
            raise ValueError(f"{label} must not contain duplicate values")
        seen.add(clean)
        cleaned.append(clean)
    return cleaned


def _replan_application_request_payload(
    payload: dict[str, Any],
    *,
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    actor_fields = {"actor", "approved_by", "revoked_by", "confirmed_by"}
    clean = {str(key): value for key, value in payload.items() if str(key) not in actor_fields}
    clean["project_id"] = project_id
    clean["run_id"] = run_id
    return clean


def _max_iterations(value: object) -> int:
    if value is None:
        return 10
    if isinstance(value, bool):
        raise ValueError("max_iterations must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max_iterations must be positive")
    return parsed


def _queue_dir(projects: ProjectStorage, project_id: str, run_id: str) -> Path:
    return internal_run_plan_queue_dir(projects.workspace_dir, project_id, run_id)


def _executor_factory(app: Any) -> ExecutorFactory | None:
    factory = app.config.get(EXECUTOR_FACTORY_CONFIG_KEY)
    return factory if callable(factory) else None


def _decide_permission(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    action: str = INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION,
) -> dict[str, Any]:
    return decide_server_permission(
        ServerPermissionStore(projects.workspace_dir),
        _RunPlanQueuePermissionPolicy(),
        action,
        project_id=project_id,
        run_id=run_id,
        actor=actor.actor,
        actor_source=actor.source,
        allow_legacy_client_flags=False,
    )


def _append_audit_or_error(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    outcome: str,
    status_code: int,
    queued_job_id: str,
    error: dict[str, Any] | None,
    permission: dict[str, Any] | None,
    waiting_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        _append_audit_event(
            projects,
            actor=actor,
            project_id=project_id,
            run_id=run_id,
            outcome=outcome,
            status_code=status_code,
            queued_job_id=queued_job_id,
            error=error,
            permission=permission,
            waiting_metadata=waiting_metadata,
        )
    except OSError as exc:
        return build_run_plan_queue_execution_summary(
            ok=False,
            terminal=False,
            error={
                "type": "audit_write_failed",
                "message": str(exc).strip() or exc.__class__.__name__,
            },
        )
    return None


def _append_replan_application_audit_or_error(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    event: str,
    bundle: RunPlanApplicationArtifactBundle | None,
    error: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        append_replan_application_audit_record(
            workspace_dir=projects.workspace_dir,
            project_id=project_id,
            run_id=run_id,
            event=event,  # type: ignore[arg-type]
            actor=actor.actor,
            actor_source=actor.source,
            bundle=bundle,
            error=error,
        )
    except OSError as exc:
        return _replan_application_error_summary(
            project_id=project_id,
            run_id=run_id,
            error={
                "type": "audit_write_failed",
                "message": str(exc).strip() or exc.__class__.__name__,
            },
            permission=None,
        )
    return None


def _append_resume_intent_validation_audit_or_error(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    event: str,
    result: ResumeIntentValidationResult | None,
    error: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        append_resume_intent_validation_audit_record(
            workspace_dir=projects.workspace_dir,
            project_id=project_id,
            run_id=run_id,
            event=event,  # type: ignore[arg-type]
            actor=actor.actor,
            actor_source=actor.source,
            result=result,
            error=error,
        )
    except OSError as exc:
        return _resume_intent_validation_error_summary(
            project_id=project_id,
            run_id=run_id,
            error={
                "type": "audit_write_failed",
                "message": str(exc).strip() or exc.__class__.__name__,
            },
            permission=None,
        )
    return None


def _append_resume_intent_execute_audit_or_error(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    event: str,
    result: ResumeIntentValidationResult | None,
    error: dict[str, Any] | None,
    permission: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        append_resume_intent_validation_audit_record(
            workspace_dir=projects.workspace_dir,
            project_id=project_id,
            run_id=run_id,
            event=event,  # type: ignore[arg-type]
            actor=actor.actor,
            actor_source=actor.source,
            result=result,
            error=error,
        )
    except OSError as exc:
        return _resume_intent_execute_error_summary(
            project_id=project_id,
            run_id=run_id,
            validation=result,
            error={
                "type": "audit_write_failed",
                "message": str(exc).strip() or exc.__class__.__name__,
            },
            permission=permission,
        )
    return None


def _append_audit_event(
    projects: ProjectStorage,
    *,
    actor: ActorContext,
    project_id: str,
    run_id: str,
    outcome: str,
    status_code: int,
    queued_job_id: str,
    error: dict[str, Any] | None,
    permission: dict[str, Any] | None,
    waiting_metadata: dict[str, Any] | None = None,
) -> None:
    permission_metadata = _permission_audit_metadata(permission, project_id=project_id, run_id=run_id)
    waiting = waiting_metadata or {"waiting_user": False, "waiting_task": "", "required_gates": []}
    record = {
        "event": "internal_run_plan_queue_execute",
        "timestamp": now_iso(),
        "actor": actor.actor,
        "actor_source": actor.source,
        "project_id": project_id,
        "run_id": run_id,
        "route": "/api/internal/run-plan/queue/execute",
        "feature_flag_enabled": True,
        "outcome": outcome,
        "status_code": int(status_code),
        "queued_job_id": queued_job_id,
        "error": error,
        "waiting_user": bool(waiting.get("waiting_user")),
        "waiting_task": str(waiting.get("waiting_task") or ""),
        "required_gates": [
            str(item).strip()
            for item in waiting.get("required_gates", [])
            if str(item).strip()
        ] if isinstance(waiting.get("required_gates"), list) else [],
        **permission_metadata,
    }
    path = _audit_path(projects)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _audit_path(projects: ProjectStorage) -> Path:
    base = (projects.workspace_dir / ".ai4s_internal" / "audit").resolve()
    path = (base / "internal_run_plan_queue_audit.jsonl").resolve()
    if path.parent != base:
        raise ValueError("internal run-plan queue audit path must stay under workspace")
    return path


def _summary_error(summary: dict[str, Any]) -> dict[str, Any] | None:
    raw_error = summary.get("error")
    if isinstance(raw_error, dict):
        return {
            "type": str(raw_error.get("type") or "execution_error"),
            "message": str(raw_error.get("message") or "execution failed"),
        }
    final_job = summary.get("final_job")
    if isinstance(final_job, dict):
        job_error = final_job.get("error")
        if isinstance(job_error, dict):
            message = str(job_error.get("reason") or job_error.get("message") or "").strip()
            if message:
                return {"type": "execution_error", "message": message}
    return None


def _waiting_audit_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    gates = summary.get("required_gates")
    return {
        "waiting_user": bool(summary.get("waiting_user")),
        "waiting_task": str(summary.get("waiting_task") or ""),
        "required_gates": [str(item).strip() for item in gates if str(item).strip()] if isinstance(gates, list) else [],
    }


def _terminal_audit_outcome(summary: dict[str, Any], *, status_code: int) -> str:
    if status_code == 200 and bool(summary.get("waiting_user")):
        return "waiting_user"
    if status_code == 200:
        return "succeeded"
    return "failed"


def _summary_error_dict(exc: BaseException) -> dict[str, Any]:
    return {
        "type": "validation_error",
        "message": str(exc).strip() or exc.__class__.__name__,
    }


def _read_current_run_plan(
    projects: ProjectStorage,
    *,
    project_id: str,
    run_id: str,
) -> RunPlan:
    run_dir = projects.run_dir(project_id, run_id)
    for filename in ("run_plan.json", "plan.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        payload = _read_json_object(path, label=filename)
        return RunPlan.model_validate(payload)
    raise FileNotFoundError("current RunPlan not found: run_plan.json or plan.json")


def _read_stage_state(
    projects: ProjectStorage,
    *,
    project_id: str,
    run_id: str,
) -> StageState | None:
    return projects.read_stage_state(project_id, run_id)


def _read_resume_intent_validation_audit_records(
    projects: ProjectStorage,
    *,
    project_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    path = projects.run_dir(project_id, run_id) / RESUME_INTENT_VALIDATION_AUDIT_REF
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            records.append({str(key): value for key, value in loaded.items()})
    return records


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _replan_application_success_summary(
    *,
    bundle: RunPlanApplicationArtifactBundle,
    memory_record: dict[str, Any],
    permission: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "project_id": bundle.project_id,
        "run_id": bundle.run_id,
        "application": {
            "application_id": bundle.application_record.application_id,
            "proposal_hash": bundle.application_record.proposal_hash,
            "selected_action": bundle.application_record.selected_action,
            "selected_operation_ids": list(bundle.application_record.selected_operation_ids),
            "result_type": bundle.application_record.result_type,
            "result_artifact_id": bundle.result_artifact_id,
            "result_ref": bundle.application_record.result_ref,
            "artifact_ids": list(bundle.artifact_ids),
            "artifacts": dict(bundle.artifacts),
            "required_gates": list(bundle.compiled.required_gates),
            "executable": False,
        },
        "audit_refs": [REPLAN_APPLICATION_AUDIT_REF],
        "memory": memory_record,
        "permission": _public_permission_decision(permission, project_id=bundle.project_id, run_id=bundle.run_id),
        "executable": False,
    }


def _replan_application_error_summary(
    *,
    project_id: str,
    run_id: str,
    error: dict[str, Any],
    permission: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "project_id": project_id,
        "run_id": run_id,
        "application": None,
        "audit_refs": [REPLAN_APPLICATION_AUDIT_REF] if project_id and run_id else [],
        "memory": None,
        "error": error,
        "executable": False,
    }
    if permission is not None:
        result["permission"] = _public_permission_decision(permission, project_id=project_id, run_id=run_id)
    return result


def _resume_intent_validation_success_summary(
    *,
    result: ResumeIntentValidationResult,
    memory_record: dict[str, Any],
    permission: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "project_id": result.project_id,
        "run_id": result.run_id,
        "validation": result.model_dump(mode="json"),
        "audit_refs": [RESUME_INTENT_VALIDATION_AUDIT_REF],
        "memory": memory_record,
        "permission": _public_permission_decision(permission, project_id=result.project_id, run_id=result.run_id),
        "executable": False,
    }


def _resume_intent_validation_error_summary(
    *,
    project_id: str,
    run_id: str,
    error: dict[str, Any],
    permission: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "project_id": project_id,
        "run_id": run_id,
        "validation": None,
        "audit_refs": [RESUME_INTENT_VALIDATION_AUDIT_REF] if project_id and run_id else [],
        "memory": None,
        "error": error,
        "executable": False,
    }
    if permission is not None:
        result["permission"] = _public_permission_decision(permission, project_id=project_id, run_id=run_id)
    return result


def _resume_intent_execute_success_summary(
    *,
    result: ResumeIntentValidationResult,
    execution: dict[str, Any],
    memory_record: dict[str, Any],
    permission: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "project_id": result.project_id,
        "run_id": result.run_id,
        "validation": result.model_dump(mode="json"),
        "execution": execution,
        "audit_refs": [RESUME_INTENT_VALIDATION_AUDIT_REF],
        "memory": memory_record,
        "permission": _public_permission_decision(permission, project_id=result.project_id, run_id=result.run_id),
        "executable": False,
    }


def _resume_intent_execute_error_summary(
    *,
    project_id: str,
    run_id: str,
    validation: ResumeIntentValidationResult | None,
    error: dict[str, Any],
    permission: dict[str, Any] | None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "project_id": project_id,
        "run_id": run_id,
        "validation": validation.model_dump(mode="json") if validation is not None else None,
        "execution": execution,
        "audit_refs": [RESUME_INTENT_VALIDATION_AUDIT_REF] if project_id and run_id else [],
        "memory": None,
        "error": error,
        "executable": False,
    }
    if permission is not None:
        result["permission"] = _public_permission_decision(permission, project_id=project_id, run_id=run_id)
    return result


def _execution_failed(execution: dict[str, Any]) -> bool:
    return execution.get("ok") is False or str(execution.get("status") or "").upper() == "FAILED"


def _execution_error_dict(execution: dict[str, Any]) -> dict[str, Any]:
    error = execution.get("error")
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("error") or "").strip()
    if not message:
        message = str(execution.get("message") or "run-plan resume execution failed").strip()
    result = {
        "type": "run_plan_resume_execution_failed",
        "message": message,
        "status": str(execution.get("status") or "").strip(),
        "failed_task": str(execution.get("failed_task") or "").strip(),
    }
    return {key: value for key, value in result.items() if value}


def _permission_error_dict(permission: dict[str, Any]) -> dict[str, Any]:
    reason = str(permission.get("reason") or "permission_denied")
    return {
        "type": "permission_denied",
        "message": f"permission denied: {reason}",
    }


def _permission_denied_summary(permission: dict[str, Any]) -> dict[str, Any]:
    return build_run_plan_queue_execution_summary(
        ok=False,
        terminal=False,
        error=_permission_error_dict(permission),
    )


def _permission_audit_metadata(
    permission: dict[str, Any] | None,
    *,
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    if permission is None:
        resource = _permission_resource(project_id, run_id)
        return {
            "permission_allowed": False,
            "permission_reason": "",
            "permission_action": INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION,
            "permission_resource": resource,
            "permission_scope": resource,
            "permission_grant_id": "",
            "permission_server_authorized": False,
        }
    resource = _permission_resource(project_id, run_id)
    return {
        "permission_allowed": bool(permission.get("allowed")),
        "permission_reason": str(permission.get("reason") or ""),
        "permission_action": str(permission.get("action") or INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION),
        "permission_resource": resource,
        "permission_scope": resource,
        "permission_grant_id": str(permission.get("grant_id") or ""),
        "permission_server_authorized": bool(permission.get("server_authorized")),
    }


def _permission_resource(project_id: str, run_id: str) -> str:
    if project_id and run_id:
        return f"project:{project_id}:run:{run_id}"
    if project_id:
        return f"project:{project_id}"
    return ""


def _public_permission_decision(permission: dict[str, Any], *, project_id: str, run_id: str) -> dict[str, Any]:
    result = {
        "allowed": bool(permission.get("allowed")),
        "reason": str(permission.get("reason") or ""),
        "action": str(permission.get("action") or INTERNAL_RUN_PLAN_QUEUE_PERMISSION_ACTION),
        "resource": _permission_resource(project_id, run_id),
        "grant_id": str(permission.get("grant_id") or ""),
        "server_authorized": bool(permission.get("server_authorized")),
    }
    return result


def _error_summary(exc: BaseException) -> dict[str, Any]:
    return build_run_plan_queue_execution_summary(
        ok=False,
        terminal=False,
        error=_summary_error_dict(exc),
    )
