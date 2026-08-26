from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.actor_identity import resolve_authenticated_actor
from ai4s_agent.llm_provider import LLMProviderError, LLMProviderManager
from ai4s_agent.llm_provider_resolution import (
    CONTROL_PLANE_ROLE,
    resolve_llm_provider_payload,
)
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.schemas import (
    AgentPlanFeedbackRequest,
    AgentPlanRevisionApplicationRequest,
)
from ai4s_agent.scientific_agent_replanner import (
    ScientificAgentReplannerConflict,
    ScientificAgentReplannerError,
    ScientificAgentReplannerOutcomeUnknown,
    ScientificAgentReplannerResponseInvalid,
    ScientificAgentReplannerService,
    ScientificAgentReplannerStale,
)


_REPLAN_FIELDS = {
    "run_id",
    "client_request_id",
    "trigger_kind",
    "baseline_proposal_id",
    "baseline_proposal_digest",
    "baseline_semantic_plan_id",
    "baseline_semantic_plan_digest",
    "baseline_run_plan_digest",
    "baseline_authorization_id",
    "baseline_authorization_digest",
    "feedback_receipt_id",
    "feedback_receipt_digest",
    "controller_execution_id",
    "controller_execution_digest",
    "controller_decision_id",
    "controller_decision_digest",
    "controller_receipt_id",
    "controller_receipt_digest",
    "tool_call_proposal_id",
    "tool_call_proposal_digest",
    "tool_call_application_receipt_id",
    "tool_call_application_receipt_digest",
    "external_llm_approved",
    "llm_provider",
}


def _payload() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _actor() -> tuple[str, str]:
    identity = resolve_authenticated_actor(request, required=True)
    if not identity.actor:
        raise PermissionError("trusted actor required")
    return identity.actor, identity.source


def _error(message: str, status: int, code: str):
    return jsonify({"ok": False, "error": message, "reason_codes": [code], "dispatched": False}), status


def register_scientific_agent_replanner_routes(
    app: Flask,
    *,
    service: ScientificAgentReplannerService,
    llm_settings: LLMSettingsStore,
    llm_providers: LLMProviderManager,
) -> None:
    app.extensions["scientific_agent_replanner_service"] = service

    @app.post("/api/projects/<project_id>/agent-plan-feedback")
    def create_agent_plan_feedback(project_id: str):
        try:
            payload = _payload()
            if set(payload).difference(
                {"run_id", "client_request_id", "feedback", "source_kind"}
            ):
                raise ValueError("unsupported feedback field")
            parsed = AgentPlanFeedbackRequest.model_validate(payload)
            actor, actor_source = _actor()
            receipt = service.create_feedback(
                project_id=project_id,
                request=parsed,
                actor=actor,
                actor_source=actor_source,
            )
            return jsonify(
                {
                    "ok": True,
                    "feedback_receipt_id": receipt.feedback_receipt_id,
                    "feedback_receipt": receipt.model_dump(mode="json"),
                    "advisory_only": True,
                    "authorized": False,
                    "dispatched": False,
                }
            ), 201
        except PermissionError:
            return _error("trusted authenticated actor required", 403, "REPLANNER_ACTOR_REQUIRED")
        except ScientificAgentReplannerConflict:
            return _error("feedback request conflicts with immutable content", 409, "REPLANNER_REQUEST_CONFLICT")
        except (ValidationError, ValueError):
            return _error("invalid dedicated feedback request", 400, "REPLANNER_FEEDBACK_INVALID")

    @app.post("/api/projects/<project_id>/agent-plan-revisions")
    def create_agent_plan_revision(project_id: str):
        try:
            payload = _payload()
            if set(payload).difference(_REPLAN_FIELDS):
                raise ValueError("unsupported replan field")
            forbidden = {"baseline_plan", "successor_plan", "dependency", "authority"}
            if forbidden.intersection(payload):
                raise ValueError("client authority field rejected")
            actor, actor_source = _actor()
            resolution = resolve_llm_provider_payload(
                payload,
                settings=llm_settings,
                providers=llm_providers,
                role=CONTROL_PLANE_ROLE,
            )
            service_payload = {key: value for key, value in payload.items() if key != "llm_provider"}
            with resolution.provider_context as provider:
                if provider is None:
                    raise LLMProviderError("Replanner provider unavailable")
                result = service.create_revision(
                    project_id=project_id,
                    payload=service_payload,
                    actor=actor,
                    actor_source=actor_source,
                    provider=provider,
                )
            revision = result.proposal
            return jsonify(
                {
                    "ok": True,
                    "revision_id": revision.revision_id,
                    "revision": revision.model_dump(mode="json"),
                    "proposed": True,
                    "review_required": revision.status == "review_required",
                    "blocking_questions": [
                        item.model_dump(mode="json") for item in revision.blocking_questions
                    ],
                    "no_material_change": revision.status == "no_material_change",
                    "applied": False,
                    "fresh_authorization_required": revision.plan_diff.material_change,
                    "dispatched": False,
                    "replayed": result.replayed,
                }
            ), 201
        except PermissionError:
            return _error("trusted authenticated actor required", 403, "REPLANNER_ACTOR_REQUIRED")
        except ScientificAgentReplannerOutcomeUnknown:
            return _error("Replanner provider outcome is unknown", 409, "REPLANNER_PROVIDER_OUTCOME_UNKNOWN")
        except ScientificAgentReplannerResponseInvalid:
            return _error("Replanner response failed strict validation", 502, "REPLANNER_RESPONSE_INVALID")
        except ScientificAgentReplannerStale:
            return _error("Replanner baseline or source is stale", 409, "REPLANNER_SOURCE_STALE")
        except ScientificAgentReplannerConflict:
            return _error("Replanner request conflicts with immutable content", 409, "REPLANNER_REQUEST_CONFLICT")
        except FileNotFoundError:
            return _error("Replanner authority not found", 404, "REPLANNER_AUTHORITY_NOT_FOUND")
        except ScientificAgentReplannerError:
            return _error("Replanner proposal was rejected", 409, "REPLANNER_PROPOSAL_REJECTED")
        except (ValidationError, ValueError, LLMProviderError):
            return _error("invalid Replanner request or provider consent", 400, "REPLANNER_REQUEST_INVALID")

    @app.get("/api/projects/<project_id>/agent-plan-revisions/<revision_id>")
    def read_agent_plan_revision(project_id: str, revision_id: str):
        try:
            revision = service.read_revision(project_id=project_id, revision_id=revision_id)
            return jsonify(
                {
                    "ok": True,
                    "revision_id": revision.revision_id,
                    "revision": revision.model_dump(mode="json"),
                    "review_required": revision.status == "review_required",
                    "no_material_change": revision.status == "no_material_change",
                    "applied": False,
                    "fresh_authorization_required": revision.plan_diff.material_change,
                    "dispatched": False,
                }
            )
        except FileNotFoundError:
            return _error("revision proposal not found", 404, "REPLANNER_PROPOSAL_NOT_FOUND")
        except ScientificAgentReplannerError:
            return _error("revision proposal failed exact verification", 409, "REPLANNER_PROPOSAL_INVALID")

    @app.post("/api/projects/<project_id>/agent-plan-revisions/<revision_id>/apply")
    def apply_agent_plan_revision(project_id: str, revision_id: str):
        try:
            payload = _payload()
            if set(payload).difference({"expected_revision_digest", "client_request_id"}):
                raise ValueError("unsupported application field")
            parsed = AgentPlanRevisionApplicationRequest.model_validate(payload)
            _actor()
            result = service.apply_revision(
                project_id=project_id,
                revision_id=revision_id,
                request=parsed,
            )
            return jsonify(
                {
                    "ok": True,
                    "applied": True,
                    "application_receipt": result.receipt.model_dump(mode="json"),
                    "successor_proposal": result.successor.model_dump(mode="json"),
                    "fresh_permission_required": result.receipt.fresh_permission_required,
                    "fresh_authorization_required": result.receipt.fresh_authorization_required,
                    "dispatched": False,
                    "replayed": result.replayed,
                }
            ), 201
        except PermissionError:
            return _error("trusted authenticated actor required", 403, "REPLANNER_ACTOR_REQUIRED")
        except FileNotFoundError:
            return _error("revision proposal not found", 404, "REPLANNER_PROPOSAL_NOT_FOUND")
        except ScientificAgentReplannerStale:
            return _error("revision application is stale", 409, "REPLANNER_APPLICATION_STALE")
        except ScientificAgentReplannerConflict:
            return _error("revision application conflicts with immutable content", 409, "REPLANNER_APPLICATION_CONFLICT")
        except (ValidationError, ValueError):
            return _error("invalid revision application", 400, "REPLANNER_APPLICATION_INVALID")

    @app.get("/api/projects/<project_id>/agent-plan-revision-applications/<receipt_id>")
    def read_agent_plan_revision_application(project_id: str, receipt_id: str):
        try:
            receipt = service.read_application(project_id=project_id, receipt_id=receipt_id)
            return jsonify(
                {
                    "ok": True,
                    "applied": True,
                    "application_receipt": receipt.model_dump(mode="json"),
                    "fresh_permission_required": receipt.fresh_permission_required,
                    "fresh_authorization_required": receipt.fresh_authorization_required,
                    "dispatched": False,
                }
            )
        except FileNotFoundError:
            return _error("revision application not found", 404, "REPLANNER_APPLICATION_NOT_FOUND")
        except ScientificAgentReplannerError:
            return _error("revision application failed exact verification", 409, "REPLANNER_APPLICATION_INVALID")
