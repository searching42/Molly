from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.actor_identity import resolve_authenticated_actor
from ai4s_agent.harness_tracing import HarnessTracer
from ai4s_agent.remote_resource_authority import (
    RemoteResourceAuthorityPolicyStore,
    RemoteResourceAuthorityService,
)
from ai4s_agent.resource_profiles import ResourceProfileStore
from ai4s_agent.routes.remote_resource_authorities import (
    register_remote_resource_authority_routes,
)
from ai4s_agent.schemas import AgentPlanAuthorizationRequest, _agent_digest_value
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationConflict,
    ScientificAgentAuthorizationDenied,
    ScientificAgentAuthorizationError,
    ScientificAgentAuthorizationService,
    ScientificAgentAuthorizationVerificationError,
)
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.storage import ProjectStorage


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _fixed_error(message: str, status: int, *, reason_code: str | None = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if reason_code:
        payload.update({"outcome": "DENY", "reason_codes": [reason_code]})
    return jsonify(payload), status


def _expected_digest_payload() -> str:
    payload = _json_object()
    if set(payload) != {"expected_proposal_digest"}:
        raise ValueError("permission evaluation accepts only expected_proposal_digest")
    return _agent_digest_value(
        payload.get("expected_proposal_digest"),
        field="expected_proposal_digest",
    )


def _authorization_payload() -> AgentPlanAuthorizationRequest:
    payload = _json_object()
    allowed = {
        "expected_proposal_digest",
        "authorization_mode",
        "requested_preauthorized_gate_ids",
        "confirmed",
        "client_request_id",
        "note",
    }
    if set(payload).difference(allowed):
        raise ValueError("authorization request contains an unsupported field")
    return AgentPlanAuthorizationRequest.model_validate(payload)


def _server_actor() -> tuple[str, str]:
    actor = resolve_authenticated_actor(request, required=True)
    # The strict body schema rejects actor aliases before this resolver is
    # called.  The authenticated resolver also excludes body/query/form and
    # X-Actor assertions, so only server-owned principals reach this route.
    if not actor.actor:
        return "", "missing"
    return actor.actor, actor.source


def register_scientific_agent_permission_routes(
    app: Flask,
    *,
    projects: ProjectStorage,
    proposal_store: ScientificAgentPlanProposalStore,
    resource_profiles: ResourceProfileStore,
    resource_authority_policy_store: RemoteResourceAuthorityPolicyStore,
    tracer: HarnessTracer | None = None,
) -> None:
    control_store = AgentPlanControlStore(storage=projects)
    resource_authorities = RemoteResourceAuthorityService(
        proposal_store=proposal_store,
        resource_profiles=resource_profiles,
        policy_store=resource_authority_policy_store,
        control_store=control_store,
    )
    service = ScientificAgentAuthorizationService(
        storage=projects,
        proposal_store=proposal_store,
        control_store=control_store,
        resource_authority_resolver=lambda publication, task_id: (
            resource_authorities.current_authority(
                publication=publication,
                task_id=task_id,
            )
        ),
        tracer=tracer,
    )
    app.extensions["scientific_agent_plan_control_store"] = control_store
    app.extensions["scientific_agent_authorization_service"] = service
    app.extensions["remote_resource_authority_policy_store"] = (
        resource_authority_policy_store
    )
    app.extensions["remote_resource_authority_service"] = resource_authorities
    register_remote_resource_authority_routes(
        app,
        service=resource_authorities,
        control_store=control_store,
    )

    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/permission-evaluations"
    )
    def evaluate_scientific_agent_permission(project_id: str, proposal_id: str):
        try:
            expected_digest = _expected_digest_payload()
            decision = service.evaluate_permission(
                project_id=project_id,
                proposal_id=proposal_id,
                expected_proposal_digest=expected_digest,
            )
        except FileNotFoundError:
            return _fixed_error("proposal not found", 404, reason_code="PROPOSAL_VERIFICATION_FAILED")
        except ScientificAgentPlanSourceChanged:
            return _fixed_error("proposal source is stale", 409, reason_code="PROPOSAL_SOURCE_STALE")
        except ScientificAgentAuthorizationVerificationError:
            return _fixed_error(
                "proposal failed current exact verification",
                409,
                reason_code="PROPOSAL_VERIFICATION_FAILED",
            )
        except (ValueError, ValidationError):
            return _fixed_error(
                "invalid permission evaluation request",
                400,
                reason_code="CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        return jsonify(
            {
                "ok": True,
                "outcome": decision.outcome.value,
                "decision_id": decision.decision_id,
                "decision": decision.model_dump(mode="json"),
                "executable": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-permission-decisions/<decision_id>"
    )
    def read_scientific_agent_permission(project_id: str, decision_id: str):
        try:
            decision = control_store.read_permission_decision(
                project_id=project_id,
                decision_id=decision_id,
            )
        except FileNotFoundError:
            return _fixed_error("permission decision not found", 404)
        except ScientificAgentAuthorizationError:
            return _fixed_error("permission decision failed exact verification", 409)
        return jsonify(
            {
                "ok": True,
                "decision_id": decision.decision_id,
                "decision": decision.model_dump(mode="json"),
                "executable": False,
            }
        )

    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/authorizations"
    )
    def authorize_scientific_agent_plan(project_id: str, proposal_id: str):
        try:
            parsed = _authorization_payload()
            actor, actor_source = _server_actor()
            if not actor:
                return _fixed_error(
                    "plan authorization requires a server-resolved actor",
                    403,
                    reason_code="AUTHORIZATION_ACTOR_REQUIRED",
                )
            authorization = service.authorize(
                project_id=project_id,
                proposal_id=proposal_id,
                request=parsed,
                actor=actor,
                actor_source=actor_source,
            )
        except FileNotFoundError:
            return _fixed_error("proposal not found", 404, reason_code="PROPOSAL_VERIFICATION_FAILED")
        except ScientificAgentAuthorizationDenied as exc:
            return jsonify(
                {
                    "ok": False,
                    "authorized": False,
                    "outcome": exc.decision.outcome.value,
                    "reason_codes": exc.decision.reason_codes,
                    "permission_decision_id": exc.decision.decision_id,
                    "decision": exc.decision.model_dump(mode="json"),
                    "dispatched": False,
                }
            ), 403
        except ScientificAgentAuthorizationConflict:
            return _fixed_error(
                "client request or authority ID is bound to different content",
                409,
                reason_code="CLIENT_REQUEST_CONFLICT",
            )
        except ScientificAgentPlanSourceChanged:
            return _fixed_error(
                "proposal source is stale",
                409,
                reason_code="PROPOSAL_SOURCE_STALE",
            )
        except ScientificAgentAuthorizationVerificationError:
            return _fixed_error(
                "proposal or authorization failed current exact verification",
                409,
                reason_code="PROPOSAL_VERIFICATION_FAILED",
            )
        except (ValidationError, ValueError):
            return _fixed_error(
                "invalid exact authorization request",
                400,
                reason_code="CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        return jsonify(
            {
                "ok": True,
                "authorized": True,
                "authorization_id": authorization.authorization_id,
                "authorization": authorization.model_dump(mode="json"),
                "start_intent_created": False,
                "dispatched": False,
                "executable": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-plan-authorizations/<authorization_id>"
    )
    def read_scientific_agent_authorization(project_id: str, authorization_id: str):
        try:
            authorization = service.verify_authorization(
                project_id=project_id,
                authorization_id=authorization_id,
                verify_current=True,
            )
        except FileNotFoundError:
            return _fixed_error("plan authorization not found", 404)
        except (ScientificAgentAuthorizationError, ScientificAgentPlanSourceChanged):
            return _fixed_error("plan authorization failed current exact verification", 409)
        return jsonify(
            {
                "ok": True,
                "authorization_id": authorization.authorization_id,
                "authorization": authorization.model_dump(mode="json"),
                "executable": False,
            }
        )

    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/approve-and-start"
    )
    def approve_and_start_scientific_agent_plan(project_id: str, proposal_id: str):
        try:
            parsed = _authorization_payload()
            actor, actor_source = _server_actor()
            if not actor:
                return _fixed_error(
                    "approve-and-start requires a server-resolved actor",
                    403,
                    reason_code="AUTHORIZATION_ACTOR_REQUIRED",
                )
            result = service.approve_and_start(
                project_id=project_id,
                proposal_id=proposal_id,
                request=parsed,
                actor=actor,
                actor_source=actor_source,
            )
        except FileNotFoundError:
            return _fixed_error("proposal not found", 404, reason_code="PROPOSAL_VERIFICATION_FAILED")
        except ScientificAgentAuthorizationDenied as exc:
            return jsonify(
                {
                    "ok": False,
                    "authorized": False,
                    "start_intent_created": False,
                    "outcome": exc.decision.outcome.value,
                    "reason_codes": exc.decision.reason_codes,
                    "permission_decision_id": exc.decision.decision_id,
                    "decision": exc.decision.model_dump(mode="json"),
                    "dispatched": False,
                }
            ), 403
        except ScientificAgentAuthorizationConflict:
            return _fixed_error(
                "client request or start-intent slot is bound to different content",
                409,
                reason_code="CLIENT_REQUEST_CONFLICT",
            )
        except ScientificAgentPlanSourceChanged:
            return _fixed_error(
                "proposal source is stale",
                409,
                reason_code="PROPOSAL_SOURCE_STALE",
            )
        except ScientificAgentAuthorizationVerificationError:
            return _fixed_error(
                "proposal or authorization failed current exact verification",
                409,
                reason_code="PROPOSAL_VERIFICATION_FAILED",
            )
        except (ValidationError, ValueError):
            return _fixed_error(
                "invalid approve-and-start request",
                400,
                reason_code="CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        return jsonify(
            {
                "ok": True,
                "authorized": True,
                "authorization_id": result.authorization.authorization_id,
                "authorization": result.authorization.model_dump(mode="json"),
                "start_intent_created": True,
                "start_intent_id": result.start_intent.start_intent_id,
                "start_intent": result.start_intent.model_dump(mode="json"),
                "permission_decision_id": result.start_decision.decision_id,
                "dispatched": False,
                "executable": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-plan-start-intents/<start_intent_id>"
    )
    def read_scientific_agent_start_intent(project_id: str, start_intent_id: str):
        try:
            intent = service.verify_start_intent(
                project_id=project_id,
                start_intent_id=start_intent_id,
                verify_current=True,
            )
        except FileNotFoundError:
            return _fixed_error("start intent not found", 404)
        except (ScientificAgentAuthorizationError, ScientificAgentPlanSourceChanged):
            return _fixed_error("start intent failed current exact verification", 409)
        return jsonify(
            {
                "ok": True,
                "start_intent_id": intent.start_intent_id,
                "start_intent": intent.model_dump(mode="json"),
                "dispatched": False,
                "executable": False,
            }
        )

    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/permission-shadow-evaluations"
    )
    def evaluate_scientific_agent_shadow(project_id: str, proposal_id: str):
        try:
            expected_digest = _expected_digest_payload()
            record = service.evaluate_shadow(
                project_id=project_id,
                proposal_id=proposal_id,
                expected_proposal_digest=expected_digest,
            )
        except FileNotFoundError:
            return _fixed_error("proposal not found", 404, reason_code="PROPOSAL_VERIFICATION_FAILED")
        except ScientificAgentPlanSourceChanged:
            return _fixed_error(
                "proposal source is stale",
                409,
                reason_code="PROPOSAL_SOURCE_STALE",
            )
        except ScientificAgentAuthorizationVerificationError:
            return _fixed_error(
                "proposal failed current exact verification",
                409,
                reason_code="PROPOSAL_VERIFICATION_FAILED",
            )
        except (ValueError, ValidationError):
            return _fixed_error(
                "invalid shadow evaluation request",
                400,
                reason_code="CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        return jsonify(
            {
                "ok": True,
                "shadow_record_id": record.shadow_record_id,
                "shadow_record": record.model_dump(mode="json"),
                "executable": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-permission-shadow/<shadow_record_id>"
    )
    def read_scientific_agent_shadow(project_id: str, shadow_record_id: str):
        try:
            record = control_store.read_shadow_record(
                project_id=project_id,
                shadow_record_id=shadow_record_id,
            )
        except FileNotFoundError:
            return _fixed_error("permission shadow record not found", 404)
        except ScientificAgentAuthorizationError:
            return _fixed_error("permission shadow record failed exact verification", 409)
        return jsonify(
            {
                "ok": True,
                "shadow_record_id": record.shadow_record_id,
                "shadow_record": record.model_dump(mode="json"),
                "executable": False,
            }
        )


__all__ = ["register_scientific_agent_permission_routes"]
