from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.actor_identity import resolve_authenticated_actor
from ai4s_agent.schemas import (
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerStartRequest,
    AgentHarnessGateApprovalRequest,
    AgentHarnessRemoteApprovalRequest,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ControllerAdvanceResult,
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerConflict,
    ScientificAgentHarnessControllerError,
    ScientificAgentHarnessControllerRecoveryRequired,
    ScientificAgentHarnessControllerVerificationError,
)


class ControllerRequestValidationError(ValueError):
    pass


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ControllerRequestValidationError("request body must be a JSON object")
    return payload


def _parse(model: type[Any], allowed: set[str]):
    payload = _json_object()
    if set(payload).difference(allowed):
        raise ControllerRequestValidationError("Controller request contains an unsupported field")
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ControllerRequestValidationError("Controller request failed strict validation") from exc


def _actor() -> tuple[str, str]:
    resolved = resolve_authenticated_actor(request, required=True)
    return resolved.actor, resolved.source


def _response(result: ControllerAdvanceResult, *, status: int = 200):
    payload: dict[str, Any] = {
        "ok": True,
        "controller_execution_id": result.execution.controller_execution_id,
        "controller_execution_digest": result.execution.execution_digest,
        "execution": result.execution.model_dump(mode="json"),
        "inspection": result.inspection.model_dump(mode="json"),
    }
    if result.decision is not None:
        payload["decision"] = result.decision.model_dump(mode="json")
    if result.receipt is not None:
        payload["receipt"] = result.receipt.model_dump(mode="json")
    return jsonify(payload), status


def _error(message: str, status: int, reason_code: str):
    return jsonify({"ok": False, "error": message, "reason_codes": [reason_code]}), status


def register_scientific_agent_harness_controller_routes(
    app: Flask,
    *,
    controller: ScientificAgentHarnessController,
) -> None:
    app.extensions["scientific_agent_harness_controller"] = controller

    @app.post(
        "/api/projects/<project_id>/agent-plan-start-intents/<start_intent_id>/controller-executions"
    )
    def create_harness_controller_execution(project_id: str, start_intent_id: str):
        try:
            parsed = _parse(
                AgentHarnessControllerStartRequest,
                {"expected_start_intent_digest", "client_request_id"},
            )
            actor, actor_source = _actor()
            if not actor:
                return _error("Controller creation requires a server-resolved actor", 403, "CONTROLLER_ACTOR_REQUIRED")
            result = controller.create(
                project_id=project_id,
                start_intent_id=start_intent_id,
                request=parsed,
                actor=actor,
                actor_source=actor_source,
            )
            return _response(result, status=201)
        except FileNotFoundError:
            return _error("Controller authority not found", 404, "CONTROLLER_AUTHORITY_NOT_FOUND")
        except ScientificAgentHarnessControllerConflict:
            return _error("Controller request conflicts with current authority", 409, "CONTROLLER_REQUEST_CONFLICT")
        except ScientificAgentHarnessControllerVerificationError:
            return _error("Controller authority failed exact verification", 409, "CONTROLLER_AUTHORITY_STALE")
        except ScientificAgentHarnessControllerRecoveryRequired:
            return _error("Controller action requires explicit recovery", 409, "CONTROLLER_RECOVERY_REQUIRED")
        except ControllerRequestValidationError:
            return _error("invalid Controller creation request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except ValueError:
            return _error("Controller authority failed current verification", 409, "CONTROLLER_AUTHORITY_STALE")

    @app.get(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>"
    )
    def read_harness_controller_execution(project_id: str, controller_execution_id: str):
        try:
            return _response(
                controller.get(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                )
            )
        except FileNotFoundError:
            return _error("Controller execution not found", 404, "CONTROLLER_EXECUTION_NOT_FOUND")
        except ScientificAgentHarnessControllerError:
            return _error("Controller execution failed current verification", 409, "CONTROLLER_AUTHORITY_STALE")

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/advance"
    )
    def advance_harness_controller_execution(project_id: str, controller_execution_id: str):
        try:
            parsed = _parse(
                AgentHarnessControllerAdvanceRequest,
                {"expected_controller_execution_digest", "client_request_id"},
            )
            return _response(
                controller.advance(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=parsed,
                )
            )
        except FileNotFoundError:
            return _error("Controller execution not found", 404, "CONTROLLER_EXECUTION_NOT_FOUND")
        except ScientificAgentHarnessControllerRecoveryRequired:
            return _error("Controller action requires explicit recovery", 409, "CONTROLLER_RECOVERY_REQUIRED")
        except ScientificAgentHarnessControllerConflict:
            return _error("Controller request conflicts with current authority", 409, "CONTROLLER_REQUEST_CONFLICT")
        except ScientificAgentHarnessControllerVerificationError:
            return _error("Controller authority failed exact verification", 409, "CONTROLLER_AUTHORITY_STALE")
        except ControllerRequestValidationError:
            return _error("invalid Controller advance request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except ValueError:
            return _error("Controller action failed exact verification", 409, "CONTROLLER_AUTHORITY_STALE")

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/gates/<gate_id>/approve"
    )
    def approve_harness_controller_gate(project_id: str, controller_execution_id: str, gate_id: str):
        try:
            parsed = _parse(
                AgentHarnessGateApprovalRequest,
                {"expected_snapshot_id", "expected_snapshot_hash", "client_request_id", "note"},
            )
            actor, _ = _actor()
            if not actor:
                return _error("Gate approval requires a server-resolved actor", 403, "CONTROLLER_ACTOR_REQUIRED")
            return _response(
                controller.approve_gate(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    gate_id=gate_id,
                    request=parsed,
                    actor=actor,
                )
            )
        except FileNotFoundError:
            return _error("Controller Gate authority not found", 404, "CONTROLLER_AUTHORITY_NOT_FOUND")
        except ControllerRequestValidationError:
            return _error("invalid Controller Gate approval request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except (ScientificAgentHarnessControllerError, ValueError):
            return _error("Gate approval conflicts with current authority", 409, "CONTROLLER_GATE_CONFLICT")

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/remote-approvals"
    )
    def approve_harness_controller_remote(project_id: str, controller_execution_id: str):
        try:
            parsed = _parse(
                AgentHarnessRemoteApprovalRequest,
                {"expected_remote_request_sha256", "client_request_id", "note"},
            )
            actor, _ = _actor()
            if not actor:
                return _error("remote approval requires a server-resolved actor", 403, "CONTROLLER_ACTOR_REQUIRED")
            return _response(
                controller.approve_remote(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=parsed,
                    actor=actor,
                )
            )
        except FileNotFoundError:
            return _error("Controller remote authority not found", 404, "CONTROLLER_AUTHORITY_NOT_FOUND")
        except ControllerRequestValidationError:
            return _error("invalid Controller remote approval request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except (ScientificAgentHarnessControllerError, ValueError):
            return _error("remote approval conflicts with current authority", 409, "CONTROLLER_REMOTE_APPROVAL_CONFLICT")

    def _advance_control_payload() -> AgentHarnessControllerAdvanceRequest:
        return _parse(
            AgentHarnessControllerAdvanceRequest,
            {"expected_controller_execution_digest", "client_request_id"},
        )

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/cancel"
    )
    def cancel_harness_controller_execution(project_id: str, controller_execution_id: str):
        try:
            return _response(
                controller.cancel(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=_advance_control_payload(),
                )
            )
        except FileNotFoundError:
            return _error("Controller remote authority not found", 404, "CONTROLLER_AUTHORITY_NOT_FOUND")
        except ControllerRequestValidationError:
            return _error("invalid Controller cancel request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except (ScientificAgentHarnessControllerError, ValueError):
            return _error("Controller cancel conflicts with current authority", 409, "CONTROLLER_CANCEL_CONFLICT")

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/<controller_execution_id>/recover"
    )
    def recover_harness_controller_execution(project_id: str, controller_execution_id: str):
        try:
            return _response(
                controller.recover(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=_advance_control_payload(),
                )
            )
        except FileNotFoundError:
            return _error("Controller recovery authority not found", 404, "CONTROLLER_AUTHORITY_NOT_FOUND")
        except ControllerRequestValidationError:
            return _error("invalid Controller recovery request", 400, "CLIENT_AUTHORITY_FIELD_REJECTED")
        except (ScientificAgentHarnessControllerError, ValueError):
            return _error("Controller recovery conflicts with current authority", 409, "CONTROLLER_RECOVERY_CONFLICT")


__all__ = ["register_scientific_agent_harness_controller_routes"]
