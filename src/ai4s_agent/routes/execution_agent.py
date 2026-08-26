from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.execution_agent import (
    ExecutionAgentConflict,
    ExecutionAgentLLMFailed,
    ExecutionAgentLLMOutcomeUnknown,
    ExecutionAgentLLMResponseInvalid,
    ExecutionAgentLLMUnavailable,
    ExecutionAgentService,
    ExecutionAgentStale,
)
from ai4s_agent.execution_agent_store import (
    ExecutionAgentStoreConflict,
    ExecutionAgentStoreError,
    ExecutionAgentStoreVerificationError,
)
from ai4s_agent.execution_agent_v2 import (
    AgentToolCallApplicationRequestV2,
    AgentToolCallApplicationReceiptV2,
    AgentToolCallProposalRequestV2,
    ExecutionAgentV2Conflict,
    ExecutionAgentV2DecisionInvalid,
    ExecutionAgentV2Error,
    ExecutionAgentV2LLMOutcomeUnknown,
    ExecutionAgentV2LLMResponseInvalid,
    ExecutionAgentV2LLMUnavailable,
    ExecutionAgentV2Stale,
    LogicalToolCompilationError,
)
from ai4s_agent.llm_provider import LLMProviderError, LLMProviderManager
from ai4s_agent.llm_provider_resolution import (
    CONTROL_PLANE_ROLE,
    resolve_llm_provider_payload,
)
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.schemas import (
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessControllerConflict,
    ScientificAgentHarnessControllerError,
    ScientificAgentHarnessControllerRecoveryRequired,
    ScientificAgentHarnessControllerVerificationError,
)


class ExecutionAgentRequestValidationError(ValueError):
    pass


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ExecutionAgentRequestValidationError(
            "request body must be a JSON object"
        )
    return payload


def _parse(model: type[Any], allowed: set[str]):
    payload = _json_object()
    if set(payload).difference(allowed):
        raise ExecutionAgentRequestValidationError(
            "Execution Agent request contains an unsupported field"
        )
    try:
        return model.model_validate(payload), payload
    except ValidationError as exc:
        raise ExecutionAgentRequestValidationError(
            "Execution Agent request failed strict validation"
        ) from exc


def _error(message: str, status: int, reason_code: str):
    return (
        jsonify({"ok": False, "error": message, "reason_codes": [reason_code]}),
        status,
    )


def _proposal_payload(result: Any, *, status: int = 200):
    publication = result.publication
    return (
        jsonify(
            {
                "ok": True,
                "observation": publication.observation.model_dump(mode="json"),
                "tool_catalog": publication.tool_catalog.model_dump(mode="json"),
                "tool_call_proposal": publication.proposal.model_dump(mode="json"),
                "applied": result.applied,
                "dispatched": result.dispatched,
                **(
                    {"llm_used": result.llm_used}
                    if hasattr(result, "llm_used")
                    else {}
                ),
            }
        ),
        status,
    )


def _v2_proposal_payload(result: Any, *, status: int = 200):
    publication = result.publication
    return (
        jsonify(
            {
                "ok": True,
                "schema_version": "execution_agent_v2_route_response.v1",
                "observation": publication.observation.model_dump(mode="json"),
                "tool_catalog": publication.tool_catalog.model_dump(mode="json"),
                "tool_call_proposal": publication.proposal.model_dump(mode="json"),
                "applied": False,
                "dispatched": False,
                "llm_used": result.llm_used,
            }
        ),
        status,
    )


def _v2_application_payload(result: Any):
    receipt: AgentToolCallApplicationReceiptV2 = result.application_receipt
    payload: dict[str, Any] = {
        "ok": True,
        "schema_version": "execution_agent_v2_route_response.v1",
        "tool_call_proposal": result.publication.proposal.model_dump(mode="json"),
        "application_receipt": receipt.model_dump(mode="json"),
        "applied": receipt.outcome.value == "applied",
        "controller_advance_called": receipt.controller_advance_called,
        "dispatch_occurred": receipt.dispatch_occurred,
        "dispatched": receipt.dispatch_occurred,
    }
    if result.controller_result is not None:
        payload["controller_inspection"] = result.controller_result.inspection.model_dump(
            mode="json"
        )
        payload["controller_decision"] = (
            result.controller_result.decision.model_dump(mode="json")
            if result.controller_result.decision is not None
            else None
        )
        payload["controller_receipt"] = (
            result.controller_result.receipt.model_dump(mode="json")
            if result.controller_result.receipt is not None
            else None
        )
    return jsonify(payload)


def register_execution_agent_routes(
    app: Flask,
    *,
    service: ExecutionAgentService,
    v2_service: Any | None = None,
    llm_settings: LLMSettingsStore,
    llm_providers: LLMProviderManager,
) -> None:
    app.extensions["execution_agent_service"] = service
    if v2_service is not None:
        app.extensions["execution_agent_v2_service"] = v2_service

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/"
        "<controller_execution_id>/execution-agent-proposals"
    )
    def create_tool_call_proposal(project_id: str, controller_execution_id: str):
        try:
            raw_payload = _json_object()
            if raw_payload.get("schema_version") == "agent_tool_call_proposal_request.v2":
                if v2_service is None:
                    raise ExecutionAgentV2Error("Execution Agent v2 is unavailable")
                parsed_v2, payload_v2 = _parse(
                    AgentToolCallProposalRequestV2,
                    {
                        "schema_version",
                        "expected_controller_execution_digest",
                        "client_request_id",
                        "external_llm_approved",
                        "llm_provider",
                    },
                )
                resolution = resolve_llm_provider_payload(
                    payload_v2,
                    settings=llm_settings,
                    providers=llm_providers,
                    role=CONTROL_PLANE_ROLE,
                )
                with resolution.provider_context as provider:
                    result = v2_service.create_proposal(
                        project_id=project_id,
                        controller_execution_id=controller_execution_id,
                        request=parsed_v2,
                        provider=provider,
                        provider_binding_digest=resolution.provider_binding_digest,
                    )
                return _v2_proposal_payload(result, status=201)
            parsed, payload = _parse(
                AgentToolCallProposalRequest,
                {
                    "expected_controller_execution_digest",
                    "client_request_id",
                    "external_llm_approved",
                    "llm_provider",
                },
            )
            resolution = resolve_llm_provider_payload(
                payload,
                settings=llm_settings,
                providers=llm_providers,
                role=CONTROL_PLANE_ROLE,
            )
            with resolution.provider_context as provider:
                if provider is None:
                    raise ExecutionAgentLLMUnavailable(
                        "execution_agent_llm_unavailable"
                    )
                result = service.create_proposal(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=parsed,
                    provider=provider,
                    provider_binding_digest=resolution.provider_binding_digest,
                )
            return _proposal_payload(result, status=201)
        except FileNotFoundError:
            return _error(
                "Execution Agent Controller authority not found",
                404,
                "EXECUTION_AGENT_AUTHORITY_NOT_FOUND",
            )
        except ExecutionAgentLLMUnavailable:
            return _error(
                "Execution Agent LLM is unavailable",
                409,
                "EXECUTION_AGENT_LLM_UNAVAILABLE",
            )
        except LLMProviderError:
            return _error(
                "Execution Agent LLM is unavailable",
                409,
                "EXECUTION_AGENT_LLM_UNAVAILABLE",
            )
        except ExecutionAgentLLMOutcomeUnknown:
            return _error(
                "Execution Agent LLM outcome is unknown",
                409,
                "EXECUTION_AGENT_LLM_OUTCOME_UNKNOWN",
            )
        except ExecutionAgentLLMResponseInvalid:
            return _error(
                "Execution Agent LLM response failed strict validation",
                502,
                "EXECUTION_AGENT_LLM_RESPONSE_INVALID",
            )
        except ExecutionAgentLLMFailed:
            return _error(
                "Execution Agent LLM request failed",
                502,
                "EXECUTION_AGENT_LLM_FAILED",
            )
        except ExecutionAgentV2LLMUnavailable:
            return _error(
                "Execution Agent v2 LLM is unavailable",
                409,
                "EXECUTION_AGENT_LLM_UNAVAILABLE",
            )
        except ExecutionAgentV2LLMOutcomeUnknown:
            return _error(
                "Execution Agent v2 LLM outcome is unknown",
                409,
                "EXECUTION_AGENT_LLM_OUTCOME_UNKNOWN",
            )
        except ExecutionAgentV2LLMResponseInvalid:
            return _error(
                "Execution Agent v2 LLM response failed strict validation",
                502,
                "EXECUTION_AGENT_LLM_RESPONSE_INVALID",
            )
        except (LogicalToolCompilationError, ExecutionAgentV2DecisionInvalid):
            return _error(
                "Execution Agent v2 decision failed server validation",
                400,
                "EXECUTION_AGENT_V2_DECISION_INVALID",
            )
        except ExecutionAgentV2Stale:
            return _error(
                "Execution Agent v2 observation is stale",
                409,
                "EXECUTION_AGENT_OBSERVATION_STALE",
            )
        except ExecutionAgentV2Conflict:
            return _error(
                "Execution Agent v2 request conflicts with existing authority",
                409,
                "EXECUTION_AGENT_REQUEST_CONFLICT",
            )
        except ExecutionAgentV2Error:
            return _error(
                "Execution Agent v2 authority is invalid",
                400,
                "EXECUTION_AGENT_V2_REQUEST_INVALID",
            )
        except ExecutionAgentRequestValidationError:
            return _error(
                "invalid Execution Agent proposal request",
                400,
                "CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        except (ExecutionAgentStale, ScientificAgentHarnessControllerVerificationError):
            return _error(
                "Execution Agent observation is stale",
                409,
                "EXECUTION_AGENT_OBSERVATION_STALE",
            )
        except (
            ExecutionAgentConflict,
            ExecutionAgentStoreConflict,
            ScientificAgentHarnessControllerConflict,
        ):
            return _error(
                "Execution Agent request conflicts with existing authority",
                409,
                "EXECUTION_AGENT_REQUEST_CONFLICT",
            )
        except (ExecutionAgentStoreVerificationError, ExecutionAgentStoreError):
            return _error(
                "Execution Agent artifact failed exact verification",
                409,
                "EXECUTION_AGENT_ARTIFACT_INVALID",
            )
        except ScientificAgentHarnessControllerError:
            return _error(
                "Execution Agent Controller authority is stale",
                409,
                "EXECUTION_AGENT_OBSERVATION_STALE",
            )
        except (ValidationError, ValueError):
            return _error(
                "Execution Agent provider or authority is invalid",
                400,
                "EXECUTION_AGENT_REQUEST_INVALID",
            )

    @app.get(
        "/api/projects/<project_id>/agent-harness-controller-executions/"
        "<controller_execution_id>/execution-agent-proposals/<tool_call_proposal_id>"
    )
    def read_tool_call_proposal(
        project_id: str,
        controller_execution_id: str,
        tool_call_proposal_id: str,
    ):
        try:
            if tool_call_proposal_id.startswith("tool-call-proposal-v2-"):
                if v2_service is None:
                    raise ExecutionAgentV2Error("Execution Agent v2 is unavailable")
                publication = v2_service.read_proposal(
                    project_id=project_id,
                    tool_call_proposal_id=tool_call_proposal_id,
                )
                current = v2_service.controller.get(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                )
                current_exact = (
                    current.execution.execution_digest
                    == publication.proposal.controller_execution_digest
                    and current.inspection.inspection_digest
                    == publication.proposal.inspection_digest
                )
                receipt = v2_service.store.read_v2_committed_receipt(
                    project_id=project_id,
                    tool_call_proposal_id=tool_call_proposal_id,
                )
                payload_v2: dict[str, Any] = {
                    "ok": True,
                    "schema_version": "execution_agent_v2_route_response.v1",
                    "observation": publication.observation.model_dump(mode="json"),
                    "tool_catalog": publication.tool_catalog.model_dump(mode="json"),
                    "tool_call_proposal": publication.proposal.model_dump(mode="json"),
                    "current": current_exact,
                    "stale": not current_exact,
                    "applied": receipt is not None,
                }
                if receipt is not None:
                    payload_v2["application_receipt"] = receipt.model_dump(mode="json")
                return jsonify(payload_v2)
            result = service.read_proposal(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
                tool_call_proposal_id=tool_call_proposal_id,
            )
            publication = result.publication
            payload: dict[str, Any] = {
                "ok": True,
                "observation": publication.observation.model_dump(mode="json"),
                "tool_catalog": publication.tool_catalog.model_dump(mode="json"),
                "tool_call_proposal": publication.proposal.model_dump(mode="json"),
                "current": result.current,
                "stale": result.stale,
                "applied": result.applied,
            }
            if result.application_receipt is not None:
                payload["application_receipt"] = (
                    result.application_receipt.model_dump(mode="json")
                )
            return jsonify(payload)
        except FileNotFoundError:
            return _error(
                "Execution Agent proposal not found",
                404,
                "EXECUTION_AGENT_PROPOSAL_NOT_FOUND",
            )
        except ExecutionAgentConflict:
            return _error(
                "Execution Agent proposal belongs to another execution",
                409,
                "EXECUTION_AGENT_REQUEST_CONFLICT",
            )
        except ExecutionAgentV2Conflict:
            return _error(
                "Execution Agent v2 proposal belongs to another execution",
                409,
                "EXECUTION_AGENT_REQUEST_CONFLICT",
            )
        except ExecutionAgentV2Error:
            return _error(
                "Execution Agent v2 proposal failed exact verification",
                409,
                "EXECUTION_AGENT_ARTIFACT_INVALID",
            )
        except (ExecutionAgentStoreError, ScientificAgentHarnessControllerError):
            return _error(
                "Execution Agent proposal failed exact verification",
                409,
                "EXECUTION_AGENT_ARTIFACT_INVALID",
            )

    @app.post(
        "/api/projects/<project_id>/agent-harness-controller-executions/"
        "<controller_execution_id>/execution-agent-proposals/"
        "<tool_call_proposal_id>/apply"
    )
    def apply_tool_call_proposal(
        project_id: str,
        controller_execution_id: str,
        tool_call_proposal_id: str,
    ):
        try:
            raw_payload = _json_object()
            if raw_payload.get("schema_version") == "agent_tool_call_application_request.v2":
                if v2_service is None:
                    raise ExecutionAgentV2Error("Execution Agent v2 is unavailable")
                parsed_v2, _ = _parse(
                    AgentToolCallApplicationRequestV2,
                    {
                        "schema_version",
                        "expected_tool_call_proposal_digest",
                        "client_request_id",
                    },
                )
                result_v2 = v2_service.apply_proposal(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    tool_call_proposal_id=tool_call_proposal_id,
                    request=parsed_v2,
                )
                return _v2_application_payload(result_v2)
            parsed, _ = _parse(
                AgentToolCallApplicationRequest,
                {"expected_tool_call_proposal_digest", "client_request_id"},
            )
            result = service.apply_proposal(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
                tool_call_proposal_id=tool_call_proposal_id,
                request=parsed,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "tool_call_proposal": result.publication.proposal.model_dump(
                    mode="json"
                ),
                "application_receipt": result.application_receipt.model_dump(
                    mode="json"
                ),
                "applied": True,
                "controller_advance_called": (
                    result.application_receipt.controller_advance_called
                ),
                "dispatch_occurred": result.application_receipt.dispatch_occurred,
                # Backward-compatible alias with corrected dispatch semantics.
                "dispatched": result.application_receipt.dispatch_occurred,
            }
            if result.controller_result is not None:
                payload["controller_inspection"] = (
                    result.controller_result.inspection.model_dump(mode="json")
                )
                payload["controller_decision"] = (
                    result.controller_result.decision.model_dump(mode="json")
                    if result.controller_result.decision is not None
                    else None
                )
                payload["controller_receipt"] = (
                    result.controller_result.receipt.model_dump(mode="json")
                    if result.controller_result.receipt is not None
                    else None
                )
            return jsonify(payload)
        except FileNotFoundError:
            return _error(
                "Execution Agent proposal not found",
                404,
                "EXECUTION_AGENT_PROPOSAL_NOT_FOUND",
            )
        except ExecutionAgentRequestValidationError:
            return _error(
                "invalid Execution Agent application request",
                400,
                "CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        except ExecutionAgentStale:
            return _error(
                "Execution Agent proposal is stale",
                409,
                "EXECUTION_AGENT_PROPOSAL_STALE",
            )
        except ExecutionAgentV2Stale:
            return _error(
                "Execution Agent v2 proposal is stale",
                409,
                "EXECUTION_AGENT_PROPOSAL_STALE",
            )
        except ScientificAgentHarnessControllerRecoveryRequired:
            return _error(
                "Controller action requires explicit recovery",
                409,
                "CONTROLLER_RECOVERY_REQUIRED",
            )
        except (
            ExecutionAgentConflict,
            ExecutionAgentStoreConflict,
            ScientificAgentHarnessControllerConflict,
        ):
            return _error(
                "Execution Agent application conflicts with current authority",
                409,
                "EXECUTION_AGENT_APPLICATION_CONFLICT",
            )
        except (
            ExecutionAgentStoreVerificationError,
            ExecutionAgentStoreError,
            ScientificAgentHarnessControllerVerificationError,
        ):
            return _error(
                "Execution Agent application failed exact verification",
                409,
                "EXECUTION_AGENT_ARTIFACT_INVALID",
            )
        except ScientificAgentHarnessControllerError:
            return _error(
                "Execution Agent Controller authority is stale",
                409,
                "EXECUTION_AGENT_PROPOSAL_STALE",
            )
        except ValidationError:
            return _error(
                "invalid Execution Agent application request",
                400,
                "CLIENT_AUTHORITY_FIELD_REJECTED",
            )
        except (LogicalToolCompilationError, ExecutionAgentV2DecisionInvalid):
            return _error(
                "Execution Agent v2 decision failed server validation",
                400,
                "EXECUTION_AGENT_V2_DECISION_INVALID",
            )
        except ExecutionAgentV2Conflict:
            return _error(
                "Execution Agent v2 application conflicts with current authority",
                409,
                "EXECUTION_AGENT_APPLICATION_CONFLICT",
            )
        except ExecutionAgentV2Error:
            return _error(
                "Execution Agent v2 application failed exact verification",
                409,
                "EXECUTION_AGENT_ARTIFACT_INVALID",
            )


__all__ = ["register_execution_agent_routes"]
