from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.remote_resource_authority import (
    RemoteResourceAuthorityConflict,
    RemoteResourceAuthorityDenied,
    RemoteResourceAuthorityError,
    RemoteResourceAuthorityService,
    RemoteResourceAuthorityStale,
)
from ai4s_agent.schemas import AgentRemoteResourceAuthorityRequest
from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore
from ai4s_agent.scientific_agent_plan import ScientificAgentPlanSourceChanged


def _payload() -> AgentRemoteResourceAuthorityRequest:
    value: Any = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return AgentRemoteResourceAuthorityRequest.model_validate(value)


def _error(message: str, status: int, reason_code: str):
    return jsonify(
        {
            "ok": False,
            "outcome": "DENY",
            "error": message,
            "reason_codes": [reason_code],
            "executable": False,
            "dispatched": False,
        }
    ), status


def register_remote_resource_authority_routes(
    app: Flask,
    *,
    service: RemoteResourceAuthorityService,
    control_store: AgentPlanControlStore,
) -> None:
    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/remote-resource-authority-evaluations"
    )
    def evaluate_remote_resource_authority(project_id: str, proposal_id: str):
        try:
            result = service.evaluate(
                project_id=project_id,
                proposal_id=proposal_id,
                request=_payload(),
            )
        except FileNotFoundError:
            return _error("proposal not found", 404, "REMOTE_RESOURCE_SOURCE_CHANGED")
        except (ScientificAgentPlanSourceChanged, RemoteResourceAuthorityStale):
            return _error(
                "proposal or resource source is stale",
                409,
                "REMOTE_RESOURCE_SOURCE_CHANGED",
            )
        except (ValidationError, ValueError):
            return _error(
                "invalid remote resource authority request",
                400,
                "REMOTE_RESOURCE_CLIENT_INJECTION",
            )
        return jsonify(
            {
                "ok": True,
                "outcome": result.decision.outcome.value,
                "reason_codes": result.decision.reason_codes,
                "decision_id": result.decision.decision_id,
                "decision": result.decision.model_dump(mode="json"),
                "authority_ids": [item.authority_id for item in result.authorities],
                "authorities": [item.model_dump(mode="json") for item in result.authorities],
                "authority_set_id": "",
                "authority_set": None,
                "executable": False,
                "dispatched": False,
            }
        )

    @app.post(
        "/api/projects/<project_id>/agent-plan-proposals/<proposal_id>/remote-resource-authorities"
    )
    def publish_remote_resource_authorities(project_id: str, proposal_id: str):
        try:
            result = service.publish(
                project_id=project_id,
                proposal_id=proposal_id,
                request=_payload(),
            )
        except FileNotFoundError:
            return _error("proposal not found", 404, "REMOTE_RESOURCE_SOURCE_CHANGED")
        except RemoteResourceAuthorityDenied as exc:
            return jsonify(
                {
                    "ok": False,
                    "outcome": "DENY",
                    "reason_codes": exc.decision.reason_codes,
                    "decision_id": exc.decision.decision_id,
                    "decision": exc.decision.model_dump(mode="json"),
                    "authority_ids": [],
                    "authorities": [],
                    "executable": False,
                    "dispatched": False,
                }
            ), 403
        except RemoteResourceAuthorityConflict:
            return _error(
                "client request is bound to different content",
                409,
                "REMOTE_RESOURCE_REQUEST_CONFLICT",
            )
        except (ScientificAgentPlanSourceChanged, RemoteResourceAuthorityStale):
            return _error(
                "proposal or resource source is stale",
                409,
                "REMOTE_RESOURCE_SOURCE_CHANGED",
            )
        except (ValidationError, RemoteResourceAuthorityError, ValueError):
            return _error(
                "invalid remote resource authority request",
                400,
                "REMOTE_RESOURCE_CLIENT_INJECTION",
            )
        return jsonify(
            {
                "ok": True,
                "outcome": result.decision.outcome.value,
                "reason_codes": result.decision.reason_codes,
                "decision_id": result.decision.decision_id,
                "authority_ids": [item.authority_id for item in result.authorities],
                "authorities": [item.model_dump(mode="json") for item in result.authorities],
                "authority_set_id": result.authority_set.authority_set_id,
                "authority_set": result.authority_set.model_dump(mode="json"),
                "executable": False,
                "dispatched": False,
            }
        ), 201

    @app.get(
        "/api/projects/<project_id>/agent-remote-resource-authority-decisions/<decision_id>"
    )
    def read_remote_resource_authority_decision(project_id: str, decision_id: str):
        try:
            decision = control_store.read_remote_resource_authority_decision(
                project_id=project_id, decision_id=decision_id
            )
        except FileNotFoundError:
            return _error("resource authority decision not found", 404, "REMOTE_RESOURCE_SOURCE_CHANGED")
        except ValueError:
            return _error("resource authority decision failed exact verification", 409, "REMOTE_RESOURCE_SOURCE_CHANGED")
        return jsonify(
            {
                "ok": True,
                "decision_id": decision.decision_id,
                "decision": decision.model_dump(mode="json"),
                "executable": False,
                "dispatched": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-remote-resource-authorities/<authority_id>"
    )
    def read_remote_resource_authority(project_id: str, authority_id: str):
        try:
            authority = service.verify_authority(
                project_id=project_id,
                authority_id=authority_id,
                verify_current=True,
            )
        except FileNotFoundError:
            return _error("remote resource authority not found", 404, "REMOTE_RESOURCE_AUTHORITY_REQUIRED")
        except (RemoteResourceAuthorityError, ScientificAgentPlanSourceChanged, ValueError):
            return _error("remote resource authority is stale", 409, "REMOTE_RESOURCE_SOURCE_CHANGED")
        return jsonify(
            {
                "ok": True,
                "authority_id": authority.authority_id,
                "authority": authority.model_dump(mode="json"),
                "executable": False,
                "dispatched": False,
            }
        )

    @app.get(
        "/api/projects/<project_id>/agent-remote-resource-authority-sets/<authority_set_id>"
    )
    def read_remote_resource_authority_set(project_id: str, authority_set_id: str):
        try:
            authority_set = service.verify_authority_set(
                project_id=project_id,
                authority_set_id=authority_set_id,
                verify_current=True,
            )
        except FileNotFoundError:
            return _error(
                "remote resource authority set not found",
                404,
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED",
            )
        except (RemoteResourceAuthorityError, ScientificAgentPlanSourceChanged, ValueError):
            return _error(
                "remote resource authority set is stale",
                409,
                "REMOTE_RESOURCE_SOURCE_CHANGED",
            )
        return jsonify(
            {
                "ok": True,
                "authority_set_id": authority_set.authority_set_id,
                "authority_set": authority_set.model_dump(mode="json"),
                "executable": False,
                "dispatched": False,
            }
        )


__all__ = ["register_remote_resource_authority_routes"]
