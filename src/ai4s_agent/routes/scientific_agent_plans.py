from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent.llm_provider import LLMProvider, LLMProviderError, LLMProviderManager
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.routes.agents import _llm_provider_from_payload
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanError,
    ScientificAgentPlanPublicationConflict,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.storage import ProjectStorage


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _error_response(message: str, status: int):
    # Route errors intentionally use fixed, non-sensitive messages.  Provider
    # and source exceptions must never become an API side channel for raw
    # response bodies, paths, stderr, or credentials.
    return jsonify({"ok": False, "error": message}), status


def register_scientific_agent_plan_routes(
    app: Flask,
    *,
    projects: ProjectStorage,
    resource_profiles: Any,
    llm_settings: LLMSettingsStore,
    llm_providers: LLMProviderManager,
) -> None:
    observation_builder = AgentProjectObservationBuilder(
        storage=projects,
        resource_profiles=resource_profiles,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=projects,
        observation_builder=observation_builder,
    )
    app.extensions["scientific_agent_plan_observation_builder"] = observation_builder
    app.extensions["scientific_agent_plan_proposal_store"] = proposal_store

    @app.post("/api/projects/<project_id>/agent-plan-proposals")
    def create_scientific_agent_plan_proposal(project_id: str):
        try:
            payload = _json_object()
            allowed = {
                "run_id",
                "goal",
                "user_constraints",
                "external_llm_approved",
                "llm_provider",
            }
            unknown = set(payload).difference(allowed)
            if unknown:
                return _error_response("unsupported planning request field", 400)
            run_id = str(payload.get("run_id") or "").strip()
            goal = str(payload.get("goal") or "").strip()
            if not run_id or not goal:
                return _error_response("run_id and goal required", 400)
            constraints = payload.get("user_constraints", [])
            if not isinstance(constraints, list) or any(not isinstance(item, str) for item in constraints):
                return _error_response("user_constraints must be a list of strings", 400)
            provider_context: AbstractContextManager[LLMProvider | None] = _llm_provider_from_payload(
                payload,
                settings=llm_settings,
                providers=llm_providers,
            )
        except (ValueError, ValidationError, LLMProviderError):
            return _error_response("invalid planning request or LLM consent/configuration", 400)

        with provider_context as provider:
            if provider is None:
                return _error_response("a configured planning LLM provider is required", 503)
            try:
                service = ScientificAgentPlanService(
                    storage=projects,
                    resource_profiles=resource_profiles,
                    observation_builder=observation_builder,
                    proposal_store=proposal_store,
                )
                proposal = service.create_proposal(
                    project_id=project_id,
                    run_id=run_id,
                    goal=goal,
                    user_constraints=constraints,
                    provider=provider,
                )
            except FileNotFoundError:
                return _error_response("run or project not found", 404)
            except ScientificAgentPlanSourceChanged:
                return _error_response("authoritative project source changed; retry planning", 409)
            except ScientificAgentPlanPublicationConflict:
                return _error_response("proposal ID is already bound to different content", 409)
            except ScientificAgentPlanError:
                return _error_response("planning proposal was rejected by server validation", 400)
            except ValueError:
                return _error_response("planning proposal was rejected by server validation", 400)
            except (LLMProviderError, OSError):
                return _error_response("planning LLM call failed", 502)
        return jsonify(
            {
                "ok": True,
                "proposal_id": proposal.proposal_id,
                "executable": False,
                "proposal": proposal.model_dump(mode="json"),
            }
        )

    @app.get("/api/projects/<project_id>/agent-plan-proposals/<proposal_id>")
    def read_scientific_agent_plan_proposal(project_id: str, proposal_id: str):
        try:
            publication = proposal_store.read(
                project_id=project_id,
                proposal_id=proposal_id,
                verify_current=True,
            )
        except FileNotFoundError:
            return _error_response("proposal not found", 404)
        except ScientificAgentPlanSourceChanged:
            return _error_response("proposal source is stale; planning must be rebuilt", 409)
        except ScientificAgentPlanError:
            return _error_response("persisted planning proposal failed verification", 409)
        return jsonify(
            {
                "ok": True,
                "proposal_id": publication.proposal.proposal_id,
                "executable": False,
                "observation": publication.observation.model_dump(mode="json"),
                "tool_catalog": publication.catalog.model_dump(mode="json"),
                "proposal": publication.proposal.model_dump(mode="json"),
            }
        )
