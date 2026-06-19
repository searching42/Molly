from __future__ import annotations

from typing import Any

from flask import jsonify, request
from pydantic import ValidationError

from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.agents.research import ResearchAgent
from ai4s_agent.chat_context import _PROJECT_STORAGES


def install_external_approval_split() -> None:
    """Split target-evidence approval from acquisition/search scope approval."""

    _install_conversation_payload_split()
    _install_modeling_plan_route_split()


def _install_conversation_payload_split() -> None:
    original_prepare = ConversationAgent.prepare_modeling_plan_payload
    if getattr(original_prepare, "_external_approval_split", False):
        return

    def prepare_modeling_plan_payload_with_split(
        self: ConversationAgent,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
        available_inputs: list[Any] | None = None,
    ) -> dict[str, Any]:
        payload = original_prepare(
            self,
            run_id=run_id,
            messages=messages,
            project_id=project_id,
            available_inputs=available_inputs,
        )
        evidence_approved = bool(payload.get("user_approved_external_search"))
        payload["user_approved_external_evidence"] = evidence_approved
        # Preserve the historical field as a compatibility alias in conversation
        # payloads, but mark search/acquisition scope as not granted by target
        # evidence approval alone. The modeling-plan API consumes the explicit
        # split fields below.
        payload["user_approved_external_search_scope"] = False
        payload["approval_semantics"] = {
            "user_approved_external_evidence": "allows cited target evidence to inform the modeling brief",
            "user_approved_external_search_scope": "allows new external search/acquisition scope; false unless separately approved",
            "user_approved_external_search": "legacy alias retained for clients that read old modeling payloads",
        }
        return payload

    prepare_modeling_plan_payload_with_split._external_approval_split = True  # type: ignore[attr-defined]
    ConversationAgent.prepare_modeling_plan_payload = prepare_modeling_plan_payload_with_split  # type: ignore[method-assign]


def _install_modeling_plan_route_split() -> None:
    import ai4s_agent.api as api_module

    original_register_routes = api_module.register_routes
    if getattr(original_register_routes, "_external_approval_split", False):
        return

    def register_routes_with_external_approval_split(app: Any, base_runs_dir: Any = None, workspace_dir: Any = None) -> None:
        original_register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
        app.view_functions["agent_modeling_plan"] = _agent_modeling_plan_with_split

    register_routes_with_external_approval_split._external_approval_split = True  # type: ignore[attr-defined]
    api_module.register_routes = register_routes_with_external_approval_split  # type: ignore[method-assign]


def _agent_modeling_plan_with_split():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "payload must be an object"}), 400

    run_id = str(payload.get("run_id") or "").strip()
    goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
    project_id = str(payload.get("project_id") or "").strip()
    if not run_id or not goal:
        return jsonify({"ok": False, "error": "run_id and goal required"}), 400

    trainability_report = payload.get("trainability_report")
    backend_recommendation = payload.get("backend_recommendation")
    model_metrics = payload.get("model_metrics")
    property_id = str(payload.get("property_id") or "").strip()
    cited_target_evidence = payload.get("cited_target_evidence", [])
    if cited_target_evidence is None:
        cited_target_evidence = []
    if not isinstance(cited_target_evidence, list):
        return jsonify({"ok": False, "error": "cited_target_evidence must be a list"}), 400
    if any(not isinstance(item, dict) for item in cited_target_evidence):
        return jsonify({"ok": False, "error": "cited_target_evidence entries must be objects"}), 400
    project_memory = payload.get("project_memory")
    previous_diagnostics = payload.get("previous_diagnostics", [])
    if project_memory is not None and not isinstance(project_memory, dict):
        return jsonify({"ok": False, "error": "project_memory must be an object"}), 400
    if previous_diagnostics is None:
        previous_diagnostics = []
    if not isinstance(previous_diagnostics, list):
        return jsonify({"ok": False, "error": "previous_diagnostics must be a list"}), 400
    if any(not isinstance(item, dict) for item in previous_diagnostics):
        return jsonify({"ok": False, "error": "previous_diagnostics entries must be objects"}), 400
    for label, value in (
        ("trainability_report", trainability_report),
        ("backend_recommendation", backend_recommendation),
        ("model_metrics", model_metrics),
    ):
        if value is not None and not isinstance(value, dict):
            return jsonify({"ok": False, "error": f"{label} must be an object"}), 400

    try:
        modeling = ModelingAgent()
        proposal = modeling.propose_modeling_plan(
            run_id=run_id,
            goal=goal,
            trainability_report=trainability_report,
            backend_recommendation=backend_recommendation,
            model_metrics=model_metrics,
        )
        target_brief = None
        if property_id or cited_target_evidence:
            if not property_id:
                property_id = _first_trainability_property(trainability_report) or "default"
            approvals = external_approval_flags(payload)
            target_evidence = ResearchAgent().prepare_target_evidence_items(
                goal=goal,
                property_id=property_id,
                cited_summaries=cited_target_evidence,
                user_approved_external_search=approvals["target_evidence"],
            )
            available_inputs_raw = payload.get("available_inputs")
            available_inputs = set(_string_list(available_inputs_raw)) if available_inputs_raw is not None else None
            target_brief = modeling.prepare_target_modeling_brief(
                run_id=run_id,
                goal=goal,
                property_id=property_id,
                trainability_report=trainability_report,
                project_memory=project_memory,
                previous_diagnostics=previous_diagnostics,
                allow_external_search=approvals["external_search_scope"],
                available_inputs=available_inputs,
                target_evidence=target_evidence,
            )
            target_brief.dataset_context["external_approval_policy"] = {
                "target_evidence": approvals["target_evidence"],
                "external_search_scope": approvals["external_search_scope"],
                "acquisition_scope": False,
            }
        outputs: dict[str, str] = {}
        if project_id:
            storage = _project_storage()
            proposal_json, proposal_md = modeling.write_proposal(storage, project_id, run_id, proposal)
            outputs = {
                "modeling_plan_proposal_json": str(proposal_json),
                "modeling_plan_proposal_md": str(proposal_md),
            }
            if target_brief is not None:
                brief_json, brief_md = modeling.write_target_modeling_brief(storage, project_id, run_id, target_brief)
                safe_property = ModelingAgent._safe_property_stem(target_brief.property_id)
                outputs.update(
                    {
                        f"target_modeling_brief_{safe_property}_json": str(brief_json),
                        f"target_modeling_brief_{safe_property}_md": str(brief_md),
                    }
                )
    except (ValidationError, ValueError) as exc:
        return jsonify({"ok": False, "error": _normalize_external_approval_error(str(exc))}), 400

    response: dict[str, Any] = {
        "ok": True,
        "proposal": proposal.model_dump(mode="json"),
        "outputs": outputs,
        "external_approval_policy": external_approval_flags(payload),
    }
    if target_brief is not None:
        response["target_modeling_brief"] = target_brief.model_dump(mode="json")
    return jsonify(response)


def external_approval_flags(payload: dict[str, Any]) -> dict[str, bool]:
    target_evidence = _approval_bool(payload, "user_approved_external_evidence")
    if target_evidence is None:
        target_evidence = _approval_bool(payload, "user_approved_external_search") or False
    external_search_scope = _approval_bool(payload, "user_approved_external_search_scope")
    if external_search_scope is None:
        # Legacy clients used user_approved_external_search for both target
        # evidence and external search. New clients should send explicit split
        # fields; when present, they take precedence.
        if "user_approved_external_evidence" in payload:
            external_search_scope = False
        else:
            external_search_scope = _approval_bool(payload, "user_approved_external_search") or False
    acquisition_scope = _approval_bool(payload, "user_confirmed_external_acquisition") or False
    return {
        "target_evidence": bool(target_evidence),
        "external_search_scope": bool(external_search_scope),
        "acquisition_scope": bool(acquisition_scope),
    }


def _approval_bool(payload: dict[str, Any], key: str) -> bool | None:
    if key not in payload:
        return None
    return _as_bool(payload.get(key))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "approved", "allow", "allowed", "project-approved"}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_trainability_property(trainability_report: object) -> str:
    if not isinstance(trainability_report, dict):
        return ""
    properties = trainability_report.get("properties")
    if not isinstance(properties, list):
        return ""
    for item in properties:
        if not isinstance(item, dict):
            continue
        property_id = str(item.get("property_id") or "").strip()
        if property_id:
            return property_id
    return ""


def _project_storage() -> Any:
    if not _PROJECT_STORAGES:
        raise ValueError("project storage is not initialized")
    return _PROJECT_STORAGES[-1]


def _normalize_external_approval_error(message: str) -> str:
    return message.replace(
        "user_approved_external_search=True",
        "user_approved_external_evidence=True",
    )
