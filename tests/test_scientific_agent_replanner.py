from __future__ import annotations

from pathlib import Path
import json

import pytest

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    CORE_SCHEMA_MODELS,
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentPlanAuthorizationRequest,
    AgentPlanFeedbackRequest,
    AgentPlanRevisionApplicationRequest,
    AgentReplanLLMResponse,
    AgentTaskDispatchIntent,
    PlannedTask,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationConflict,
    ScientificAgentAuthorizationDenied,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.scientific_agent_replanner import (
    ScientificAgentReplannerConflict,
    ScientificAgentReplannerOutcomeUnknown,
    ScientificAgentReplannerService,
    ScientificAgentReplannerStale,
    ScientificAgentReplannerStore,
    canonical_plan_diff,
    plan_semantic_projection,
)
from ai4s_agent.storage import ProjectStorage
from tests.execution_agent_test_support import local_controller_execution


NOW = "2026-08-02T00:00:00Z"


class NoController:
    def __getattr__(self, name: str):
        raise AssertionError(f"Replanner unexpectedly called Controller.{name}")


class CountingProvider(StubLLMProvider):
    def __init__(self, response):
        super().__init__(response=response)
        self.calls = 0

    def complete_json(self, **kwargs):
        self.calls += 1
        return super().complete_json(**kwargs)


def _baseline(tmp_path: Path):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: NOW)
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {"count": 8, "seed": 1}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce reviewable candidates"],
        rationales=["Use one registered local task."],
        assumptions=[],
        questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Generate a bounded candidate set",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="baseline-proposal",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: NOW,
    )
    authorization = authorization_service.authorize(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="baseline-authorization",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    service = ScientificAgentReplannerService(
        storage=storage,
        proposal_store=proposal_store,
        observation_builder=builder,
        authorization_service=authorization_service,
        control_store=control_store,
        controller=NoController(),
        clock=lambda: NOW,
    )
    return storage, proposal_store, authorization_service, proposal, authorization, service


def _revision_payload(proposal, authorization, feedback, *, request_id="revision-1"):
    return {
        "run_id": proposal.run_id,
        "client_request_id": request_id,
        "trigger_kind": "explicit_user_feedback",
        "baseline_proposal_id": proposal.proposal_id,
        "baseline_proposal_digest": proposal.proposal_digest,
        "baseline_semantic_plan_id": proposal.semantic_plan_id,
        "baseline_semantic_plan_digest": proposal.semantic_plan_digest,
        "baseline_run_plan_digest": _agent_digest(
            proposal.run_plan.model_dump(mode="json")
        ),
        "baseline_authorization_id": authorization.authorization_id,
        "baseline_authorization_digest": authorization.authorization_digest,
        "feedback_receipt_id": feedback.feedback_receipt_id,
        "feedback_receipt_digest": feedback.feedback_receipt_digest,
        "external_llm_approved": True,
    }


@pytest.mark.pr_fast
def test_explicit_feedback_compiles_diff_applies_successor_and_requires_fresh_authority(
    tmp_path,
) -> None:
    storage, proposal_store, authorization_service, baseline, old_authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1",
            client_request_id="feedback-1",
            feedback="Reduce the candidate count to four.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    response = AgentReplanLLMResponse(
        rationale_summary="Use a smaller bounded candidate set.",
        option_patch={"generate_candidates": {"count": 4}},
    )
    provider = CountingProvider(response=response.model_dump(mode="json"))
    created = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(baseline, old_authorization, feedback),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=provider,
    )
    assert provider.calls == 1
    revision = created.proposal
    assert revision.status == "review_required"
    assert revision.review_only is True
    assert revision.executable is False
    assert revision.successor_candidate is not None
    assert revision.successor_candidate.compiled_task_options["generate_candidates"]["count"] == 4
    paths = {item.path for item in revision.plan_diff.changes}
    assert "option.raw_planner_options" in paths
    assert "option.effective_planner_options" in paths
    assert "option.compiled_task_options" in paths

    replay = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(baseline, old_authorization, feedback),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=provider,
    )
    assert replay.replayed is True
    assert replay.proposal.model_dump(mode="json") == revision.model_dump(mode="json")
    assert provider.calls == 1

    applied = service.apply_revision(
        project_id="project-1",
        revision_id=revision.revision_id,
        request=AgentPlanRevisionApplicationRequest(
            expected_revision_digest=revision.revision_digest,
            client_request_id="application-1",
        ),
    )
    assert applied.dispatched is False
    assert applied.receipt.fresh_permission_required is True
    assert applied.receipt.fresh_authorization_required is True
    assert applied.successor.proposal_digest != baseline.proposal_digest
    assert proposal_store.read(
        project_id="project-1", proposal_id=baseline.proposal_id
    ).proposal.proposal_digest == baseline.proposal_digest
    assert authorization_service.verify_authorization(
        project_id="project-1", authorization_id=old_authorization.authorization_id
    ).authorization_digest == old_authorization.authorization_digest

    decision = authorization_service.evaluate_permission(
        project_id="project-1",
        proposal_id=applied.successor.proposal_id,
        expected_proposal_digest=applied.successor.proposal_digest,
    )
    assert decision.proposal_digest == applied.successor.proposal_digest
    fresh = authorization_service.authorize(
        project_id="project-1",
        proposal_id=applied.successor.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=applied.successor.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="successor-authorization",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert fresh.authorization_digest != old_authorization.authorization_digest
    assert fresh.proposal_digest == applied.successor.proposal_digest
    with pytest.raises((ScientificAgentAuthorizationConflict, ScientificAgentAuthorizationDenied)):
        authorization_service.authorize(
            project_id="project-1",
            proposal_id=applied.successor.proposal_id,
            request=AgentPlanAuthorizationRequest(
                expected_proposal_digest=baseline.proposal_digest,
                authorization_mode=AgentAuthorizationMode.STEPWISE,
                requested_preauthorized_gate_ids=[],
                confirmed=True,
                client_request_id="old-digest-reuse",
            ),
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        )

    persisted = b"".join(
        path.read_bytes()
        for path in (tmp_path / "workspace" / "projects" / "project-1").rglob("*")
        if path.is_file() and "private_feedback" not in str(path)
    )
    assert b"Reduce the candidate count" not in persisted


def test_no_change_is_immutable_and_cannot_be_applied(tmp_path) -> None:
    _, _, _, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1",
            client_request_id="feedback-no-change",
            feedback="Keep the current plan.",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    created = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(
            baseline, authorization, feedback, request_id="revision-no-change"
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(
                rationale_summary="No material plan change is needed.", no_change=True
            ).model_dump(mode="json")
        ),
    )
    assert created.proposal.status == "no_material_change"
    assert created.proposal.successor_candidate is None
    assert created.proposal.plan_diff.changes == []
    with pytest.raises(ScientificAgentReplannerConflict):
        service.apply_revision(
            project_id="project-1",
            revision_id=created.proposal.revision_id,
            request=AgentPlanRevisionApplicationRequest(
                expected_revision_digest=created.proposal.revision_digest,
                client_request_id="no-change-apply",
            ),
        )


def test_canonical_diff_is_deterministic_and_dimensioned(tmp_path) -> None:
    _, _, _, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-diff", feedback="Use six candidates."
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    revision = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(baseline, authorization, feedback, request_id="revision-diff"),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(
                option_patch={"generate_candidates": {"count": 6}}
            ).model_dump(mode="json")
        ),
    ).proposal
    successor = revision.successor_candidate
    assert successor is not None
    rebuilt = canonical_plan_diff(baseline=baseline, successor=successor, created_at="later")
    assert rebuilt.plan_diff_digest == revision.plan_diff.plan_diff_digest
    assert plan_semantic_projection(baseline) != plan_semantic_projection(successor)
    assert rebuilt.model_dump(mode="json", exclude={"created_at"}) == revision.plan_diff.model_dump(
        mode="json", exclude={"created_at"}
    )


def test_request_conflict_and_llm_authority_injection_fail_closed(tmp_path) -> None:
    _, _, _, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-conflict", feedback="Use four candidates."
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    payload = _revision_payload(baseline, authorization, feedback, request_id="revision-conflict")
    service.create_revision(
        project_id="project-1",
        payload=payload,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(
                option_patch={"generate_candidates": {"count": 4}}
            ).model_dump(mode="json")
        ),
    )
    with pytest.raises(ScientificAgentReplannerConflict):
        service.create_revision(
            project_id="project-1",
            payload={**payload, "baseline_authorization_digest": "sha256:" + "0" * 64},
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            provider=StubLLMProvider(response={"no_change": True}),
        )

    with pytest.raises(Exception):
        AgentReplanLLMResponse.model_validate(
            {"option_patch": {}, "dispatch": True, "scientific_success": True}
        )


def test_replanner_generated_schemas_equal_pydantic_source() -> None:
    names = {
        "agent_plan_feedback_request",
        "agent_plan_feedback_receipt",
        "agent_plan_replan_request",
        "agent_replan_llm_response",
        "agent_replanner_observation",
        "agent_plan_diff",
        "agent_plan_revision_proposal",
        "agent_plan_revision_application_request",
        "agent_plan_revision_application_receipt",
    }
    root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    for name in sorted(names):
        assert json.loads((root / f"{name}.schema.json").read_text(encoding="utf-8")) == (
            CORE_SCHEMA_MODELS[name].model_json_schema()
        )


@pytest.mark.pr_fast
def test_replanner_api_separates_feedback_proposal_and_application(tmp_path) -> None:
    from ai4s_agent.app import create_app

    workspace = tmp_path / "api-workspace"
    storage = ProjectStorage(workspace_dir=workspace)
    storage.create_project("project-1", name="Project", created_at=NOW)
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    app.config.update(TESTING=True, AI4S_AGENT_AUTHORIZATION_OWNER="alice")
    client = app.test_client()
    baseline_response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {"count": 8, "seed": 1}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce reviewable candidates"],
        rationales=["Use a bounded registered task."],
        assumptions=[],
        questions=[],
    )
    planned = client.post(
        "/api/projects/project-1/agent-plan-proposals",
        json={
            "run_id": "run-1",
            "goal": "Generate a bounded candidate set",
            "user_constraints": [],
            "client_request_id": "api-baseline",
            "llm_provider": {
                "provider": "stub",
                "stub_response": baseline_response.model_dump(mode="json"),
            },
        },
    )
    assert planned.status_code == 200, planned.get_json()
    proposal = planned.get_json()["proposal"]
    authorized = client.post(
        f"/api/projects/project-1/agent-plan-proposals/{proposal['proposal_id']}/authorizations",
        json={
            "expected_proposal_digest": proposal["proposal_digest"],
            "authorization_mode": "stepwise",
            "requested_preauthorized_gate_ids": [],
            "confirmed": True,
            "client_request_id": "api-authorization",
        },
    )
    assert authorized.status_code == 200, authorized.get_json()
    authorization = authorized.get_json()["authorization"]

    feedback_response = client.post(
        "/api/projects/project-1/agent-plan-feedback",
        json={
            "run_id": "run-1",
            "client_request_id": "api-feedback",
            "feedback": "Reduce the candidate count to four.",
        },
    )
    assert feedback_response.status_code == 201
    feedback = feedback_response.get_json()["feedback_receipt"]
    body = {
        "run_id": "run-1",
        "client_request_id": "api-revision",
        "trigger_kind": "explicit_user_feedback",
        "baseline_proposal_id": proposal["proposal_id"],
        "baseline_proposal_digest": proposal["proposal_digest"],
        "baseline_semantic_plan_id": proposal["semantic_plan_id"],
        "baseline_semantic_plan_digest": proposal["semantic_plan_digest"],
        "baseline_run_plan_digest": _agent_digest(proposal["run_plan"]),
        "baseline_authorization_id": authorization["authorization_id"],
        "baseline_authorization_digest": authorization["authorization_digest"],
        "feedback_receipt_id": feedback["feedback_receipt_id"],
        "feedback_receipt_digest": feedback["feedback_receipt_digest"],
        "external_llm_approved": True,
        "llm_provider": {
            "provider": "stub",
            "stub_response": AgentReplanLLMResponse(
                option_patch={"generate_candidates": {"count": 4}}
            ).model_dump(mode="json"),
        },
    }
    created = client.post("/api/projects/project-1/agent-plan-revisions", json=body)
    assert created.status_code == 201, created.get_json()
    created_payload = created.get_json()
    assert created_payload["review_required"] is True
    assert created_payload["applied"] is False
    assert created_payload["dispatched"] is False
    revision = created_payload["revision"]

    injection = client.post(
        "/api/projects/project-1/agent-plan-revisions",
        json={**body, "client_request_id": "api-injection", "successor_plan": {}},
    )
    assert injection.status_code == 400

    applied = client.post(
        f"/api/projects/project-1/agent-plan-revisions/{revision['revision_id']}/apply",
        json={
            "expected_revision_digest": revision["revision_digest"],
            "client_request_id": "api-application",
        },
    )
    assert applied.status_code == 201, applied.get_json()
    applied_payload = applied.get_json()
    assert applied_payload["fresh_permission_required"] is True
    assert applied_payload["fresh_authorization_required"] is True
    assert applied_payload["dispatched"] is False
    receipt = applied_payload["application_receipt"]
    read = client.get(
        "/api/projects/project-1/agent-plan-revision-applications/"
        + receipt["application_receipt_id"]
    )
    assert read.status_code == 200
    assert read.get_json()["application_receipt"] == receipt


def test_provider_checkpoint_recovery_never_calls_provider_twice(tmp_path) -> None:
    storage, proposal_store, authorization_service, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-crash", feedback="Use five candidates."
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    crashed = False

    def fault(phase: str) -> None:
        nonlocal crashed
        if phase == "after_provider_outcome" and not crashed:
            crashed = True
            raise RuntimeError("simulated process crash")

    service.store = ScientificAgentReplannerStore(storage=storage, fault_injector=fault)
    provider = CountingProvider(
        response=AgentReplanLLMResponse(
            option_patch={"generate_candidates": {"count": 5}}
        ).model_dump(mode="json")
    )
    payload = _revision_payload(
        baseline, authorization, feedback, request_id="revision-crash"
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        service.create_revision(
            project_id="project-1",
            payload=payload,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            provider=provider,
        )
    recovered = service.create_revision(
        project_id="project-1",
        payload=payload,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=provider,
    )
    assert recovered.proposal.status == "review_required"
    assert provider.calls == 1


def test_unknown_provider_outcome_is_stable_and_never_retried(tmp_path) -> None:
    _, _, _, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-timeout", feedback="Review the plan."
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )

    class TimeoutProvider:
        def __init__(self):
            self.calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            raise OSError("timeout with unknown provider outcome")

    provider = TimeoutProvider()
    payload = _revision_payload(
        baseline, authorization, feedback, request_id="revision-timeout"
    )
    for _ in range(2):
        with pytest.raises(ScientificAgentReplannerOutcomeUnknown):
            service.create_revision(
                project_id="project-1",
                payload=payload,
                actor="alice",
                actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
                provider=provider,
            )
    assert provider.calls == 1


def test_application_recovers_successor_published_before_receipt(tmp_path) -> None:
    storage, _, _, baseline, authorization, service = _baseline(tmp_path)
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-apply-crash", feedback="Use three candidates."
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    revision = service.create_revision(
        project_id="project-1",
        payload=_revision_payload(
            baseline, authorization, feedback, request_id="revision-apply-crash"
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(
                option_patch={"generate_candidates": {"count": 3}}
            ).model_dump(mode="json")
        ),
    ).proposal
    crashed = False

    def fault(phase: str) -> None:
        nonlocal crashed
        if phase == "after_successor_proposal" and not crashed:
            crashed = True
            raise RuntimeError("successor committed before receipt")

    service.store = ScientificAgentReplannerStore(storage=storage, fault_injector=fault)
    request = AgentPlanRevisionApplicationRequest(
        expected_revision_digest=revision.revision_digest,
        client_request_id="application-crash",
    )
    with pytest.raises(RuntimeError, match="successor committed before receipt"):
        service.apply_revision(
            project_id="project-1", revision_id=revision.revision_id, request=request
        )
    recovered = service.apply_revision(
        project_id="project-1", revision_id=revision.revision_id, request=request
    )
    assert recovered.receipt.successor_proposal_digest == revision.successor_proposal_digest
    assert recovered.dispatched is False


def test_feedback_same_request_replays_original_bytes(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path / "feedback-workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    store = ScientificAgentReplannerStore(storage=storage)
    request = AgentPlanFeedbackRequest(
        run_id="run-1", client_request_id="feedback-replay", feedback="A private suggestion."
    )
    first = store.create_feedback(
        project_id="project-1",
        request=request,
        actor="alice",
        actor_source="trusted",
        created_at="2026-08-02T00:00:00Z",
    )
    replay = store.create_feedback(
        project_id="project-1",
        request=request,
        actor="alice",
        actor_source="trusted",
        created_at="2026-08-02T00:01:00Z",
    )
    assert replay.model_dump(mode="json") == first.model_dump(mode="json")


def test_complete_diff_surfaces_every_plan_semantic_dimension(tmp_path) -> None:
    _, _, _, baseline, _, _ = _baseline(tmp_path)
    extra_task = PlannedTask(
        task_id="extra_task",
        depends_on=[baseline.run_plan.tasks[0].task_id],
        required_artifacts=["artifact-x"],
        output_artifacts=["artifact-y"],
    )
    changed_run_plan = baseline.run_plan.model_copy(
        update={
            "requested_tasks": [*baseline.run_plan.requested_tasks, "extra_task"],
            "tasks": [*baseline.run_plan.tasks, extra_task],
            "available_artifacts": ["artifact-x"],
            "missing_artifacts": ["artifact-z"],
        }
    )
    changed_response = baseline.validated_llm_response.model_copy(
        update={
            "requested_tool_ids": [
                *baseline.validated_llm_response.requested_tool_ids,
                "extra_tool",
            ]
        }
    )
    successor = baseline.model_copy(
        update={
            "run_plan": changed_run_plan,
            "validated_llm_response": changed_response,
            "planner_options": {**baseline.planner_options, "extra_tool": {}},
            "effective_planner_options": {
                **baseline.effective_planner_options,
                "extra_task": {},
            },
            "compiled_task_options": {
                **baseline.compiled_task_options,
                "extra_task": {},
            },
            "selected_artifacts": ["artifact-x"],
            "selected_profiles": ["profile-x"],
            "dispatch_intents": [
                *baseline.dispatch_intents,
                AgentTaskDispatchIntent(
                    task_id="extra_task", execution_route="local_executor"
                ),
            ],
            "limits": {"max_steps": 2},
            "required_gates": ["review_gate"],
            "stop_conditions": ["stop after two steps"],
            "success_criteria": ["verified output exists"],
            "tool_catalog_digest": "sha256:" + "1" * 64,
            "semantic_plan_digest": "sha256:" + "2" * 64,
        }
    )
    diff = canonical_plan_diff(
        baseline=baseline, successor=successor, created_at=NOW
    )
    dimensions = {item.dimension for item in diff.changes}
    assert dimensions == {
        "task",
        "dependency",
        "option",
        "artifact",
        "route_profile_resource",
        "budget",
        "gate",
        "semantic",
    }
    paths = {item.path for item in diff.changes}
    assert {
        "task.ordered_roster",
        "dependency.edges",
        "option.effective_planner_options",
        "artifact.selected_artifact_ids",
        "route_profile_resource.dispatch_intents",
        "budget.limits",
        "gate.required_gates",
        "semantic.success_criteria",
    }.issubset(paths)


def test_current_verified_controller_terminal_can_trigger_no_change(tmp_path) -> None:
    storage, control_store, controller, snapshot = local_controller_execution(tmp_path)
    proposal = controller.proposal_store.read(
        project_id="project-1",
        proposal_id=snapshot.execution.proposal_id,
        verify_current=False,
    ).proposal
    authorization = control_store.read_authorization(
        project_id="project-1",
        authorization_id=snapshot.execution.authorization_id,
    )
    assert snapshot.receipt is not None
    decision = control_store.read_harness_controller_decision(
        project_id="project-1", decision_id=snapshot.receipt.decision_id
    )
    service = ScientificAgentReplannerService(
        storage=storage,
        proposal_store=controller.proposal_store,
        observation_builder=controller.proposal_store.observation_builder,
        authorization_service=controller.authorization_service,
        control_store=control_store,
        controller=controller,
        clock=lambda: NOW,
    )
    payload = {
        "run_id": proposal.run_id,
        "client_request_id": "controller-terminal-revision",
        "trigger_kind": "controller_terminal",
        "baseline_proposal_id": proposal.proposal_id,
        "baseline_proposal_digest": proposal.proposal_digest,
        "baseline_semantic_plan_id": proposal.semantic_plan_id,
        "baseline_semantic_plan_digest": proposal.semantic_plan_digest,
        "baseline_run_plan_digest": _agent_digest(proposal.run_plan.model_dump(mode="json")),
        "baseline_authorization_id": authorization.authorization_id,
        "baseline_authorization_digest": authorization.authorization_digest,
        "controller_execution_id": snapshot.execution.controller_execution_id,
        "controller_execution_digest": snapshot.execution.execution_digest,
        "controller_decision_id": decision.decision_id,
        "controller_decision_digest": decision.decision_digest,
        "controller_receipt_id": snapshot.receipt.receipt_id,
        "controller_receipt_digest": snapshot.receipt.receipt_digest,
        "external_llm_approved": True,
    }
    result = service.create_revision(
        project_id="project-1",
        payload=payload,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        provider=StubLLMProvider(
            response=AgentReplanLLMResponse(no_change=True).model_dump(mode="json")
        ),
    )
    assert result.proposal.status == "no_material_change"
    names = {item.name for item in result.proposal.observation.source_bindings}
    assert {"controller_execution", "controller_decision", "controller_receipt"}.issubset(names)

    with pytest.raises(ScientificAgentReplannerStale):
        service.create_revision(
            project_id="project-1",
            payload={
                **payload,
                "client_request_id": "controller-failed-mismatch",
                "trigger_kind": "controller_failed",
            },
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            provider=StubLLMProvider(response={"no_change": True}),
        )
