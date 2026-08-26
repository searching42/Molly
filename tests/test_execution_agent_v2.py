from __future__ import annotations

from dataclasses import replace
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent.autonomy_authority import evaluate_authority
from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.execution_agent_v2 import (
    AgentExecutionLLMResponseV2,
    AgentExecutionV2Classification,
    AgentExecutionV2DecisionType,
    AgentToolCallApplicationRequestV2,
    AgentToolCallProposalRequestV2,
    ExecutionAgentV2Service,
    ExecutionAgentV2DecisionInvalid,
    ExecutionAgentV2Store,
    ExecutionAgentV2LLMOutcomeUnknown,
    ExecutionAgentV2Stale,
    LogicalToolCompilationError,
    build_execution_v2_tool_catalog,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStoreVerificationError
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.llm_provider import LLMProviderError
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessGateApprovalRequest,
    AgentHarnessControllerStartRequest,
    AgentPlanAuthorizationRequest,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_conversation import (
    ScientificAgentConversationSessionService,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage
from tests.execution_agent_test_support import (
    CountingStubProvider,
    NoRemoteAuthorities,
    NoRemoteLifecycle,
    NOW,
    execution_agent_service,
    local_controller_execution,
)


pytestmark = pytest.mark.pr_fast


def _clean_controller_fixture(tmp_path: Path):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    dataset = run_dir / "inputs" / "dataset.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/dataset.csv"
    )
    plan_response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["clean_dataset"],
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={
            "clean_dataset": {
                "min_numeric_ratio": 0.5,
                "min_nonempty": 1,
                "drop_empty_target_rows": False,
                "strict_smiles_cleaning": True,
            }
        },
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a cleaned dataset"],
        rationales=["Execute the exact registered local task roster."],
        assumptions=[],
        questions=[],
    )
    observation_builder = AgentProjectObservationBuilder(
        storage=storage,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=observation_builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=observation_builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Clean one exact dataset",
        user_constraints=[],
        provider=StubLLMProvider(response=plan_response.model_dump(mode="json")),
        client_request_id="plan-v2-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: NOW,
    )
    approved = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="authorization-v2-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    current = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="controller-v2-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert current.inspection.current_task_id == "clean_dataset"
    return storage, controller, current


def _generate_controller_fixture(tmp_path: Path, *, approve_gate: bool = True):
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at=NOW)
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={
            "generate_candidates": {
                "backend": "deterministic_stub",
                "count": 8,
                "seed": 0,
            }
        },
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce a bounded candidate set"],
        rationales=["Use the registered deterministic local candidate generator."],
        assumptions=[],
        questions=[],
    )
    observation_builder = AgentProjectObservationBuilder(
        storage=storage,
        clock=lambda: NOW,
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=observation_builder,
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=observation_builder,
        proposal_store=proposal_store,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Generate one bounded candidate set",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="plan-v2-generate-1",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: NOW,
    )
    approved = authorization_service.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="authorization-v2-generate-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposal_store.registry),
        remote_executions=NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    created = controller.create(
        project_id="project-1",
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="controller-v2-generate-1",
        ),
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    assert created.inspection.next_action.value in {
        "prepare_local_gate",
        "wait_for_gate",
    }
    if not approve_gate:
        return storage, controller, created
    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    current = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=created.execution.controller_execution_id,
        gate_id="gate_5_final_threshold",
        request=AgentHarnessGateApprovalRequest(
            expected_snapshot_id=snapshot["snapshot_id"],
            expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
            client_request_id="controller-v2-generate-gate-1",
            note="Approve the exact deterministic fixture Gate.",
        ),
        actor="alice",
    )
    assert current.inspection.next_action.value == "execute_local_task"
    return storage, controller, current


def _v2_service(storage: ProjectStorage, controller: ScientificAgentHarnessController):
    return ExecutionAgentV2Service(
        controller=controller,
        store=ExecutionAgentV2Store(storage=storage),
        clock=lambda: NOW,
    )


def _tool_call_response(
    *,
    tool_id: str = "clean_dataset",
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision_type": "TOOL_CALL",
        "tool_id": tool_id,
        "arguments": arguments
        if arguments is not None
        else {
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
            "drop_empty_target_rows": False,
            "strict_smiles_cleaning": True,
        },
        "expected_outcome": "Apply the registered bounded local task.",
        "confidence": 0.8,
    }


def _proposal_request(digest: str, request_id: str) -> AgentToolCallProposalRequestV2:
    return AgentToolCallProposalRequestV2(
        expected_controller_execution_digest=digest,
        client_request_id=request_id,
        external_llm_approved=True,
        llm_provider={"provider": "stub"},
    )


def test_v2_catalog_and_response_are_closed_world() -> None:
    catalog = build_execution_v2_tool_catalog()
    assert [item.tool_id for item in catalog.tools] == [
        "clean_dataset",
        "confirm_extracted_dataset",
        "filter_rank",
        "generate_candidates",
    ]
    assert all(item.executable is False for item in catalog.tools)
    assert all("adapter" not in item.model_dump(mode="json") for item in catalog.tools)
    generation = catalog.get("generate_candidates")
    assert generation.argument_schema["additionalProperties"] is False
    assert "backend" not in generation.argument_schema["properties"]
    assert generation.server_bound_argument_keys == ["backend"]

    valid = AgentExecutionLLMResponseV2.model_validate(_tool_call_response())
    assert valid.decision_type is AgentExecutionV2DecisionType.TOOL_CALL
    with pytest.raises(ValidationError):
        AgentExecutionLLMResponseV2.model_validate(
            {
                "decision_type": "TOOL_CALL",
                "tool_id": "clean_dataset",
                "expected_outcome": "bounded",
                "confidence": 0.8,
            }
        )
    with pytest.raises(ValidationError):
        AgentExecutionLLMResponseV2.model_validate(
            {
                **_tool_call_response(),
                "arguments": {"output_path": "/tmp/not-a-logical-argument"},
            }
        )
    with pytest.raises(ValidationError):
        AgentExecutionLLMResponseV2.model_validate(
            {**_tool_call_response(), "selected_tool_id": "shell"}
        )


def test_v2_authority_relation_and_boundary_are_server_derived() -> None:
    grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(
                minimum=1,
                maximum=32,
            )
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    subset = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(
                minimum=1,
                maximum=16,
            )
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    expansion = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(
                minimum=1,
                maximum=64,
            )
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    safe = evaluate_authority(grant, subset, changes=[{"dimension": "option", "path": "option.count"}])
    unsafe = evaluate_authority(
        grant,
        expansion,
        changes=[{"dimension": "option", "path": "option.count"}],
    )
    boundary = evaluate_authority(
        grant,
        subset,
        changes=[
            {
                "dimension": "semantic",
                "boundary": SemanticBoundary.SCIENTIFIC_CONFIRMATION.value,
            }
        ],
    )
    assert safe.relation is AuthorityRelation.SUBSET and safe.auto_apply is True
    assert unsafe.relation is AuthorityRelation.EXPANSION and unsafe.auto_apply is False
    assert boundary.semantic_boundary is SemanticBoundary.SCIENTIFIC_CONFIRMATION
    assert boundary.auto_apply is False

    historical_exact = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(allowed_values=[8])
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    historical_different = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(allowed_values=[4])
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    exact_evaluation = evaluate_authority(
        historical_exact,
        historical_different,
        changes=[{"dimension": "option", "path": "option.count"}],
    )
    assert exact_evaluation.relation is not AuthorityRelation.SUBSET
    assert exact_evaluation.auto_apply is False

    scope_baseline = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(
                minimum=1,
                maximum=8,
            )
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    scope_candidate = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["generate_candidates"],
        allowed_effect_classes=["compute"],
        parameter_bounds={
            "generate_candidates.count": AutonomyParameterBound(allowed_values=[4])
        },
        valid_until="9999-12-31T00:00:00Z",
    )
    scope_evaluation = evaluate_authority(
        scope_baseline,
        scope_candidate,
        changes=[{"dimension": "option", "path": "option.count"}],
    )
    assert scope_evaluation.relation is AuthorityRelation.SUBSET
    assert scope_evaluation.auto_apply is True


def test_v2_valid_tool_call_is_one_provider_call_and_binds_compiler(tmp_path: Path) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    provider = CountingStubProvider(response=_tool_call_response())
    result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-1"),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = result.publication.proposal
    assert result.llm_used is True
    assert provider.calls == 1
    assert proposal.decision_type is AgentExecutionV2DecisionType.TOOL_CALL
    assert proposal.classification is AgentExecutionV2Classification.AUTO_APPLY
    assert proposal.authority_relation is AuthorityRelation.SUBSET
    assert proposal.semantic_boundary is SemanticBoundary.NONE
    assert proposal.authority_auto_apply is True
    assert proposal.compilation is not None
    assert proposal.compilation.executable is False
    assert proposal.compilation.controller_options_match is True
    assert proposal.server_compiled_operation.value == "controller_advance"


def test_v2_create_apply_ignores_advancing_wall_clock(tmp_path: Path) -> None:
    storage, controller, current = _generate_controller_fixture(tmp_path)
    ticks = 0

    def advancing_clock() -> str:
        nonlocal ticks
        ticks += 1
        return f"2026-08-01T00:00:{ticks:02d}Z"

    service = ExecutionAgentV2Service(
        controller=controller,
        store=ExecutionAgentV2Store(storage=storage),
        clock=advancing_clock,
    )
    proposal_result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-advancing-clock"),
        provider=CountingStubProvider(
            response={
                "decision_type": "TOOL_CALL",
                "tool_id": "generate_candidates",
                "arguments": {"count": 8, "seed": 0},
                "expected_outcome": "Generate a bounded deterministic candidate set.",
                "confidence": 0.82,
            }
        ),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposal_result.publication.proposal
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="application-v2-advancing-clock",
        ),
    )
    assert ticks > 1
    assert proposal_result.publication.observation.created_at != applied.application_receipt.created_at
    assert applied.application_receipt.outcome.value == "applied"


def test_v2_response_checkpoint_reuses_frozen_context_and_rejects_changed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    request = _proposal_request(
        current.execution.execution_digest,
        "proposal-v2-response-checkpoint",
    )
    provider = CountingStubProvider(response=_tool_call_response())

    def fail_publication(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulate crash after response checkpoint")

    monkeypatch.setattr(service.store, "publish_v2_proposal", fail_publication)
    with pytest.raises(RuntimeError, match="response checkpoint"):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    assert provider.calls == 1

    changed_inspection = type(current.inspection).model_validate(
        {
            **current.inspection.model_dump(mode="json"),
            "next_action": AgentHarnessControllerAction.WAIT_FOR_GATE.value,
            "inspection_digest": "",
        }
    )
    monkeypatch.setattr(
        controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: replace(current, inspection=changed_inspection),
    )

    class ExplodingProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response=_tool_call_response())
            self.calls = 0

        def complete_json(self, **kwargs: Any):
            self.calls += 1
            raise AssertionError("frozen response recovery must not call the provider")

    recovering_provider = ExplodingProvider()
    with pytest.raises(ExecutionAgentV2Stale):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=request,
            provider=recovering_provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    assert recovering_provider.calls == 0


def test_v2_generate_candidates_reaches_controller_once_and_replays(
    tmp_path: Path,
) -> None:
    storage, controller, current = _generate_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    provider = CountingStubProvider(
        response={
            "decision_type": "TOOL_CALL",
            "tool_id": "generate_candidates",
            "arguments": {"count": 8, "seed": 0},
            "expected_outcome": "Generate a bounded deterministic candidate set.",
            "confidence": 0.82,
        }
    )
    proposal_result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(
            current.execution.execution_digest,
            "proposal-v2-generate-success",
        ),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposal_result.publication.proposal
    assert provider.calls == 1
    assert proposal.classification is AgentExecutionV2Classification.AUTO_APPLY
    assert proposal.authority_relation is AuthorityRelation.SUBSET
    assert proposal.semantic_boundary is SemanticBoundary.NONE

    before = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="application-v2-generate-success",
        ),
    )
    after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    assert applied.application_receipt.outcome.value == "applied"
    assert applied.application_receipt.controller_advance_called is True
    assert applied.application_receipt.dispatch_occurred is True
    assert applied.controller_result is not None
    assert len(after) == len(before) + 1
    assert sum(item.dispatch_occurred for item in after) == 1

    replay = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="application-v2-generate-success",
        ),
    )
    replay_receipts = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    assert replay.application_receipt.application_receipt_id == (
        applied.application_receipt.application_receipt_id
    )
    assert len(replay_receipts) == len(after)
    assert provider.calls == 1


def test_v2_gate_boundary_is_server_human_boundary_without_effect(
    tmp_path: Path,
) -> None:
    storage, controller, current = _generate_controller_fixture(
        tmp_path,
        approve_gate=False,
    )
    service = _v2_service(storage, controller)

    class ExplodingProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response={})
            self.calls = 0

        def complete_json(self, **kwargs: Any):
            self.calls += 1
            raise AssertionError("v2 must not call the provider at a Gate")

    provider = ExplodingProvider()
    result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(
            current.execution.execution_digest,
            "proposal-v2-gate-boundary",
        ),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "exploding"}),
    )
    before = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=result.publication.proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=(
                result.publication.proposal.tool_call_proposal_digest
            ),
            client_request_id="application-v2-gate-boundary",
        ),
    )
    after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    assert provider.calls == 0
    assert result.publication.proposal.decision_type is AgentExecutionV2DecisionType.ASK_USER
    assert result.publication.proposal.classification is AgentExecutionV2Classification.REQUIRE_HUMAN
    assert applied.application_receipt.outcome.value == "user_action_required"
    assert applied.application_receipt.controller_advance_called is False
    assert sum(item.dispatch_occurred for item in after) == sum(
        item.dispatch_occurred for item in before
    ) == 0


def test_v2_changed_gated_option_requires_fresh_gate_without_effect(
    tmp_path: Path,
) -> None:
    storage, controller, current = _generate_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    provider = CountingStubProvider(
        response={
            "decision_type": "TOOL_CALL",
            "tool_id": "generate_candidates",
            "arguments": {"count": 16, "seed": 0},
            "expected_outcome": "Generate a bounded deterministic candidate set.",
            "confidence": 0.82,
        }
    )
    result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(
            current.execution.execution_digest,
            "proposal-v2-gated-successor",
        ),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert provider.calls == 1
    assert result.publication.proposal.compilation is not None
    assert result.publication.proposal.compilation.authority_relation is AuthorityRelation.SUBSET
    assert result.publication.proposal.compilation.controller_options_match is False
    assert result.publication.proposal.classification is AgentExecutionV2Classification.REQUIRE_HUMAN
    assert result.publication.proposal.fresh_permission_required is True
    assert result.publication.proposal.fresh_authorization_required is True

    before = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=result.publication.proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=result.publication.proposal.tool_call_proposal_digest,
            client_request_id="application-v2-gated-successor",
        ),
    )
    after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    assert applied.application_receipt.outcome.value == "user_action_required"
    assert applied.application_receipt.controller_create_called is False
    assert applied.application_receipt.fresh_permission_required is True
    assert applied.application_receipt.fresh_authorization_required is True
    assert applied.application_receipt.dispatch_occurred is False
    assert len(after) == len(before)


def test_conversation_routes_non_deterministic_step_to_v2_and_counts_one_call(
    tmp_path: Path,
) -> None:
    storage, controller, current = _generate_controller_fixture(tmp_path)
    conversations = ConversationStore(projects=storage)
    conversations.create_conversation(
        "project-1",
        conversation_id="conversation-v2",
        title="Execution Agent v2",
    )
    proposal_store = controller.proposal_store
    service = ScientificAgentConversationSessionService(
        projects=storage,
        conversations=conversations,
        plan_service=ScientificAgentPlanService(
            storage=storage,
            proposal_store=proposal_store,
        ),
        proposal_store=proposal_store,
        authorization_service=controller.authorization_service,
        controller=controller,
        execution_agent=execution_agent_service(
            storage=storage,
            controller=controller,
        ),
        execution_agent_v2=_v2_service(storage, controller),
        clock=lambda: NOW,
    )
    state = service._transition(
        project_id="project-1",
        conversation_id="conversation-v2",
        status="running",
        reason_code="EXECUTION_AGENT_STEP",
        updates={
            "run_id": current.execution.run_id,
            "proposal_id": current.execution.proposal_id,
            "proposal_digest": current.execution.proposal_digest,
            "authorization_id": current.execution.authorization_id,
            "authorization_digest": current.execution.authorization_digest,
            "start_intent_id": current.execution.start_intent_id,
            "start_intent_digest": current.execution.start_intent_digest,
            "controller_execution_id": current.execution.controller_execution_id,
            "controller_execution_digest": current.execution.execution_digest,
            "controller_status": current.inspection.status.value,
            "current_task_id": current.inspection.current_task_id,
        },
        event_type="test.v2.execution_agent_step",
    )
    provider = CountingStubProvider(
        response={
            "decision_type": "TOOL_CALL",
            "tool_id": "generate_candidates",
            "arguments": {"count": 8, "seed": 0},
            "expected_outcome": "Generate a bounded deterministic candidate set.",
            "confidence": 0.82,
        }
    )
    result, final_state, _stop_reason = service._auto_progress(
        project_id="project-1",
        conversation_id="conversation-v2",
        state=state,
        controller_result=current,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert provider.calls == 1
    assert result is not None
    assert final_state["autonomy_budget_usage"]["llm_calls"] == 1
    receipts = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    assert sum(item.dispatch_occurred for item in receipts) == 1


def test_v2_bounded_option_successor_reaches_controller_exactly_once(tmp_path: Path) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    arguments = _tool_call_response()["arguments"]
    arguments["min_nonempty"] = 2
    provider = CountingStubProvider(response=_tool_call_response(arguments=arguments))
    result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-2"),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = result.publication.proposal
    assert provider.calls == 1
    assert proposal.compilation is not None
    assert proposal.compilation.authority_relation is AuthorityRelation.SUBSET
    assert proposal.compilation.controller_options_match is False
    assert proposal.classification is AgentExecutionV2Classification.AUTO_APPLY

    baseline_before = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="application-v2-2",
        ),
    )
    assert applied.application_receipt.outcome.value == "applied"
    assert applied.application_receipt.controller_advance_called is False
    assert applied.application_receipt.controller_create_called is True
    assert applied.application_receipt.successor_proposal_id
    assert applied.application_receipt.successor_authorization_id
    assert applied.application_receipt.successor_start_intent_id
    assert applied.application_receipt.successor_controller_execution_id
    assert applied.application_receipt.successor_authority_evaluation_id
    assert applied.application_receipt.dispatch_occurred is True
    assert applied.controller_result is not None
    assert applied.controller_result.execution.controller_execution_id != current.execution.controller_execution_id
    assert applied.controller_result.execution.proposal_id == applied.application_receipt.successor_proposal_id
    successor_authorization = controller.authorization_service.verify_authorization(
        project_id="project-1",
        authorization_id=applied.application_receipt.successor_authorization_id,
        verify_current=False,
    )
    assert successor_authorization.compiled_task_options["clean_dataset"]["min_nonempty"] == 2
    baseline_after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
    )
    after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=applied.application_receipt.successor_controller_execution_id,
    )
    assert len(baseline_after) == len(baseline_before)
    assert len(after) == 1

    replay = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequestV2(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="application-v2-2",
        ),
    )
    replay_after = controller.control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=applied.application_receipt.successor_controller_execution_id,
    )
    assert replay.application_receipt.application_receipt_id == applied.application_receipt.application_receipt_id
    assert len(replay_after) == len(after)
    assert provider.calls == 1


def test_v2_terminal_human_boundary_does_not_call_provider(tmp_path: Path) -> None:
    storage, _control_store, controller, current = local_controller_execution(tmp_path)
    service = _v2_service(storage, controller)

    class ExplodingProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response={})
            self.calls = 0

        def complete_json(self, **kwargs: Any):
            self.calls += 1
            raise AssertionError("Execution Agent provider must not cross a human boundary")

    provider = ExplodingProvider()
    result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-terminal"),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "exploding"}),
    )
    assert provider.calls == 0
    assert result.llm_used is False
    assert result.publication.proposal.decision_type is AgentExecutionV2DecisionType.ASK_USER
    assert result.publication.proposal.classification is AgentExecutionV2Classification.REQUIRE_HUMAN
    assert current.inspection.next_action.value in {
        "prepare_local_gate",
        "wait_for_gate",
        "complete_execution",
        "stop_task_terminal",
    }
    assert result.publication.proposal.authority_auto_apply is False


def test_v2_unknown_tool_and_argument_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)

    unknown_tool = CountingStubProvider(
        response=_tool_call_response(tool_id="python.execute")
    )
    with pytest.raises(LogicalToolCompilationError):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=_proposal_request(current.execution.execution_digest, "proposal-v2-unknown-tool"),
            provider=unknown_tool,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    assert unknown_tool.calls == 1

    for forbidden_field, forbidden_value in (
        ("shell", "echo forbidden"),
        ("output_path", "/tmp/not-a-logical-argument"),
        ("adapter", "physical.adapter"),
        ("host", "worker.internal"),
        ("api_key", "secret"),
    ):
        unknown_argument = _tool_call_response()
        unknown_argument["arguments"] = {
            **unknown_argument["arguments"],
            forbidden_field: forbidden_value,
        }
        with pytest.raises(ValidationError):
            AgentExecutionLLMResponseV2.model_validate(unknown_argument)

    payload = current.inspection.model_dump(mode="python")
    payload.update(
        {
            "next_action": "future.controller.action",
            "inspection_digest": "",
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unknown_inspection = type(current.inspection).model_construct(**payload)
        object.__setattr__(
            unknown_inspection,
            "inspection_digest",
            _agent_digest(unknown_inspection.semantic_material()),
        )
    altered = replace(current, inspection=unknown_inspection)
    monkeypatch.setattr(
        controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: altered,
    )
    with pytest.raises(ExecutionAgentV2DecisionInvalid):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=_proposal_request(
                current.execution.execution_digest,
                "proposal-v2-unknown-action",
            ),
            provider=None,
            provider_binding_digest="",
        )


def test_v2_request_schema_is_versioned_and_forged_digest_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentToolCallProposalRequestV2.model_validate(
            {
                "schema_version": "agent_tool_call_proposal_request.v1",
                "expected_controller_execution_digest": "sha256:" + "a" * 64,
                "client_request_id": "request-1",
                "external_llm_approved": True,
            }
        )
    response = AgentExecutionLLMResponseV2.model_validate(_tool_call_response())
    payload = response.model_dump(mode="json")
    payload["arguments"]["min_nonempty"] = 2
    changed = AgentExecutionLLMResponseV2.model_validate(payload)
    assert changed.model_dump(mode="json") != response.model_dump(mode="json")


def test_v2_serialized_proposal_tamper_and_stale_inspection_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)
    proposal_result = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-tamper"),
        provider=CountingStubProvider(response=_tool_call_response()),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposal_result.publication.proposal
    fresh_response = _tool_call_response()
    fresh_response["expected_outcome"] = "Use a distinct immutable decision for stale checking."
    fresh = service.create_proposal(
        project_id="project-1",
        controller_execution_id=current.execution.controller_execution_id,
        request=_proposal_request(current.execution.execution_digest, "proposal-v2-stale"),
        provider=CountingStubProvider(response=fresh_response),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    target = service.store._publication_target(
        project_id="project-1",
        root_name="agent_execution_agent_v2_proposals",
        artifact_id=proposal.tool_call_proposal_id,
        create_root=False,
    )
    assert target is not None
    decision_path = target / "decision.json"
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8").replace(
            "Apply the registered bounded local task.",
            "forged decision",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExecutionAgentStoreVerificationError):
        service.read_proposal(
            project_id="project-1",
            tool_call_proposal_id=proposal.tool_call_proposal_id,
        )

    stale_inspection = type(current.inspection).model_validate(
        {
            **current.inspection.model_dump(mode="json"),
            "next_action": AgentHarnessControllerAction.WAIT_FOR_GATE.value,
            "inspection_digest": "",
        }
    )
    altered = replace(current, inspection=stale_inspection)
    monkeypatch.setattr(
        controller,
        "read_execution_agent_snapshot",
        lambda **_kwargs: altered,
    )
    with pytest.raises(ExecutionAgentV2Stale):
        service.apply_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            tool_call_proposal_id=fresh.publication.proposal.tool_call_proposal_id,
            request=AgentToolCallApplicationRequestV2(
                expected_tool_call_proposal_digest=(
                    fresh.publication.proposal.tool_call_proposal_digest
                ),
                client_request_id="application-v2-stale",
            ),
        )


def test_v2_provider_unknown_outcome_is_checkpointed_without_retry(tmp_path: Path) -> None:
    storage, controller, current = _clean_controller_fixture(tmp_path)
    service = _v2_service(storage, controller)

    class UnknownProvider(StubLLMProvider):
        def __init__(self) -> None:
            super().__init__(response=_tool_call_response())
            self.calls = 0

        def complete_json(self, **kwargs: Any):
            self.calls += 1
            raise LLMProviderError("provider outcome crossed the external boundary")

    provider = UnknownProvider()
    request = _proposal_request(
        current.execution.execution_digest,
        "proposal-v2-unknown-outcome",
    )
    with pytest.raises(ExecutionAgentV2LLMOutcomeUnknown):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "unknown"}),
        )
    with pytest.raises(ExecutionAgentV2LLMOutcomeUnknown):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=current.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "unknown"}),
        )
    assert provider.calls == 1
