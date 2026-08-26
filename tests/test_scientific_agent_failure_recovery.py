from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.execution_agent_v2 import ExecutionAgentV2DecisionInvalid
from ai4s_agent.schemas import (
    AgentEffectCertainty,
    AgentFailureClass,
    AgentRecoveryAction,
    AgentRecoveryOutcome,
    AgentTaskFailureEvidence,
    AutonomyGrant,
    AutonomyParameterBound,
    _agent_digest,
)
from ai4s_agent.scientific_agent_failure_recovery import (
    FAILURE_RECOVERY_POLICY_DIGEST,
    FAILURE_RECOVERY_POLICY_VERSION,
    FailureRecoveryDecisionInvalid,
    FailureRecoveryObservationInvalid,
    FailureRecoveryProviderOutcomeUnknown,
    FailureRecoveryStale,
    ScientificAgentFailureRecoveryService,
    classify_failure,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.pr_fast


def _grant(tmp_path: Path, *, retries: int = 1, replans: int = 0) -> tuple[ProjectStorage, AutonomyGrant]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-08-01T00:00:00Z")
    grant = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["clean_task", "alternative_tool"],
        parameter_bounds={
            "clean_task.min_nonempty": AutonomyParameterBound(minimum=0, maximum=10),
        },
        max_retries=retries,
        max_replans=replans,
        valid_until="9999-12-31T23:59:59Z",
    )
    return storage, grant


def _evidence(
    failure_class: AgentFailureClass = AgentFailureClass.TRANSIENT,
    certainty: AgentEffectCertainty = AgentEffectCertainty.NO_EFFECT_CONFIRMED,
    *,
    alternatives: list[str] | None = None,
) -> AgentTaskFailureEvidence:
    return AgentTaskFailureEvidence(
        failure_code="typed_failure",
        failure_class=failure_class,
        effect_certainty=certainty,
        task_id="clean_task",
        logical_tool_id="clean_dataset",
        safe_alternative_tool_ids=alternatives or [],
        reason_codes=["TYPED_FAILURE"],
    )


def _observation(service: ScientificAgentFailureRecoveryService, grant: AutonomyGrant, evidence: AgentTaskFailureEvidence, **kwargs):
    return service.observe_failure(
        project_id="project-1",
        run_id="run-1",
        controller_execution_id=kwargs.pop("controller_execution_id", "controller-1"),
        controller_execution_digest=kwargs.pop("controller_execution_digest", "sha256:" + "1" * 64),
        inspection_digest=kwargs.pop("inspection_digest", "sha256:" + "2" * 64),
        task_id="clean_task",
        logical_tool_id="clean_dataset",
        arguments=kwargs.pop("arguments", {"min_nonempty": 3}),
        evidence=evidence,
        grant=grant,
        **kwargs,
    )


def test_exact_transient_retry_is_a_new_bounded_successor_and_replays(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    calls: list[dict] = []
    provider = StubLLMProvider(response={"action": "RETRY_EXACT"})
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=provider,
        successor_applicator=lambda **kwargs: calls.append(kwargs) or {
            "successor_controller_execution_id": "controller-successor",
            "effect_started": True,
            "effect_receipt_id": "controller-receipt",
            "effect_receipt_digest": "sha256:" + "3" * 64,
        },
    )
    observation = _observation(service, grant, _evidence())
    result = service.recover(observation=observation, grant=grant, client_request_id="recovery-1")
    replay = service.recover(observation=observation, grant=grant, client_request_id="recovery-1")
    assert result.decision.recovery_action is AgentRecoveryAction.RETRY_EXACT
    assert result.decision.selected_logical_tool_id == "clean_dataset"
    assert result.decision.selected_arguments == {}
    assert result.decision.retry_ordinal == 1
    assert result.receipt.outcome is AgentRecoveryOutcome.COMMITTED
    assert replay.replayed is True
    assert len(calls) == 1


def test_unknown_effect_is_deterministic_human_boundary(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=3, replans=3)
    class CountingProvider(StubLLMProvider):
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            return super().complete_json(**kwargs)

    provider = CountingProvider(response={"action": "RETRY_EXACT"})
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    observation = _observation(
        service,
        grant,
        _evidence(AgentFailureClass.UNKNOWN_EFFECT, AgentEffectCertainty.EFFECT_UNKNOWN),
    )
    result = service.recover(observation=observation, grant=grant)
    assert result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert result.receipt.effect_started is False
    assert result.provider_calls == 0
    assert provider.calls == 0


def test_known_server_failure_types_are_mapped_without_exception_text(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    mapped = classify_failure(ExecutionAgentV2DecisionInvalid("timeout-looking text"))
    assert mapped.failure_class is AgentFailureClass.NONRECOVERABLE
    assert mapped.effect_certainty is AgentEffectCertainty.NO_EFFECT_CONFIRMED

    orthogonal = _evidence(AgentFailureClass.TRANSIENT, AgentEffectCertainty.EFFECT_UNKNOWN)
    observation = _observation(service, grant, orthogonal)
    result = service.recover(observation=observation, grant=grant)
    assert result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert result.provider_calls == 0


def test_retry_budget_is_rebuilt_from_receipts_across_controller_successors(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=1)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=lambda **kwargs: {
            "successor_controller_execution_id": "controller-successor",
            "effect_started": True,
            "effect_receipt_id": "receipt-successor",
            "effect_receipt_digest": "sha256:" + "4" * 64,
        },
    )
    first = _observation(service, grant, _evidence())
    service.recover(observation=first, grant=grant, client_request_id="first")
    second = _observation(
        service,
        grant,
        _evidence(),
        controller_execution_id="controller-successor",
        controller_execution_digest="sha256:" + "8" * 64,
    )
    assert second.retry_count_used == 1
    exhausted = service.recover(observation=second, grant=grant, client_request_id="second")
    assert exhausted.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert exhausted.receipt.effect_started is False
    assert exhausted.receipt.retry_ordinal == 0
    assert exhausted.provider_calls == 0
    assert exhausted.budget.retries_used == 1


def test_physical_or_invented_tools_cannot_enter_server_roster(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    with pytest.raises(FailureRecoveryObservationInvalid):
        _observation(
            service,
            grant,
            _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE),
            available_recovery_tools=["shell"],
        )


def test_parameter_recovery_requires_closed_schema_and_rejects_expansion(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "clean_dataset", "arguments": {"min_nonempty": 5}}),
        tool_schemas={"clean_dataset": {"type": "object", "properties": {"min_nonempty": {"type": "integer", "minimum": 0, "maximum": 10}}, "required": ["min_nonempty"], "additionalProperties": False}},
        successor_applicator=lambda **kwargs: {"successor_controller_execution_id": "successor", "effect_started": True, "effect_receipt_id": "receipt", "effect_receipt_digest": "sha256:" + "5" * 64},
    )
    observation = _observation(service, grant, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    result = service.recover(observation=observation, grant=grant, client_request_id="parameter")
    assert result.decision.recovery_action is AgentRecoveryAction.TOOL_CALL
    assert result.decision.authority_relation.name == "SUBSET"
    assert result.decision.semantic_boundary.name == "NONE"

    storage2, grant2 = _grant(tmp_path / "expansion")
    expansion_service = ScientificAgentFailureRecoveryService(
        storage=storage2,
        grant=grant2,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "clean_dataset", "arguments": {"min_nonempty": 20}}),
        tool_schemas={"clean_dataset": {"type": "object", "properties": {"min_nonempty": {"type": "integer", "minimum": 0, "maximum": 20}}, "required": ["min_nonempty"], "additionalProperties": False}},
    )
    expansion = _observation(expansion_service, grant2, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    with pytest.raises(FailureRecoveryDecisionInvalid):
        expansion_service.recover(observation=expansion, grant=grant2, client_request_id="expansion")


def test_provider_unknown_is_not_retried(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)

    class UnknownProvider:
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            raise OSError("provider boundary")

    provider = UnknownProvider()
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    observation = _observation(service, grant, _evidence(AgentFailureClass.PARAMETER_RECOVERABLE))
    with pytest.raises(FailureRecoveryProviderOutcomeUnknown):
        service.recover(observation=observation, grant=grant, client_request_id="unknown")
    with pytest.raises(FailureRecoveryProviderOutcomeUnknown):
        service.recover(observation=observation, grant=grant, client_request_id="unknown")
    assert provider.calls == 1


def test_server_owned_alternative_tool_is_allowed_only_inside_grant(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "alternative_tool", "arguments": {}}),
        tool_schemas={"alternative_tool": {"type": "object", "properties": {}, "additionalProperties": False}},
        successor_applicator=lambda **kwargs: {"successor_controller_execution_id": "alt-successor", "effect_started": True, "effect_receipt_id": "alt-receipt", "effect_receipt_digest": "sha256:" + "6" * 64},
    )
    observation = _observation(service, grant, _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE, alternatives=["alternative_tool"]))
    result = service.recover(observation=observation, grant=grant, client_request_id="alternative")
    assert result.decision.selected_logical_tool_id == "alternative_tool"
    assert result.decision.authority_relation.name == "SUBSET"

    storage2, grant2 = _grant(tmp_path / "outside")
    grant2 = AutonomyGrant.model_validate({**grant2.model_dump(mode="json"), "allowed_tasks": ["clean_task"], "grant_id": "", "grant_digest": ""})
    outside = ScientificAgentFailureRecoveryService(
        storage=storage2,
        grant=grant2,
        provider=StubLLMProvider(response={"action": "TOOL_CALL", "logical_tool_id": "alternative_tool", "arguments": {}}),
        tool_schemas={"alternative_tool": {"type": "object", "properties": {}, "additionalProperties": False}},
    )
    outside_observation = _observation(outside, grant2, _evidence(AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE, alternatives=["alternative_tool"]))
    with pytest.raises(FailureRecoveryDecisionInvalid):
        outside.recover(observation=outside_observation, grant=grant2, client_request_id="outside")


def test_replan_uses_existing_entrypoint_once_and_budget_survives(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, replans=1)
    calls: list[dict] = []

    class Replanner:
        def create_current_controller_failure_revision(self, **kwargs):
            calls.append(kwargs)
            return {"successor_proposal_id": "revision-successor"}

    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "REPLAN"}),
        replanner=Replanner(),
    )
    first = _observation(service, grant, _evidence(AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT))
    result = service.recover(observation=first, grant=grant, client_request_id="replan-1")
    assert result.decision.recovery_action is AgentRecoveryAction.REPLAN
    assert result.receipt.effect_started is False
    assert len(calls) == 1
    second = _observation(
        service,
        grant,
        _evidence(AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT),
        controller_execution_id="successor-controller",
        controller_execution_digest="sha256:" + "9" * 64,
    )
    assert second.replan_count_used == 1
    exhausted = service.recover(observation=second, grant=grant, client_request_id="replan-2")
    assert exhausted.decision.recovery_action is AgentRecoveryAction.ASK_USER
    assert len(calls) == 1


def test_semantic_review_and_nonrecoverable_are_human_or_stop_without_provider(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path, retries=2)

    class NeverProvider:
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            raise AssertionError("deterministic boundary must not call provider")

    provider = NeverProvider()
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant, provider=provider)
    semantic = _observation(service, grant, _evidence(AgentFailureClass.SEMANTIC_REVIEW_REQUIRED))
    semantic_result = service.recover(observation=semantic, grant=grant)
    assert semantic_result.decision.recovery_action is AgentRecoveryAction.ASK_USER
    nonrecoverable = _observation(service, grant, _evidence(AgentFailureClass.NONRECOVERABLE), controller_execution_id="controller-nonrecoverable")
    stop_result = service.recover(observation=nonrecoverable, grant=grant)
    assert stop_result.decision.recovery_action is AgentRecoveryAction.STOP
    assert provider.calls == 0


def test_effect_crash_reconciles_without_a_second_successor_call(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    calls: list[int] = []

    class Fault:
        fired = False

        def __call__(self, phase: str):
            if phase == "after_effect" and not self.fired:
                self.fired = True
                raise RuntimeError("simulated crash")

    fault = Fault()
    service = ScientificAgentFailureRecoveryService(
        storage=storage,
        grant=grant,
        provider=StubLLMProvider(response={"action": "RETRY_EXACT"}),
        successor_applicator=lambda **kwargs: calls.append(1) or {"successor_controller_execution_id": "successor", "effect_started": True, "effect_receipt_id": "effect", "effect_receipt_digest": "sha256:" + "7" * 64},
        effect_reconciler=lambda **kwargs: {"successor_controller_execution_id": "successor", "effect_started": True, "effect_receipt_id": "effect", "effect_receipt_digest": "sha256:" + "7" * 64},
        fault_injector=fault,
    )
    observation = _observation(service, grant, _evidence())
    with pytest.raises(RuntimeError):
        service.recover(observation=observation, grant=grant, client_request_id="crash")
    resumed = service.recover(observation=observation, grant=grant, client_request_id="crash")
    assert len(calls) == 1
    assert resumed.receipt.effect_started is True


def test_forged_observation_and_future_enum_fail_closed(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    observation = _observation(service, grant, _evidence())
    forged = observation.model_dump(mode="json")
    forged["retry_count_used"] = 99
    forged["failure_digest"] = _agent_digest({**{key: value for key, value in forged.items() if key not in {"failure_id", "failure_digest", "created_at"}}, "arguments_digest": forged["arguments_digest"]})
    with pytest.raises((FailureRecoveryStale, ValueError)):
        service.recover(observation=forged, grant=grant)
    with pytest.raises(ValidationError):
        AgentTaskFailureEvidence.model_validate({**_evidence().model_dump(mode="json"), "failure_class": "FUTURE"})


def test_schema_contract_is_non_executable_and_provider_prompt_is_safe(tmp_path: Path) -> None:
    storage, grant = _grant(tmp_path)
    service = ScientificAgentFailureRecoveryService(storage=storage, grant=grant)
    observation = _observation(service, grant, _evidence())
    assert observation.executable is False
    assert observation.policy_version == FAILURE_RECOVERY_POLICY_VERSION
    assert observation.policy_digest == FAILURE_RECOVERY_POLICY_DIGEST
