from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest

from ai4s_agent.execution_agent import (
    ExecutionAgentConflict,
    ExecutionAgentLLMOutcomeUnknown,
    ExecutionAgentLLMResponseInvalid,
    ExecutionAgentStale,
    build_execution_tool_catalog,
)
from ai4s_agent.execution_agent_store import (
    ExecutionAgentStoreConflict,
    ExecutionAgentStoreVerificationError,
)
from ai4s_agent.llm_provider import OpenAICompatibleProvider, StubLLMProvider
from ai4s_agent.schemas import (
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerAction,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    AgentToolCallApplicationOutcome,
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
    LLMInvocationRecord,
    LLMProviderConfig,
    _agent_digest,
)
from ai4s_agent.scientific_agent_harness_controller import ControllerAdvanceResult
from tests.execution_agent_test_support import (
    CountingStubProvider,
    execution_agent_service,
    local_controller_execution,
)


def _proposal_request(
    execution_digest: str,
    *,
    request_id: str = "execution-agent-proposal-1",
) -> AgentToolCallProposalRequest:
    return AgentToolCallProposalRequest(
        expected_controller_execution_digest=execution_digest,
        client_request_id=request_id,
        external_llm_approved=True,
        llm_provider={"provider": "stub"},
    )


def _provider(tool_id: str) -> CountingStubProvider:
    return CountingStubProvider(
        response={
            "selected_tool_id": tool_id,
            "decision_summary": "Select one bounded server operation.",
        }
    )


class _CapturingProvider(CountingStubProvider):
    def __init__(self) -> None:
        super().__init__(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause this bounded turn.",
            }
        )
        self.messages: list[dict[str, str]] = []

    def complete_json(self, **kwargs: Any):
        self.messages = kwargs["messages"]
        return super().complete_json(**kwargs)


@pytest.mark.pr_fast
def test_proposal_and_application_are_two_phase_and_exactly_once(tmp_path) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    provider = _provider("controller.advance_current.v1")
    request = _proposal_request(initial.execution.execution_digest)

    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    replayed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert proposed == replayed
    assert provider.calls == 1
    assert proposed.applied is False and proposed.dispatched is False
    before_decisions = control_store.list_harness_controller_decisions(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )

    proposal = proposed.publication.proposal
    application_request = AgentToolCallApplicationRequest(
        expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
        client_request_id="apply-proposal-1",
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=application_request,
    )
    replayed_application = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=application_request,
    )
    assert applied.application_receipt == replayed_application.application_receipt
    assert applied.application_receipt.outcome == AgentToolCallApplicationOutcome.APPLIED
    after_decisions = control_store.list_harness_controller_decisions(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(after_decisions) == len(before_decisions) + 1
    audit = service.read_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert audit.applied is True
    assert audit.stale is True


def test_pause_application_never_calls_controller_advance(tmp_path) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    provider = _provider("agent.pause_current.v1")
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    before = control_store.list_harness_controller_decisions(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="apply-pause-1",
        ),
    )
    after = control_store.list_harness_controller_decisions(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert after == before
    assert applied.application_receipt.outcome == AgentToolCallApplicationOutcome.PAUSED
    assert applied.application_receipt.side_effect_attempted is False


def test_provider_prompt_contains_only_safe_observation_and_catalog(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    provider = _CapturingProvider()
    service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert len(provider.messages) == 2
    user_message = provider.messages[1]["content"]
    assert set(json.loads(user_message)) == {
        "observation",
        "tool_catalog",
    }
    for forbidden in (
        "Inspect one exact dataset",
        "SMILES,value",
        "CCO,1.0",
        "inputs/dataset.csv",
        "alice",
        "authorization note",
        "api_key",
    ):
        assert forbidden not in user_message


def test_same_request_different_safe_provider_binding_conflicts(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    request = _proposal_request(initial.execution.execution_digest)
    service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=_provider("agent.pause_current.v1"),
        provider_binding_digest=_agent_digest({"model": "model-a"}),
    )
    with pytest.raises((ExecutionAgentConflict, ExecutionAgentStoreConflict)):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=_provider("agent.pause_current.v1"),
            provider_binding_digest=_agent_digest({"model": "model-b"}),
        )


def test_different_requests_publish_identical_semantic_proposal_bytes(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    first_provider = _provider("agent.pause_current.v1")
    second_provider = _provider("agent.pause_current.v1")
    first = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id="semantic-proposal-a",
        ),
        provider=first_provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    second = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id="semantic-proposal-b",
        ),
        provider=second_provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert first.publication == second.publication
    assert first_provider.calls == second_provider.calls == 1


class _MarkdownProvider(StubLLMProvider):
    def complete_json(self, **kwargs: Any) -> LLMInvocationRecord:
        parsed = {
            "selected_tool_id": "agent.pause_current.v1",
            "decision_summary": "Pause this bounded turn.",
        }
        return LLMInvocationRecord(
            provider="stub",
            model="stub",
            prompt_version=kwargs["prompt_version"],
            response_id="markdown",
            raw_response={"response": "```json\n{}\n```"},
            parsed_output=parsed,
        )


class _InvocationMetadataProvider:
    def __init__(self, *, provider: str, model: str, response_id: str) -> None:
        self.provider = provider
        self.model = model
        self.response_id = response_id
        self.calls = 0

    def complete_json(self, **kwargs: Any) -> LLMInvocationRecord:
        self.calls += 1
        parsed = {
            "selected_tool_id": "agent.pause_current.v1",
            "decision_summary": "Pause this bounded turn.",
        }
        return LLMInvocationRecord(
            provider=self.provider,
            model=self.model,
            prompt_version=kwargs["prompt_version"],
            response_id=self.response_id,
            raw_response={"response": parsed},
            parsed_output=parsed,
        )


@pytest.mark.parametrize(
    ("configured_model", "expected_model"),
    [
        ("", "default"),
        ("Qwen/Qwen3-32B", None),
    ],
)
def test_openai_compatible_provider_metadata_projection_accepts_real_contracts(
    tmp_path,
    configured_model,
    expected_model,
) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    response = {
        "selected_tool_id": "agent.pause_current.v1",
        "decision_summary": "Pause this bounded turn.",
    }
    calls = 0

    def transport(url, payload, headers, timeout):
        del url, headers, timeout
        nonlocal calls
        calls += 1
        assert payload["model"] == (configured_model or "default")
        return {
            "choices": [{"message": {"content": json.dumps(response)}}],
        }

    provider = OpenAICompatibleProvider(
        config=LLMProviderConfig(
            provider="openai_compatible",
            endpoint="https://example.test/v1",
            model=configured_model,
        ),
        transport=transport,
    )
    proposed = execution_agent_service(
        storage=storage,
        controller=controller,
    ).create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id=(
                "provider-contract-namespaced"
                if configured_model
                else "provider-contract-default"
            ),
        ),
        provider=provider,
        provider_binding_digest=_agent_digest({"model": configured_model}),
    )
    proposal = proposed.publication.proposal
    assert calls == 1
    assert proposal.llm_response_id == "unavailable"
    assert proposal.llm_response_id_digest.startswith("sha256:")
    if expected_model is None:
        assert proposal.llm_model == proposal.llm_model_digest
        assert configured_model not in json.dumps(proposal.model_dump(mode="json"))
    else:
        assert proposal.llm_model == expected_model
        assert proposal.llm_model_digest.startswith("sha256:")


def test_provider_metadata_rejection_is_checkpointed_and_never_recalled(
    tmp_path,
) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    provider = _InvocationMetadataProvider(
        provider="unsupported_provider",
        model="model",
        response_id="response",
    )
    request = _proposal_request(
        initial.execution.execution_digest,
        request_id="invalid-provider-metadata-1",
    )
    for _ in range(2):
        with pytest.raises(
            ExecutionAgentLLMResponseInvalid,
            match="execution_agent_llm_response_invalid",
        ):
            service.create_proposal(
                project_id="project-1",
                controller_execution_id=initial.execution.controller_execution_id,
                request=request,
                provider=provider,
                provider_binding_digest=_agent_digest({"provider": "invalid"}),
            )
    assert provider.calls == 1
    request_root = (
        storage.project_dir("project-1")
        / "agent_execution_agent_requests"
        / initial.execution.controller_execution_id
        / "requests"
        / request.client_request_id
    )
    assert (request_root / "llm_response_rejected.json").is_file()


@pytest.mark.parametrize(
    "provider",
    [
        CountingStubProvider(
            response={
                "selected_tool_id": "unknown.tool.v1",
                "decision_summary": "Select one bounded operation.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Read /private/tmp/secret before pausing.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Contact user@example.com before pausing.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Connect to https://worker.internal now.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Use 192.0.2.1 for this operation.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Set API_KEY=privatevalue before pausing.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "ValueError: private provider failure.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Invoke bash --force for this turn.",
            }
        ),
        CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause.",
                "arguments": {"command": "run"},
            }
        ),
        _MarkdownProvider(),
    ],
)
def test_invalid_llm_output_never_publishes_proposal(tmp_path, provider) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    with pytest.raises(ExecutionAgentLLMResponseInvalid):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=_proposal_request(initial.execution.execution_digest),
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "invalid-test"}),
        )
    proposal_root = storage.project_dir("project-1") / "agent_execution_agent_proposals"
    assert not proposal_root.exists() or list(proposal_root.iterdir()) == []


def test_response_checkpoint_recovers_without_second_llm_call(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    phases: list[str] = []

    def fail_once(phase: str) -> None:
        phases.append(phase)
        if phase == "after_llm_response_checkpoint":
            raise RuntimeError("checkpoint crash")

    provider = _provider("agent.pause_current.v1")
    crashing = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=fail_once,
    )
    request = _proposal_request(initial.execution.execution_digest)
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        crashing.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    recovered = execution_agent_service(storage=storage, controller=controller)
    result = recovered.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert result.publication.proposal.selected_tool_id == "agent.pause_current.v1"
    assert provider.calls == 1


@pytest.mark.parametrize(
    "crash_phase",
    ["after_reservation", "after_observation_frozen"],
)
def test_pre_llm_checkpoint_crashes_resume_with_one_provider_call(
    tmp_path,
    crash_phase,
) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)

    def crash(phase: str) -> None:
        if phase == crash_phase:
            raise RuntimeError("pre-LLM crash")

    provider = _provider("agent.pause_current.v1")
    request = _proposal_request(initial.execution.execution_digest)
    crashing = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=crash,
    )
    with pytest.raises(RuntimeError, match="pre-LLM crash"):
        crashing.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    recovered = execution_agent_service(storage=storage, controller=controller)
    recovered.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert provider.calls == 1


@pytest.mark.parametrize(
    "crash_phase",
    [
        "after_execution_agent_proposal_file_2",
        "after_execution_agent_proposal_rename",
    ],
)
def test_proposal_publication_crash_replays_without_second_llm_call(
    tmp_path,
    crash_phase,
) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)

    def crash(phase: str) -> None:
        if phase == crash_phase:
            raise RuntimeError("proposal publication crash")

    provider = _provider("agent.pause_current.v1")
    request = _proposal_request(initial.execution.execution_digest)
    service = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=crash,
    )
    with pytest.raises(RuntimeError, match="proposal publication crash"):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    recovered = execution_agent_service(storage=storage, controller=controller)
    result = recovered.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=request,
        provider=provider,
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert result.publication.proposal.selected_tool_id == "agent.pause_current.v1"
    assert provider.calls == 1


class _StateDriftProvider(CountingStubProvider):
    def __init__(self, *, controller, initial) -> None:
        super().__init__(
            response={
                "selected_tool_id": "controller.advance_current.v1",
                "decision_summary": "Commit one bounded Controller action.",
            }
        )
        self.controller = controller
        self.initial = initial

    def complete_json(self, **kwargs: Any):
        result = super().complete_json(**kwargs)
        self.controller.advance(
            project_id="project-1",
            controller_execution_id=self.initial.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=(
                    self.initial.execution.execution_digest
                ),
                client_request_id="drift-during-llm-1",
            ),
        )
        return result


def test_controller_source_drift_after_llm_call_aborts_proposal(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    with pytest.raises(ExecutionAgentStale):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=_proposal_request(initial.execution.execution_digest),
            provider=_StateDriftProvider(controller=controller, initial=initial),
            provider_binding_digest=_agent_digest({"provider": "drift-test"}),
        )
    root = storage.project_dir("project-1") / "agent_execution_agent_proposals"
    assert not root.exists() or list(root.iterdir()) == []


def test_crash_after_llm_start_is_unknown_and_never_retries(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)

    def crash(phase: str) -> None:
        if phase == "after_llm_request_started":
            raise RuntimeError("request start crash")

    provider = _provider("agent.pause_current.v1")
    request = _proposal_request(initial.execution.execution_digest)
    service = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=crash,
    )
    with pytest.raises(RuntimeError, match="request start crash"):
        service.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    recovered = execution_agent_service(storage=storage, controller=controller)
    with pytest.raises(ExecutionAgentLLMOutcomeUnknown):
        recovered.create_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=request,
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
    assert provider.calls == 0


@pytest.mark.parametrize(
    ("crash_phase", "expected_outcome"),
    [
        ("before_controller_call", AgentToolCallApplicationOutcome.APPLIED),
        ("after_controller_advance", AgentToolCallApplicationOutcome.RECONCILED),
        ("before_application_receipt", AgentToolCallApplicationOutcome.RECONCILED),
        (
            "after_execution_agent_application_receipt_file_1",
            AgentToolCallApplicationOutcome.RECONCILED,
        ),
        (
            "after_execution_agent_application_receipt_rename",
            AgentToolCallApplicationOutcome.APPLIED,
        ),
        ("after_application_receipt", AgentToolCallApplicationOutcome.APPLIED),
    ],
)
def test_application_crash_boundaries_preserve_one_controller_effect(
    tmp_path,
    crash_phase,
    expected_outcome,
) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    base_service = execution_agent_service(storage=storage, controller=controller)
    proposed = base_service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider("controller.advance_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    request = AgentToolCallApplicationRequest(
        expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
        client_request_id="apply-crash-1",
    )

    def crash(phase: str) -> None:
        if phase == crash_phase:
            raise RuntimeError("application crash")

    before_receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    crashing = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=crash,
    )
    with pytest.raises(RuntimeError, match="application crash"):
        crashing.apply_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
            request=request,
        )
    recovered = execution_agent_service(storage=storage, controller=controller)
    result = recovered.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=request,
    )
    after_receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(after_receipts) == len(before_receipts) + 1
    assert result.application_receipt.outcome == expected_outcome


def test_manual_controller_advance_makes_proposal_stale(tmp_path) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider("controller.advance_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    controller.advance(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="manual-controller-advance-1",
        ),
    )
    proposal = proposed.publication.proposal
    with pytest.raises(ExecutionAgentConflict):
        service.apply_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
                client_request_id="stale-apply-1",
            ),
        )
    receipts = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert all(
        item.decision_id != proposal.tool_call_proposal_id for item in receipts
    )


def test_no_effect_receipt_is_published_under_controller_execution_lock(
    tmp_path,
) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    at_receipt = threading.Event()
    release_receipt = threading.Event()

    def hold_before_receipt(phase: str) -> None:
        if phase == "before_application_receipt":
            at_receipt.set()
            assert release_receipt.wait(timeout=10)

    service = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=hold_before_receipt,
    )
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider("agent.pause_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    manual_started = threading.Event()

    def apply_pause():
        return service.apply_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=(
                    proposal.tool_call_proposal_digest
                ),
                client_request_id="locked-pause-apply-1",
            ),
        )

    def manual_advance():
        manual_started.set()
        return controller.advance(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=(
                    initial.execution.execution_digest
                ),
                client_request_id="locked-manual-advance-1",
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        application_future = pool.submit(apply_pause)
        assert at_receipt.wait(timeout=10)
        controller_future = pool.submit(manual_advance)
        assert manual_started.wait(timeout=10)
        assert controller_future.done() is False
        release_receipt.set()
        application = application_future.result(timeout=10)
        controller_result = controller_future.result(timeout=10)
    receipt = application.application_receipt
    assert receipt.before_inspection_digest == receipt.after_inspection_digest
    assert receipt.before_inspection_digest == proposal.inspection_digest
    assert controller_result.receipt is not None


def test_proposal_byte_tampering_fails_exact_read(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider("agent.pause_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    path = (
        storage.project_dir("project-1")
        / "agent_execution_agent_proposals"
        / proposal.tool_call_proposal_id
        / "tool_call_proposal.json"
    )
    path.write_bytes(path.read_bytes().replace(b'"review_only"', b'"review_fake"'))
    with pytest.raises(
        (ExecutionAgentStoreConflict, ExecutionAgentStoreVerificationError)
    ):
        service.read_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
        )


class _SnapshotOnlyController:
    def __init__(self, snapshot, control_store) -> None:
        self.snapshot = snapshot
        self.control_store = control_store
        self.advance_calls = 0

    def read_execution_agent_snapshot(self, **_: Any):
        return self.snapshot

    @contextmanager
    def execution_agent_snapshot_session(self, **_: Any):
        yield self.snapshot

    def advance(self, **_: Any):  # pragma: no cover - test requires no call
        self.advance_calls += 1
        raise AssertionError("user-boundary tool called Controller.advance")


def _snapshot_with_action(
    initial: ControllerAdvanceResult,
    *,
    status: AgentHarnessControllerStatus,
    action: AgentHarnessControllerAction,
) -> ControllerAdvanceResult:
    prior = initial.inspection
    inspection = AgentHarnessControllerInspection(
        controller_execution_id=prior.controller_execution_id,
        controller_execution_digest=prior.controller_execution_digest,
        status=status,
        current_task_index=prior.current_task_index,
        current_task_id=prior.current_task_id,
        current_slot_id=prior.current_slot_id,
        next_action=action,
        facts=prior.facts,
        source_roster_digest=prior.source_roster_digest,
        inspected_at=prior.inspected_at,
    )
    return ControllerAdvanceResult(
        execution=initial.execution,
        inspection=inspection,
        receipt=initial.receipt,
    )


@pytest.mark.parametrize(
    ("status", "action", "tool_id"),
    [
        (
            AgentHarnessControllerStatus.WAITING_GATE,
            AgentHarnessControllerAction.WAIT_FOR_GATE,
            "user.request_gate_approval.v1",
        ),
        (
            AgentHarnessControllerStatus.WAITING_REMOTE_APPROVAL,
            AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
            "user.request_remote_approval.v1",
        ),
        (
            AgentHarnessControllerStatus.RECOVERY_REQUIRED,
            AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
            "user.request_recovery.v1",
        ),
    ],
)
def test_user_boundary_tools_only_publish_no_effect_receipts(
    tmp_path,
    status,
    action,
    tool_id,
) -> None:
    storage, control_store, _, initial = local_controller_execution(tmp_path)
    snapshot = _snapshot_with_action(initial, status=status, action=action)
    controller = _SnapshotOnlyController(snapshot, control_store)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(initial.execution.execution_digest),
        provider=_provider(tool_id),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    catalog, _ = build_execution_tool_catalog(snapshot)
    assert {item.tool_id for item in catalog.tools} == {
        "agent.pause_current.v1",
        tool_id,
    }
    proposal = proposed.publication.proposal
    applied = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="apply-user-boundary-1",
        ),
    )
    assert applied.application_receipt.outcome == (
        AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED
    )
    assert applied.application_receipt.controller_advance_called is False
    assert controller.advance_calls == 0


def test_user_boundary_receipt_is_adopted_after_snapshot_changes(tmp_path) -> None:
    storage, control_store, _, initial = local_controller_execution(tmp_path)
    gate_snapshot = _snapshot_with_action(
        initial,
        status=AgentHarnessControllerStatus.WAITING_GATE,
        action=AgentHarnessControllerAction.WAIT_FOR_GATE,
    )
    controller = _SnapshotOnlyController(gate_snapshot, control_store)

    def crash(phase: str) -> None:
        if phase == "after_application_receipt":
            raise RuntimeError("user-boundary application crash")

    crashing = execution_agent_service(
        storage=storage,
        controller=controller,
        fault_injector=crash,
    )
    proposed = crashing.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id="gate-boundary-crash-proposal-1",
        ),
        provider=_provider("user.request_gate_approval.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    with pytest.raises(RuntimeError, match="user-boundary application crash"):
        crashing.apply_proposal(
            project_id="project-1",
            controller_execution_id=initial.execution.controller_execution_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=(
                    proposal.tool_call_proposal_digest
                ),
                client_request_id="gate-boundary-crash-request-a",
            ),
        )
    assert crashing.store.read_committed_application_receipt(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    ) is None

    controller.snapshot = initial
    recovered = execution_agent_service(storage=storage, controller=controller)
    result = recovered.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            client_request_id="gate-boundary-recovery-request-b",
        ),
    )
    assert result.application_receipt.outcome == (
        AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED
    )
    assert result.application_receipt.before_inspection_digest == (
        result.application_receipt.after_inspection_digest
    )
    assert result.application_receipt.controller_advance_called is False
    assert result.application_receipt.dispatch_occurred is False
    assert controller.advance_calls == 0
    assert recovered.store.read_committed_application_receipt(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    ) == result.application_receipt


def test_sequential_turn_reaches_stable_terminal_observation(tmp_path) -> None:
    storage, _, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    first = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id="terminal-turn-1",
        ),
        provider=_provider("controller.advance_current.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    first_proposal = first.publication.proposal
    service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=first_proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=(
                first_proposal.tool_call_proposal_digest
            ),
            client_request_id="terminal-apply-1",
        ),
    )
    second = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=_proposal_request(
            initial.execution.execution_digest,
            request_id="terminal-turn-2",
        ),
        provider=_provider("agent.observe_terminal.v1"),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    assert {item.tool_id for item in second.publication.tool_catalog.tools} == {
        "agent.observe_terminal.v1",
        "agent.pause_current.v1",
    }
    second_proposal = second.publication.proposal
    observed = service.apply_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=second_proposal.tool_call_proposal_id,
        request=AgentToolCallApplicationRequest(
            expected_tool_call_proposal_digest=(
                second_proposal.tool_call_proposal_digest
            ),
            client_request_id="terminal-apply-2",
        ),
    )
    assert observed.application_receipt.outcome == (
        AgentToolCallApplicationOutcome.TERMINAL_OBSERVED
    )
    assert observed.application_receipt.controller_advance_called is False
