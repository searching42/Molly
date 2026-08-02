from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from ai4s_agent.schemas import (
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
    _agent_digest,
)
from tests.execution_agent_test_support import (
    CountingStubProvider,
    execution_agent_service,
    local_controller_execution,
    reopen_local_controller,
)


def _proposal_worker(
    workspace_dir: str,
    controller_execution_id: str,
    controller_execution_digest: str,
    ready,
    results,
) -> None:
    try:
        storage, controller = reopen_local_controller(Path(workspace_dir))
        service = execution_agent_service(storage=storage, controller=controller)
        provider = CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause this bounded turn.",
            }
        )
        ready.wait(timeout=10)
        proposed = service.create_proposal(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            request=AgentToolCallProposalRequest(
                expected_controller_execution_digest=controller_execution_digest,
                client_request_id="cross-process-proposal-1",
                external_llm_approved=True,
                llm_provider={"provider": "stub"},
            ),
            provider=provider,
            provider_binding_digest=_agent_digest({"provider": "stub"}),
        )
        results.put(
            (
                "ok",
                provider.calls,
                proposed.publication.proposal.tool_call_proposal_id,
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", type(exc).__name__, ""))


def _application_worker(
    workspace_dir: str,
    controller_execution_id: str,
    proposal_id: str,
    proposal_digest: str,
    client_request_id: str,
    ready,
    results,
) -> None:
    try:
        storage, controller = reopen_local_controller(Path(workspace_dir))
        service = execution_agent_service(storage=storage, controller=controller)
        ready.wait(timeout=10)
        applied = service.apply_proposal(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            tool_call_proposal_id=proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=proposal_digest,
                client_request_id=client_request_id,
            ),
        )
        results.put(
            (
                "ok",
                applied.application_receipt.application_receipt_id,
                applied.application_receipt.controller_receipt_id,
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", type(exc).__name__, ""))


@pytest.mark.pr_fast
def test_same_proposal_request_across_processes_calls_provider_once(tmp_path) -> None:
    storage, _, _, initial = local_controller_execution(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    args = (
        str(storage.workspace_dir),
        initial.execution.controller_execution_id,
        initial.execution.execution_digest,
        ready,
        results,
    )
    processes = [
        context.Process(target=_proposal_worker, args=args),
        context.Process(target=_proposal_worker, args=args),
    ]
    for process in processes:
        process.start()
    ready.set()
    observed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert all(item[0] == "ok" for item in observed)
    assert sum(int(item[1]) for item in observed) == 1
    assert len({item[2] for item in observed}) == 1


@pytest.mark.parametrize(
    "client_request_ids",
    [
        ("apply-process-same", "apply-process-same"),
        ("apply-process-a", "apply-process-b"),
    ],
)
def test_application_requests_across_processes_have_one_effect(
    tmp_path,
    client_request_ids,
) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentToolCallProposalRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="application-concurrency-proposal-1",
            external_llm_approved=True,
            llm_provider={"provider": "stub"},
        ),
        provider=CountingStubProvider(
            response={
                "selected_tool_id": "controller.advance_current.v1",
                "decision_summary": "Commit one bounded Controller action.",
            }
        ),
        provider_binding_digest=_agent_digest({"provider": "stub"}),
    )
    proposal = proposed.publication.proposal
    before = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_application_worker,
            args=(
                str(storage.workspace_dir),
                initial.execution.controller_execution_id,
                proposal.tool_call_proposal_id,
                proposal.tool_call_proposal_digest,
                client_request_ids[index],
                ready,
                results,
            ),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    ready.set()
    observed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert all(item[0] == "ok" for item in observed)
    assert len({item[1] for item in observed}) == 1
    assert len({item[2] for item in observed}) == 1
    after = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(after) == len(before) + 1
    application_receipts = service.store.application_receipts_for_proposal(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert len(application_receipts) == 1
