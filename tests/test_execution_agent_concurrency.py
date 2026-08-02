from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from ai4s_agent.schemas import (
    AgentHarnessControllerAdvanceRequest,
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
                applied.application_receipt.outcome.value,
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", type(exc).__name__, ""))


def _crashing_application_worker(
    workspace_dir: str,
    controller_execution_id: str,
    proposal_id: str,
    proposal_digest: str,
    results,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_controller_advance":
            raise RuntimeError("application crash")

    try:
        storage, controller = reopen_local_controller(Path(workspace_dir))
        service = execution_agent_service(
            storage=storage,
            controller=controller,
            fault_injector=crash,
        )
        service.apply_proposal(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            tool_call_proposal_id=proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=proposal_digest,
                client_request_id="apply-crash-request-a",
            ),
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", type(exc).__name__, str(exc)))


def _crashing_no_effect_application_worker(
    workspace_dir: str,
    controller_execution_id: str,
    proposal_id: str,
    proposal_digest: str,
    results,
) -> None:
    def crash(phase: str) -> None:
        if phase == "after_application_receipt":
            raise RuntimeError("no-effect application crash")

    try:
        storage, controller = reopen_local_controller(Path(workspace_dir))
        service = execution_agent_service(
            storage=storage,
            controller=controller,
            fault_injector=crash,
        )
        service.apply_proposal(
            project_id="project-1",
            controller_execution_id=controller_execution_id,
            tool_call_proposal_id=proposal_id,
            request=AgentToolCallApplicationRequest(
                expected_tool_call_proposal_digest=proposal_digest,
                client_request_id="no-effect-crash-request-a",
            ),
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", type(exc).__name__, str(exc)))


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


@pytest.mark.pr_fast
def test_different_request_new_process_reconciles_proposal_scoped_effect(
    tmp_path,
) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentToolCallProposalRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="cross-request-crash-proposal-1",
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
    crashed = context.Queue()
    first = context.Process(
        target=_crashing_application_worker,
        args=(
            str(storage.workspace_dir),
            initial.execution.controller_execution_id,
            proposal.tool_call_proposal_id,
            proposal.tool_call_proposal_digest,
            crashed,
        ),
    )
    first.start()
    first_result = crashed.get(timeout=30)
    first.join(timeout=30)
    assert first.exitcode == 0
    assert first_result[:2] == ("error", "RuntimeError")

    # This unrelated corrupt publication must not be scanned while resolving
    # or applying the proposal-scoped receipt authority.
    corrupt = (
        storage.project_dir("project-1")
        / "agent_execution_agent_application_receipts"
        / "unrelated-corrupt-receipt"
    )
    corrupt.mkdir(parents=True)
    (corrupt / "application_receipt.json").write_text("{", encoding="utf-8")

    ready = context.Event()
    recovered = context.Queue()
    second = context.Process(
        target=_application_worker,
        args=(
            str(storage.workspace_dir),
            initial.execution.controller_execution_id,
            proposal.tool_call_proposal_id,
            proposal.tool_call_proposal_digest,
            "apply-recovery-request-b",
            ready,
            recovered,
        ),
    )
    second.start()
    ready.set()
    second_result = recovered.get(timeout=30)
    second.join(timeout=30)
    assert second.exitcode == 0
    assert second_result[0] == "ok"
    assert second_result[3] == "reconciled"

    reopened_storage, reopened_controller = reopen_local_controller(
        storage.workspace_dir
    )
    reopened_service = execution_agent_service(
        storage=reopened_storage,
        controller=reopened_controller,
    )
    pointer_receipt = reopened_service.store.read_committed_application_receipt(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert pointer_receipt is not None
    assert pointer_receipt.application_receipt_id == second_result[1]
    assert pointer_receipt.controller_receipt_id == second_result[2]
    application_root = (
        storage.project_dir("project-1")
        / "agent_execution_agent_applications"
        / proposal.tool_call_proposal_id
    )
    assert (application_root / "controller_call_started.json").is_file()
    assert (application_root / "controller_effect_observed.json").is_file()
    assert (application_root / "application_receipt_committed.json").is_file()
    read = reopened_service.read_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert read.applied is True
    after = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(after) == len(before) + 1


@pytest.mark.pr_fast
def test_no_effect_receipt_is_adopted_after_controller_advances(
    tmp_path,
) -> None:
    storage, control_store, controller, initial = local_controller_execution(tmp_path)
    service = execution_agent_service(storage=storage, controller=controller)
    proposed = service.create_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentToolCallProposalRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="no-effect-cross-request-proposal-1",
            external_llm_approved=True,
            llm_provider={"provider": "stub"},
        ),
        provider=CountingStubProvider(
            response={
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause this bounded turn.",
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
    crashed = context.Queue()
    first = context.Process(
        target=_crashing_no_effect_application_worker,
        args=(
            str(storage.workspace_dir),
            initial.execution.controller_execution_id,
            proposal.tool_call_proposal_id,
            proposal.tool_call_proposal_digest,
            crashed,
        ),
    )
    first.start()
    first_result = crashed.get(timeout=30)
    first.join(timeout=30)
    assert first.exitcode == 0
    assert first_result[:2] == ("error", "RuntimeError")
    assert service.store.read_committed_application_receipt(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    ) is None

    controller.advance(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=initial.execution.execution_digest,
            client_request_id="manual-advance-after-no-effect-crash",
        ),
    )

    ready = context.Event()
    recovered = context.Queue()
    second = context.Process(
        target=_application_worker,
        args=(
            str(storage.workspace_dir),
            initial.execution.controller_execution_id,
            proposal.tool_call_proposal_id,
            proposal.tool_call_proposal_digest,
            "no-effect-recovery-request-b",
            ready,
            recovered,
        ),
    )
    second.start()
    ready.set()
    second_result = recovered.get(timeout=30)
    second.join(timeout=30)
    assert second.exitcode == 0
    assert second_result[0] == "ok"
    assert second_result[3] == "paused"

    reopened_storage, reopened_controller = reopen_local_controller(
        storage.workspace_dir
    )
    reopened_service = execution_agent_service(
        storage=reopened_storage,
        controller=reopened_controller,
    )
    pointer_receipt = reopened_service.store.read_committed_application_receipt(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert pointer_receipt is not None
    assert pointer_receipt.application_receipt_id == second_result[1]
    assert pointer_receipt.controller_advance_called is False
    assert pointer_receipt.dispatch_occurred is False
    read = reopened_service.read_proposal(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert read.applied is True
    assert read.stale is True
    after = control_store.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=initial.execution.controller_execution_id,
    )
    assert len(after) == len(before) + 1
    application_receipts = reopened_service.store.application_receipts_for_proposal(
        project_id="project-1",
        tool_call_proposal_id=proposal.tool_call_proposal_id,
    )
    assert [item.application_receipt_id for item in application_receipts] == [
        pointer_receipt.application_receipt_id
    ]
