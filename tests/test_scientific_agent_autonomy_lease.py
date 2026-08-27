from __future__ import annotations

from dataclasses import dataclass
import json
from multiprocessing import Barrier, Queue, get_context
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent.app import create_app
from ai4s_agent.schemas import (
    AgentHarnessControllerAction,
    AutonomyGrant,
    AutonomyLeaseV1,
    _agent_digest,
)
from ai4s_agent.scientific_agent_autonomy_lease import (
    AUTONOMY_LEASE_POLICY_DIGEST,
    AUTONOMY_LEASE_POLICY_VERSION,
    AutonomyLeaseActiveBudgetExhausted,
    AutonomyLeaseConflict,
    AutonomyLeaseError,
    AutonomyLeaseExpired,
    AutonomyLeaseReconciliationRequired,
    AutonomyLeaseRemoteBudgetExhausted,
    AutonomyLeaseRemoteBudgetEnforcementUnavailable,
    AutonomyLeaseService,
    AutonomyLeaseStale,
)
from ai4s_agent.scientific_agent_failure_recovery_runtime import (
    ScientificAgentAutonomyGrantStore,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.pr_fast


@dataclass
class _FakeClock:
    value: str

    def __call__(self) -> str:
        return self.value


@dataclass
class _FakeMonotonic:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _fixed_clock() -> str:
    return "2026-08-27T00:00:00Z"


def _grant(
    *,
    valid_from: str = "2026-08-26T00:00:00Z",
    valid_until: str = "2099-01-01T00:00:00Z",
    active_seconds: float = 10.0,
    remote_seconds: float = 10.0,
) -> AutonomyGrant:
    return AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["task-1"],
        aggregate_budget={
            "active_execution_seconds": active_seconds,
            "remote_runtime_seconds": remote_seconds,
        },
        valid_from=valid_from,
        valid_until=valid_until,
        created_at="2026-08-26T00:00:00Z",
    )


def _fixture(
    tmp_path: Path,
    *,
    grant: AutonomyGrant | None = None,
    clock: _FakeClock | None = None,
    lease_ttl_seconds: float = 3600,
) -> tuple[
    ProjectStorage,
    ScientificAgentAutonomyGrantStore,
    AutonomyLeaseService,
    AutonomyGrant,
    _FakeClock,
    _FakeMonotonic,
]:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project(
        "project-1",
        name="Project",
        created_at="2026-08-26T00:00:00Z",
    )
    selected_grant = grant or _grant()
    selected_clock = clock or _FakeClock("2026-08-27T00:00:00Z")
    grant_store = ScientificAgentAutonomyGrantStore(
        storage=storage,
        clock=selected_clock,
    )
    grant_store.publish_server_grant(
        grant=selected_grant,
        authority_epoch="epoch-1",
        actor="owner-1",
        actor_source="server:test",
        run_id="run-1",
    )
    monotonic = _FakeMonotonic()
    service = AutonomyLeaseService(
        storage=storage,
        grant_source=grant_store,
        lease_ttl_seconds=lease_ttl_seconds,
        operation_reservation_seconds=5,
        remote_operation_reservation_seconds=5,
        clock=selected_clock,
        monotonic=monotonic,
    )
    return storage, grant_store, service, selected_grant, selected_clock, monotonic


def _controller_pair(
    *,
    action: str = "execute_local_task",
    decision_id: str = "decision-1",
    controller_execution_id: str = "controller-1",
):
    execution = SimpleNamespace(
        project_id="project-1",
        run_id="run-1",
        controller_execution_id=controller_execution_id,
        conversation_id="conversation-1",
        task_slots=[SimpleNamespace(task_id="task-1")],
    )
    decision = SimpleNamespace(
        decision_id=decision_id,
        action_kind=action,
        executable=True,
        task_index=0,
    )
    return execution, decision


def _reserve_worker(workspace: str, barrier: Barrier, results: Queue, operation_id: str) -> None:
    storage = ProjectStorage(workspace_dir=Path(workspace))
    grant_store = ScientificAgentAutonomyGrantStore(
        storage=storage,
        clock=_fixed_clock,
    )
    service = AutonomyLeaseService(
        storage=storage,
        grant_source=grant_store,
        lease_ttl_seconds=3600,
        operation_reservation_seconds=1,
        remote_operation_reservation_seconds=1,
        clock=_fixed_clock,
    )
    barrier.wait(10)
    try:
        reservation = service.reserve_usage(
            project_id="project-1",
            run_id="run-1",
            operation_id=operation_id,
            controller_execution_id=f"controller-{operation_id}",
            usage_kind="ACTIVE_EXECUTION",
            reserved_seconds=1,
        )
    except AutonomyLeaseError as exc:
        results.put(("error", exc.reason_code))
    else:
        results.put(("ok", reservation.operation_id))


def _crash_after_reservation_worker(workspace: str, results: Queue) -> None:
    storage = ProjectStorage(workspace_dir=Path(workspace))
    grant_store = ScientificAgentAutonomyGrantStore(
        storage=storage,
        clock=_fixed_clock,
    )
    service = AutonomyLeaseService(
        storage=storage,
        grant_source=grant_store,
        lease_ttl_seconds=3600,
        operation_reservation_seconds=1,
        remote_operation_reservation_seconds=1,
        clock=_fixed_clock,
    )
    reservation = service.reserve_usage(
        project_id="project-1",
        run_id="run-1",
        operation_id="crashed-operation",
        controller_execution_id="controller-crashed",
        usage_kind="ACTIVE_EXECUTION",
        reserved_seconds=1,
    )
    results.put((reservation.lease_id, reservation.operation_id))


def test_lease_contract_is_closed_world_and_digest_bound(tmp_path: Path) -> None:
    storage, grant_store, service, grant, _clock, _monotonic = _fixture(tmp_path)
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    replay = AutonomyLeaseV1.model_validate(lease.model_dump(mode="json"))
    assert replay == lease
    assert replay.lease_digest == lease.lease_digest
    assert lease.policy_version == AUTONOMY_LEASE_POLICY_VERSION
    assert lease.policy_digest == AUTONOMY_LEASE_POLICY_DIGEST

    with pytest.raises(Exception):
        AutonomyLeaseV1.model_validate(
            {
                **lease.model_dump(mode="json"),
                "unexpected": True,
            }
        )
    with pytest.raises(Exception):
        AutonomyLeaseV1.model_validate(
            {
                **lease.model_dump(mode="json"),
                "lease_id": "",
                "lease_digest": "",
                "valid_from": "2026-08-28T00:00:00Z",
                "valid_until": "2026-08-27T00:00:00Z",
            }
        )
    with pytest.raises(Exception):
        AutonomyLeaseV1.model_validate(
            {
                **lease.model_dump(mode="json"),
                "lease_id": "",
                "lease_digest": "",
                "max_active_execution_seconds": -1,
            }
        )

    foreign = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["task-foreign"],
        aggregate_budget={
            "active_execution_seconds": 10,
            "remote_runtime_seconds": 10,
        },
        valid_until="2099-01-01T00:00:00Z",
        created_at="2026-08-26T00:00:00Z",
    )
    with pytest.raises(AutonomyLeaseStale):
        service._validate_lease_against_grant(
            lease=lease,
            grant=foreign,
            authority_epoch="epoch-1",
        )
    with pytest.raises(AutonomyLeaseStale):
        service.verify_current_lease(
            project_id="project-1",
            run_id="run-1",
            grant_id="foreign-grant",
            usage_kind="ACTIVE_EXECUTION",
        )


def test_historical_grant_binding_without_new_provenance_remains_readable(
    tmp_path: Path,
) -> None:
    storage, grant_store, service, _grant_value, _clock, _monotonic = _fixture(tmp_path)
    path = storage.project_dir("project-1") / "agent-autonomy" / "autonomy_grants.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["grants"]:
        record.pop("actor", None)
        record.pop("actor_source", None)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    binding = grant_store.resolve_current(project_id="project-1", run_id="run-1")
    assert binding is not None
    assert binding.actor == ""
    assert binding.actor_source == ""
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    assert lease.created_by_source == "server:autonomy-lease"


def test_lease_validity_uses_half_open_window_and_future_evidence(tmp_path: Path) -> None:
    clock = _FakeClock("2026-08-26T23:59:59Z")
    grant = _grant(
        valid_from="2026-08-27T00:00:00Z",
        valid_until="2026-08-27T00:00:10Z",
    )
    _storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path,
        grant=grant,
        clock=clock,
    )
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    assert service.read_budget_evidence(
        project_id="project-1",
        run_id="run-1",
        lease_id=lease.lease_id,
    ).validity_status == "NOT_YET_VALID"
    with pytest.raises(AutonomyLeaseError) as not_yet:
        service.verify_current_lease(
            project_id="project-1",
            run_id="run-1",
            usage_kind="ACTIVE_EXECUTION",
        )
    assert not_yet.value.reason_code == "AUTONOMY_LEASE_NOT_YET_VALID"

    clock.value = "2026-08-27T00:00:00Z"
    service.verify_current_lease(
        project_id="project-1",
        run_id="run-1",
        usage_kind="ACTIVE_EXECUTION",
    )
    clock.value = "2026-08-27T00:00:10Z"
    with pytest.raises(AutonomyLeaseExpired):
        service.verify_current_lease(
            project_id="project-1",
            run_id="run-1",
            usage_kind="ACTIVE_EXECUTION",
        )


def test_lease_limits_are_narrower_than_grant_and_action_roster_is_closed(
    tmp_path: Path,
) -> None:
    grant_value = _grant(active_seconds=10, remote_seconds=20)
    _storage, _grant_store, service, _grant_result, _clock, _monotonic = _fixture(
        tmp_path,
        grant=grant_value,
    )
    service_with_tighter_caps = AutonomyLeaseService(
        storage=service.storage,
        grant_source=service.grant_source,
        lease_ttl_seconds=3600,
        max_active_execution_seconds=3,
        max_remote_runtime_seconds=4,
        operation_reservation_seconds=5,
        remote_operation_reservation_seconds=5,
        clock=service.clock,
    )
    lease = service_with_tighter_caps.ensure_current_lease(
        project_id="project-1",
        run_id="run-1",
    )
    assert lease.max_active_execution_seconds == 3
    assert lease.max_remote_runtime_seconds == 4
    assert service.usage_kind_for_controller_action(
        AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
    ) == "ACTIVE_EXECUTION"
    assert service.usage_kind_for_controller_action(
        AgentHarnessControllerAction.ADOPT_COMPLETED_TASK
    ) == "ACTIVE_EXECUTION"
    for action in (
        AgentHarnessControllerAction.WAIT_FOR_GATE,
        AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
        AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
        AgentHarnessControllerAction.CANCEL_EXECUTION,
        AgentHarnessControllerAction.COMPLETE_EXECUTION,
    ):
        assert service.usage_kind_for_controller_action(action) is None


def test_expired_lease_denies_effect_at_controller_boundary(tmp_path: Path) -> None:
    _storage, _grant_store, service, _grant_value, clock, _monotonic = _fixture(
        tmp_path,
    )
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    clock.value = lease.valid_until
    execution, decision = _controller_pair()
    with pytest.raises(AutonomyLeaseExpired):
        service.begin_controller_effect(execution=execution, decision=decision)
    assert service.store.list_reservations(
        project_id="project-1",
        lease_id=lease.lease_id,
    ) == []


def test_expired_lease_blocks_deterministic_fastpath_before_adoption(
    tmp_path: Path,
) -> None:
    _storage, _grant_store, service, _grant_value, clock, _monotonic = _fixture(
        tmp_path
    )
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    clock.value = lease.valid_until
    execution, decision = _controller_pair(
        action=AgentHarnessControllerAction.ADOPT_COMPLETED_TASK.value,
        decision_id="deterministic-fastpath-adoption",
    )
    with pytest.raises(AutonomyLeaseExpired):
        service.begin_controller_effect(execution=execution, decision=decision)
    assert service.store.list_reservations(
        project_id="project-1",
        lease_id=lease.lease_id,
    ) == []


def test_started_controller_effect_exception_persists_unknown_effect(
    tmp_path: Path,
) -> None:
    storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path
    )
    execution, decision = _controller_pair()
    operation = service.begin_controller_effect(execution=execution, decision=decision)
    assert operation is not None

    # The production Controller exception path calls this after begin_effect
    # has already published its immutable STARTED checkpoint.
    service.mark_unknown_effect(reservation=operation.reservation)
    reconciliation_path = (
        storage.project_dir("project-1")
        / "agent-autonomy-leases"
        / "reconciliations"
        / operation.reservation.lease_id
        / f"{operation.reservation.operation_id}.json"
    )
    assert json.loads(reconciliation_path.read_text(encoding="utf-8"))["effect_state"] == "UNKNOWN_EFFECT"
    with pytest.raises(AutonomyLeaseReconciliationRequired) as blocked:
        service.reconcile_controller_effect(
            execution=execution,
            decision=decision,
        )
    assert blocked.value.reason_code == "AUTONOMY_LEASE_RECONCILIATION_REQUIRED"
    assert service.store.list_receipts(
        project_id="project-1",
        lease_id=operation.reservation.lease_id,
    ) == []


@pytest.mark.parametrize(
    "action",
    [
        AgentHarnessControllerAction.DISPATCH_REMOTE_TASK,
        AgentHarnessControllerAction.REFRESH_REMOTE_TASK,
        AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS,
    ],
)
def test_remote_controller_effect_fails_closed_without_runtime_evidence(
    tmp_path: Path,
    action: AgentHarnessControllerAction,
) -> None:
    _storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path
    )
    execution, decision = _controller_pair(action=action.value)
    with pytest.raises(AutonomyLeaseRemoteBudgetEnforcementUnavailable) as blocked:
        service.begin_controller_effect(execution=execution, decision=decision)
    assert blocked.value.reason_code == (
        "AUTONOMY_REMOTE_BUDGET_ENFORCEMENT_UNAVAILABLE"
    )
    assert service.store.list_leases(project_id="project-1") == []


def test_lease_and_recovery_flags_are_independent_for_grant_and_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI4S_AGENT_AUTONOMY_LEASE_ENABLED", "true")
    monkeypatch.setenv("AI4S_AGENT_FAILURE_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("AI4S_AGENT_FAILURE_RECOVERY_MAX_RETRIES", "0")
    monkeypatch.setenv("AI4S_AGENT_FAILURE_RECOVERY_MAX_REPLANS", "0")
    monkeypatch.setenv("AI4S_AGENT_AUTONOMY_MAX_ACTIVE_EXECUTION_SECONDS", "1")
    monkeypatch.setenv("AI4S_AGENT_AUTONOMY_OPERATION_RESERVATION_SECONDS", "1")
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
    )
    issuer = app.extensions["scientific_agent_autonomy_grant_issuer"]
    assert issuer.enabled is True
    assert issuer.max_retries == 0
    assert issuer.max_replans == 0
    assert app.extensions["scientific_agent_failure_recovery_grant_issuer"] is issuer
    conversation_service = app.extensions[
        "scientific_agent_conversation_session_service"
    ]
    assert conversation_service.controller.autonomy_lease_service is app.extensions[
        "scientific_agent_autonomy_lease_service"
    ]

    created_at = now_iso()
    authorization_digest = _agent_digest({"test": "lease-without-recovery"})
    authorization = SimpleNamespace(
        project_id="project-1",
        run_id="run-1",
        authorization_id="authorization-1",
        authorization_digest=authorization_digest,
        actor="alice",
        task_ids=["task-1"],
        profile_bindings=[],
        compiled_task_options={},
        created_at=created_at,
    )
    start_intent = SimpleNamespace(
        project_id="project-1",
        run_id="run-1",
        authorization_id="authorization-1",
        authorization_digest=authorization_digest,
    )
    binding = issuer.issue_from_approved_chain(
        SimpleNamespace(authorization=authorization, start_intent=start_intent)
    )
    assert binding is not None
    lease = app.extensions["scientific_agent_autonomy_lease_service"].ensure_current_lease(
        project_id="project-1",
        run_id="run-1",
    )
    assert lease.grant_id == binding.grant.grant_id
    controller = conversation_service.controller
    execution, first_decision = _controller_pair(
        decision_id="lease-without-recovery-effect-1",
        controller_execution_id="lease-without-recovery-controller-1",
    )
    first = controller.autonomy_lease_service.begin_controller_effect(
        execution=execution,
        decision=first_decision,
    )
    assert first is not None
    _second_execution, second_decision = _controller_pair(
        decision_id="lease-without-recovery-effect-2",
        controller_execution_id="lease-without-recovery-controller-2",
    )
    with pytest.raises(AutonomyLeaseActiveBudgetExhausted):
        controller.autonomy_lease_service.begin_controller_effect(
            execution=execution,
            decision=second_decision,
        )


def test_human_waiting_is_not_active_usage(tmp_path: Path) -> None:
    storage, _grant_store, service, _grant_value, clock, monotonic = _fixture(
        tmp_path,
        lease_ttl_seconds=4 * 3600,
    )
    execution, decision = _controller_pair()
    operation = service.begin_controller_effect(execution=execution, decision=decision)
    assert operation is not None
    monotonic.value = 2.0
    service.finish_controller_effect(operation=operation, reconcile_only=False)
    clock.value = "2026-08-27T03:00:00Z"
    evidence = service.read_budget_evidence(
        project_id="project-1",
        run_id="run-1",
    )
    assert evidence.active_execution_seconds_used == pytest.approx(2.0)
    assert evidence.remote_runtime_seconds_used == 0
    assert evidence.validity_status == "ACTIVE"
    # A later wall-clock expiry is independent of the already committed
    # active interval; waiting time cannot be retroactively charged.
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    clock.value = lease.valid_until
    assert service.read_budget_evidence(
        project_id="project-1",
        run_id="run-1",
        lease_id=lease.lease_id,
    ).validity_status == "EXPIRED"


def test_known_effect_reconciles_once_after_receipt_crash(tmp_path: Path) -> None:
    _storage, _grant_store, service, _grant, _clock, _monotonic = _fixture(tmp_path)
    execution, decision = _controller_pair()
    operation = service.begin_controller_effect(execution=execution, decision=decision)
    assert operation is not None
    first = service.reconcile_controller_effect(
        execution=execution,
        decision=decision,
    )
    second = service.reconcile_controller_effect(
        execution=execution,
        decision=decision,
    )
    assert first is not None
    assert second == first
    assert first.usage_seconds == operation.reservation.reserved_seconds
    assert len(
        service.store.list_receipts(
            project_id="project-1",
            lease_id=operation.reservation.lease_id,
        )
    ) == 1


def test_started_effect_checkpoint_is_not_reclaimed_as_not_started(
    tmp_path: Path,
) -> None:
    storage, _grant_store, service, _grant, _clock, _monotonic = _fixture(tmp_path)
    execution, decision = _controller_pair()
    operation = service.begin_controller_effect(execution=execution, decision=decision)
    assert operation is not None
    start_path = (
        storage.project_dir("project-1")
        / "agent-autonomy-leases"
        / "starts"
        / operation.reservation.lease_id
        / f"{operation.reservation.operation_id}.json"
    )
    assert json.loads(start_path.read_text(encoding="utf-8"))["effect_state"] == "STARTED"
    reservation_path = (
        storage.project_dir("project-1")
        / "agent-autonomy-leases"
        / "reservations"
        / operation.reservation.lease_id
        / f"{operation.reservation.operation_id}.json"
    )
    reservation_payload = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation_payload["owner_pid"] = 999_999_991
    reservation_path.write_text(
        json.dumps(reservation_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert service.reclaim_orphaned_reservations(
        project_id="project-1",
        lease_id=operation.reservation.lease_id,
    ) == 0
    with pytest.raises(AutonomyLeaseReconciliationRequired):
        service.reserve_usage(
            project_id="project-1",
            run_id="run-1",
            operation_id=operation.reservation.operation_id,
            controller_execution_id=execution.controller_execution_id,
            task_id="task-1",
            usage_kind="ACTIVE_EXECUTION",
            reserved_seconds=5,
        )


def test_unknown_effect_and_conflicting_replay_fail_closed(tmp_path: Path) -> None:
    _storage, _grant_store, service, _grant, clock, _monotonic = _fixture(tmp_path)
    reservation = service.reserve_usage(
        project_id="project-1",
        run_id="run-1",
        operation_id="operation-1",
        controller_execution_id="controller-1",
        usage_kind="ACTIVE_EXECUTION",
        reserved_seconds=2,
    )
    service.mark_unknown_effect(reservation=reservation)
    with pytest.raises(AutonomyLeaseReconciliationRequired):
        service.reserve_usage(
            project_id="project-1",
            run_id="run-1",
            operation_id="operation-1",
            controller_execution_id="controller-1",
            usage_kind="ACTIVE_EXECUTION",
            reserved_seconds=2,
        )

    # A committed receipt is exact-replayable, but the same operation cannot
    # be rebound to different usage bytes.
    second = service.reserve_usage(
        project_id="project-1",
        run_id="run-1",
        operation_id="operation-2",
        controller_execution_id="controller-2",
        usage_kind="ACTIVE_EXECUTION",
        reserved_seconds=2,
    )
    clock.value = "2026-08-27T00:00:01Z"
    receipt = service.commit_usage(
        reservation=second,
        usage_seconds=1,
        started_at="2026-08-27T00:00:00Z",
        ended_at="2026-08-27T00:00:00Z",
    )
    assert service.commit_usage(
        reservation=second,
        usage_seconds=1,
        started_at=receipt.started_at,
        ended_at=receipt.ended_at,
    ) == receipt
    with pytest.raises(AutonomyLeaseConflict):
        service.commit_usage(
            reservation=second,
            usage_seconds=0.5,
            started_at=receipt.started_at,
            ended_at=receipt.ended_at,
        )


def test_multiprocess_reservation_is_atomic(tmp_path: Path) -> None:
    grant_value = _grant(active_seconds=1, remote_seconds=10)
    _storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path,
        grant=grant_value,
        clock=_FakeClock("2026-08-27T00:00:00Z"),
    )
    workspace = str(tmp_path / "workspace")
    context = get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_worker,
            args=(workspace, barrier, results, f"operation-{index}"),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    observed = sorted(results.get(timeout=5) for _ in processes)
    assert observed.count(("ok", "operation-0")) + observed.count(
        ("ok", "operation-1")
    ) == 1
    assert observed.count(("error", "AUTONOMY_ACTIVE_BUDGET_EXHAUSTED")) == 1
    # A failed competitor never reached an effect boundary; only one
    # reservation is authoritative.
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    assert len(
        service.store.list_reservations(
            project_id="project-1",
            lease_id=lease.lease_id,
        )
    ) == 1


def test_restart_reclaims_dead_reservation_without_spending_budget(tmp_path: Path) -> None:
    grant_value = _grant(active_seconds=1, remote_seconds=10)
    _storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path,
        grant=grant_value,
        clock=_FakeClock("2026-08-27T00:00:00Z"),
    )
    lease = service.ensure_current_lease(project_id="project-1", run_id="run-1")
    context = get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_crash_after_reservation_worker,
        args=(str(tmp_path / "workspace"), results),
    )
    process.start()
    process.join(20)
    assert process.exitcode == 0
    child_lease_id, operation_id = results.get(timeout=5)
    assert child_lease_id == lease.lease_id
    assert operation_id == "crashed-operation"
    assert service.reclaim_orphaned_reservations(
        project_id="project-1",
        lease_id=lease.lease_id,
    ) == 1
    replacement = service.reserve_usage(
        project_id="project-1",
        run_id="run-1",
        operation_id="replacement-operation",
        controller_execution_id="controller-replacement",
        usage_kind="ACTIVE_EXECUTION",
        reserved_seconds=1,
    )
    assert replacement.reserved_seconds == 1
    evidence = service.read_budget_evidence(
        project_id="project-1",
        run_id="run-1",
    )
    assert evidence.active_execution_seconds_used == 0
    assert evidence.active_execution_seconds_reserved == 1


def test_remote_runtime_has_independent_budget(tmp_path: Path) -> None:
    grant_value = _grant(active_seconds=10, remote_seconds=2)
    _storage, _grant_store, service, _grant_value, _clock, _monotonic = _fixture(
        tmp_path,
        grant=grant_value,
    )
    service.record_remote_runtime(
        project_id="project-1",
        run_id="run-1",
        operation_id="remote-1",
        controller_execution_id="controller-remote-1",
        task_id="task-1",
        started_at="2026-08-27T00:00:00Z",
        ended_at="2026-08-27T00:00:02Z",
    )
    with pytest.raises(AutonomyLeaseRemoteBudgetExhausted):
        service.record_remote_runtime(
            project_id="project-1",
            run_id="run-1",
            operation_id="remote-2",
            controller_execution_id="controller-remote-2",
            task_id="task-1",
            started_at="2026-08-27T00:00:02Z",
            ended_at="2026-08-27T00:00:03Z",
        )
    evidence = service.read_budget_evidence(
        project_id="project-1",
        run_id="run-1",
    )
    assert evidence.active_execution_seconds_remaining == 10
    assert evidence.remote_runtime_seconds_used == 2
    with pytest.raises(AutonomyLeaseConflict):
        service.record_remote_runtime(
            project_id="project-1",
            run_id="run-1",
            operation_id="remote-1",
            controller_execution_id="controller-remote-1",
            task_id="task-1",
            started_at="2026-08-27T00:00:01Z",
            ended_at="2026-08-27T00:00:03Z",
        )


def test_forged_reservation_bytes_fail_closed(tmp_path: Path) -> None:
    storage, _grant_store, service, _grant, _clock, _monotonic = _fixture(tmp_path)
    reservation = service.reserve_usage(
        project_id="project-1",
        run_id="run-1",
        operation_id="operation-forge",
        controller_execution_id="controller-forge",
        usage_kind="ACTIVE_EXECUTION",
        reserved_seconds=2,
    )
    path = (
        storage.project_dir("project-1")
        / "agent-autonomy-leases"
        / "reservations"
        / reservation.lease_id
        / f"{reservation.operation_id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reserved_seconds"] = 10
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AutonomyLeaseConflict):
        service.read_budget_evidence(project_id="project-1", run_id="run-1")
